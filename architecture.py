#!/usr/bin/env python3
"""
Class for large network with multiple branches, cost functions, training stages.
"""

import keras
from keras.models import Model
from keras.layers import Input, merge, Dense
from keras.optimizers import Adam, Adadelta, RMSprop, SGD
from keras.utils.visualize_util import plot
import tensorflow as tf
import keras.backend as K
import numpy as np
from tqdm import tqdm

import simple_network as network_params


MODELS_FOLDER = 'models'
REPLAY_RATE = 0.08
RECORD_RATE = 0.0005


def make_trainable(model, trainable):
    """
    Freeze or unfreeze network weights
    :param model: particular model (layers of neural net)
    :param trainable: if true freeze, else unfreeze
    :return: None
    """
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable
        if type(l) == type(model):
            make_trainable(l, trainable)


def metric_acc(x, y):
    correct = (x > 0) == (y > 0)
    acc = np.mean(correct)
    return acc


def loss_wasserstein(y_true, y_pred):
    loss = K.mean(tf.mul(y_true, y_pred))
    return loss


def loss_diff(y_true, y_pred):
    # true gradients
    grad_true_y = y_true[:-1, :] - y_true[1:, :]
    grad_true_x = y_true[:, :-1] - y_true[:, 1:]
    grad_pred_y = y_pred[:-1, :] - y_pred[1:, :]
    grad_pred_x = y_pred[:, :-1] - y_pred[:, 1:]

    grad_cost = tf.reduce_mean(tf.abs(grad_true_x - grad_pred_x)) + tf.reduce_mean(tf.abs(grad_true_y - grad_pred_y))

    return grad_cost


def sq_diff_loss(y_true, y_pred):
    mse = keras.objectives.mean_squared_error(y_true, y_pred)
    grad = loss_diff(y_true, y_pred)

    return 0.9*mse + 0.1*grad


class MultiNetwork(object):
    def __init__(self, **kwargs):
        self.models_folder = kwargs.get('models_folder', MODELS_FOLDER)

        self.structure = kwargs.get('structure', network_params.DEFAULT_STRUCTURE)

        self.pred_ahead = True

        # branches of network
        self.encoder = None
        self.decoder = None
        self.generator = None
        self.state_predictor = None
        self.action_mapper = None
        self.action_predictor = None
        self.state_sampler = None

        self.discriminator = None
        self.screen_discriminator = None
        self.state_discriminator = None

        # full networks
        self.autoencoder = None
        self.autoencoder_gen = None
        self.autoencoder_disc = None
        self.autoencoder_gan = None

        self.screen_predictor = None
        self.screen_predictor_g = None
        self.screen_predictor_d = None

        self.state_assigner = None
        self.future_sampler_g = None
        self.future_sampler_d = None

        self.build_branches()
        self.build_networks()

    def build_branches(self):
        self.encoder = self.build_branch(self.structure['encoder'])
        self.generator = self.build_branch(self.structure['generator'])
        self.decoder = self.build_branch(self.structure['decoder'])

        self.discriminator = self.build_branch(self.structure['discriminator'])

        self.state_predictor = self.build_branch(self.structure['state_predictor'])

        # self.physics_predictor = self.build_physics_predictor()
        # self.action_mapper = self.build_action_mapper()
        # self.action_predictor = self.build_action_predictor()
        #
        # self.state_sampler = self.build_state_sampler()
        # self.decoder = self.build_decoder()

    def build_networks(self):
        self.build_autoencoder()
        self.build_autoencoder_gen()
        self.build_ae_gan()
        self.build_screen_predictor()

    def build_branch(self, structure):
        input_shape = structure.get('input_shape')
        output_shape = structure.get('output_shape')
        name = structure.get('name')

        layers = structure.get('layers')

        input_layer = Input(shape=input_shape)
        x = input_layer

        for layer in layers:
            layer_constructor = layer.get('type')
            pos_args = layer.get(network_params.POSITIONAL_ARGS, [])
            key_args = layer.get(network_params.KEYWORD_ARGS, {})
            print('Building: ', layer_constructor, pos_args, key_args)
            x = layer_constructor(*pos_args, **key_args)(x)

        branch = Model(input_layer, x, name=name)
        branch.summary()
        test_data = np.zeros([1] + list(input_shape))
        print('test data shape:', test_data.shape)

        # res = branch.predict(test_data)
        # print('result shape:', res.shape)

        if not branch.output_shape[1:] == output_shape:
            raise ValueError('Bad output shape! Expected: {0} Actual: {1}'.format(output_shape, branch.output_shape))

        plot(branch, to_file='{0}/{1}.png'.format(self.models_folder, name), show_layer_names=True, show_shapes=True)

        return branch

    def build_autoencoder_gen(self):
        input_img = Input(shape=network_params.INPUT_IMAGE_SHAPE)
        input_noise = Input(shape=[network_params.NOISE_SIZE])
        z = self.encoder(input_img)

        z_noise = merge([input_noise, z], mode='concat')

        screen_recon = self.generator(z_noise)

        self.autoencoder_gen = Model(input=[input_img, input_noise], output=screen_recon)
        # self.autoencoder_gen.compile(optimizer=Adam(lr=0.0001), loss=sq_diff_loss)
        self.autoencoder_gen.compile(optimizer=Adam(lr=0.0001), loss='mse')
        # self.autoencoder_gen.summary()
        plot(self.autoencoder_gen, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder_gen'),
             show_layer_names=True,
             show_shapes=True)

    def build_autoencoder(self):
        input_img = Input(shape=network_params.INPUT_IMAGE_SHAPE)
        z = self.encoder(input_img)

        screen_recon = self.decoder(z)

        self.autoencoder = Model(input_img, screen_recon)
        self.autoencoder.compile(optimizer=Adam(lr=0.0002), loss='mse')
        # self.autoencoder_gen.summary()
        plot(self.autoencoder, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder'), show_layer_names=True,
             show_shapes=True)

    def build_ae_gan(self):
        input_img = Input(shape=network_params.INPUT_IMAGE_SHAPE)
        input_noise = Input(shape=[network_params.NOISE_SIZE])
        # z_disc = self.encoder_disc(input_img)
        screen_disc = self.discriminator(input_img)
        # screen_disc = self.screen_discriminator(z_disc)

        self.autoencoder_disc = Model(input_img, screen_disc)
        self.autoencoder_disc.trainable = True
        self.compile_disc_ent()
        # self.compile_disc_was()

        # self.autoencoder_disc.summary()
        plot(self.autoencoder_disc, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder_disc'),
             show_layer_names=True,
             show_shapes=True)

        screen_recon = self.autoencoder_gen([input_img, input_noise])
        fakeness = self.autoencoder_disc(screen_recon)

        self.autoencoder_gan = Model(input=[input_img, input_noise], output=[screen_recon, fakeness])
        self.autoencoder_disc.trainable = False
        self.compile_gan_ent()
        # self.compile_gan_was()

        # self.autoencoder_gan.summary()
        plot(self.autoencoder_gan, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder_gan'),
             show_layer_names=True,
             show_shapes=True)

    def build_screen_predictor(self):
        input_img = Input(shape=network_params.INPUT_IMAGE_SHAPE)
        z = self.encoder(input_img)
        z_next = self.state_predictor(z)
        screen_next = self.decoder(z_next)

        self.screen_predictor = Model(input_img, screen_next)
        self.screen_predictor.compile(optimizer='adam', loss='mse')

        plot(self.screen_predictor, to_file='{0}/{1}.png'.format(self.models_folder, 'screen_predictor'),
             show_layer_names=True,
             show_shapes=True)

    def compile_disc_mse(self):
        # self.autoencoder_disc.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
        self.autoencoder_disc.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='mse', metrics=['accuracy'])
        # self.autoencoder_disc.compile(optimizer=Adam(lr=0.0002), loss='mse', metrics=['accuracy'])
        # self.autoencoder_disc.compile(optimizer=Adam(lr=0.0002), loss=loss_wasserstein, metrics=['accuracy'])

    def compile_disc_was(self):
        # self.autoencoder_disc.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss=loss_wasserstein, metrics=['accuracy'])
        self.autoencoder_disc.compile(optimizer='adadelta', loss=loss_wasserstein, metrics=['accuracy'])

    def compile_disc_ent(self):
        # self.autoencoder_disc.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        self.autoencoder_disc.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        # self.autoencoder_disc.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
        # self.autoencoder_disc.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    def compile_gan_ent(self):
        self.autoencoder_gan.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                                     loss=['mse', 'binary_crossentropy'],
                                     loss_weights=[0.0, 1.0],
                                     metrics={'model_2': 'accuracy'})

    def compile_gan_was(self):
        self.autoencoder_gan.compile(optimizer=Adam(lr=0.0002, beta_1=0.5),
                                     loss=['mse', loss_wasserstein],
                                     loss_weights=[0.0, 1.0],
                                     metrics={'model_2': 'accuracy'})

    def build_physics_predictor(self):
        return self

    def build_action_predictor(self):
        return self

    def build_state_sampler(self):
        return self

    def train_network(self):
        # stages:
        # 1) encoder/decoder
        return self

    def train_epoch_ae_discriminator(self, batch_getter, batches_per_epoch, exp_cont=None, fake_rate=0.6, noise=0.25):
        fake_rate = 0.2 + 0.6*np.random.random()
        loss = 0
        for i in range(batches_per_epoch):
            fake = np.random.random() < fake_rate
            real_images = batch_getter()
            loss += self.train_batch_ae_discriminator(real_images, fake=fake, noise=noise, exp_replayer=exp_cont)[0]

        av_loss = loss / batches_per_epoch
        return av_loss

    def train_batch_ae_discriminator(self, real_images, fake, exp_replayer=None, noise=0.0):
        if not self.autoencoder_disc.trainable:
            make_trainable(self.autoencoder_disc, True)
            self.compile_disc_was()
            # self.autoencoder_gan_compile()

            # raise ValueError('Discriminator must be trainable')

        batch_size = real_images.shape[0]

        if fake:
            if exp_replayer is not None and exp_replayer.is_ready() and np.random.random() < REPLAY_RATE:
                images = exp_replayer.get_batch()
                # print('replaying!')
            else:
                images = self.autoencoder_gen.predict(real_images)
                if exp_replayer is not None and np.random.random() < RECORD_RATE:
                    exp_replayer.add_batch(images)
                    # print('recording!')
            labels = -np.ones(batch_size)
        else:
            images = real_images
            labels = np.ones(batch_size)

        loss = self.autoencoder_disc.train_on_batch(images, labels)
        # loss = self.autoencoder_disc.train_on_batch(images, labels + noise*np.random.randn(batch_size))

        return loss

    def train_epoch_gan_simple(self, batch_getter, batches_per_epoch):
        d_losses = []
        g_losses = []
        for _ in tqdm(range(batches_per_epoch)):
            d_loss = 1
            g_loss = 1
            # while d_loss > -0.6:
            for _ in range(1):
                real_images = batch_getter()
                noise = np.random.normal(0, 1, (real_images.shape[0], network_params.NOISE_SIZE))
                fake_images = self.autoencoder_gen.predict([0*real_images, noise])
                batch_size = real_images.shape[0]
                images = np.concatenate((real_images, fake_images))

                labels = np.zeros([images.shape[0], 1])
                labels[:images.shape[0]//2] = 1.0

                # labels = np.ones([images.shape[0], 1])
                # labels[images.shape[0]//2:] = -1 #last sort of working
                # labels *= -1 # good

                self.autoencoder_disc.trainable = True
                d_loss, d_acc = self.autoencoder_disc.train_on_batch(images, labels)
                # d_loss_real, d_acc_real = self.autoencoder_disc.train_on_batch(real_images, -np.ones(batch_size))
                # d_loss_fake, d_acc_fake = self.autoencoder_disc.train_on_batch(fake_images, np.ones(batch_size))
                # d_loss_real, d_acc_real = self.autoencoder_disc.train_on_batch(real_images, np.ones(batch_size))
                # d_loss_fake, d_acc_fake = self.autoencoder_disc.train_on_batch(fake_images, np.zeros(batch_size))
                # d_loss = (d_loss_real + d_loss_fake)/2
                self.autoencoder_disc.trainable = False

            d_losses.append(d_loss)

            # while g_loss > -0.2:
            for _ in range(1):
                real_images = batch_getter()
                noise = np.random.normal(0, 1, (real_images.shape[0], network_params.NOISE_SIZE))
                # labels = -np.ones(real_images.shape[0])
                labels = np.ones(real_images.shape[0])
                metrics = self.autoencoder_gan.train_on_batch([0*real_images, noise], [real_images, labels])
                total_loss, recon_loss, g_loss, g_acc = metrics

                g_losses.append(g_loss)

        return d_losses, g_losses

    def train_predictor(self, ep_getter, batches):
        losses = []
        for _ in tqdm(range(batches)):
            ep = ep_getter()
            if self.pred_ahead:
                loss = self.screen_predictor.train_on_batch(ep[0:19], ep[1:20])
            else:
                loss = self.screen_predictor.train_on_batch(ep[0:19], ep[0:19])
            losses.append(loss)

        return losses

    def test_ae_discriminator(self, batch_getter, batches):
        loss = 0
        acc = 0

        for i in range(batches//2):
            real_images = batch_getter()
            fake_images = self.autoencoder_gen.predict(real_images)
            images = np.concatenate((real_images, fake_images))
            batch_size = images.shape[0]
            labels = -np.ones(batch_size)
            labels[0:batch_size//2] = 1

            # todo: accumulate list, not scalar
            # correct = np.round(self.autoencoder_disc.predict(images)).astype('int').reshape(batch_size) == -labels
            x = self.autoencoder_disc.predict(images).reshape(batch_size)
            y = -labels
            # correct = (self.autoencoder_disc.predict(images).reshape(batch_size) > 0) == (-labels > 0)
            # acc += np.sum(correct)/batch_size
            acc += metric_acc(x, y)
            # loss += self.autoencoder_disc.test_on_batch(images, labels)[1]

        av_acc = acc / (batches//2)
        # av_loss = loss / (batches//2)
        return av_acc

    def test_ae_discriminator_experience(self, batch_getter, exp_replayer, batches):
        if not exp_replayer.is_ready():
            return 0

        loss = 0
        acc = 0

        for i in range(batches // 2):
            real_images = batch_getter()
            fake_images = exp_replayer.get_batch()
            images = np.concatenate((real_images, fake_images))
            batch_size = images.shape[0]
            labels = -np.ones(batch_size)
            labels[0:batch_size // 2] = 1

            # todo: accumulate list, not scalar
            # correct = np.round(self.autoencoder_disc.predict(images)).astype('int').reshape(batch_size) == -labels
            x = self.autoencoder_disc.predict(images).reshape(batch_size)
            y = -labels
            # correct = (self.autoencoder_disc.predict(images).reshape(batch_size) > 0) == (-labels > 0)
            # acc += np.sum(correct)/batch_size
            acc += metric_acc(x, y)
            # loss += self.autoencoder_disc.test_on_batch(images, labels)[1]

        av_acc = acc / (batches // 2)
        # av_loss = loss / (batches//2)
        return av_acc



    # def train_batch_ae_gan(self, real_images):
    #     batch_size = 32
    #
    #     if self.autoencoder_disc.trainable:
    #         make_trainable(self.autoencoder_disc, False)
    #         self.autoencoder_disc.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy')
    #         # raise ValueError('Discriminator must not be trainable')
    #
    #     labels = np.ones((batch_size,))
    #
    #     loss = self.autoencoder_gan.train_on_batch(real_images, labels)
    #
    #     return loss
    #
    # def show_reconstruction(self):
    #     return self
    #
    # def show_predictions(self):
    #     return self


if __name__ == '__main__':
    mn = MultiNetwork()


