#!/usr/bin/env python3
"""
Class for large network with multiple branches, cost functions, training stages.
"""

import keras
from keras.models import Model
from keras.layers import Input
from keras.optimizers import Adam, Adadelta, RMSprop, SGD
from keras.utils.visualize_util import plot
import tensorflow as tf
import keras.backend as K
import numpy as np

import simple_network as network_params


MODELS_FOLDER = 'models'
REPLAY_RATE = 0.04
RECORD_RATE = 0.004


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
    loss = K.mean(y_true * y_pred)
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

        # branches of network
        self.encoder = None
        self.decoder = None
        self.physics_predictor = None
        self.action_mapper = None
        self.action_predictor = None
        self.state_sampler = None

        self.encoder_disc = None
        self.screen_discriminator = None
        self.state_discriminator = None

        # full networks
        self.autoencoder_gen = None
        self.autoencoder_disc = None
        self.autoencoder_gan = None

        self.screen_predictor_g = None
        self.screen_predictor_d = None

        self.state_assigner = None
        self.future_sampler_g = None
        self.future_sampler_d = None

        self.build_branches()
        self.build_networks()

    def build_branches(self):
        self.encoder = self.build_branch(self.structure['encoder'])
        self.decoder = self.build_branch(self.structure['decoder'])

        self.encoder_disc = self.build_branch(self.structure['encoder'])
        self.screen_discriminator = self.build_branch(self.structure['screen_discriminator'])

        # self.physics_predictor = self.build_physics_predictor()
        # self.action_mapper = self.build_action_mapper()
        # self.action_predictor = self.build_action_predictor()
        #
        # self.state_sampler = self.build_state_sampler()
        # self.decoder = self.build_decoder()

    def build_networks(self):
        self.build_autoencoder()
        self.build_ae_gan()

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

    def build_autoencoder(self):
        input_img = Input(shape=network_params.INPUT_IMAGE_SHAPE)
        z = self.encoder(input_img)
        screen_recon = self.decoder(z)

        self.autoencoder_gen = Model(input_img, screen_recon)
        # self.autoencoder_gen.compile(optimizer=Adam(lr=0.0001), loss=sq_diff_loss)
        self.autoencoder_gen.compile(optimizer=Adam(lr=0.0001), loss='mse')
        # self.autoencoder_gen.summary()
        plot(self.autoencoder_gen, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder_gen'), show_layer_names=True,
             show_shapes=True)

    def build_ae_gan(self):
        input_img = Input(shape=network_params.INPUT_IMAGE_SHAPE)
        z_disc = self.encoder_disc(input_img)
        screen_disc = self.screen_discriminator(z_disc)

        self.autoencoder_disc = Model(input_img, screen_disc)
        self.compile_disc_was()
        # self.autoencoder_disc.summary()
        plot(self.autoencoder_disc, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder_disc'),
             show_layer_names=True,
             show_shapes=True)

        screen_recon = self.autoencoder_gen(input_img)
        fakeness = self.autoencoder_disc(screen_recon)

        self.autoencoder_gan = Model(input=[input_img], output=[screen_recon, fakeness])
        self.compile_gan()

        # self.autoencoder_gan.summary()
        plot(self.autoencoder_gan, to_file='{0}/{1}.png'.format(self.models_folder, 'autoencoder_gan'),
             show_layer_names=True,
             show_shapes=True)

    def compile_disc_mse(self):
        # self.autoencoder_disc.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
        self.autoencoder_disc.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])
        # self.autoencoder_disc.compile(optimizer=Adam(lr=0.0002), loss='mse', metrics=['accuracy'])
        # self.autoencoder_disc.compile(optimizer=Adam(lr=0.0002), loss=loss_wasserstein, metrics=['accuracy'])

    def compile_disc_was(self):
        self.autoencoder_disc.compile(optimizer=Adam(lr=0.0002), loss=loss_wasserstein, metrics=['accuracy'])

    def compile_disc_ent(self):
        self.autoencoder_disc.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    def compile_gan(self):
        self.autoencoder_gan.compile(optimizer=Adam(lr=0.0001),
                                     loss=['mse', 'binary_crossentropy'],
                                     loss_weights=[0.0, 1.0],
                                     metrics={'model_1': 'mse', 'model_2': 'accuracy'})

    def compile_gan_was(self):
        self.autoencoder_gan.compile(optimizer=Adam(lr=0.0001),
                                     loss=['mse', loss_wasserstein],
                                     loss_weights=[0.0, 1.0],
                                     metrics={'model_1': 'mse', 'model_2': 'accuracy'})

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


