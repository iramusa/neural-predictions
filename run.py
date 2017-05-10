#!/usr/bin/env python3

# %% imports
import sys
import os
import datetime

import numpy as np
import pandas as pd
import time
from PIL import Image
from keras.optimizers import Adam, Adadelta, RMSprop
# from scipy.misc import imsave

import architecture
import simple_network as network_params
from simple_container import DataContainer
from mnist_container import MNISTContainer

EXPERIMENTS_FOLDER = 'experiments'
DATA_FOLDER = 'data-toy'
RECONSTRUCTIONS_FOLDER = 'reconstructions'
PLOTS_FOLDER = 'plots'
MODELS_FOLDER = 'models'
LOAD_MODELS_FOLDER = 'load_models'
LOG_TO_FILE = False
LOG_FILE = 'log.log'
ERR_FILE = 'err.log'

GAME = 'simple'
BATCH_SIZE = 32
BATCHES_PER_EPOCH = 600


class Experiment(object):
    def __init__(self, output_folder='pure_ae', description='', epochs=200, mnist_run=True, **kwargs):
        datetag = datetime.datetime.now().strftime('%y-%m-%d_%H:%M')
        self.output_folder = '{0}/{1}_{2}'.format(EXPERIMENTS_FOLDER, output_folder, datetag)
        self.reconstructions_folder = '{0}/{1}'.format(self.output_folder, RECONSTRUCTIONS_FOLDER)
        self.plots_folder = '{0}/{1}'.format(self.output_folder, PLOTS_FOLDER)
        self.models_folder = '{0}/{1}'.format(self.output_folder, MODELS_FOLDER)

        self.make_directories()

        # divert prints to log file, print line by line so the file can be read real time
        self.log_file = open('{0}/{1}'.format(self.output_folder, LOG_FILE), 'w', buffering=1)
        # self.err_file = open('{0}/{1}'.format(self.output_folder, ERR_FILE), 'w', buffering=1)

        if LOG_TO_FILE:
            self.stdout = sys.stdout  # copy just in case
            self.stderr = sys.stderr  # copy just in case
            sys.stdout = self.log_file
            sys.stderr = self.log_file

        self.name = kwargs.get('name', 'noname')
        self.game = kwargs.get('game', GAME)
        self.description = description

        print('Initialising the experiment {0} in folder {1} on game {2}.\nDescription: {3}'.format(self.name,
                                                                                                    self.output_folder,
                                                                                                    self.game,
                                                                                                    self.description))

        file_train = "{0}/{1}-{2}.tfrecords".format(DATA_FOLDER, GAME, 'train')
        file_valid = "{0}/{1}-{2}.tfrecords".format(DATA_FOLDER, GAME, 'valid')

        self.epochs = epochs
        self.batch_size = kwargs.get('batch_size', BATCH_SIZE)

        if mnist_run:
            self.train_gen = MNISTContainer(batch_size=self.batch_size,
                                            im_shape=network_params.INPUT_IMAGE_SHAPE,
                                            valid=False)
            self.valid_gen = MNISTContainer(batch_size=self.batch_size,
                                            im_shape=network_params.INPUT_IMAGE_SHAPE,
                                            valid=True)
        else:
            self.train_gen = DataContainer(file_train, batch_size=self.batch_size,
                                           im_shape=network_params.INPUT_IMAGE_SHAPE,
                                           ep_len_read=20, episodes=1000)
            self.valid_gen = DataContainer(file_valid, batch_size=self.batch_size,
                                           im_shape=network_params.INPUT_IMAGE_SHAPE,
                                           ep_len_read=20, episodes=200)

        print('Containers started.')

        self.network = architecture.MultiNetwork(models_folder=self.models_folder)
        print('Networks built.')

        self.losses = {'ae_train': [],
                       'ae_valid': [],
                       # 'ae_disc_train': [],
                       # 'ae_disc_valid': [],
                       # 'ae_gan_train': [],
                       # 'ae_gan_valid': [],
                       }

    def train_ae(self, epochs=5, model_checkpoint=False):
        print('Training autoencoder for {0} epochs.'.format(epochs))
        history = self.network.autoencoder_gen.fit_generator(self.train_gen.generate_ae(),
                                                             samples_per_epoch=BATCHES_PER_EPOCH * BATCH_SIZE,
                                                             nb_epoch=epochs,
                                                             max_q_size=5,
                                                             validation_data=self.valid_gen.generate_ae(),
                                                             nb_val_samples=4 * BATCH_SIZE)

        self.losses['ae_train'] += history.history['loss']
        self.losses['ae_valid'] += history.history['val_loss']

        self.save_losses()
        self.save_ae_recons('AE')
        self.save_models()

        if model_checkpoint:
            epochs_so_far = len(self.losses['ae_train'])
            print('Model checkpoint reached. Saving the model after {0} epochs.'.format(epochs_so_far))
            fpath = '{0}/ae_gen_{1}.hdf5'.format(self.models_folder, epochs_so_far)
            self.network.autoencoder_gen.save_weights(fpath)

    def train_ae_disc(self, epochs=5, noise=0.25):
        print('Training discriminator for {0} epochs.'.format(epochs))

        n_batches_train = int(BATCHES_PER_EPOCH)
        n_batches_valid = 16
        train_losses = []
        validation_losses = []

        for i in range(epochs):
            train_losses.append(self.network.train_epoch_ae_discriminator(self.train_gen.get_batch_images, n_batches_train, noise=noise))
            validation_losses.append(self.network.test_ae_discriminator(self.valid_gen.get_batch_images, n_batches_valid))
            print('disc train losses:', train_losses[-1])
            print('disc valid acc:', validation_losses[-1])

        #     batch_loss = 0
        #     for j in range(n_batches_train):
        #         real_images = self.train_gen.get_batch_images()
        #         batch_loss += self.network.train_batch_ae_discriminator(real_images)[1]
        #
        #     train_losses.append(batch_loss/n_batches_train)
        #
        #     batch_loss = 0
        #     for j in range(n_batches_valid):
        #         real_images = self.valid_gen.get_batch_images()
        #         batch_loss += self.network.train_batch_ae_discriminator(real_images, test=True)[1]
        #
        #     validation_losses.append(batch_loss/n_batches_valid)

        # self.losses[''] += train_losses
        # self.losses[''] += validation_losses

    def train_ae_gan(self, epochs=5, model_checkpoint=False):
        print('Training generator for {0} epochs.'.format(epochs))
        if self.network.autoencoder_disc.trainable is True:
            architecture.make_trainable(self.network.autoencoder_disc, False)
            self.network.compile_gan()
            # self.network.autoencoder_disc_compile()
            # raise ValueError('Discriminator must not be trainable')

        history = self.network.autoencoder_gan.fit_generator(self.train_gen.generate_ae_gan_mo(),
                                                             samples_per_epoch=BATCHES_PER_EPOCH * BATCH_SIZE,
                                                             nb_epoch=epochs,
                                                             max_q_size=5,
                                                             validation_data=self.valid_gen.generate_ae_gan_mo(),
                                                             nb_val_samples=4 * BATCH_SIZE)

        self.losses['ae_train'] += history.history['loss']
        self.losses['ae_valid'] += history.history['val_loss']

        # self.save_losses()
        self.save_ae_recons('GAN')
        self.save_models()

        if model_checkpoint:
            epochs_so_far = len(self.losses['ae_train'])
            print('Model checkpoint reached. Saving the model after {0} epochs.'.format(epochs_so_far))
            fpath = '{0}/ae_gen_{1}.hdf5'.format(self.models_folder, epochs_so_far)
            self.network.autoencoder_gen.save_weights(fpath)

    def save_ae_recons(self, label):
        N_SAMPLES = 5
        im_med = self.train_gen.im_med
        im_valid = self.valid_gen.get_batch_images()
        im_recon = self.network.autoencoder_gen.predict(im_valid)

        # print('im_valid', im_valid.shape)
        # print('im_recon', im_recon.shape)
        # print('im_med', im_med.shape)

        pairs = []
        for i in range(N_SAMPLES):
            pairs.append(np.concatenate([im_valid[i, ...] + im_med, im_recon[i, ...] + im_med], axis=0))

        tiled = np.concatenate(pairs, axis=1)

        # return to viewable representation
        tiled *= 255
        tiled = tiled.astype('uint8')

        if tiled.shape[2] == 1:
            tiled = tiled.reshape(tiled.shape[:2])

        epochs_so_far = len(self.losses['ae_train'])
        print('Saving new reconstructions after {0} epochs.'.format(epochs_so_far))
        # imsave('{0}/{1}.png'.format(self.reconstructions_folder, epochs_so_far), tiled)

        # print('tiled', tiled.shape)
        tiled = Image.fromarray(tiled)
        tiled.save('{0}/{1}{2}.png'.format(self.reconstructions_folder, label, epochs_so_far))

    def save_losses(self):
        # TODO arrays must be the same lengths to use this constructor
        losses = pd.DataFrame.from_dict(self.losses)
        losses.to_csv('{0}/losses.csv'.format(self.output_folder))

    def save_plots(self):
        print('NOT Generating plots.')

    def save_models(self):
        epochs_so_far = len(self.losses['ae_train'])
        print('Saving models after {0} epochs'.format(epochs_so_far))
        fpath = '{0}/encoder_{1}.hdf5'.format(self.models_folder, epochs_so_far)
        self.network.encoder.save_weights(fpath)
        fpath = '{0}/decoder_{1}.hdf5'.format(self.models_folder, epochs_so_far)
        self.network.decoder.save_weights(fpath)
        fpath = '{0}/autoencoder_disc_{1}.hdf5'.format(self.models_folder, epochs_so_far)
        self.network.autoencoder_disc.save_weights(fpath)
        fpath = '{0}/screen_discriminator_{1}.hdf5'.format(self.models_folder, epochs_so_far)
        self.network.screen_discriminator.save_weights(fpath)

    def load_models(self, tag):
        print('Loading models')
        fpath = '{0}/encoder_{1}.hdf5'.format(LOAD_MODELS_FOLDER, tag)
        self.network.encoder.load_weights(fpath)
        fpath = '{0}/decoder_{1}.hdf5'.format(LOAD_MODELS_FOLDER, tag)
        self.network.decoder.load_weights(fpath)
        fpath = '{0}/autoencoder_disc_{1}.hdf5'.format(LOAD_MODELS_FOLDER, tag)
        self.network.autoencoder_disc.load_weights(fpath)
        fpath = '{0}/screen_discriminator_{1}.hdf5'.format(LOAD_MODELS_FOLDER, tag)
        self.network.screen_discriminator.load_weights(fpath)

    def finish(self):
        print('Finishing.')
        self.save_plots()
        self.save_models()

    def make_directories(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        if not os.path.exists(self.reconstructions_folder):
            os.makedirs(self.reconstructions_folder)
        if not os.path.exists(self.models_folder):
            os.makedirs(self.models_folder)
        if not os.path.exists(self.plots_folder):
            os.makedirs(self.plots_folder)

    def adaptive_gan_train(self):
        n_batches_valid = 64
        for i in range(1500):
            loss = 1.0
            acc = 0.0
            while acc < 0.94:
            # while loss > 0.4 or acc < 0.9:
                print('Training discriminator')
                loss = self.network.train_epoch_ae_discriminator(self.train_gen.get_batch_images, BATCHES_PER_EPOCH, noise=0.0)
                acc = self.network.test_ae_discriminator(self.valid_gen.get_batch_images, n_batches_valid)
                print('D loss:', loss)
                print('D acc:', acc)

            # self.network.autoencoder_disc_compile_ent()
            acc = 0
            loss_recon = 1.0
            loss = 1.0
            loss_ent = 1.0
            if self.network.autoencoder_disc.trainable is True:
                architecture.make_trainable(self.network.autoencoder_disc, False)
                self.network.compile_gan()
            while acc < 0.92 or loss_recon > 0.02:
            # while loss_ent > 0.1:
                print('Training generator')
                history = self.network.autoencoder_gan.fit_generator(self.train_gen.generate_ae_gan_mo(),
                                                                     samples_per_epoch=BATCHES_PER_EPOCH * BATCH_SIZE,
                                                                     nb_epoch=1,
                                                                     max_q_size=5,
                                                                     validation_data=self.valid_gen.generate_ae_gan_mo(),
                                                                     nb_val_samples=16 * BATCH_SIZE)
                loss = history.history['val_loss'][-1]
                acc = history.history['val_model_2_acc'][-1]
                loss_ent = history.history['val_model_2_loss'][-1]
                loss_recon = history.history['val_model_1_loss'][-1]
                print('D loss:', loss)
                print('D acc:', acc)
                self.losses['ae_train'] += history.history['loss']
                self.losses['ae_valid'] += history.history['val_loss']

            self.save_ae_recons('GAN')

            if not i % 10:
                self.save_models()

    def run_experiment(self):
        if 'adaptive_train' in self.output_folder:
            for i in range(10):
                self.train_ae(epochs=10)
            self.adaptive_gan_train()

        if 'pure_gan' in self.output_folder:
            # self.train_ae_gan(epochs=15)
            for i in range(100):
                self.train_ae_disc(epochs=8)
                # time.sleep(3)
                self.train_ae_gan(epochs=4)
                # time.sleep(3)

        if 'pure_ae' in self.output_folder:
            for i in range(100):
                self.train_ae(epochs=10)

        if 'ae_gan' in self.output_folder:
            self.train_ae(epochs=60)
            for i in range(50):
                time.sleep(3)
                self.train_ae_disc(epochs=2)
                time.sleep(3)
                self.train_ae_gan(epochs=5)
                time.sleep(3)
                self.train_ae(epochs=5)

        if 'ae_gan_mix' in self.output_folder:
            self.train_ae(epochs=4)
            for i in range(250):
                time.sleep(3)
                self.train_ae_disc(epochs=2)
                time.sleep(3)
                self.train_ae_gan(epochs=4)
                time.sleep(3)
                self.train_ae(epochs=4)

        self.finish()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        output_folder = sys.argv[1]
        exp = Experiment(output_folder=output_folder)
    else:
        exp = Experiment()

    exp.run_experiment()
    exp.finish()
