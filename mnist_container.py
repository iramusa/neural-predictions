#!/usr/bin/env python3

import numpy as np
from keras.datasets import mnist


class MNISTContainer(object):
    def __init__(self, batch_size, im_shape, valid):
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.im_med = np.zeros(self.im_shape)
        self.images = None

        self.load_images(valid=valid)

        # self.im_med = np.median(self.images, axis=0)

    def load_images(self, valid=False):
        (x_train, _), (x_test, _) = mnist.load_data()
        if valid:
            self.images = x_test/255
        else:
            self.images = x_train/255

    def get_n_random_images(self, n, subtract_median=True):
        rolls = np.random.randint(0, self.images.shape[0], n)

        random_ims = self.images[rolls, ...]

        return random_ims.reshape((n,) + self.im_shape)

    def get_batch_images(self):
        return self.get_n_random_images(self.batch_size)

    def generate_ae(self):
        while True:
            images = self.get_batch_images()
            yield (images, images)

    def generate_ae_gan(self):
        while True:
            images = self.get_batch_images()
            labels = np.ones((images.shape[0],))
            yield (images, labels)

    def generate_ae_gan_mo(self):
        while True:
            images = self.get_batch_images()
            labels = np.ones((images.shape[0],))
            yield (images, [images, labels])

