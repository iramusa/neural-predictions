#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


def read_and_decode_single_example(filename):
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    return image


class DataContainer(object):
    def __init__(self, file, batch_size, im_shape, ep_len_read=20, episodes=200):
        self.file = file
        self.ep_len_read = ep_len_read
        self.ep_len_gen = ep_len_read
        self.episodes = episodes
        self.total_images = self.ep_len_read * self.episodes
        self.batch_size = batch_size
        self.im_shape = im_shape
        self.im_med = np.zeros(self.im_shape)
        self.images = None

        self.single_image = read_and_decode_single_example(file)

        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        tf.train.start_queue_runners(sess=self.sess)

        self.load_images()

        # TODO get a median image, not episode
        self.im_med = np.median(self.images, axis=0)[0]

        # del self.single_image
        # self.sess.close()
        # mean based shift
        # self.im_med = np.mean(self.images, axis=0) * (1.0/255.0)

    def set_ep_len(self, ep_len):
        self.ep_len_gen = ep_len

    def load_images(self):
        episodes = [self.read_episodes() for _ in range(self.episodes)]
        self.images = np.array(episodes)

    # def get_single_image(self, reshape=False):
    #     """Returns the next image from the file.
    #
    #     :return: float image with values between 0 and 1 -- np.array of shape im_shape
    #
    #     """
    #     im = self.sess.run(self.single_image)
    #
    #     if reshape:
    #         im = np.reshape(im, self.im_shape)
    #
    #     return im

    def read_episodes(self):
        """Returns n images in the order they are stored in the file.

        :param n: number of images
        :param subtract_median: if true median image is subtracted
        :return: float images with values between 0 and 1 or -1 and +1 if median was subtracted
        """
        ims = []
        for i in range(self.ep_len_read):
            im = self.sess.run(self.single_image)
            ims.append(im)

        ims = np.array(ims)
        ims = np.reshape(ims, [ims.shape[0]] + list(self.im_shape))

        return ims

    def get_n_random_images(self, n):
        ep_rolls = np.random.randint(0, self.episodes, n)
        t_rolls = np.random.randint(0, self.ep_len_read, n)

        random_ims = self.images[ep_rolls, t_rolls, ...]
        return random_ims

    def get_batch_images(self):
        return self.get_n_random_images(self.batch_size)

    def get_n_random_episodes(self, n, ep_len=None):
        if ep_len is None:
            ep_len = self.ep_len_gen

        ep_rolls = np.random.randint(0, self.episodes, n)
        random_eps = self.images[ep_rolls, :ep_len, ...]

        return random_eps

    def get_batch_episodes(self):
        self.get_n_random_episodes(self.batch_size)

    def get_n_batches_images(self, n=10):
        ims = []
        for i in range(n):
            im = self.get_batch_images()
            ims.append(im)

        ims = np.concatenate(ims, axis=0)

        return ims

    def generate_ae(self):
        while True:
            images = self.get_batch_images()
            yield (images, images)

    def generate_ae_gan(self):
        while True:
            images = self.get_batch_images()
            labels = np.ones((images.shape[0],))
            yield (images, labels)


