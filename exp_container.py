#!/usr/bin/env python3

import numpy as np


class ExperienceContainer(object):
    def __init__(self, max_batches=400):
        self.max_batches = max_batches
        self.batch_list = []

    def is_ready(self):
        if len(self.batch_list) > 0:
            return True
        else:
            return False

    def add_batch(self, batch):
        self.batch_list.append(batch)
        if len(self.batch_list) > self.max_batches:
            self.batch_list.pop(np.random.randint(0, len(self.batch_list)))

    def get_batch(self):
        return self.batch_list[np.random.randint(0, len(self.batch_list))]

    def generate_ae(self):
        while True:
            images = self.get_batch()
            yield (images, images)

    def generate_ae_gan(self):
        while True:
            images = self.get_batch()
            labels = -np.ones((images.shape[0],))
            yield (images, labels)

    def generate_ae_gan_mo(self):
        while True:
            images = self.get_batch()
            labels = -np.ones((images.shape[0],))
            yield (images, [images, labels])

