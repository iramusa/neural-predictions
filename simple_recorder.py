#!/usr/bin/env python3
"""
Records images from simple sim
"""
import simple_sim
import numpy as np
import os
import time
import tensorflow as tf

# SIM = 'simple'
SIM = 'single'
FILENAME = SIM + '-valid'
# FILENAME = SIM + '-train'
EPISODES_TRAIN = 1000
EPISODES_VALID = 200
EP_LEN = 20


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class Record(object):
    def __init__(self):
        self.world = None
        self.ep = 0
        self.t = 0
        self.obs_list = []
        filename = os.path.join('data-toy/', FILENAME + '.tfrecords')
        print('Writing', filename)
        self.writer = tf.python_io.TFRecordWriter(filename)

    def run(self):
        total_episodes = EPISODES_TRAIN if 'train' in FILENAME else EPISODES_VALID
        while self.ep < total_episodes:
            print('Episode:', self.ep)
            self.world = simple_sim.World()
            for _ in range(EP_LEN):
                self.world.run()
                observation = self.world.draw()
                self.obs_list.append(observation)
                self.write_record()
                self.obs_list = []
                self.t += 1

            self.ep += 1

    def write_record(self):
        im = self.obs_list[0]

        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(28),
            'width': _int64_feature(28),
            'depth': _int64_feature(1),
            'timestep': _int64_feature(self.t),
            'episode': _int64_feature(self.ep),
            'image_raw': _bytes_feature(im.tobytes()),
        }))

        self.writer.write(example.SerializeToString())

    def close(self):
        self.writer.close()


if __name__ == '__main__':

    rec = Record()

    rec.run()
    rec.close()


