# -*- coding: utf-8 -*-

import tensorflow as tf
from gntp.kernels.base import BaseKernel

import logging

logger = logging.getLogger(__name__)


class RBFKernel(BaseKernel):
    def __init__(self, slope=1.0):
        super().__init__()
        self.slope = slope

    def elementwise(self, x, y):
        emb_size_x = x.get_shape()[-1]
        emb_size_y = y.get_shape()[-1]

        a = tf.reshape(x, [-1, emb_size_x])
        b = tf.reshape(y, [-1, emb_size_y])

        l2 = tf.reduce_sum(tf.square(a - b), 1)

        l2 = tf.clip_by_value(l2, 1e-6, 1000)
        l2 = tf.sqrt(l2)
        return tf.exp(- l2 * self.slope)

    def pairwise(self, x, y):
        dim_x, emb_size_x = x.get_shape()[:-1], x.get_shape()[-1]
        dim_y, emb_size_y = y.get_shape()[:-1], y.get_shape()[-1]

        a = tf.reshape(x, [-1, emb_size_x])
        b = tf.reshape(y, [-1, emb_size_y])

        c = - 2 * tf.matmul(a, tf.transpose(b))
        na = tf.reduce_sum(tf.square(a), 1, keepdims=True)
        nb = tf.reduce_sum(tf.square(b), 1, keepdims=True)

        l2 = (c + tf.transpose(nb)) + na

        l2 = tf.clip_by_value(l2, 1e-6, 1000)
        l2 = tf.sqrt(l2)

        sim = tf.exp(- l2 * self.slope)
        return tf.reshape(sim, dim_x.concatenate(dim_y))
