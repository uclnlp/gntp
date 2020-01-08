# -*- coding: utf-8 -*-

import tensorflow as tf
from gntp.kernels.base import BaseKernel

import logging

logger = logging.getLogger(__name__)


class LinearKernel(BaseKernel):
    def __init__(self):
        super().__init__()

    def pairwise(self, x, y):
        dim_x, emb_size_x = x.get_shape()[:-1], x.get_shape()[-1]
        dim_y, emb_size_y = y.get_shape()[:-1], y.get_shape()[-1]

        a = tf.reshape(x, [-1, emb_size_x])
        b = tf.reshape(y, [-1, emb_size_y])

        dot = tf.einsum('xe,ye->xy', a, b)
        sdot = tf.nn.sigmoid(dot)
        return tf.reshape(sdot, dim_x.concatenate(dim_y))

    def elementwise(self, x, y):
        emb_size_x = x.get_shape()[-1]
        emb_size_y = y.get_shape()[-1]

        a = tf.reshape(x, [-1, emb_size_x])
        b = tf.reshape(y, [-1, emb_size_y])

        dot = tf.einsum('xe,xe->x', a, b)
        return tf.nn.sigmoid(dot)
