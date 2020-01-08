# -*- coding: utf-8 -*-

import tensorflow as tf

from gntp.kernels.base import BaseKernel

import logging

logger = logging.getLogger(__name__)


class CosineKernel(BaseKernel):
    def __init__(self):
        super().__init__()

    def pairwise(self, x, y):
        """
        :param x: [AxE] Tensor
        :param y: [AxE] Tensor
        :return: [A] Tensor
        """
        dim_x, emb_size_x = x.get_shape()[:-1], x.get_shape()[-1]
        dim_y, emb_size_y = y.get_shape()[:-1], y.get_shape()[-1]

        a = tf.reshape(x, [-1, emb_size_x])
        b = tf.reshape(y, [-1, emb_size_y])

        norm_a = tf.nn.l2_normalize(a, dim=1)
        norm_b = tf.nn.l2_normalize(b, dim=1)

        res = tf.einsum('xe,ye->xy', norm_a, norm_b)

        # The cosine similarity has values in [-1, 1], need to project it into [0, 1]
        squashed_res = (1.0 + res) * 0.5

        return tf.reshape(squashed_res, dim_x.concatenate(dim_y))

    def elementwise(self, x, y):
        """
        :param x: [AxE] Tensor
        :param y: [BxE] Tensor
        :return: [AxB] Tensor
        """
        dim_x, emb_size_x = x.get_shape()[:-1], x.get_shape()[-1]
        dim_y, emb_size_y = y.get_shape()[:-1], y.get_shape()[-1]

        a = tf.reshape(x, [-1, emb_size_x])
        b = tf.reshape(y, [-1, emb_size_y])

        norm_a = tf.nn.l2_normalize(a, dim=1)
        norm_b = tf.nn.l2_normalize(b, dim=1)

        res = tf.einsum('xe,xe->x', norm_a, norm_b)

        # The cosine similarity has values in [-1, 1], need to project it into [0, 1]
        squashed_res = (1.0 + res) * 0.5

        return squashed_res
