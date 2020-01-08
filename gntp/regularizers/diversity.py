# -*- coding: utf-8 -*-

import tensorflow as tf


def diversity(kernel, matrix):
    n = matrix.get_shape()[0]
    similarities = kernel.pairwise(matrix, matrix) * (1.0 - tf.eye(n))
    return tf.reduce_mean(similarities)
