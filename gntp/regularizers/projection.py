# -*- coding: utf-8 -*-

import tensorflow as tf


def renorm_update(var_matrix, norm=1.0, axis=1):
    row_norms = tf.sqrt(tf.reduce_sum(tf.square(var_matrix), axis=axis))
    scaled = var_matrix * tf.expand_dims(norm / row_norms, axis=axis)
    return var_matrix.assign(scaled)


def clip_update(var_matrix, lower_bound=0.0, upper_bound=1.0):
    clipped = tf.minimum(upper_bound, tf.maximum(var_matrix, lower_bound))
    return var_matrix.assign(clipped)
