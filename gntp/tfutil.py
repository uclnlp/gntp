# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

from gntp.util import is_tensor


def tile_left(atom_elem, target_shp):
    """
    Given a tensor of shape [D_1, D_2, .., D_n, E], and a shape
    [X, D_1, D_2, .., D_n], add the extra dimension via tiling and
    create a new tensor with shape [X, D_1, D_2, .., D_n, E].

    :param atom_elem: Tensor with shape [D_1, D_2, .., D_n, E]
    :param target_shp: Shape [X, D_1, D_2, .., D_n,]
    :return: Tensor with shape [X, D_1, D_2, .., D_n, E]
    """
    res = atom_elem
    if is_tensor(atom_elem):
        tensor_shp = atom_elem.get_shape()
        nb_vectors = np.prod(tensor_shp[:-1])
        nb_target_vectors = np.prod(target_shp)
        embedding_size = tensor_shp[-1]
        res_3d = tf.reshape(atom_elem, [1, -1, embedding_size])
        res_3d = tf.tile(res_3d, [nb_target_vectors // nb_vectors, 1, 1])
        final_shp = target_shp.concatenate([embedding_size])
        res = tf.reshape(res_3d, final_shp)
    return res


def tile_right(atom_elem, target_shp):
    """
    Given a tensor of shape [D_1, D_2, .., D_n, E], and a shape
    [D_1, D_2, .., D_n, X], add the extra dimension via tiling and
    create a new tensor with shape [D_1, D_2, .., D_n, X, E].

    :param atom_elem: Tensor with shape [D_1, D_2, .., D_n, E]
    :param target_shp: Shape [D_1, D_2, .., D_n, X]
    :return: Tensor with shape [D_1, D_2, .., D_n, X, E]
    """
    res = atom_elem
    if is_tensor(atom_elem):
        tensor_shp = atom_elem.get_shape()
        nb_vectors = np.prod(tensor_shp[:-1])
        nb_target_vectors = np.prod(target_shp)
        embedding_size = tensor_shp[-1]
        res_3d = tf.reshape(atom_elem, [-1, 1, embedding_size])
        res_3d = tf.tile(res_3d, [1, nb_target_vectors // nb_vectors, 1])
        final_shp = target_shp.concatenate([embedding_size])
        res = tf.reshape(res_3d, final_shp)
    return res


def smooth_maximum(logits, alpha=1.0, axis=0):
    scores = tf.exp(alpha * logits)
    return tf.reduce_sum(logits * scores, axis=axis) / tf.reduce_sum(scores, axis=axis)


def smooth_minimum(logits, alpha=1.0, axis=0):
    pass
    # In [53]: p = tf.pow(a, k)
    #
    # In [54]: tf.pow(1/tf.reduce_sum(1/p), 1.0/k)
