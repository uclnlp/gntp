# -*- coding: utf-8 -*-

import logging

import numpy as np
import tensorflow as tf

from typing import Optional, Union

logger = logging.getLogger(__name__)


def create_mask(mask_indices: Union[tf.Tensor, np.ndarray],
                mask_shape: Union[tf.TensorShape, np.ndarray],
                indices: Optional[Union[tf.Tensor, np.ndarray]] = None):
    if indices is None:
        res = _create_mask(mask_indices, mask_shape)
    else:
        res = _create_mask2(mask_indices, mask_shape, indices)
    return res


def _create_mask(mask_indices: Union[tf.Tensor, np.ndarray],
                 mask_shape: Union[tf.TensorShape, np.ndarray]):
    """
    Create a {0, 1} mask with shape mask_shape for masking the kernel matrix with.

    mask_indices contains the indices of the facts we aim at masking,
    and it is structured as follows:
         [ 6 -1 -1 22 -1 -1  3 -1 -1  7 .. ]

    Where negative values correspond to goals we do not want to mask.

    :param mask_indices: Indices (in the [FE] facts matrix) of the facts we want to mask.
    :param mask_shape: Shape of the mask.
    :return: Mask.
    """
    # If there is no indices to mask, return an empty mask
    # if np.all(np.array(mask_indices) < 0):
    #     return None
    if tf.equal(tf.reduce_sum(tf.cast(tf.less(mask_indices, 0), tf.int32)), tf.size(mask_indices)).numpy():
        return None

    fact_dim, goal_dim = mask_shape[0], mask_shape[-1]

    condition = tf.greater(mask_indices, -1)

    # Indices of positive examples
    ones_x = tf.boolean_mask(mask_indices, condition)

    # Their position in the batch
    ones_y = tf.cast(tf.squeeze(tf.where(condition)), ones_x.dtype)

    # Sparse matrix that is 1 in the cells we aim at masking, and 0 everywhere else.
    if len(mask_shape) == 2:
        ones = tf.ones(tf.size(ones_x), dtype=tf.float32)

        _indices = tf.transpose(tf.stack([ones_x, ones_y]))

        mask = tf.scatter_nd(indices=_indices, updates=ones,
                             shape=[fact_dim, goal_dim])
    else:
        # This code is based on the following observation:
        #  If we have a kernel such as [F, A, B, G], where F is the fact dimension and G
        #  is the goal dimension, and we need to mask Fact 5 for Goal 1, we can create a
        #  mask [F, A * B, G] with [5, :, 1] = 0, reshape it to [F, A, B, G], and use it
        #  for masking the kernel [F, A, B, G].
        nb_int_dims = np.prod(mask_shape[1:-1])
        ones = tf.ones(ones_x.shape[0] * nb_int_dims, dtype=tf.float32)

        ones_x_tiled = tf.tile(ones_x, [nb_int_dims])
        ind = tf.reshape(tf.tile(tf.reshape(tf.range(nb_int_dims), [-1, 1]), [1, ones_x.shape[0]]), [-1])

        ones_y_tiled = tf.tile(ones_y, [nb_int_dims])

        ind = tf.cast(ind, ones_x_tiled.dtype)
        ones_y_tiled = tf.cast(ones_y_tiled, ones_x_tiled.dtype)

        _indices = tf.transpose(tf.stack([ones_x_tiled, ind, ones_y_tiled]))

        mask = tf.scatter_nd(indices=_indices, updates=ones,
                             shape=[fact_dim, nb_int_dims, goal_dim])
        mask = tf.reshape(mask, mask_shape)
    return 1 - mask


def _create_mask2(mask_indices: Union[tf.Tensor, np.ndarray],
                  mask_shape: Union[tf.TensorShape, np.ndarray],
                  indices: Union[tf.Tensor, np.ndarray]):
    # If there is no indices to mask, return an empty mask

    if tf.equal(tf.reduce_sum(tf.cast(tf.less(mask_indices, 0), tf.int32)), tf.size(mask_indices)).numpy():
        return None

    fact_dim = indices.shape[0]
    goal_dim = mask_shape[-1]

    nb_goals = np.prod(mask_shape[1:])

    tile_goals = 1 if nb_goals == goal_dim else nb_goals // goal_dim

    mask_shape_2d = [fact_dim, int(nb_goals)]

    _fact_indices = tf.reshape(indices, [fact_dim, -1])

    if tile_goals > 1:
        mask_indices = tf.reshape(tf.tile(tf.reshape(mask_indices, [1, -1]), [tile_goals, 1]), [-1])

    condition = tf.greater(mask_indices, -1)
    # Indices of positive examples
    ones_x = tf.boolean_mask(mask_indices, condition)
    # Their position in the batch

    ones_y = tf.cast(tf.squeeze(tf.where(condition)), tf.int32)

    ones_x_tiled = tf.tile(ones_x, [fact_dim])
    ind = tf.reshape(tf.tile(tf.reshape(tf.range(fact_dim), [-1, 1]), [1, ones_x.shape[0]]), [-1])
    ones_y_tiled = tf.tile(ones_y, [fact_dim])

    indices_and_ones_y = tf.transpose(tf.stack([ind, ones_y_tiled]))

    gathered_results = tf.gather_nd(_fact_indices, indices_and_ones_y)

    where_equal = tf.where(tf.equal(gathered_results, ones_x_tiled))

    r1 = tf.gather(ind, where_equal)
    r2 = tf.gather(ones_y_tiled, where_equal)
    _indices = tf.concat([r1, r2], axis=1)

    if len(_indices.numpy()) == 0:
        return None

    ones = tf.ones(len(_indices.numpy()), dtype=tf.float32)
    mask = tf.scatter_nd(indices=_indices, updates=ones, shape=mask_shape_2d)
    mask = tf.reshape(mask, mask_shape)

    return 1 - mask
