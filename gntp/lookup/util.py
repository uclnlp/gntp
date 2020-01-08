# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import gntp

from gntp.lookup.base import BaseLookupIndex

from typing import List, Union, Optional

import logging

logger = logging.getLogger(__name__)


def reshape(tensor: Union[tf.Tensor, str],
            embedding_size: int):
    return tf.reshape(tensor, [-1, embedding_size]) if gntp.is_tensor(tensor) else tensor


def find_best_heads(index: BaseLookupIndex,
                    atoms: List[Union[tf.Tensor, str]],
                    goals: List[Union[tf.Tensor, str]],
                    goal_shape: tf.TensorShape,
                    k: int = 10,
                    is_training: bool = False,
                    goal_indices: Optional[List[Union[np.ndarray, str]]] = None,
                    position: int = None):

    if isinstance(index, gntp.lookup.SymbolLookupIndex) and goal_indices is not None:
        assert position is not None
        atom_indices = index.query_sym(data_indices=goal_indices,
                                       k=k,
                                       is_training=is_training,
                                       position=position)

        actual_k = atom_indices.shape[-1]
        new_shp = tf.TensorShape(actual_k).concatenate(goal_shape[:-1])
        atom_indices = tf.cast(tf.reshape(tf.transpose(atom_indices), new_shp), tf.int32)
    else:
        embedding_size = goal_shape[-1]
        new_goals = [reshape(ge, embedding_size) for ge in goals]

        ground_goals = [ge for fe, ge in zip(atoms, new_goals) if gntp.is_tensor(fe) and gntp.is_tensor(ge)]

        max_dim = max([gg.get_shape()[0] for gg in ground_goals])

        ground_goals = [tf.tile(goal, [max_dim // goal.get_shape()[0], 1]) for goal in ground_goals]

        # [G, 3 E], or e.g. [K, 2 3] if facts or goals contains a variable
        goals_2d = tf.concat(ground_goals, axis=1)

        # Facts in 'facts_2d' most relevant to the query in 'goals_2d'
        atom_indices = index.query(goals_2d.numpy(),
                                   k=k,
                                   is_training=is_training)

        actual_k = atom_indices.shape[-1]
        new_shp = tf.TensorShape(actual_k).concatenate(goal_shape[:-1])
        atom_indices = tf.cast(tf.reshape(tf.transpose(atom_indices), new_shp), tf.int32)
    return atom_indices
