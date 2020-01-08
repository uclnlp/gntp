# -*- coding: utf-8 -*-

import copy
import gntp
import numpy as np

import tensorflow as tf

from typing import List, Union


def k_max(goal: List[Union[tf.Tensor, str]],
          proof_state: gntp.ProofState,
          k: int = 10):
    new_proof_state = proof_state

    goal_variables = {goal_elem for goal_elem in goal if gntp.is_variable(goal_elem)}

    if len(goal_variables) > 0:
        # Check goal for variables
        scores = proof_state.scores
        substitutions = copy.copy(proof_state.substitutions)

        index_substitutions = None
        if proof_state.index_substitutions is not None:
            index_substitutions = copy.copy(proof_state.index_substitutions)

        scores_shp = scores.get_shape()  # [K, R, G]
        k_size = scores_shp[0]  # K

        # R * G
        # _batch_size = tf.reduce_prod(scores_shp[1:])
        _batch_size = np.prod(scores_shp[1:])

        # [ k, R, G ]
        new_scores_shp = tf.TensorShape(k).concatenate(scores_shp[1:])

        # [K, R, G] -> [K, R * G]
        scores_2d = tf.reshape(scores, [k_size, -1])

        # [R * G, K]
        scores_2d_t = tf.transpose(scores_2d)
        # [ R * G, k]

        scores_2d_top_k_t, scores_2d_top_k_idxs_t = tf.nn.top_k(scores_2d_t, k=k)

        # [ k, R * G ]
        scores_2d_top_k = tf.transpose(scores_2d_top_k_t)
        # [ k, R * G ]
        scores_2d_top_k_idxs = tf.transpose(scores_2d_top_k_idxs_t)

        # [k, R, G]
        scores_top_k = tf.reshape(scores_2d_top_k, new_scores_shp)

        scores_top_k_idxs = tf.reshape(scores_2d_top_k_idxs, new_scores_shp)

        # [[ 4 23 10  1 29], [14 30 14 20 21]]
        coordinates_lhs = scores_2d_top_k_idxs
        # [[ 4 14] [23 30] [10 14] [ 1 20] [29 21]],
        coordinates_lhs = tf.reshape(coordinates_lhs, [1, -1])
        # [[ 4 14 23 30 10 14  1 20 29 21]]

        # [0, 1, 2, 3, 4]
        coordinates_rhs = tf.reshape(tf.range(0, _batch_size), [1, -1])
        # [[0], [1], [2], [3], [4]]
        coordinates_rhs = tf.tile(coordinates_rhs, [1, k])
        # [[0 0] [1 1] [2 2] [3 3] [4 4]]

        k_2d_coordinates = tf.concat([coordinates_lhs, coordinates_rhs], axis=0)
        # [[ 4 14 23 30 10 14  1 20 29 21]
        #  [ 0  0  1  1  2  2  3  3  4  4]]
        k_2d_coordinates = tf.transpose(k_2d_coordinates)

        # Assume we unified [KE, KE, KE] and [GE, GE, X] - we get X/KGE.
        # Reshape X such that we have X/K'GE instead.
        for goal_variable in {g for g in goal_variables if g in substitutions}:
            var_tensor = substitutions[goal_variable]

            if gntp.is_tensor(var_tensor):
                # Variable is going to be [K, R, G, E]
                substitution_shp = var_tensor.get_shape()
                embedding_size = substitution_shp[-1]

                assert substitution_shp[0] == k_size

                new_substitution_shp = new_scores_shp.concatenate([embedding_size])

                # Reshape to [K, R * G, E]
                substitution_3d = tf.reshape(var_tensor, [k_size, -1, embedding_size])

                new_substitution_3d = tf.gather_nd(substitution_3d, k_2d_coordinates)
                new_substitution_3d = tf.reshape(new_substitution_3d, [k, -1, embedding_size])

                substitutions[goal_variable] = tf.reshape(new_substitution_3d, new_substitution_shp)

            if index_substitutions is not None:
                index_var_tensor = index_substitutions[goal_variable]

                if index_var_tensor is np.ndarray:
                    tmp_2d = tf.reshape(index_var_tensor, [k_size, -1])
                    new_tmp_2d = np.take(tmp_2d, k_2d_coordinates)
                    new_tmp_2d = np.reshape(new_tmp_2d, [k, -1])
                    index_substitutions[goal_variable] = np.reshape(new_tmp_2d, new_scores_shp)

        index_mappers = copy.deepcopy(proof_state.index_mappers)
        index_kb = proof_state.index_kb
        if index_mappers is not None:
            index_mappers[-len(scores_top_k_idxs.shape)] = scores_top_k_idxs

        new_proof_state = gntp.ProofState(scores=scores_top_k,
                                          substitutions=substitutions,
                                          index_mappers=index_mappers,
                                          index_kb=index_kb,
                                          index_substitutions=index_substitutions)
    return new_proof_state
