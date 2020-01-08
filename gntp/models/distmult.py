# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
from gntp.models.lfm import BaseLatentFeatureModel

logger = logging.getLogger(__name__)


class DistMult(BaseLatentFeatureModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(emb_rel: tf.Tensor,
              emb_arg1: tf.Tensor,
              emb_arg2: tf.Tensor):
        """
        Scoring function for DistMult.

        :param emb_rel: [B, E] Tensor
        :param emb_arg1: [B, E] Tensor
        :param emb_arg2: [B, E] Tensor
        :return:
        """
        return tf.einsum("ij,ij->i", emb_rel, emb_arg1 * emb_arg2)

    @staticmethod
    def score_sp(emb_rel: tf.Tensor,
                 emb_arg1: tf.Tensor,
                 all_emb_arg2: tf.Tensor):
        """
        Scoring function for DistMult.

        :param emb_rel: [B, 2E] Tensor
        :param emb_arg1: [B, 2E] Tensor
        :param all_emb_arg2: [N, 2E] Tensor
        :return:
        """
        # [B, N] Tensor
        score = tf.einsum("ij,nj->in", emb_rel * emb_arg1, all_emb_arg2)
        # [B, N] Tensor
        return score

    @staticmethod
    def score_po(emb_rel: tf.Tensor,
                 all_emb_arg1: tf.Tensor,
                 emb_arg2: tf.Tensor):
        """
        Scoring function for DistMult.

        :param emb_rel: [B, 2E] Tensor
        :param all_emb_arg1: [N, 2E] Tensor
        :param emb_arg2: [B, 2E] Tensor
        :return:
        """
        # [B, N] Tensor
        score = tf.einsum("nj,ij->in", all_emb_arg1, emb_arg2 * emb_rel)
        # [B, N] Tensor
        return score

    def L3(self,
           emb_vec: tf.Tensor):
        """
        L3 Regularization term - |u|^3
        :param emb_vec: [N, E] Tensor
        :return:
        """
        # [N, E], [N, E] Tensors
        vec_abs = tf.abs
        # [N, E] Tensor
        mod_vec = tf.abs(emb_vec)
        mod_vec_cube = tf.pow(mod_vec, 3)
        # Scalar
        res = tf.reduce_sum(mod_vec_cube, [0, 1])
        return res
