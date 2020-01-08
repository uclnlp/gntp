# -*- coding: utf-8 -*-

import logging
import tensorflow as tf

from gntp.models.lfm import BaseLatentFeatureModel

logger = logging.getLogger(__name__)


class ComplEx(BaseLatentFeatureModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(emb_rel: tf.Tensor,
              emb_arg1: tf.Tensor,
              emb_arg2: tf.Tensor):
        """
        Scoring function for ComplEx.

        :param emb_rel: [B, 2E] Tensor
        :param emb_arg1: [B, 2E] Tensor
        :param emb_arg2: [B, 2E] Tensor
        :return:
        """
        # [B, E], [B, E] Tensors
        rel_real, rel_img = tf.split(emb_rel, 2, axis=1)
        # [B, E], [B, E] Tensors
        arg1_real, arg1_img = tf.split(emb_arg1, 2, axis=1)
        # [B, E], [B, E] Tensors
        arg2_real, arg2_img = tf.split(emb_arg2, 2, axis=1)

        # [B] Tensor
        score1 = tf.einsum("ij,ij->i", rel_real * arg1_real, arg2_real)
        # [B] Tensor
        score2 = tf.einsum("ij,ij->i", rel_real * arg1_img, arg2_img)
        # [B] Tensor
        score3 = tf.einsum("ij,ij->i", rel_img * arg1_real, arg2_img)
        # [B] Tensor
        score4 = tf.einsum("ij,ij->i", rel_img * arg1_img, arg2_real)

        # [B] Tensor
        return score1 + score2 + score3 - score4

    @staticmethod
    def score_sp(emb_rel: tf.Tensor,
                 emb_arg1: tf.Tensor,
                 all_emb_arg2: tf.Tensor):
        """
        Scoring function for ComplEx.

        :param emb_rel: [B, 2E] Tensor
        :param emb_arg1: [B, 2E] Tensor
        :param all_emb_arg2: [N, 2E] Tensor
        :return:
        """
        # [B, E], [B, E] Tensors
        rel_real, rel_img = tf.split(emb_rel, 2, axis=1)
        # [B, E], [B, E] Tensors
        arg1_real, arg1_img = tf.split(emb_arg1, 2, axis=1)
        # [N, E], [N, E] Tensors
        all_arg2_real, all_arg2_img = tf.split(all_emb_arg2, 2, axis=1)

        # [B, N] Tensor
        score1 = tf.einsum("ij,nj->in", rel_real * arg1_real, all_arg2_real)
        # [B, N] Tensor
        score2 = tf.einsum("ij,nj->in", rel_real * arg1_img, all_arg2_img)
        # [B, N] Tensor
        score3 = tf.einsum("ij,nj->in", rel_img * arg1_real, all_arg2_img)
        # [B, N] Tensor
        score4 = tf.einsum("ij,nj->in", rel_img * arg1_img, all_arg2_real)

        # [B, N] Tensor
        return score1 + score2 + score3 - score4

    @staticmethod
    def score_po(emb_rel: tf.Tensor,
                 all_emb_arg1: tf.Tensor,
                 emb_arg2: tf.Tensor):
        """
        Scoring function for ComplEx.

        :param emb_rel: [B, 2E] Tensor
        :param all_emb_arg1: [N, 2E] Tensor
        :param emb_arg2: [B, 2E] Tensor
        :return:
        """
        # [B, E], [B, E] Tensors
        rel_real, rel_img = tf.split(emb_rel, 2, axis=1)
        # [N, E], [N, E] Tensors
        all_arg1_real, all_arg1_img = tf.split(all_emb_arg1, 2, axis=1)
        # [B, E], [B, E] Tensors
        arg2_real, arg2_img = tf.split(emb_arg2, 2, axis=1)

        # [B, N] Tensor
        score1 = tf.einsum("nj,ij->in", all_arg1_real, arg2_real * rel_real)
        # [B, N] Tensor
        score2 = tf.einsum("nj,ij->in", all_arg1_img, arg2_img * rel_real)
        # [B, N] Tensor
        score3 = tf.einsum("nj,ij->in", all_arg1_real, arg2_img * rel_img)
        # [B, N] Tensor
        score4 = tf.einsum("nj,ij->in", all_arg1_img, arg2_real * rel_img)

        # [B, N] Tensor
        return score1 + score2 + score3 - score4

    def L3(self,
           emb_vec: tf.Tensor):
        """
        L3 Regularization term - |u|^3
        :param emb_vec: [N, E] Tensor
        :return:
        """
        # [N, E], [N, E] Tensors
        vec_real, vec_img = tf.split(emb_vec, 2, axis=1)
        # [N, E] Tensor
        mod_vec = tf.sqrt(tf.pow(vec_real, 2) + tf.pow(vec_img, 2))
        mod_vec_cube = tf.pow(mod_vec, 3)
        # Scalar
        res = tf.reduce_sum(mod_vec_cube, [0, 1])
        return res
