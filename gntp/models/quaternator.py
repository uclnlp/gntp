# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
from gntp.models.lfm import BaseLatentFeatureModel

logger = logging.getLogger(__name__)


class Quaternator(BaseLatentFeatureModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def score(emb_rel, emb_arg1, emb_arg2):
        rel_r, rel_x, rel_y, rel_z = tf.split(emb_rel, 4, axis=1)
        arg1_r, arg1_x, arg1_y, arg1_z = tf.split(emb_arg1, 4, axis=1)
        arg2_r, arg2_x, arg2_y, arg2_z = tf.split(emb_arg2, 4, axis=1)

        score1 = tf.einsum("ij,ij->i", arg1_r * arg2_r, rel_x)
        score2 = tf.einsum("ij,ij->i", arg1_r * arg2_x, rel_r)
        score3 = tf.einsum("ij,ij->i", arg1_r * arg2_y, rel_z)
        score4 = tf.einsum("ij,ij->i", arg1_r * arg2_z, rel_y)
        score5 = tf.einsum("ij,ij->i", arg1_x * arg2_r, rel_r)
        score6 = tf.einsum("ij,ij->i", arg1_x * arg2_x, rel_x)
        score7 = tf.einsum("ij,ij->i", arg1_x * arg2_y, rel_y)
        score8 = tf.einsum("ij,ij->i", arg1_x * arg2_z, rel_z)
        score9 = tf.einsum("ij,ij->i", arg1_y * arg2_r, rel_z)
        score10 = tf.einsum("ij,ij->i", arg1_y * arg2_x, rel_y)
        score11 = tf.einsum("ij,ij->i", arg1_y * arg2_y, rel_x)
        score12 = tf.einsum("ij,ij->i", arg1_y * arg2_z, rel_r)
        score13 = tf.einsum("ij,ij->i", arg1_z * arg2_r, rel_y)
        score14 = tf.einsum("ij,ij->i", arg1_z * arg2_x, rel_z)
        score15 = tf.einsum("ij,ij->i", arg1_z * arg2_y, rel_r)
        score16 = tf.einsum("ij,ij->i", arg1_z * arg2_z, rel_x)

        return (score1 - score2 + score3 - score4 +
                score5 + score6 - score7 - score8 -
                score9 + score10 + score11 - score12 +
                score13 + score14 + score15 + score16)
