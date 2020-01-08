# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
from gntp.models.lfm import BaseLatentFeatureModel

logger = logging.getLogger(__name__)


class TransE(BaseLatentFeatureModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    def negative_l2_distance(x1, x2, axis=1):
        """
        Negative L2 Distance.

        .. math:: L = - \\sqrt{\\sum_i (x1_i - x2_i)^2}

        :param x1: First term.
        :param x2: Second term.
        :param axis: Reduction Indices.
        :return: Similarity Value.
        """

        distance = tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), axis=axis))
        return - distance

    @staticmethod
    def score(emb_rel, emb_arg1, emb_arg2):
        return TransE.negative_l2_distance(emb_arg1 + emb_rel, emb_arg2)
