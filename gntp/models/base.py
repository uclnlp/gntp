# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
import gntp
from abc import ABC

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    def __init__(self):
        pass

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def loss(target_inputs, goal_scores, aggregator: str = 'sum'):
        losses = gntp.logistic_loss(goal_scores, target_inputs)
        name_to_aggregator = {
            'sum': tf.reduce_sum,
            'mean': tf.reduce_mean
        }
        if aggregator not in name_to_aggregator:
            raise NotImplementedError

        aggregator_op = name_to_aggregator[aggregator]
        return aggregator_op(losses)
