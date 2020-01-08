# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


def clip(x: tf.Tensor,
         epsilon: float = 1e-10):
    return tf.clip_by_value(x, epsilon, 1.0)


def logistic_loss(probabilities: tf.Tensor,
                  targets: np.ndarray,
                  epsilon: float = 1e-10):
    # this is sigmoid_cross_entropy without logits
    loss = - targets * tf.log(clip(probabilities, epsilon)) - (1 - targets) * tf.log(clip(1.0 - probabilities, epsilon))
    return loss
