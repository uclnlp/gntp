# -*- coding: utf-8 -*-

import tensorflow as tf


def entropy_logits(logits):
    return tf.reduce_sum(- tf.log(tf.nn.softmax(logits)) * tf.nn.softmax(logits))


def entropy(probabilities):
    return tf.reduce_sum(- tf.log(probabilities) * probabilities)
