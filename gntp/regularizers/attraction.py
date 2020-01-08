# -*- coding: utf-8 -*-

import tensorflow as tf


def attractor(kernel, attractors, attracted):
    # attractors is AxE, attracted is BxE
    pairwise_dissimilarities = 1.0 - kernel.pairwise(attractors, attracted)
    # c is AxB
    min_dissimilarities = tf.reduce_min(pairwise_dissimilarities, axis=0)
    # d is B
    mean_min_dissimilarity = tf.reduce_mean(min_dissimilarities, axis=0)
    return mean_min_dissimilarity
