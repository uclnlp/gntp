# -*- coding: utf-8 -*-

import tensorflow as tf


def attention(logits, mask):
    attention_mask = tf.nn.softmax(mask)
    clause_embeddings = tf.einsum('cr,re->ce', attention_mask, logits)
    return clause_embeddings


def sparse_softmax(logits, exponent=None, eps=1e-6, axis=-1):
    _logits = tf.nn.relu(logits) + eps
    _logits = _logits ** exponent if exponent is not None else _logits
    normalizer = tf.reduce_sum(_logits, axis=axis)
    res = tf.einsum('cr,c->cr', _logits, 1.0 / normalizer)
    res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
    return res
