# -*- coding: utf-8 -*-

import logging
import tensorflow as tf
from abc import abstractmethod
from gntp.models.base import BaseModel

logger = logging.getLogger(__name__)


class BaseLatentFeatureModel(BaseModel):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def score(emb_rel: tf.Tensor,
              emb_arg1: tf.Tensor,
              emb_arg2: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @staticmethod
    # @abstractmethod
    def score_sp(emb_rel: tf.Tensor,
                 emb_arg1: tf.Tensor,
                 all_emb_arg2: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    @staticmethod
    # @abstractmethod
    def score_po(emb_rel: tf.Tensor,
                 all_emb_arg1: tf.Tensor,
                 emb_arg2: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def multiclass_loss(self,
                        p_emb: tf.Tensor,
                        s_emb: tf.Tensor,
                        o_emb: tf.Tensor,
                        all_emb: tf.Tensor) -> tf.Tensor:
        # [B]
        x_ijk = self.score(p_emb, s_emb, o_emb)
        # [N,
        # [B, N]
        x_ij = self.score_sp(p_emb, s_emb, all_emb)
        x_jk = self.score_po(p_emb, all_emb, o_emb)
        # [B]
        lse_x_ij = tf.reduce_logsumexp(x_ij, 1)
        lse_x_jk = tf.reduce_logsumexp(x_jk, 1)
        # [B]
        losses = - x_ijk + lse_x_ij - x_ijk + lse_x_jk
        # Scalar
        loss = tf.reduce_mean(losses)
        return loss

    def predict(self,
                emb_rel: tf.Tensor,
                emb_arg1: tf.Tensor,
                emb_arg2: tf.Tensor,
                is_probability: bool = True) -> tf.Tensor:
        logits = self.score(emb_rel, emb_arg1, emb_arg2)
        return tf.sigmoid(logits) if is_probability is True else logits
