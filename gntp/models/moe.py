# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import gntp
from gntp.neuralkb import NeuralKB
from gntp.models.base import BaseModel

from typing import Optional, Any

import logging

logger = logging.getLogger(__name__)

layers = tf.keras.layers


class MoE(BaseModel):
    def __init__(self,
                 ntp_model,
                 aux_model,
                 moe_mixer_type,
                 model_type,
                 mixed_losses,
                 evaluation_mode,
                 mixed_losses_aggregator_type=None,
                 is_no_ntp0=False,
                 entity_embedding_size=None,
                 kernel_parameters=None):
        super().__init__()

        self.ntp_model = ntp_model
        self.aux_model = aux_model
        self.moe_mixer_type = moe_mixer_type
        self.model_type = model_type
        self.aux_proj = None

        self.kernel_parameters = kernel_parameters

        if model_type == 'moe2':
            self.aux_proj = layers.Dense(entity_embedding_size)

        self.mixed_losses = mixed_losses
        self.evaluation_mode = evaluation_mode
        self.mixed_losses_aggregator_type = mixed_losses_aggregator_type
        self.is_no_ntp0 = is_no_ntp0

    def get_trainable_variables(self,
                                neural_kb: NeuralKB,
                                is_rules_only: bool = False,
                                is_facts_only: bool = False,
                                is_entities_only: bool = False,
                                is_rules_entities_only: bool = False):

        res = neural_kb.get_trainable_variables(is_rules_only=is_rules_only,
                                                is_facts_only=is_facts_only,
                                                is_entities_only=is_entities_only,
                                                is_rules_entities_only=is_rules_entities_only)

        res += self.aux_proj.trainable_variables if self.aux_proj is not None else []

        if self.kernel_parameters is not None:
            res += self.kernel_parameters

        return res

    def predict(self,
                goal_predicates: np.ndarray,
                goal_subjects: np.ndarray,
                goal_objects: np.ndarray,
                neural_kb: NeuralKB,
                target_inputs: Optional[np.ndarray] = None,
                mask_indices: Optional[np.ndarray] = None,
                is_training: bool = True,
                is_auxiliary: bool = True,
                is_model: bool = True,
                support_explanations: bool = False,
                only_ntp0: bool = False):
        """
        :param goal_predicates: [G] int32 vector
        :param goal_subjects: [G] int32 vector
        :param goal_objects: [G] int32 vector
        :param neural_kb:
        :param target_inputs:
        :param mask_indices: [G] vector containing the index of the fact we want to mask in the Neural KB.
        :param is_training: Flag denoting whether it is the training phase or not.
        :param is_auxiliary:
        :param is_model:
        :param support_explanations:
        :param only_ntp0:
        :return:
        """
        proof_states = None

        p_emb = tf.nn.embedding_lookup(neural_kb.relation_embeddings, goal_predicates)
        s_emb = tf.nn.embedding_lookup(neural_kb.entity_embeddings, goal_subjects)
        o_emb = tf.nn.embedding_lookup(neural_kb.entity_embeddings, goal_objects)

        model_p_emb, model_s_emb, model_o_emb = p_emb, s_emb, o_emb
        aux_p_emb, aux_s_emb, aux_o_emb = p_emb, s_emb, o_emb

        if neural_kb.aux_entity_embeddings is not None and neural_kb.aux_predicate_embeddings is not None:
            aux_p_emb = tf.nn.embedding_lookup(neural_kb.aux_predicate_embeddings, goal_predicates)
            aux_s_emb = tf.nn.embedding_lookup(neural_kb.aux_entity_embeddings, goal_subjects)
            aux_o_emb = tf.nn.embedding_lookup(neural_kb.aux_entity_embeddings, goal_objects)

        if self.aux_proj:
            aux_p_emb = self.aux_proj(p_emb)
            aux_s_emb = self.aux_proj(s_emb)
            aux_o_emb = self.aux_proj(o_emb)

        model_scores = None
        if self.ntp_model is not None and is_model is True:

            is_pos = None
            if self.mixed_losses and is_training:
                is_pos = target_inputs

            def npy(tensor: Any) -> np.ndarray:
                return tensor.numpy() if gntp.is_tensor(tensor) else tensor

            model_scores, (proof_states, _) = self.ntp_model.predict(goal_predicate_embeddings=model_p_emb,
                                                                     goal_subject_embeddings=model_s_emb,
                                                                     goal_object_embeddings=model_o_emb,
                                                                     goal_predicates=npy(goal_predicates),
                                                                     goal_subjects=npy(goal_subjects),
                                                                     goal_objects=npy(goal_objects),
                                                                     neural_facts_kb=neural_kb.neural_facts_kb,
                                                                     neural_rules_kb=neural_kb.neural_rules_kb,
                                                                     mask_indices=mask_indices,
                                                                     is_training=is_training,
                                                                     mixed_losses=is_pos is not None,
                                                                     target_inputs=is_pos,
                                                                     aggregator_type=self.mixed_losses_aggregator_type,
                                                                     no_ntp0=self.is_no_ntp0,
                                                                     only_ntp0=only_ntp0,
                                                                     support_explanations=support_explanations)

        aux_scores = None
        if self.aux_model is not None and is_auxiliary is True:
            is_probability = True

            # During testing, if the model is just a Neural Link Predictor,
            # do not use a non-linearity on the output layer if it is a ranking task:
            # it is not needed, and it might give non-accurate results due to the non-linearity saturating.
            if is_training is False and self.model_type in {'nlp'} and self.evaluation_mode in {'ranking'}:
                is_probability = False

            aux_scores = self.aux_model.predict(aux_p_emb, aux_s_emb, aux_o_emb,
                                                is_probability=is_probability)

        assert (model_scores is not None) or (aux_scores is not None)

        if model_scores is None:
            res = aux_scores
        elif aux_scores is None:
            res = model_scores
        else:
            if self.moe_mixer_type == 'mean':
                res = (model_scores + aux_scores) / 2.0
            else:
                res = tf.maximum(model_scores, aux_scores)

        return res, proof_states
