# -*- coding: utf-8 -*-

import tensorflow as tf

import gntp

from gntp.base import ProofState
from gntp.models.base import BaseModel
from gntp.indices import SymbolIndices
from gntp.kernels.base import BaseKernel

from gntp.lookup import LookupIndexStore

from typing import List, Union, Optional, Tuple

import logging
import numpy as np

logger = logging.getLogger(__name__)


class NTP(BaseModel):
    def __init__(self,
                 kernel: Optional[BaseKernel] = None,
                 max_depth: int = 1,
                 k_max: Optional[int] = None,
                 retrieve_k_facts: Optional[int] = None,
                 retrieve_k_rules: Optional[int] = None,
                 index_refresh_rate: Optional[int] = None,
                 index_store: Optional[LookupIndexStore] = None,
                 unification_type: Optional[str] = None,
                 facts_kb: Optional[List[np.ndarray]] = None
                 ):
        super().__init__()
        self.kernel = kernel if kernel is not None else gntp.kernels.RBFKernel()

        self.max_depth = max_depth
        self.k_max = k_max

        self.retrieve_k_facts = retrieve_k_facts
        self.retrieve_k_rules = retrieve_k_rules

        self.index_refresh_rate = index_refresh_rate
        self.index_store = index_store

        self.initializer = tf.random_uniform_initializer(-1.0, 1.0)
        self.neural_facts_kb = self.neural_rules_kb = None

        if unification_type is None:
            unification_type = 'classic'

        unification_type_to_op = {
            'classic': gntp.unify,
            'joint': gntp.joint_unify
        }

        assert unification_type in unification_type_to_op
        self.unification_op = unification_type_to_op.get(unification_type)

        self.facts_kb = facts_kb

    def predict(self,
                goal_predicate_embeddings: tf.Tensor,
                goal_subject_embeddings: tf.Tensor,
                goal_object_embeddings: tf.Tensor,
                neural_facts_kb: List[tf.Tensor],
                neural_rules_kb: List[List[List[Union[tf.Tensor, str]]]],
                goal_predicates: Optional[np.ndarray] = None,
                goal_subjects: Optional[np.ndarray] = None,
                goal_objects: Optional[np.ndarray] = None,
                mask_indices: Optional[np.ndarray] = None,
                is_training: bool = False,
                target_inputs: Optional[np.ndarray] = None,
                mixed_losses: bool = False,
                aggregator_type: str = 'mean',
                no_ntp0: bool = False,
                only_ntp0: bool = False,
                support_explanations: bool = False,
                unification_score_aggregation: str = 'min',
                multimax: bool = False,
                tensorboard: bool = False) -> Tuple[tf.TensorShape, Tuple[List[ProofState], Optional[np.ndarray]]]:
        """
        Computes the goal scores of input triples (provided as embeddings).

        :param goal_predicate_embeddings: [G, E] tensor of predicate embeddings.
        :param goal_subject_embeddings: [G, E] tensor of subject embeddings.
        :param goal_object_embeddings: [G, E] tensor of object embeddings.
        :param neural_facts_kb: [[K, E], [K, E], [K, E]] tensor list
        :param neural_rules_kb: [[[s_1, s_2, .., s_n]]] list, where each [s_1, s_2, .., s_n] is an atom
        :param goal_predicates: [G] tensor of predicates.
        :param goal_subjects: [G] tensor of subjects.
        :param goal_objects: [G] tensor of objects.
        :param mask_indices: [G] vector containing the index of the fact we want to mask in the Neural KB.
        :param is_training: Flag denoting whether it is the training phase or not.
        :param target_inputs: [G] {0, 1} vector or None.
        :param mixed_losses: If true, then the average proof score is used for
                        0 goals, and the maximum proof score for 1 goals.
        :param aggregator_type: value in {'mean', 'sum'} - only for cases where is_pos is defined.
        :param no_ntp0: Boolean flag, specifies whether to use NTP0 or not.
        :param only_ntp0:
        :param support_explanations: collect the info necessary to provide explanations later on
        :param unification_score_aggregation: 'min' for minimum, 'mul' for multiplication, 'minmul' for (min+mul)/2
        :param multimax: propagate gradients through ALL maximum paths, not just one
        :param tensorboard: should we collect tensorboard data?
        :return: [G] goal scores.
        """
        goals = [
            goal_predicate_embeddings,
            goal_subject_embeddings,
            goal_object_embeddings
        ]

        neural_kb = neural_rules_kb + [[neural_facts_kb]]

        goal_indices = index_substitutions = None
        if goal_predicates is not None and goal_subjects is not None and goal_objects is not None and self.facts_kb:
            goal_indices = [
                SymbolIndices(indices=goal_predicates, is_fact=True),
                SymbolIndices(indices=goal_subjects, is_fact=True),
                SymbolIndices(indices=goal_objects, is_fact=True)
            ]
            index_substitutions = {}

        start_proof_state = gntp.ProofState(substitutions={},
                                            scores=None,
                                            index_mappers=None,
                                            index_substitutions=index_substitutions)

        ntp_params = gntp.NTPParams(kernel=self.kernel,
                                    max_depth=self.max_depth,
                                    k_max=self.k_max,
                                    mask_indices=mask_indices,
                                    retrieve_k_facts=self.retrieve_k_facts,
                                    retrieve_k_rules=self.retrieve_k_rules,
                                    unification_op=self.unification_op,
                                    index_refresh_rate=self.index_refresh_rate,
                                    index_store=self.index_store,
                                    is_training=is_training,
                                    support_explanations=support_explanations,
                                    unification_score_aggregation=unification_score_aggregation,
                                    facts_kb=self.facts_kb if self.facts_kb and goal_indices else None)

        proof_states = gntp.neural_or(neural_kb=neural_kb,
                                      goals=goals,
                                      proof_state=start_proof_state,
                                      ntp_params=ntp_params,
                                      goal_indices=goal_indices,
                                      no_ntp0=no_ntp0,
                                      only_ntp0=only_ntp0)

        goal_scores_lst = []
        num_proofs_lst = []

        new_target_inputs = target_inputs
        for proof_state in proof_states:
            axis = tf.constant(np.arange(len(proof_state.scores.shape) - 1))
            num_proofs_lst.append(np.prod(proof_state.scores.shape[:-1]))

            proof_goal_scores_lst = []
            if mixed_losses and is_training:
                scores_pos = tf.reduce_max(proof_state.scores, axis=axis)
                scores_neg = tf.reduce_sum(proof_state.scores, axis=axis)
                proof_goal_scores = scores_pos * target_inputs + scores_neg * (1 - target_inputs)
            else:
                proof_goal_scores = tf.reduce_max(proof_state.scores, axis=axis)

                if multimax:
                    # # @pminervini if you want an epsilon here, it's as easy as:
                    # epsilon = 0.0000000001
                    # maxima_where = tf.greater(proof_state.scores, proof_goal_scores - epsilon)

                    proof_state_scores_unstacked = tf.unstack(proof_state.scores, axis=-1)
                    maxima_where = tf.equal(proof_goal_scores, proof_state.scores)
                    maxima_where_unstacked = tf.unstack(maxima_where, axis=-1)

                    proof_goal_scores_lst = [tf.boolean_mask(pss, mw)
                                             for pss, mw in zip(proof_state_scores_unstacked, maxima_where_unstacked)]

                # NOTE this only works when using maxing, otherwise it's a ruckus
                if ntp_params.support_explanations:
                    scores = proof_state.scores.numpy()
                    raw_indices = scores.reshape(-1, scores.shape[-1]).argmax(0)
                    argmax_indices = np.column_stack(np.unravel_index(raw_indices, scores[..., 0].shape))
                    full_indices = np.concatenate(
                        (argmax_indices, np.reshape(np.arange(scores.shape[-1]), [scores.shape[-1], 1])), axis=1)
                    reduce_max_test = np.asarray([scores[tuple(index)] for index in full_indices])
                    assert np.all(proof_goal_scores.numpy() == reduce_max_test)
                    proof_state.index_coordinates = full_indices
                    # index_kb is collected in the reverse order
                    proof_state.index_kb.reverse()

            if not multimax:
                goal_scores_lst += [proof_goal_scores]
            else:
                goal_scores_lst += [proof_goal_scores_lst]

        if mixed_losses and is_training:
            concat = tf.concat([tf.reshape(g, [1, -1]) for g in goal_scores_lst], 0)
            scores_pos = tf.reduce_max(concat, axis=0)

            normalizer = 1.0
            if aggregator_type == 'mean':
                # normalizer = tf.reduce_sum(num_proofs_lst)
                normalizer = np.sum(num_proofs_lst)

            scores_neg = tf.reduce_sum(concat, axis=0) / tf.cast(normalizer, tf.float32)
            goal_scores = scores_pos * target_inputs + scores_neg * (1 - target_inputs)
        else:

            last_dim_size = proof_states[0].scores.shape[-1]
            if multimax:
                per_batch_stuff = [tf.concat([tf.reshape(g[i], [1, -1]) for g in goal_scores_lst], 1)
                                   for i in range(last_dim_size)]
                maxima = [tf.reduce_max(l) for l in per_batch_stuff]
                maxima_where = [tf.equal(l, m) for l, m in zip(per_batch_stuff, maxima)]

                how_many_maxima = [tf.reduce_sum(tf.cast(item, dtype=tf.int32)).numpy() for item in maxima_where]

                new_target_inputs = np.concatenate([target_in * np.ones([how_many])
                                                    for how_many, target_in in zip(how_many_maxima, target_inputs)])

                if tensorboard:
                    this_many = tf.reduce_sum(how_many_maxima)

                    with tf.contrib.summary.always_record_summaries():
                        tf.contrib.summary.scalar('multimax_gain_total',
                                                  this_many.numpy() / last_dim_size.value)
                        tf.contrib.summary.scalar('multimax_gain_positive',
                                                  np.sum(new_target_inputs) / np.sum(target_inputs))
                        tf.contrib.summary.scalar('multimax_gain_negative',
                                                  ((this_many.numpy() - np.sum(new_target_inputs)) /
                                                   (last_dim_size.value - np.sum(target_inputs))))
                goal_scores = tf.concat([tf.boolean_mask(l, mw)
                                         for l, mw in zip(per_batch_stuff, maxima_where)],
                                        axis=0)
            else:
                maximum_paths = tf.concat([tf.reshape(g, [1, -1]) for g in goal_scores_lst], 0)
                goal_scores = tf.reduce_max(maximum_paths, axis=0)

            # this is just a check
            if ntp_params.support_explanations:
                goal_scores_check = [goal_score.numpy() for goal_score in goal_scores_lst]
                goal_scores_check_stacked = np.column_stack(goal_scores_check)

                argmax_indices = np.argmax(goal_scores_check_stacked, axis=1)
                argmax_indices = np.transpose(np.vstack((np.arange(goal_scores_check_stacked.shape[0]),
                                                         argmax_indices)))

                goal_scores_check = np.asarray([goal_scores_check_stacked[tuple(i)] for i in argmax_indices])
                assert np.all(goal_scores_check == goal_scores.numpy())

        return goal_scores, (proof_states, new_target_inputs)
