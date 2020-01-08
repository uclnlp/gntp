# -*- coding: utf-8 -*-

import numpy as np

from typing import List, Union

import gntp

from gntp.indices import SymbolIndices
from gntp.neuralkb import NeuralKB
from gntp.kernels import BaseKernel
from gntp.lookup.base import BaseLookupIndex

from typing import Optional

import logging

logger = logging.getLogger(__name__)


class SymbolLookupIndex(BaseLookupIndex):
    def __init__(self,
                 neural_kb: Optional[NeuralKB] = None,
                 kernel: Optional[BaseKernel] = None,
                 **args):
        super().__init__()
        # print('SymbolLookupIndex({})'.format(type(data)))

        self.neural_kb = neural_kb
        self.kernel = kernel

        assert self.neural_kb is not None
        assert self.kernel is not None

        self.kernel_value_cache = {}

        self.times_queried = 0

    def query_sym(self,
                  data_indices: List[Union[SymbolIndices, str]],
                  position: int,
                  k: int,
                  is_training: bool):
        neural_kb = self.neural_kb.neural_rules_kb + [[self.neural_kb.neural_facts_kb]]
        fact_position = len(self.neural_kb.neural_rules_kb)
        facts_kb = self.neural_kb.facts_kb

        rule = neural_kb[position]
        rule_head = rule[0]

        rule_is_fact = fact_position == position

        assert rule_is_fact == (len(rule) == 1)

        nb_rules = rule_head[0].shape[0]
        nb_goals = np.prod(data_indices[0].indices.shape)
        goal_rule_scores = np.ones([nb_goals, nb_rules])

        for ge_idx, goal_elem in enumerate(data_indices):
            ge_key = rhe_key = None

            if isinstance(goal_elem, SymbolIndices):
                ge_key = (fact_position, 0, ge_idx if ge_idx < 2 else 1)
                if not goal_elem.is_fact:
                    ge_key = (goal_elem.rule_idx, goal_elem.atom_idx, ge_idx)

            if not gntp.is_variable(rule_head[ge_idx]):
                rhe_key = (fact_position, 0, ge_idx if ge_idx < 2 else 1)
                if not rule_is_fact:
                    rhe_key = (position, 0, ge_idx)

            if ge_key is not None and rhe_key is not None:
                kernel_matrix = self.compute_kernel(rhe_key, ge_key)

                goal_indices = goal_elem.indices
                goal_indices_1d = goal_indices.reshape([-1])

                slow = False

                if slow is True:
                    local_goal_rule_scores = np.zeros([nb_goals, nb_rules])

                    for goal_idx, goal_symbol in enumerate(goal_indices_1d):
                        for rule_idx in range(nb_rules):
                            rule_sym_idx = facts_kb[ge_idx][rule_idx] if rule_is_fact else rule_idx
                            new_value = kernel_matrix[rule_sym_idx, goal_symbol]
                            local_goal_rule_scores[goal_idx, rule_idx] = new_value

                    goal_rule_scores = np.minimum(goal_rule_scores, local_goal_rule_scores)
                else:
                    k_x, k_y = [], []

                    for goal_idx, goal_symbol in enumerate(goal_indices_1d):
                        for rule_idx in range(nb_rules):
                            rule_sym_idx = facts_kb[ge_idx][rule_idx] if rule_is_fact else rule_idx
                            k_x += [rule_sym_idx]
                            k_y += [goal_symbol]

                    local_goal_rule_scores = kernel_matrix[k_x, k_y].reshape([nb_goals, nb_rules])
                    goal_rule_scores = np.minimum(goal_rule_scores, local_goal_rule_scores)

        indices = np.argsort(goal_rule_scores)

        top_k_indices = indices[:, :k].reshape(data_indices[0].indices.shape + (k,))

        self.times_queried += 1 if is_training else 0

        return top_k_indices

    def compute_kernel(self, ge_key, rhe_key):
        cache_key = (ge_key, rhe_key)
        cache_inv_key = (rhe_key, ge_key)

        if cache_key in self.kernel_value_cache:
            res = self.kernel_value_cache[cache_key]
        else:
            neural_kb = self.neural_kb.neural_rules_kb + [[self.neural_kb.neural_facts_kb]]
            fact_position = len(self.neural_kb.neural_rules_kb)

            ge_rule_id, ge_atom_id, ge_arg_id = ge_key
            rhe_rule_id, rhe_atom_id, rhe_arg_id = rhe_key

            ge_emb = self.neural_kb.relation_embeddings if ge_arg_id == 0 else self.neural_kb.entity_embeddings
            if ge_rule_id != fact_position:
                ge_emb = neural_kb[ge_rule_id][ge_atom_id][ge_arg_id]

            rhe_emb = self.neural_kb.relation_embeddings if rhe_arg_id == 0 else self.neural_kb.entity_embeddings
            if rhe_rule_id != fact_position:
                rhe_emb = neural_kb[rhe_rule_id][rhe_atom_id][rhe_arg_id]

            res = self.kernel.pairwise(ge_emb, rhe_emb).numpy()

            self.kernel_value_cache[cache_key] = res
            self.kernel_value_cache[cache_inv_key] = res.T
        return res

    def query(self,
              data: np.ndarray,
              k: int = 10,
              is_training: bool = False):
        raise NotImplementedError

    def build_index(self,
                    data: np.ndarray):
        raise NotImplementedError
