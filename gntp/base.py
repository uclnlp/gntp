# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf
from termcolor import colored

import gntp
from gntp.kernels import BaseKernel
from gntp.lookup import LookupIndexStore

from gntp.kernels import RBFKernel

import logging

from typing import List, Dict, Optional, Callable

logger = logging.getLogger(__name__)


class ProofState:
    def __init__(self,
                 substitutions: Dict[str, tf.Tensor],
                 scores: Optional[tf.Tensor],
                 index_substitutions: Optional[Dict[str, np.ndarray]] = None,
                 index_mappers: Optional[Dict[int, np.ndarray]] = None,
                 index_kb: Optional[List[int]] = None):
        self._substitutions = substitutions
        self._scores = scores
        self._index_substitutions = index_substitutions
        self._index_mappers = index_mappers
        self._index_coordinates = None
        self._index_kb = index_kb

    @property
    def substitutions(self):
        return self._substitutions

    @property
    def scores(self):
        return self._scores

    @property
    def index_substitutions(self):
        return self._index_substitutions

    @property
    def index_mappers(self):
        return self._index_mappers

    @property
    def index_coordinates(self):
        return self._index_coordinates

    @index_coordinates.setter
    def index_coordinates(self, value):
        self._index_coordinates = value

    @property
    def index_kb(self):
        return self._index_kb

    @index_kb.setter
    def index_kb(self, value):
        self._index_kb = value

    @staticmethod
    def _t(t):
        if gntp.is_tensor(t):
            content = '{0}, {1:.2f}, {2:.2f}'.format(t.get_shape(), tf.reduce_min(t), tf.reduce_max(t))
        else:
            content = t
        return 'Tensor({})'.format(content)

    @staticmethod
    def _d(d):
        return '{' + ' '.join([key + ': ' + ProofState._t(tensor) for key, tensor in d.items()]) + '}'

    def __str__(self):
        return '{0}: {1}\n{2}: {3}'.format(colored('PS', 'green'), ProofState._t(self.scores),
                                           colored('SUB', 'blue'), ProofState._d(self.substitutions))


class NTPParams:
    def __init__(self,
                 kernel: Optional[BaseKernel] = None,
                 max_depth: int = 1,
                 k_max: Optional[int] = None,
                 mask_indices: Optional[np.ndarray] = None,
                 retrieve_k_facts: Optional[int] = None,
                 retrieve_k_rules: Optional[int] = None,
                 unification_op: Callable = None,
                 index_refresh_rate: Optional[int] = None,
                 index_store: Optional[LookupIndexStore] = None,
                 is_training: bool = False,
                 support_explanations: bool = False,
                 unification_score_aggregation: str = 'min',
                 facts_kb: Optional[List[np.ndarray]] = None):

        self._kernel = kernel if kernel is not None else RBFKernel()
        self._max_depth = max_depth
        self._k_max = k_max
        self._mask_indices = mask_indices
        self._retrieve_k_facts = retrieve_k_facts
        self._retrieve_k_rules = retrieve_k_rules
        self._unification_op = unification_op
        self._index_refresh_rate = index_refresh_rate
        self._index_store = index_store
        self._is_training = is_training
        self._support_explanations = support_explanations
        self._unification_score_aggregation = unification_score_aggregation
        self._facts_kb = facts_kb
        # For internal use only
        self.traversed_rule_indexes = set()

    @property
    def kernel(self):
        return self._kernel

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def k_max(self):
        return self._k_max

    @property
    def mask_indices(self):
        return self._mask_indices

    @property
    def retrieve_k_facts(self):
        return self._retrieve_k_facts

    @property
    def retrieve_k_rules(self):
        return self._retrieve_k_rules

    @property
    def unification_op(self):
        return self._unification_op

    @property
    def index_refresh_rate(self):
        return self._index_refresh_rate

    @property
    def index_store(self):
        return self._index_store

    @property
    def is_training(self):
        return self._is_training

    @property
    def support_explanations(self):
        return self._support_explanations

    @property
    def facts_kb(self):
        return self._facts_kb

    @property
    def unification_score_aggregation(self):
        return self._unification_score_aggregation
