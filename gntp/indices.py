# -*- coding: utf-8 -*-

import numpy as np

import logging

logger = logging.getLogger(__name__)


class SymbolIndices:
    def __init__(self,
                 indices: np.ndarray,
                 is_fact: bool = False,
                 rule_idx: int = 0,
                 atom_idx: int = 0):
        self.indices = indices
        self.is_fact = is_fact
        self.rule_idx = rule_idx
        self.atom_idx = atom_idx

    def __repr__(self):
        return 'SymbolIndices({}, {}, {}, {})'.format(self.indices.shape, self.is_fact, self.rule_idx, self.atom_idx)

    def __str__(self):
        return 'SymbolIndices({}, {}, {}, {})'.format(self.indices.shape, self.is_fact, self.rule_idx, self.atom_idx)
