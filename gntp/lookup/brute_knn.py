# -*- coding: utf-8 -*-

import numpy as np
from gntp.lookup.base import BaseLookupIndex
import logging

logger = logging.getLogger(__name__)


class BruteKNNLookupIndex(BaseLookupIndex):
    def __init__(self,
                 data: np.ndarray):
        super().__init__()

        self.times_queried = 0

        # creating the index
        self.index = None
        self.build_index(data)

    def query(self,
              data: np.ndarray,
              k: int = 10,
              is_training: bool = False):
        nb_instances = data.shape[0]
        # Euclidean distances
        res = []
        for i in range(nb_instances):
            sqd = np.sqrt(((self.index - data[i, :]) ** 2).sum(axis=1))
            indices = np.argsort(sqd)  # this can be sped up with argpartition
            top_k_indices = indices[:k].tolist()
            res += [top_k_indices]
        self.times_queried += 1 if is_training else 0
        res = np.array(res)
        return res

    def build_index(self,
                    data: np.ndarray):
        self.index = data
        self.times_queried = 0
