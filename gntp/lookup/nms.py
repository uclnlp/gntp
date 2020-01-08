# -*- coding: utf-8 -*-

import numpy as np
import nmslib
from gntp.lookup.base import BaseLookupIndex
import logging

logger = logging.getLogger(__name__)


class NMSLookupIndex(BaseLookupIndex):
    def __init__(self,
                 data: np.ndarray,
                 method='hnsw', space='l2', num_threads=4, m=15, efc=100, efs=100):
        super().__init__()

        self.method = method
        self.space = space
        self.num_threads = num_threads
        self.m = m
        self.efc = efc
        self.efs = efs

        self.times_queried = 0

        self._index_time_params = {
            'M': self.m,
            'indexThreadQty': self.num_threads,
            'efConstruction': self.efc,
            'post': 0
        }

        self._query_time_params = {
            'efSearch': self.efs
        }

        # creating the index
        self.index = None
        self.build_index(data)

    def query(self,
              data: np.ndarray,
              k: int = 10,
              is_training: bool = False):
        neighbours = self.index.knnQueryBatch(data, k=k, num_threads=self.num_threads)
        neighbour_indexes = np.array([index for index, _ in neighbours])
        self.times_queried += 1 if is_training else 0
        return neighbour_indexes

    def build_index(self,
                    data: np.ndarray):
        index = nmslib.init(method=self.method, space=self.space,
                            data_type=nmslib.DataType.DENSE_VECTOR)
        index.addDataPointBatch(data)
        index.createIndex(self._index_time_params, print_progress=False)
        index.setQueryTimeParams(self._query_time_params)
        self.index = index
        self.times_queried = 0
