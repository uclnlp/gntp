# -*- coding: utf-8 -*-

import numpy as np
from gntp.lookup.base import BaseLookupIndex
import logging
logger = logging.getLogger(__name__)


class RandomLookupIndex(BaseLookupIndex):
    def __init__(self,
                 data: np.ndarray,
                 random_state: np.random.RandomState):
        super().__init__()

        self.times_queried = 0

        # creating the index
        self.random_state = random_state
        self.index = None
        self.build_index(data)

    def query(self,
              data: np.ndarray,
              k: int = 10,
              is_training: bool = False):
        return self.random_state.choice(self.index, size=[data.shape[0], k])

    def build_index(self, data):
        self.index = np.arange(data.shape[0])
