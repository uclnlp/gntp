# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np

import logging

logger = logging.getLogger(__name__)


class BaseLookupIndex(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def query(self,
              data: np.ndarray,
              k: int = 10,
              is_training: bool = False):
        raise NotImplementedError

    @abstractmethod
    def build_index(self,
                    data: np.ndarray):
        raise NotImplementedError
