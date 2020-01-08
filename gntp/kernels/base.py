# -*- coding: utf-8 -*-

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseKernel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def pairwise(self, x, y):
        raise NotImplementedError

    @abstractmethod
    def elementwise(self, x, y):
        raise NotImplementedError
