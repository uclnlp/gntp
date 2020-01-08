# -*- coding: utf-8 -*-

from gntp.lookup.base import BaseLookupIndex
from gntp.lookup.nms import NMSLookupIndex
from gntp.lookup.faiss import FAISSLookupIndex
from gntp.lookup.random import RandomLookupIndex
from gntp.lookup.exact_knn import ExactKNNLookupIndex
from gntp.lookup.brute_knn import BruteKNNLookupIndex
from gntp.lookup.symbol import SymbolLookupIndex

from gntp.lookup.util import find_best_heads
from gntp.lookup.indexstore import LookupIndexStore


def get_index_class_by_name(name):
    name_to_lookup_index_class = {
        'nmslib': NMSLookupIndex,
        'faiss': FAISSLookupIndex,
        'faiss-cpu': FAISSLookupIndex,
        'random': RandomLookupIndex,
        'exact': ExactKNNLookupIndex,
        'brute': BruteKNNLookupIndex,
        'symbol': SymbolLookupIndex,
    }

    if name not in name_to_lookup_index_class:
        raise ModuleNotFoundError

    return name_to_lookup_index_class[name]


__all__ = [
    'BaseLookupIndex',
    'NMSLookupIndex',
    'find_best_heads',
    'LookupIndexStore'
]
