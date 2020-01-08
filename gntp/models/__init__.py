# # -*- coding: utf-8 -*-

from gntp.models.ntp import NTP
from gntp.models.moe import MoE

from gntp.models.transe import TransE
from gntp.models.distmult import DistMult
from gntp.models.complex import ComplEx
from gntp.models.quaternator import Quaternator


def get_model_by_name(name):
    name_to_class = {
        'complex': ComplEx,
        'distmult': DistMult,
        'transe': TransE,
        'quaternions': Quaternator,
        'quaternator': Quaternator
    }
    if name not in name_to_class:
        raise ModuleNotFoundError
    model_class = name_to_class[name]
    return model_class()


__all__ = [
    'NTP',
    'TransE',
    'DistMult',
    'ComplEx',
    'MoE',
    'Quaternator',
    'get_model_by_name'
]
