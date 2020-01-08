# -*- coding: utf-8 -*-

from gntp.types import Tensor
from gntp.base import ProofState, NTPParams
from gntp.neuralkb import NeuralKB
from gntp import kernels
from gntp.prover import neural_or, unify, joint_unify
from gntp.util import is_variable, is_tensor, atom_to_str
from gntp.losses import logistic_loss
from gntp.io import read_triples, read_labeled_triples, triples_to_vectors, read_mentions
from gntp.util import make_batches, make_epoch_batches, generate_indices, corrupt_triples
from gntp import models
from gntp.parse import parse_clause
from gntp.masking import create_mask

from gntp import evaluation
from gntp import explanation

from gntp.kmax import k_max
from gntp.decode import decode, decode_v2

from gntp.mentions import pad_sequences

from gntp.attention import sparse_softmax
from gntp.regularizers.entropy import entropy_logits
from gntp.regularizers.diversity import diversity

from gntp import prover
from gntp import lookup
from gntp import readers
from gntp import mentions


__all__ = [
    'Tensor',
    'kernels',
    'ProofState',
    'NTPParams',
    'NeuralKB',
    'kernels',
    'neural_or',
    'unify',
    'joint_unify',
    'is_variable',
    'is_tensor',
    'atom_to_str',
    'logistic_loss',
    'read_triples',
    'read_labeled_triples',
    'triples_to_vectors',
    'read_mentions',
    'make_batches',
    'make_epoch_batches',
    'generate_indices',
    'corrupt_triples',
    'models',
    'parse_clause',
    'create_mask',
    'evaluation',
    'explanation',
    'k_max',
    'decode',
    'decode_v2',
    'pad_sequences',
    'sparse_softmax',
    'entropy_logits',
    'diversity',
    'prover',
    'lookup',
    'readers',
    'mentions',
]
