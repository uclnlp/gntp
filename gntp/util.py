# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import gntp
from gntp.indices import SymbolIndices

from termcolor import colored
from collections.abc import Iterable


def is_tensor(atom_elem):
    return isinstance(atom_elem, tf.Tensor) or isinstance(atom_elem, tf.Variable)


def is_variable(atom_elem):
    return isinstance(atom_elem, str) and atom_elem.isupper()


def atom_to_str(atom):
    if isinstance(atom, Iterable):
        def _to_show(e):
            return e.get_shape() if is_tensor(e) else e.shape if isinstance(e, np.ndarray) else e
        body_str = str([_to_show(e) for e in atom])
        body_str = body_str.replace('Dimension', '')
    else:
        body_str = str(atom)
    return '{} {}'.format(colored('A', 'red'), body_str)


def make_batches(size, batch_size):
    nb_batch = int(np.ceil(size / float(batch_size)))
    res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]
    return res


def make_epoch_batches(size, batch_size, examples_per_epoch, drop_last=False):
    start = 0
    epoch = 1
    res = []
    while start < size:
        end = start + batch_size
        if end > epoch * examples_per_epoch:
            end = epoch * examples_per_epoch
            epoch += 1
            if not drop_last and end != start:
                res.append((start, end))
        else:
            res.append((start, end))
        start = end
    return res


def generate_indices(random_state, n_samples, candidate_indices):
    shuffled_indices = candidate_indices[random_state.permutation(len(candidate_indices))]
    return shuffled_indices[np.arange(n_samples) % len(shuffled_indices)]


def corrupt_triples(random_state,
                    Xs_batch, Xp_batch, Xo_batch,
                    Xs, Xp, Xo,
                    entity_indices,
                    corrupt_subject=False,
                    corrupt_object=False):
    true_triples = {(s, p, o) for s, p, o in zip(Xs, Xp, Xo)}

    res_Xs, res_Xp, res_Xo = [], [], []
    for s, p, o in zip(Xs_batch, Xp_batch, Xo_batch):
        corrupt_s = s
        corrupt_p = p
        corrupt_o = o

        done = False
        while not done:
            if corrupt_subject is True:
                corrupt_s = entity_indices[random_state.choice(entity_indices.shape[0])]
            if corrupt_object is True:
                corrupt_o = entity_indices[random_state.choice(entity_indices.shape[0])]
            done = (corrupt_s, corrupt_p, corrupt_o) not in true_triples

        res_Xs += [corrupt_s]
        res_Xp += [corrupt_p]
        res_Xo += [corrupt_o]

    return np.array(res_Xs), np.array(res_Xp), np.array(res_Xo)


def traverse_substitutions(element, substitutions, nb_iterations=32):
    res = element
    passes = 0
    while is_variable(element) and element in substitutions and passes < nb_iterations:
        res = substitutions.get(element, element)
        passes += 1
    return res


def tile_left_np(np_tensor, shape):
    res = np_tensor
    if isinstance(np_tensor, np.ndarray):
        tensor_shp = np_tensor.shape
        nb_entries = np.prod(tensor_shp)
        nb_target_entries = np.prod(shape)
        res_2d = np.reshape(np_tensor, [1, -1])
        res_2d = np.tile(res_2d, [nb_target_entries // nb_entries, 1])
        res = np.reshape(res_2d, shape)
    elif gntp.is_tensor(np_tensor):
        tensor_shp = np_tensor.get_shape()
        nb_vectors = np.prod(tensor_shp)
        nb_target_vectors = np.prod(shape)
        res_2d = tf.reshape(np_tensor, [1, -1])
        res_2d = tf.tile(res_2d, [nb_target_vectors // nb_vectors, 1])
        res = tf.reshape(res_2d, shape)
    elif isinstance(np_tensor, SymbolIndices):
        old_tensor = np_tensor.indices
        is_fact = np_tensor.is_fact
        rule_idx = np_tensor.rule_idx
        atom_idx = np_tensor.atom_idx
        res_tensor = tile_left_np(old_tensor, shape)
        res = SymbolIndices(indices=res_tensor, is_fact=is_fact, rule_idx=rule_idx, atom_idx=atom_idx)
    return res


def tile_right_np(np_tensor, shape):
    res = np_tensor
    if isinstance(np_tensor, np.ndarray):
        tensor_shp = np_tensor.shape
        nb_entries = np.prod(tensor_shp)
        nb_target_entries = np.prod(shape)
        res_2d = np.reshape(np_tensor, [-1, 1])
        res_2d = np.tile(res_2d, [1, nb_target_entries // nb_entries])
        res = np.reshape(res_2d, shape)
    return res
