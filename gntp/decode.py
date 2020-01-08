# -*- coding: utf-8 -*-

import numpy as np

import tensorflow as tf

import gntp

import logging

logger = logging.getLogger(__name__)


def decode(neural_rule, relation_embeddings, index_to_relation, kernel):
    nb_rules = None
    ret = []

    new_neural_rule = []
    for atom_idx, atom in enumerate(neural_rule):
        p, s, o = atom

        new_atom = [p, s, o]

        if gntp.is_tensor(p):
            similarities = kernel.pairwise(p, relation_embeddings)

            values, indices = tf.nn.top_k(similarities, 1)

            values_np, indices_np = values.numpy(), indices.numpy()
            values_np, indices_np = values_np.reshape([-1]), indices_np.reshape([-1])

            new_atom[0] = (indices_np, values_np)
        else:
            indices_np = np.array(p).reshape([-1])
            values_np = np.ones(indices_np.shape[0])

            new_atom[0] = (indices_np, values_np)
        new_neural_rule += [new_atom]

        nb_rules = values_np.shape[0]

    for rule_id in range(nb_rules):
        rule_str = ''
        rule_score = 1.0

        for atom_idx, atom in enumerate(new_neural_rule):
            is_head = atom_idx == 0
            p, s, o = atom
            indices_np, values_np = p

            suffix = ':- ' if is_head else ''
            rule_str += '{}({}, {}) {}'.format(index_to_relation[indices_np[rule_id]], s, o, suffix)
            rule_score = np.min([rule_score, values_np[rule_id]])

        ret.append('{}\t[{}]\t{}'.format(rule_score, rule_id, rule_str))
        print('{}\t{}'.format(rule_score, rule_str))

    return ret


def decode_v2(neural_rule, relation_embeddings,
              index_to_relation,
              nb_predicates, kernel):
    nb_rules = None

    new_neural_rule = []

    for atom_idx, atom in enumerate(neural_rule):
        is_head = atom_idx == 0
        p, s, o = atom

        new_atom = [p, s, o]

        if gntp.is_tensor(p):

            embeddings = relation_embeddings
            if is_head is True:
                embeddings = relation_embeddings[:nb_predicates]

            similarities = kernel.pairwise(p, embeddings)

            values, indices = tf.nn.top_k(similarities, 1)

            values_np, indices_np = values.numpy(), indices.numpy()
            values_np, indices_np = values_np.reshape([-1]), indices_np.reshape([-1])

            new_atom[0] = (indices_np, values_np)
        else:
            indices_np = np.array(p).reshape([-1])
            values_np = np.ones(indices_np.shape[0])

            new_atom[0] = (indices_np, values_np)
        new_neural_rule += [new_atom]

        nb_rules = values_np.shape[0]

    for rule_id in range(nb_rules):
        rule_str = ''
        rule_score = 1.0

        for atom_idx, atom in enumerate(new_neural_rule):
            is_head = atom_idx == 0
            p, s, o = atom
            indices_np, values_np = p

            suffix = ':- ' if is_head else ''
            rule_str += '{}({}, {}) {}'.format(index_to_relation[indices_np[rule_id]], s, o, suffix)
            rule_score = np.min([rule_score, values_np[rule_id]])

        print('{}\t{}'.format(rule_score, rule_str))

    return
