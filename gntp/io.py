# -*- coding: utf-8 -*-

import gzip
import bz2

import numpy as np

import logging

logger = logging.getLogger(__name__)


def iopen(file, *args, **kwargs):
    f = open
    if file.endswith('.gz'):
        f = gzip.open
    elif file.endswith('.bz2'):
        f = bz2.open
    return f(file, *args, **kwargs)


def read_triples(path):
    triples = []
    with iopen(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples


def read_mentions(path):
    triples = []
    with iopen(path, 'rt') as f:
        for line in f.readlines():
            s, m, o, counts = line.split('\t')
            s, m, o = s.strip(), m.strip(), o.strip()
            nb_counts = int(counts.strip())
            triples += [(s, m, o, nb_counts)]
    return triples


def read_labeled_triples(path):
    labeled_triples = []
    with iopen(path, 'rt') as f:
        for line in f.readlines():
            s, p, o, l = line.split()
            labeled_triples += [((s.strip(), p.strip(), o.strip()), int(l.strip()))]
    return labeled_triples


def triples_to_vectors(triples, entity_to_idx, predicate_to_idx):
    Xs = np.array([entity_to_idx[s] for (s, p, o) in triples], dtype=np.int32)
    Xp = np.array([predicate_to_idx[p] for (s, p, o) in triples], dtype=np.int32)
    Xo = np.array([entity_to_idx[o] for (s, p, o) in triples], dtype=np.int32)
    return Xs, Xp, Xo
