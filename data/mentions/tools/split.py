#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import gzip
import bz2

import argparse
import logging
import sys


def iopen(file, *args, **kwargs):
    _open = open
    if file.endswith('.gz'):
        _open = gzip.open
    elif file.endswith('.bz2'):
        _open = bz2.open
    return _open(file, *args, **kwargs)


def read_triples(path):
    triples = []
    with iopen(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples


def main(argv):
    argparser = argparse.ArgumentParser('Knowledge Graph Splitter',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('triples', type=str, default='/dev/stdin')

    argparser.add_argument('--kept', type=argparse.FileType('w'), default='/dev/stdout')
    argparser.add_argument('--held-out', type=argparse.FileType('w'), default='/dev/stdout')
    argparser.add_argument('--held-out-size', action='store', type=int, default=1000)

    argparser.add_argument('--seed', action='store', type=int, default=0)

    args = argparser.parse_args(argv)

    triples_path = args.triples

    kept_fd = args.kept

    held_out_fd = args.held_out
    held_out_size = args.held_out_size

    assert held_out_size > 0

    seed = args.seed

    triples = read_triples(triples_path)
    nb_triples = len(triples)

    nb_kept_triples = nb_triples - held_out_size
    assert nb_kept_triples > 0

    random_state = np.random.RandomState(seed)
    permutation = random_state.permutation(nb_triples).tolist()

    shuffled_triples = [triples[i] for i in permutation]

    kept_triples = shuffled_triples[:nb_kept_triples]
    held_out_triples = shuffled_triples[nb_kept_triples:]

    kept_fd.writelines(['\t'.join(symbol for symbol in triple) + '\n' for triple in kept_triples])
    held_out_fd.writelines(['\t'.join(symbol for symbol in triple) + '\n' for triple in held_out_triples])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
