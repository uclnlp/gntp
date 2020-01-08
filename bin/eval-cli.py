#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import numpy as np

from gntp.training.data import Data

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)


def compute_stats(rank_l_lst, rank_r_lst):
    counter = 0.0
    mrr = 0.0

    hits = dict()
    hits_at = [1, 3, 5, 10]

    for hits_at_value in hits_at:
        hits[hits_at_value] = 0.0

    def hits_at_n(n_, rank):
        if rank <= n_:
            hits[n_] = hits.get(n_, 0) + 1

    for rank_l, rank_r in zip(rank_l_lst, rank_r_lst):
        counter += 2.0

        mrr += 1.0 / rank_l
        mrr += 1.0 / rank_r

        for n in hits_at:
            hits_at_n(n, rank_l)
            hits_at_n(n, rank_r)

    mrr /= counter
    for n in hits_at:
        hits[n] /= counter

    metrics = dict()
    metrics['MRR'] = mrr
    for n in hits_at:
        metrics['hits@{}'.format(n)] = hits[n]

    return metrics


def main(argv):
    argparser = argparse.ArgumentParser('NTP 2.0', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    argparser.add_argument('--train', action='store', type=str, default=None)
    argparser.add_argument('--dev', action='store', type=str, default=None)
    argparser.add_argument('--test', action='store', type=str, default=None)
    argparser.add_argument('--rankings', action='store', type=str, default=None)

    argparser.add_argument('--per-relation', '-r', action='store_true')

    argparser.add_argument('--min-frequency', '--min', action='store', type=int, default=None)
    argparser.add_argument('--max-frequency', '--max', action='store', type=int, default=None)

    argparser.add_argument('--both-min-frequency', '--both-min', action='store', type=int, default=None)
    argparser.add_argument('--both-max-frequency', '--both-max', action='store', type=int, default=None)

    args = argparser.parse_args(argv)

    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    rankings_path = args.rankings

    is_per_relation = args.per_relation

    min_frequency = args.min_frequency
    max_frequency = args.max_frequency

    both_min_frequency = args.both_min_frequency
    both_max_frequency = args.both_max_frequency

    data = Data(train_path=train_path, dev_path=dev_path, test_path=test_path) if train_path else None

    if min_frequency or max_frequency or both_min_frequency or both_max_frequency:
        assert data is not None

    entity_to_frequency = {}
    for s, p, o in data.train_triples if data else []:
        if s not in entity_to_frequency:
            entity_to_frequency[s] = 0

        if o not in entity_to_frequency:
            entity_to_frequency[o] = 0

        entity_to_frequency[s] += 1
        entity_to_frequency[o] += 1

    if rankings_path is not None:
        with open(rankings_path, 'rt') as f:
            lines = list(f.readlines())

        triples = []
        rank_l_lst, rank_r_lst = [], []

        for line in lines:
            s, p, o, rank_l, rank_r = line.split('\t')
            rank_l, rank_r = int(rank_l), int(rank_r)

            add_this_triple = True

            if min_frequency is not None:
                s_freq, o_freq = entity_to_frequency.get(s, 0), entity_to_frequency.get(o, 0)
                if s_freq < min_frequency and o_freq < min_frequency:
                    add_this_triple = False

            if max_frequency is not None:
                s_freq, o_freq = entity_to_frequency.get(s, 0), entity_to_frequency.get(o, 0)
                if s_freq > max_frequency and o_freq > max_frequency:
                    add_this_triple = False

            if both_min_frequency is not None:
                s_freq, o_freq = entity_to_frequency.get(s, 0), entity_to_frequency.get(o, 0)
                if s_freq < both_min_frequency or o_freq < both_min_frequency:
                    add_this_triple = False

            if both_max_frequency is not None:
                s_freq, o_freq = entity_to_frequency.get(s, 0), entity_to_frequency.get(o, 0)
                if s_freq > both_max_frequency or o_freq > both_max_frequency:
                    add_this_triple = False

            if add_this_triple:
                triples += [(s, p, o)]
                rank_l_lst += [rank_l]
                rank_r_lst += [rank_r]

        p_set = sorted({p for (_, p, _) in triples})

        if is_per_relation:
            for selected_p in p_set:
                p_rank_l_lst = [rank_l for rank_l, (_, p, _) in zip(rank_l_lst, triples) if p == selected_p]
                p_rank_r_lst = [rank_r for rank_r, (_, p, _) in zip(rank_r_lst, triples) if p == selected_p]
                p_metrics = compute_stats(p_rank_l_lst, p_rank_r_lst)
                print('{}\t({})\t{}'.format(selected_p, len(p_rank_l_lst), p_metrics))
        else:
            metrics = compute_stats(rank_l_lst, rank_r_lst)
            print('({})\t{}'.format(len(rank_l_lst), metrics))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
