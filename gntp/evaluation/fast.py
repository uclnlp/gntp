# -*- coding: utf-8 -*-

import numpy as np

from gntp.util import make_batches

from typing import List, Tuple, Dict, Callable, Optional


def fast_evaluate(test_triples: List[Tuple[str, str, str]],
                  all_triples: List[Tuple[str, str, str]],
                  entity_to_index: Dict[str, int],
                  predicate_to_index: Dict[str, int],
                  scoring_function: Callable,
                  batch_size: int,
                  save_ranks_path: Optional[str] = None) -> Dict[str, float]:
    xs = np.array([entity_to_index.get(s) for (s, _, _) in test_triples])
    xp = np.array([predicate_to_index.get(p) for (_, p, _) in test_triples])
    xo = np.array([entity_to_index.get(o) for (_, _, o) in test_triples])

    index_to_entity = {index: entity for entity, index in entity_to_index.items()}
    index_to_predicate = {index: predicate for predicate, index in predicate_to_index.items()}

    sp_to_o, po_to_s = {}, {}
    for s, p, o in all_triples:
        s_idx, p_idx, o_idx = entity_to_index.get(s), predicate_to_index.get(p), entity_to_index.get(o)
        sp_key = (s_idx, p_idx)
        po_key = (p_idx, o_idx)

        if sp_key not in sp_to_o:
            sp_to_o[sp_key] = []
        if po_key not in po_to_s:
            po_to_s[po_key] = []

        sp_to_o[sp_key] += [o_idx]
        po_to_s[po_key] += [s_idx]

    assert xs.shape == xp.shape == xo.shape
    nb_test_triples = xs.shape[0]

    batches = make_batches(nb_test_triples, batch_size)

    hits = dict()
    hits_at = [1, 3, 5, 10]

    for hits_at_value in hits_at:
        hits[hits_at_value] = 0.0

    def hits_at_n(n_, rank):
        if rank <= n_:
            hits[n_] = hits.get(n_, 0) + 1

    counter = 0
    mrr = 0.0

    ranks_l, ranks_r = [], []
    for start, end in batches:
        batch_xs = xs[start:end]
        batch_xp = xp[start:end]
        batch_xo = xo[start:end]

        batch_size = batch_xs.shape[0]
        counter += batch_size * 2

        scores_sp, scores_po = scoring_function(batch_xp, batch_xs, batch_xo)

        batch_size = batch_xs.shape[0]
        for elem_idx in range(batch_size):
            s_idx, p_idx, o_idx = batch_xs[elem_idx], batch_xp[elem_idx], batch_xo[elem_idx]

            # Code for the filtered setting
            sp_key = (s_idx, p_idx)
            po_key = (p_idx, o_idx)

            o_to_remove = sp_to_o[sp_key]
            s_to_remove = po_to_s[po_key]

            for tmp_o_idx in o_to_remove:
                if tmp_o_idx != o_idx:
                    scores_sp[elem_idx, tmp_o_idx] = - np.infty

            for tmp_s_idx in s_to_remove:
                if tmp_s_idx != s_idx:
                    scores_po[elem_idx, tmp_s_idx] = - np.infty
            # End of code for the filtered setting

            rank_l = 1 + np.argsort(np.argsort(- scores_po[elem_idx, :]))[s_idx]
            rank_r = 1 + np.argsort(np.argsort(- scores_sp[elem_idx, :]))[o_idx]

            if save_ranks_path is not None:
                s_s, s_p, s_o = index_to_entity[s_idx], index_to_predicate[p_idx], index_to_entity[o_idx]
                with open(save_ranks_path, 'a+') as f:
                    print('{}\t{}\t{}\t{}\t{}'.format(s_s, s_p, s_o, rank_l, rank_r), file=f)

            ranks_l += [rank_l]
            ranks_r += [rank_r]

            mrr += 1.0 / rank_l
            mrr += 1.0 / rank_r

            for n in hits_at:
                hits_at_n(n, rank_l)

            for n in hits_at:
                hits_at_n(n, rank_r)

    counter = float(counter)

    mrr /= counter

    for n in hits_at:
        hits[n] /= counter

    metrics = dict()
    metrics['MRR'] = mrr
    for n in hits_at:
        metrics['hits@{}'.format(n)] = hits[n]

    return metrics
