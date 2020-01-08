# -*- coding: utf-8 -*-

import numpy as np

from tqdm import tqdm

from gntp.util import make_batches

from typing import List, Tuple, Dict, Callable, Optional


def evaluate(test_triples: List[Tuple[str, str, str]],
             all_triples: List[Tuple[str, str, str]],
             entity_to_index: Dict[str, int],
             predicate_to_index: Dict[str, int],
             entity_indices: np.ndarray,
             scoring_function: Callable,
             batch_size: int,
             save_ranks_path: Optional[str] = None) -> Dict[str, float]:

    index_to_entity = {index: entity for entity, index in entity_to_index.items()}
    index_to_predicate = {index: predicate for predicate, index in predicate_to_index.items()}

    test_triples = {(entity_to_index[s], predicate_to_index[p], entity_to_index[o]) for s, p, o in test_triples}
    all_triples = {(entity_to_index[s], predicate_to_index[p], entity_to_index[o]) for s, p, o in all_triples}

    entities = entity_indices.tolist()

    hits = dict()
    hits_at = [1, 3, 5, 10]

    for hits_at_value in hits_at:
        hits[hits_at_value] = 0.0

    def hits_at_n(n_, rank):
        if rank <= n_:
            hits[n_] = hits.get(n_, 0) + 1

    counter = 0
    mrr = 0.0

    for s_idx, p_idx, o_idx in tqdm(test_triples):
        corrupted_subject = [(entity, p_idx, o_idx) for entity in entities if (entity, p_idx, o_idx) not in all_triples or entity == s_idx]
        corrupted_object = [(s_idx, p_idx, entity) for entity in entities if (s_idx, p_idx, entity) not in all_triples or entity == o_idx]

        index_l = corrupted_subject.index((s_idx, p_idx, o_idx))
        index_r = corrupted_object.index((s_idx, p_idx, o_idx))

        nb_corrupted_l = len(corrupted_subject)
        # nb_corrupted_r = len(corrupted_object)

        corrupted = corrupted_subject + corrupted_object

        nb_corrupted = len(corrupted)

        batches = make_batches(nb_corrupted, batch_size)

        scores_lst = []
        for start, end in batches:
            batch = np.array(corrupted[start:end])
            x_sub, x_pred, x_obj = batch[:, 0], batch[:, 1], batch[:, 2]

            scores_np = scoring_function(x_pred, x_sub, x_obj)
            scores_lst += scores_np.tolist()

        scores_l = scores_lst[:nb_corrupted_l]
        scores_r = scores_lst[nb_corrupted_l:]

        rank_l = 1 + np.argsort(np.argsort(- np.array(scores_l)))[index_l]
        counter += 1

        for n in hits_at:
            hits_at_n(n, rank_l)

        mrr += 1.0 / rank_l

        rank_r = 1 + np.argsort(np.argsort(- np.array(scores_r)))[index_r]
        counter += 1

        if save_ranks_path is not None:
            s_s, s_p, s_o = index_to_entity[s_idx], index_to_predicate[p_idx], index_to_entity[o_idx]
            with open(save_ranks_path, 'a+') as f:
                print('{}\t{}\t{}\t{}\t{}'.format(s_s, s_p, s_o, rank_l, rank_r), file=f)

        for n in hits_at:
            hits_at_n(n, rank_r)

        mrr += 1.0 / rank_r

    counter = float(counter)

    mrr /= counter

    for n in hits_at:
        hits[n] /= counter

    metrics = dict()
    metrics['MRR'] = mrr
    for n in hits_at:
        metrics['hits@{}'.format(n)] = hits[n]

    return metrics
