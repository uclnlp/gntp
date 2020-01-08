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
             batch_size: int) -> Dict[str, float]:

    test_triples = {(entity_to_index[s], predicate_to_index[p], entity_to_index[o]) for s, p, o in test_triples}
    all_triples = {(entity_to_index[s], predicate_to_index[p], entity_to_index[o]) for s, p, o in all_triples}

    entities = entity_indices.tolist()

    hits = dict()
    hits_at = [1, 3, 5, 10]

    for hits_at_value in hits_at:
        hits[hits_at_value] = 0.0

    def hits_at_n(n, rank):
        if rank <= n:
            hits[n] = hits.get(n, 0) + 1

    counter = 0
    MRR = 0.0

    with open('output.txt', 'a') as f:

        for s, p, o in tqdm(test_triples):
            corrupted_subject = [(entity, p, o) for entity in entities if (entity, p, o) not in all_triples or entity == s]
            corrupted_object = [(s, p, entity) for entity in entities if (s, p, entity) not in all_triples or entity == o]

            index_l = corrupted_subject.index((s, p, o))
            index_r = corrupted_object.index((s, p, o))

            nb_corrupted_l = len(corrupted_subject)
            nb_corrupted_r = len(corrupted_object)

            corrupted = corrupted_subject + corrupted_object

            nb_corrupted = len(corrupted)

            batches = make_batches(nb_corrupted, batch_size)

            scores_lst = []
            for start, end in batches:
                batch = np.array(corrupted[start:end])
                Xs, Xp, Xo = batch[:, 0], batch[:, 1], batch[:, 2]

                scores_np = scoring_function(Xp, Xs, Xo)
                scores_lst += scores_np.tolist()

            scores_l = scores_lst[:nb_corrupted_l]
            scores_r = scores_lst[nb_corrupted_l:]

            rank_l = 1 + np.argsort(np.argsort(- np.array(scores_l)))[index_l]
            counter += 1

            for n in hits_at:
                hits_at_n(n, rank_l)

            MRR += 1.0 / rank_l

            rank_r = 1 + np.argsort(np.argsort(- np.array(scores_r)))[index_r]
            counter += 1

            for n in hits_at:
                hits_at_n(n, rank_r)

            MRR += 1.0 / rank_r

            f.write('{}({}, {})\t{}\t{}\n'.format(p, s, o, rank_l, rank_r))
            f.flush()

    counter = float(counter)

    MRR /= counter

    for n in hits_at:
        hits[n] /= counter

    metrics = dict()
    metrics['MRR'] = MRR
    for n in hits_at:
        metrics['hits@{}'.format(n)] = hits[n]

    return metrics
