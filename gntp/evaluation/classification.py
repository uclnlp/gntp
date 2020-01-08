# -*- coding: utf-8 -*-

from gntp.util import make_batches
import numpy as np

from tqdm import tqdm


def evaluate_classification(test_triples, test_labels,
                            entity_to_index, predicate_to_index,
                            scoring_function, batch_size,
                            cut_point=None):

    test_triples = np.array([
        (entity_to_index[s], predicate_to_index[p], entity_to_index[o])
        for s, p, o in test_triples
    ])
    test_labels = np.array(test_labels)

    nb_triples = test_triples.shape[0]
    batches = make_batches(nb_triples, batch_size)

    scores_lst = []

    for start, end in tqdm(batches):
        batch = test_triples[start:end]

        Xs = batch[:, 0]
        Xp = batch[:, 1]
        Xo = batch[:, 2]

        scores = scoring_function(Xp, Xs, Xo)

        scores_lst += scores.tolist()

    scores_np = np.array(scores_lst)

    np.set_printoptions(threshold=np.nan, linewidth=256)
    print('SCORES', scores_np[:128])

    def accuracy(fun_cut_point):
        gold_labels = test_labels >= 0
        predicted_labels = scores_np >= fun_cut_point

        accuracy = np.mean(np.array(gold_labels) == np.array(predicted_labels))
        return accuracy

    if cut_point is None:
        scores_sorted_np = np.array(sorted(scores_lst))
        cut_points_np = (scores_sorted_np[1:] + scores_sorted_np[:-1]) / 2.0

        best_cut_point, best_cut_point_accuracy = None, None

        for local_cut_point in cut_points_np:
            accuracy_value = accuracy(local_cut_point)
            if best_cut_point_accuracy is None or accuracy_value > best_cut_point_accuracy:
                best_cut_point = local_cut_point
                best_cut_point_accuracy = accuracy_value

        cut_point = best_cut_point

    accuracy_value = accuracy(cut_point)
    return accuracy_value, cut_point
