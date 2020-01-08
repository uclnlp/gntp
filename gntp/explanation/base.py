# -*- coding: utf-8 -*-

import numpy as np

from gntp.neuralkb import NeuralKB
from gntp.base import ProofState
from typing import List


def decode_per_path_type_proof_states_indices(proof_states: List[ProofState]):
    all_paths = []
    batch_size = int(proof_states[0].scores.shape[-1])
    for batch_no in range(batch_size):
        per_batch_paths = []
        for proof_state in proof_states:
            # reconstruction coordinates are correct on dimensions where index_mappers do not exist
            index_coordinates = np.copy(proof_state.index_coordinates[batch_no])

            single_path = decode_single_path(proof_state, index_coordinates, batch_no)

            per_batch_paths.append(single_path)
        per_batch_paths.sort(key=lambda tup: -tup[0])
        all_paths.append(per_batch_paths)
    return all_paths


# iterate over each dimension backwards, skip the batch dimension
def decode_single_path(proof_state, index_coordinates, batch_no):
    # reconstruction coordinates are correct on dimensions where index_mappers do not exist
    reconstructed_coordinates = np.copy(proof_state.index_coordinates[batch_no])
    rank = len(proof_state.scores.shape)
    single_path = []
    for dim in range(-2, -rank - 1, -1):

        if dim in proof_state.index_mappers:
            # index_coordinate always correctly indexes mappers
            mapper_coordinates = tuple(index_coordinates[dim:])
            # got a number out, and THAT's the thing we're looking for - the newly reconstructed index
            reconstructed_coordinates[dim] = proof_state.index_mappers[dim][mapper_coordinates]

        coordinate = reconstructed_coordinates[dim]
        kb_element_dim = proof_state.index_kb[dim + 1]
        coordinate = (coordinate,) if kb_element_dim == -1 else (kb_element_dim, coordinate)

        single_path.append(coordinate)

    score = proof_state.scores.numpy()[tuple(index_coordinates)]

    return score, single_path


def decode_proof_states_indices(proof_states: List[ProofState], top_k=10):
    all_paths = []
    batch_size = int(proof_states[0].scores.shape[-1])
    for batch_no in range(batch_size):

        dtype = [('score', float), ('index', int), ('proof_state_index', int)]
        scores_and_indices = []

        for i, proof_state in enumerate(proof_states):

            # scores_shape = proof_state.scores.numpy().shape
            scores = np.reshape(proof_state.scores.numpy()[..., batch_no], [-1])

            ind = np.arange(scores.shape[0])
            s = scores
            proof_state_indices = np.ones_like(ind, dtype=np.int32) * i

            if top_k is not None:
                # this is in lower complexity, but not not how like tensorflow does it :(
                # ind = np.argpartition(scores, -top_k)[-top_k:]  # get unsorted indices of top k scores
                # ind = ind[np.argsort(scores[ind])]             # get the indices of those scores when sorted

                ind = np.argsort(scores)[::-1][:top_k]
                s = scores[ind]
                proof_state_indices = np.ones_like(ind, dtype=np.int32) * i

            scores_and_indices.append(np.column_stack((s, ind, proof_state_indices)))

        scores_and_indices = np.vstack(scores_and_indices)
        scores_and_indices = [(s, i, pi) for s, i, pi in scores_and_indices]
        scores_and_indices = np.array(scores_and_indices, dtype=dtype)
        scores_and_indices = np.sort(scores_and_indices, order='score')

        if top_k is not None:
            scores_and_indices = scores_and_indices[-top_k:]
        scores_and_indices = scores_and_indices[::-1]

        per_batch_paths = []
        for score, index, proof_state_index in scores_and_indices:
            proof_state = proof_states[proof_state_index]
            scores_shape = proof_state.scores.numpy()[..., batch_no].shape
            indices = np.unravel_index(index, scores_shape) + (batch_no,)
            score_path = decode_single_path(proof_state, indices, batch_no)

            # all_score_paths.sort(key=lambda x: -x[0])
            per_batch_paths.append(score_path)

        # # just a check it's all good
        # aaa = [np.reshape(proof_state.scores.numpy()[..., batch_no], [-1]) for proof_state in proof_states]
        # aaa = np.concatenate(aaa)
        # aaa = np.sort(aaa)[-top_k:][::-1]
        # sss = np.array([s for s, _ in per_batch_paths])
        # assert np.all(aaa == sss)

        all_paths.append(per_batch_paths)
    return all_paths


# TODO fetch the goal!
def decode_paths(all_paths, neural_kb: NeuralKB):
    data = neural_kb.data
    facts = ["{}({}, {})".format(p, s, o) for (s, p, o) in data.train_triples]
    rules = [['[{}] {}'.format(i, rule) for i in range(int(rule.weight))] for rule in data.clauses]

    decoded_paths = []
    for batch_paths in all_paths:
        decoded_batch_path = []

        for path in batch_paths:
            decoded_path = []
            score = path[0]
            for coord in path[1]:
                if len(coord) == 1:
                    decoded_path.append(facts[coord[0]])
                elif len(coord) == 2:
                    decoded_path.append(rules[coord[0]][coord[1]])

            decoded_batch_path.append((score, decoded_path))

        decoded_paths.append(decoded_batch_path)
    # pdb.set_trace()
    return decoded_paths


def explain(test_triples,
            entity_to_index, predicate_to_index,
            scoring_function):

    res = []

    for s, p, o in test_triples:
        s_idx, p_idx, o_idx = entity_to_index[s], predicate_to_index[p], entity_to_index[o]
        np_s, np_p, np_o = np.array([s_idx]), np.array([p_idx]), np.array([o_idx])

        np_scores = scoring_function(np_p, np_s, np_o)
        score = np_scores[0]

        print(score)

        res += [score]

    return res
