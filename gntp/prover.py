# -*- coding: utf-8 -*-

import copy

import numpy as np
import tensorflow as tf

from gntp.base import ProofState, NTPParams
from gntp.indices import SymbolIndices
from gntp.util import is_variable, is_tensor

from gntp.tfutil import tile_left, tile_right
from gntp.util import tile_left_np

from typing import List, Optional, Union, Any

import gntp
import pdb

DEBUG = False


def print_debug(print_depth, what_to_print):
    if DEBUG:
        print('\t' * print_depth + what_to_print)


def to_show(atom_elem):
    return atom_elem.get_shape() if is_tensor(atom_elem) else atom_elem


def joint_unify(atom: List[Union[tf.Tensor, str]],
                goal: List[Union[tf.Tensor, str]],
                proof_state: ProofState,
                ntp_params: NTPParams,
                is_fact: bool = False,
                top_indices: Optional[tf.Tensor] = None,
                goal_indices: Optional[List[Union[SymbolIndices, str]]] = None) -> ProofState:

    # symbol-wise unify and min-pooling
    substitutions = copy.copy(proof_state.substitutions)
    index_substitutions = copy.copy(proof_state.index_substitutions)

    scores = proof_state.scores

    f_k = top_indices.shape[0] if top_indices is not None else None

    initial_scores_shp = scores.get_shape()
    goal = [tile_left(elem, initial_scores_shp) for elem in goal]

    if goal_indices is not None:
        goal_indices = [tile_left_np(elem, initial_scores_shp) for elem in goal_indices]

    atom_tensors_lst = []
    goal_tensors_lst = []

    for atom_index, (atom_elem, goal_elem) in enumerate(zip(atom, goal)):
        goal_indices_elem = goal_indices[atom_index] if goal_indices is not None else None

        if is_variable(atom_elem):
            if atom_elem not in substitutions:
                substitutions.update({atom_elem: goal_elem})

                if index_substitutions is not None and goal_indices_elem is not None:
                    # print('XXX', type(goal_indices_elem))
                    index_substitutions.update({atom_elem: goal_indices_elem})

        elif is_variable(goal_elem):
            if is_tensor(atom_elem):
                atom_shp = atom_elem.get_shape()
                scores_shp = scores.get_shape()

                embedding_size = atom_shp[-1]
                substitution_shp = scores_shp.concatenate([embedding_size])

                if top_indices is None:
                    atom_elem = tile_right(atom_elem, scores_shp)
                else:
                    f_atom_elem = tf.gather(atom_elem, tf.reshape(top_indices, [-1]))
                    atom_elem = tf.reshape(f_atom_elem, substitution_shp)

            if goal_elem not in substitutions:
                substitutions.update({goal_elem: atom_elem})

                if index_substitutions is not None and goal_indices_elem is not None:
                    # print('XXY', type(top_indices))
                    index_substitutions.update({goal_indices_elem: top_indices})

        elif is_tensor(atom_elem) and is_tensor(goal_elem):
            atom_tensors_lst += [atom_elem]
            goal_tensors_lst += [goal_elem]

        atom_elem = tf.concat(atom_tensors_lst, axis=-1)
        goal_elem = tf.concat(goal_tensors_lst, axis=-1)

        goal_elem_shp = goal_elem.get_shape()
        embedding_size = goal_elem_shp[-1]

        if top_indices is None:
            similarities = ntp_params.kernel.pairwise(atom_elem, goal_elem)
        else:
            # Replicate each sub-goal by the number of facts it will be unified with
            f_goal_elem = tf.reshape(goal_elem, [-1, 1, embedding_size])
            f_goal_elem = tf.tile(f_goal_elem, [1, f_k, 1])
            f_goal_elem = tf.reshape(f_goal_elem, [-1, embedding_size])

            # Move the "most relevant fact dimension per sub-goal" dimension from first to last (IIRC)
            f_top_indices = tf.transpose(top_indices, list(range(1, len(top_indices.shape))) + [0])

            # For each sub-goal, lookup the most relevant facts
            f_new_atom_elem = tf.gather(atom_elem, tf.reshape(f_top_indices, [-1]))

            # Compute the kernel between each (repeated) sub-goal and its most relevant facts
            f_values = ntp_params.kernel.elementwise(f_new_atom_elem, f_goal_elem)

            # New shape that similarities should acquire (i.e. [k, g1, .., gn])
            f_scatter_shp = tf.TensorShape(f_k).concatenate(top_indices.shape[1:])

            # Here similarities have shape [g1 .. gn, k]
            f_values = tf.reshape(f_values, [-1, f_k])
            # Transpose and move the k dimension such that we have [k, g1 .. gn]
            f_values = tf.transpose(f_values, (1, 0))

            # Reshape similarity values
            similarities = tf.reshape(f_values, f_scatter_shp)

        # Reshape the kernel accordingly
        similarities_shp = similarities.get_shape()

        goal_shp = goal_elem.get_shape()[:-1]
        k_shp = atom_elem.get_shape()[:-1] if top_indices is None else tf.TensorShape([f_k])

        target_shp = k_shp.concatenate(initial_scores_shp)

        if similarities_shp != target_shp:
            nb_similarities = tf.size(similarities)

            # nb_targets = tf.reduce_prod(target_shp)
            # nb_goals = tf.reduce_prod(goal_shp)
            nb_targets = np.prod(target_shp)
            nb_goals = np.prod(goal_shp)

            similarities = tf.reshape(similarities, [-1, 1, nb_goals])
            similarities = tf.tile(similarities, [1, nb_targets // nb_similarities, 1])

        similarities = tf.reshape(similarities, target_shp)

        if ntp_params.mask_indices is not None and is_fact:
            # Mask away the similarities to facts that correspond to goals (used for the LOO loss)
            mask_indices = ntp_params.mask_indices
            mask = gntp.create_mask(mask_indices=mask_indices, mask_shape=target_shp, indices=top_indices)

            if mask is not None:
                similarities *= mask

        similarities_shp = similarities.get_shape()
        scores_shp = scores.get_shape()

        if similarities_shp != scores_shp:
            new_scores_shp = tf.TensorShape([1]).concatenate(scores_shp)
            scores = tf.reshape(scores, new_scores_shp)

        if ntp_params.unification_score_aggregation == 'min':
            scores = tf.minimum(similarities, scores)
        elif ntp_params.unification_score_aggregation == 'mul':
            scores = similarities * scores
        elif ntp_params.unification_score_aggregation == 'minmul':
            scores = (tf.minimum(similarities, scores) + similarities * scores) / 2

    index_mappers = copy.deepcopy(proof_state.index_mappers)
    if top_indices is not None and index_mappers is not None:
        index_mappers[-len(top_indices.shape)] = top_indices

    index_kb = proof_state.index_kb[:] if proof_state.index_kb is not None else proof_state.index_kb

    proof_state = ProofState(substitutions=substitutions,
                             scores=scores,
                             index_substitutions=index_substitutions,
                             index_mappers=index_mappers,
                             index_kb=index_kb)

    return proof_state


def unify(atom: List[Union[tf.Tensor, str]],
          goal: List[Union[tf.Tensor, str]],
          proof_state: ProofState,
          ntp_params: NTPParams,
          is_fact: bool = False,
          top_indices: Optional[tf.Tensor] = None,
          goal_indices: Optional[List[Union[SymbolIndices, str]]] = None) -> ProofState:

    # symbol-wise unify and min-pooling
    substitutions = copy.copy(proof_state.substitutions)
    index_substitutions = copy.copy(proof_state.index_substitutions)

    scores = proof_state.scores

    f_k = top_indices.shape[0] if top_indices is not None else None

    initial_scores_shp = scores.get_shape()

    goal = [tile_left(elem, initial_scores_shp) for elem in goal]

    if goal_indices is not None:
        goal_indices = [tile_left_np(elem, initial_scores_shp) for elem in goal_indices]

    for atom_index, (atom_elem, goal_elem) in enumerate(zip(atom, goal)):
        goal_indices_elem = goal_indices[atom_index] if (goal_indices is not None) else None

        if is_variable(atom_elem):
            if atom_elem not in substitutions:
                substitutions.update({atom_elem: goal_elem})

                if index_substitutions is not None and goal_indices_elem is not None:
                    # print('XXX', type(goal_indices_elem))
                    index_substitutions.update({atom_elem: goal_indices_elem})

        elif is_variable(goal_elem):
            if is_tensor(atom_elem):
                atom_shp = atom_elem.get_shape()
                scores_shp = scores.get_shape()

                embedding_size = atom_shp[-1]
                substitution_shp = scores_shp.concatenate([embedding_size])

                if top_indices is None:
                    atom_elem = tile_right(atom_elem, scores_shp)
                else:
                    f_atom_elem = tf.gather(atom_elem, tf.reshape(top_indices, [-1]))
                    atom_elem = tf.reshape(f_atom_elem, substitution_shp)

            if goal_elem not in substitutions:
                substitutions.update({goal_elem: atom_elem})

                if index_substitutions is not None and goal_indices_elem is not None:
                    # print('XXY', type(top_indices))
                    tmp = top_indices.copy() if isinstance(top_indices, np.ndarray) else top_indices.numpy()
                    linear_top_indices = tmp.reshape([-1])
                    sym_top_indices = np.array([ntp_params.facts_kb[atom_index][idx] for idx in linear_top_indices])
                    sym_top_indices = np.reshape(sym_top_indices, top_indices.shape)
                    sym_indices = SymbolIndices(indices=sym_top_indices, is_fact=True)
                    index_substitutions.update({goal_indices_elem: sym_indices})

        elif is_tensor(atom_elem) and is_tensor(goal_elem):
            goal_elem_shp = goal_elem.get_shape()
            embedding_size = goal_elem_shp[-1]

            if top_indices is None:
                similarities = ntp_params.kernel.pairwise(atom_elem, goal_elem)
            else:
                # Replicate each sub-goal by the number of facts it will be unified with
                f_goal_elem = tf.reshape(goal_elem, [-1, 1, embedding_size])
                f_goal_elem = tf.tile(f_goal_elem, [1, f_k, 1])
                f_goal_elem = tf.reshape(f_goal_elem, [-1, embedding_size])

                # Move the "most relevant fact dimension per sub-goal" dimension from first to last
                f_top_indices = tf.transpose(top_indices, list(range(1, len(top_indices.shape))) + [0])

                # For each sub-goal, lookup the most relevant facts
                f_new_atom_elem = tf.gather(atom_elem, tf.reshape(f_top_indices, [-1]))

                # Compute the kernel between each (repeated) sub-goal and its most relevant facts
                f_values = ntp_params.kernel.elementwise(f_new_atom_elem, f_goal_elem)

                # New shape that similarities should acquire (i.e. [k, g1, .., gn])
                f_scatter_shp = tf.TensorShape(f_k).concatenate(top_indices.shape[1:])

                # Here similarities have shape [g1 .. gn, k]
                f_values = tf.reshape(f_values, [-1, f_k])
                # Transpose and move the k dimension such that we have [k, g1 .. gn]
                f_values = tf.transpose(f_values, (1, 0))

                # Reshape similarity values
                similarities = tf.reshape(f_values, f_scatter_shp)

            # Reshape the kernel accordingly
            similarities_shp = similarities.get_shape()

            goal_shp = goal_elem.get_shape()[:-1]
            k_shp = atom_elem.get_shape()[:-1] if top_indices is None else tf.TensorShape([f_k])

            target_shp = k_shp.concatenate(initial_scores_shp)

            if similarities_shp != target_shp:
                nb_similarities = tf.size(similarities)

                # nb_targets = tf.reduce_prod(target_shp)
                # nb_goals = tf.reduce_prod(goal_shp)
                nb_targets = np.prod(target_shp)
                nb_goals = np.prod(goal_shp)

                similarities = tf.reshape(similarities, [-1, 1, nb_goals])
                similarities = tf.tile(similarities, [1, nb_targets // nb_similarities, 1])

            similarities = tf.reshape(similarities, target_shp)

            if ntp_params.mask_indices is not None and is_fact:
                # Mask away the similarities to facts that correspond to goals (used for the LOO loss)
                mask_indices = ntp_params.mask_indices
                mask = gntp.create_mask(mask_indices=mask_indices, mask_shape=target_shp, indices=top_indices)

                if mask is not None:
                    similarities *= mask

            similarities_shp = similarities.get_shape()
            scores_shp = scores.get_shape()

            if similarities_shp != scores_shp:
                new_scores_shp = tf.TensorShape([1]).concatenate(scores_shp)
                scores = tf.reshape(scores, new_scores_shp)

            if ntp_params.unification_score_aggregation == 'min':
                scores = tf.minimum(similarities, scores)
            elif ntp_params.unification_score_aggregation == 'mul':
                scores = similarities * scores
            elif ntp_params.unification_score_aggregation == 'minmul':
                scores = (tf.minimum(similarities, scores) + similarities * scores) / 2

    index_mappers = copy.deepcopy(proof_state.index_mappers)
    if top_indices is not None and index_mappers is not None:
        index_mappers[-len(top_indices.shape)] = top_indices

    index_kb = proof_state.index_kb[:] if proof_state.index_kb is not None else proof_state.index_kb

    proof_state = ProofState(substitutions=substitutions,
                             scores=scores,
                             index_substitutions=index_substitutions,
                             index_mappers=index_mappers,
                             index_kb=index_kb)

    return proof_state


def substitute(atom: List[Union[tf.Tensor, SymbolIndices, str]],
               proof_state: ProofState,
               is_indices: bool = False) -> List[Union[tf.Tensor, SymbolIndices, str]]:
    """
    Implements the SUBSTITUTION method for an atom, given the proof state.

    This is done by traversing through the atom and replacing symbols
    according to the substitution.

    Example:
        atom: [GE, X, Y]
        proof_state:
            scores: [RG
            substitution: {X/GE}

    The result is:
        atom: [RGE, RGE, Y]

    :param atom: Atom.
    :param proof_state: Proof state.
    :param is_indices: is_indices.
    :return: New atom, matching the proof scores.
    """
    scores_shp = proof_state.scores.get_shape()

    if is_indices is True:
        def _process(atom_elem):
            # if atom element is a variable, replace it as specified by the substitution
            index_substitutions = proof_state.index_substitutions if proof_state.index_substitutions else {}
            res = index_substitutions.get(atom_elem, atom_elem) if is_variable(atom_elem) else atom_elem
            return tile_left_np(res, scores_shp)
        new_atom = [_process(atom_elem) for atom_elem in atom]
    else:
        def _process(atom_elem):
            # if atom element is a variable, replace it as specified by the substitution
            res = proof_state.substitutions.get(atom_elem, atom_elem) if is_variable(atom_elem) else atom_elem
            return tile_left(res, scores_shp)
        new_atom = [_process(atom_elem) for atom_elem in atom]
    return new_atom


def neural_and(neural_kb: List[List[List[Union[tf.Tensor, str]]]],
               goals: List[List[Union[tf.Tensor, str]]],
               proof_state: ProofState,
               ntp_params: NTPParams,
               depth: int,
               print_depth: Optional[int] = None,
               goal_indices: Optional[List[List[Union[SymbolIndices, str]]]] = None) -> List[ProofState]:
    """
    Implements the neural AND operator.

    The neural AND operator has the following definition:

    1) AND(_, _, FAIL) = FAIL
    2) AND(_, 0, _) = FAIL
    3) AND([], _, S) = S
    4) AND(g : G, d, S) = [ S'' | S'' \in AND(G, d, S')
                               for S' \in OR(SUBSTITUTE(G, S_psi), d - 1, S) ]

    assume the list of atoms g : G encodes a rule, where g is the head (e.g. [RE, X, Y])
    and G is the body (e.g. [RE, X, Z], [RE, Z, Y]).

    This method proceeds as follows:
    - First it replaces variables in "head" using the current substitution set.
    - Then it calls the OR operator on the new atom.

    Then AND operator is then called recursively on the body G of the rule.

    :param neural_kb: Neural Knowledge Base.
    :param goals: List of atoms.
    :param proof_state: Proof state.
    :param ntp_params: NTP Parameters
    :param depth: Current depth in the proof tree (increased by one when calling neural_or)
    :param print_depth: an auxiliary variable used to print out the depth of the call
    :param goal_indices:
    :return: List of proof states.
    """
    # 1) (upstream unification failed) and 2) (depth == max_depth)
    proof_states = []

    if len(goals) == 0:  # 3)
        print_debug(print_depth + 1, 'SUCCESS')
        proof_states = [proof_state]

    elif depth < ntp_params.max_depth:  # 4)
        goal, sub_goals = goals[0], goals[1:]

        new_goal = substitute(goal, proof_state)

        new_goal_index, sub_goal_index = None, None
        if goal_indices is not None:
            goal_index, sub_goal_index = goal_indices[0], goal_indices[1:]
            new_goal_index = substitute(goal_index, proof_state, is_indices=True)

        # print('NEW_GOAL', gntp.atom_to_str(new_goal))
        # print('NEW_GOAL_INDEX', gntp.atom_to_str(new_goal_index))

        or_proof_states = neural_or(neural_kb=neural_kb,
                                    goals=new_goal,
                                    proof_state=proof_state,
                                    ntp_params=ntp_params,
                                    depth=depth + 1,
                                    print_depth=print_depth + 1,
                                    goal_indices=new_goal_index)

        for i, or_proof_state in enumerate(or_proof_states):
            print_debug(print_depth + 1, 'AND -- {}/{}'.format(i + 1, len(or_proof_states)))

            proof_states += neural_and(neural_kb=neural_kb,
                                       goals=sub_goals,
                                       proof_state=or_proof_state,
                                       ntp_params=ntp_params,
                                       depth=depth,
                                       print_depth=print_depth + 1,
                                       goal_indices=sub_goal_index)

    return proof_states


def neural_or(neural_kb: List[List[List[Union[tf.Tensor, str]]]],
              goals: List[Union[tf.Tensor, str]],
              proof_state: ProofState,
              ntp_params: NTPParams,
              depth: int = 0,
              no_ntp0: bool = False,
              only_ntp0: bool = False,
              print_depth: int = 0,
              goal_indices: Optional[List[Union[SymbolIndices, str]]] = None) -> List[ProofState]:
    """
    Implements the neural OR operator.

    It is defined as follows:

    OR(G, d, S) = [ S' | S' \in AND(HEAD, d, UNIFY(HEAD, GOAL, S))
                         for HEAD <- BODY in KB ]

    Assume we have a goal of shape [GE, GE, GE],
    and a rule such as [[RE, X, Y], [RE, X, Z], [RE, Z, Y]].

    This method iterates through all rules (note - facts are just rules with an empty body),
    and unifies the goal (e.g. [GE, GE, GE]) with the head of the rule (e.g. [RE, X, Y]).

    The result of unification is a [RG] tensor of proof scores, and a new set of substitutions
    compatible with the proof scores, i.e. X/RGE and Y/RGE.

    Then, the body of the rule (if present) is reshaped so to match the new proof scores [RG].
    For instance, if the body was [[RE, X, Z], [RE, Z, Y]], the RE tensors are reshaped to GE.

    :param neural_kb: Neural Knowledge Base.
    :param goals: Atom, e.g. [GE, GE, GE].
    :param proof_state: Proof state.
    :param ntp_params: NTP Parameters.
    :param depth: Current depth in the proof tree [default: 0].
    :param no_ntp0: Boolean, decide whether not to unify with facts or not.
    :param only_ntp0: Boolean, decide whether only unify with facts or not.
    :param print_depth: an auxiliary variable used to print out the depth of the call
    :param goal_indices: [[int], [int], [int]] goal indices.
    :return: List of proof states.
    """
    print_debug(print_depth, 'OR')

    if proof_state.scores is None:
        index_mappers = index_kb = None
        if ntp_params.support_explanations:
            index_mappers = dict()
            index_kb = []

        initial_scores = tf.ones(shape=goals[0].get_shape()[:-1], dtype=goals[0].dtype)

        proof_state = ProofState(scores=initial_scores,
                                 substitutions=proof_state.substitutions,
                                 index_mappers=index_mappers,
                                 index_kb=index_kb,
                                 index_substitutions=proof_state.index_substitutions)

    scores_shp = proof_state.scores.get_shape()
    embedding_size = goals[0].get_shape()[-1]

    goals = [tile_left(elem, scores_shp) for elem in goals]
    goal_shp = scores_shp.concatenate([embedding_size])

    if goal_indices is not None:
        goal_indices = [tile_left_np(elem, scores_shp) for elem in goal_indices]

    proof_states = []

    for rule_index, rule in enumerate(neural_kb):
        # Assume we unify with a rule, e.g. [[RE X Y], [RE Y X]]
        heads, bodies = rule[0], rule[1:]

        is_fact = len(bodies) == 0

        if is_fact and no_ntp0:
            continue

        if not is_fact and only_ntp0:
            continue

        k = ntp_params.retrieve_k_facts if is_fact else ntp_params.retrieve_k_rules
        top_indices = None

        if k is not None:
            index_store = ntp_params.index_store
            index = index_store.get_or_create(atoms=heads,
                                              goals=goals,
                                              index_refresh_rate=ntp_params.index_refresh_rate,
                                              position=rule_index,
                                              is_training=ntp_params.is_training)

            top_indices = gntp.lookup.find_best_heads(index=index,
                                                      atoms=heads,
                                                      goals=goals,
                                                      goal_shape=goal_shp,
                                                      k=k,
                                                      is_training=ntp_params.is_training,
                                                      goal_indices=goal_indices,
                                                      position=rule_index)

            k = top_indices.shape[0]

        rule_vars = {e for atom in rule for e in atom if is_variable(e)}
        applied_before = bool(proof_state.substitutions.keys() & rule_vars)

        # In case we reached the maximum recursion depth, do not proceed
        # Also, avoid cycles
        if (depth < ntp_params.max_depth or is_fact) and not applied_before:
            # Unify the goal, e.g. [GE GE GE], with the head of the rule [RE X Y]
            unification_op = ntp_params._unification_op if ntp_params._unification_op else unify

            new_proof_state = unification_op(atom=heads,
                                             goal=goals,
                                             proof_state=proof_state,
                                             ntp_params=ntp_params,
                                             is_fact=is_fact,
                                             top_indices=top_indices,
                                             goal_indices=goal_indices)

            # Differentiable k-max
            # if k-max-ing is on, and we're processing facts (empty body)
            if ntp_params.k_max and is_fact:
                # Check whether k is < than the number of facts
                if ntp_params.k_max < new_proof_state.scores.get_shape()[0]:
                    new_proof_state = gntp.k_max(goals, new_proof_state, k=ntp_params.k_max)

            # The new proof state will be of shape [RG]
            # We now need the body [RE Y X] to match the new proof state as well
            scores_shp = new_proof_state.scores.get_shape()

            # Reshape the rest of the body, so it matches the shape of the head,
            # the current substitutions, and the proof score.
            if k is None:
                def normalize_atom(atom):
                    return [tile_right(elem, scores_shp) for elem in atom]
                new_bodies = [normalize_atom(body_atom) for body_atom in bodies]
            else:
                f_top_indices = tf.transpose(top_indices, list(range(1, len(top_indices.shape))) + [0])

                def normalize_atom_elem(_atom_elem):
                    res = _atom_elem
                    if is_tensor(res):
                        f_atom_elem = tf.gather(_atom_elem, tf.reshape(f_top_indices, [-1]))
                        f_atom_shp = scores_shp.concatenate([embedding_size])
                        res = tf.reshape(f_atom_elem, f_atom_shp)
                    return res

                def normalize_atom(_atom):
                    return [normalize_atom_elem(elem) for elem in _atom]

                new_bodies = [normalize_atom(body_atom) for body_atom in bodies]

            print_facts_or_rules = 'facts' if rule_index + 1 == len(neural_kb) else 'rules {}'.format(rule_index)
            print_debug(print_depth + 1, 'AND - {}'.format(print_facts_or_rules))

            # I really dislike coding fact position as -1, but well...it is
            if new_proof_state.index_kb is not None:
                new_proof_state.index_kb += [-1 if rule_index + 1 == len(neural_kb) else rule_index]

            new_body_indices = []
            for atom in new_bodies:
                atom_indices = []
                # Enumeration starts at 1 because the atom indexed at 0 is the one in the head of the rule
                for atom_idx, atom_elem in enumerate(atom, 1):
                    sym_atom_indices = atom_elem

                    def npy(tensor: Any) -> np.ndarray:
                        return tensor.numpy() if gntp.is_tensor(tensor) else tensor

                    if is_tensor(atom_elem):
                        sym_atom_indices = SymbolIndices(indices=npy(top_indices),
                                                         is_fact=False,
                                                         rule_idx=rule_index,
                                                         atom_idx=atom_idx)

                    atom_indices += [sym_atom_indices]
                new_body_indices += [atom_indices]

            body_proof_states = neural_and(neural_kb=neural_kb,
                                           goals=new_bodies,
                                           proof_state=new_proof_state,
                                           ntp_params=ntp_params,
                                           depth=depth,
                                           print_depth=print_depth + 1,
                                           goal_indices=new_body_indices)

            if body_proof_states:
                proof_states += body_proof_states
    return proof_states
