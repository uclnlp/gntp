#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Do not remove this line - it seems to fix the following error on the UCLCS cluster:
# ImportError: dlopen: cannot load any more object with static TLS
import nmslib

import os
import sys

import argparse

import numpy as np

import multiprocessing

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from gntp.evaluation.classification import evaluate_classification
from gntp.evaluation import evaluate, evaluate_per_predicate
from gntp.explanation.base import decode_paths, decode_proof_states_indices

import gntp

from gntp.neuralkb import NeuralKB
from gntp.training.data import Data
from gntp.training.batcher import Batcher

import logging
import time

logger = logging.getLogger(os.path.basename(sys.argv[0]))
logger.setLevel(logging.INFO)

np.set_printoptions(linewidth=48, precision=5, suppress=True)


def entropy_regularizer(rules_kb):
    return [gntp.entropy_logits(term) for rule in rules_kb for atom in rule for term in atom if gntp.is_tensor(term)]


def diversity_regularizer(kernel, neural_rules_kb):
    # Enforce diversity only on rule heads
    return [gntp.diversity(kernel, rule_graph[0][0]) for rule_graph in neural_rules_kb]


def main(argv):
    argparser = argparse.ArgumentParser('NTP 2.0', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    argparser.add_argument('--train', required=True, action='store', type=str)

    argparser.add_argument('--dev', action='store', type=str, default=None)
    argparser.add_argument('--test', action='store', type=str, default=None)

    argparser.add_argument('--test-I', action='store', type=str, default=None)
    argparser.add_argument('--test-II', action='store', type=str, default=None)

    argparser.add_argument('--all-path', action='store', type=str, default=None)

    argparser.add_argument('--clauses', '-c', action='store', type=str, default=None)
    argparser.add_argument('--mentions', action='store', type=str, default=None)
    argparser.add_argument('--mentions-min', action='store', type=int, default=1)

    # model params
    argparser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    argparser.add_argument('--batch-size', '-b', action='store', type=int, default=10)
    argparser.add_argument('--test-batch-size', action='store', type=int, default=None)

    argparser.add_argument('--k-max', '-m', action='store', type=int, default=None)
    argparser.add_argument('--max-depth', '-M', action='store', type=int, default=1)

    # training params
    argparser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    argparser.add_argument('--only-rules-epochs', action='store', type=int, default=0)
    argparser.add_argument('--only-facts-epochs', action='store', type=int, default=0)
    argparser.add_argument('--only-entities-epochs', action='store', type=int, default=0)
    argparser.add_argument('--only-ntp0-epochs', action='store', type=int, default=0)
    argparser.add_argument('--only-rules-entities-epochs', action='store', type=int, default=0)

    argparser.add_argument('--only-dev', action='store_true')

    argparser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.001)
    argparser.add_argument('--clip', action='store', type=float, default=1.0)
    argparser.add_argument('--l2', action='store', type=float, default=0.001)

    argparser.add_argument('--kernel', action='store', type=str, default='rbf',
                           choices=['linear', 'rbf', 'cosine'])

    argparser.add_argument('--auxiliary-loss-model', '--auxiliary-model', '--aux-model',
                           action='store', type=str, default='complex')
    argparser.add_argument('--auxiliary-epochs', '--aux-epochs', action='store', type=int, default=0)

    argparser.add_argument('--corrupted-pairs', '--corruptions', '-C', action='store', type=int, default=1)
    argparser.add_argument('--all', '-a', action='store_true')

    argparser.add_argument('--retrieve-k-facts', '-F', action='store', type=int, default=None)
    argparser.add_argument('--retrieve-k-rules', '-R', action='store', type=int, default=None)

    argparser.add_argument('--index-type', '-i', action='store', type=str, default='nmslib',
                           choices=['nmslib', 'faiss', 'symbol', 'exact'])

    argparser.add_argument('--index-refresh-rate', '-I', action='store', type=int, default=100)

    argparser.add_argument('--nms-m', action='store', type=int, default=15)
    argparser.add_argument('--nms-efc', action='store', type=int, default=100)
    argparser.add_argument('--nms-efs', action='store', type=int, default=100)
    argparser.add_argument('--nms-space', action='store', type=str, default='l2')

    argparser.add_argument('--evaluation-mode', '-E', action='store', type=str, default='ranking',
                           choices=['ranking', 'countries', 'ntn', 'none'])

    argparser.add_argument('--decode', '-D', action='store_true')
    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--model-type', '--model', action='store', type=str, default='moe',
                           choices=['nlp', 'ntp', 'moe', 'moe2', 'moe3'])

    argparser.add_argument('--moe-mixer', '--mixer', action='store', type=str, default='mean',
                           choices=['mean', 'max'])

    argparser.add_argument('--mixed-losses', action='store_true')
    argparser.add_argument('--mixed-losses-aggregator', action='store', type=str, default='mean',
                           choices=['mean', 'sum'])

    argparser.add_argument('--initializer', action='store', type=str, default='uniform',
                           choices=['uniform', 'xavier'])

    argparser.add_argument('--rule-embeddings-type', '--rule-type', '-X', action='store', type=str,
                           default='standard', choices=['standard', 'attention', 'sparse-attention'])
    argparser.add_argument('--entropy-regularization', action='store', type=float, default=None)

    argparser.add_argument('--unification-type', '-U', action='store', type=str,
                           default='classic', choices=['classic', 'joint'])

    argparser.add_argument('--no-ntp0', action='store_true')
    argparser.add_argument('--train-slope', action='store_true')

    argparser.add_argument('--input-type', action='store', type=str, default='standard',
                           choices=['standard', 'reciprocal'])

    argparser.add_argument('--save', action='store', type=str, default=None)
    argparser.add_argument('--load', action='store', type=str, default=None)

    argparser.add_argument('--explanation', '--explain', action='store', type=str,
                           default=None, choices=['train', 'dev', 'test'])

    argparser.add_argument('--ranking-per-predicate', action='store_true')
    argparser.add_argument('--debug', action='store_true')

    argparser.add_argument('--save-ranks-prefix', action='store', type=str, default=None)

    argparser.add_argument('--check-path', action='store', type=str, default=None)
    argparser.add_argument('--check-interval', action='store', type=int, default=1000)

    # argparser.add_argument('--controller-key-size', action='store', type=int, default=None)

    args = argparser.parse_args(argv)

    check_path = args.check_path
    check_interval = args.check_interval

    save_path = args.save
    load_path = args.load

    is_ranking_per_predicate = args.ranking_per_predicate
    is_debug = args.debug

    is_explanation = args.explanation

    nb_epochs = args.epochs
    nb_only_rules_epochs = args.only_rules_epochs
    nb_only_facts_epochs = args.only_facts_epochs
    nb_only_entities_epochs = args.only_entities_epochs
    nb_only_ntp0_epochs = args.only_ntp0_epochs
    nb_only_rules_entities_epochs = args.only_rules_entities_epochs

    is_only_dev = args.only_dev

    nb_aux_epochs = args.auxiliary_epochs

    import pprint
    pprint.pprint(vars(args))

    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    test_I_path = args.test_I
    test_II_path = args.test_II

    _all_path = args.all_path

    clauses_path = args.clauses
    mentions_path = args.mentions
    mentions_min = args.mentions_min

    entity_embedding_size = predicate_embedding_size = args.embedding_size
    symbol_embedding_size = args.embedding_size

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size

    seed = args.seed

    learning_rate = args.learning_rate
    clip_value = args.clip
    l2_weight = args.l2
    kernel_name = args.kernel

    aux_loss_model = args.auxiliary_loss_model

    nb_corrupted_pairs = args.corrupted_pairs
    is_all = args.all

    if test_batch_size is None:
        test_batch_size = batch_size * (1 + nb_corrupted_pairs * 2 * (2 if is_all else 1))

    index_type = args.index_type
    index_refresh_rate = args.index_refresh_rate

    retrieve_k_facts = args.retrieve_k_facts
    retrieve_k_rules = args.retrieve_k_rules

    nms_m = args.nms_m
    nms_efc = args.nms_efc
    nms_efs = args.nms_efs
    nms_space = args.nms_space

    k_max = args.k_max
    max_depth = args.max_depth

    evaluation_mode = args.evaluation_mode

    has_decode = args.decode

    model_type = args.model_type
    moe_mixer_type = args.moe_mixer

    mixed_losses = args.mixed_losses
    mixed_losses_aggregator_type = args.mixed_losses_aggregator

    initializer_name = args.initializer

    rule_embeddings_type = args.rule_embeddings_type
    entropy_regularization_weight = args.entropy_regularization

    unification_type = args.unification_type

    is_no_ntp0 = args.no_ntp0
    is_train_slope = args.train_slope
    input_type = args.input_type

    save_ranks_prefix = args.save_ranks_prefix

    # fire up eager
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.enable_eager_execution(config=config)

    # set the seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)

    kernel_slope = 1.0
    kernel_parameters = []

    if is_train_slope is True:
        kernel_slope = tfe.Variable(1.0, dtype=tf.float32)
        kernel_parameters += [kernel_slope]

    kernel = gntp.kernels.get_kernel_by_name(kernel_name, slope=kernel_slope)

    clauses = []
    if clauses_path:
        with open(clauses_path, 'r') as f:
            clauses += [gntp.parse_clause(line.strip()) for line in f.readlines()]

    mention_counts = gntp.read_mentions(mentions_path) if mentions_path else []
    mentions = [(s, pattern, o) for s, pattern, o, c in mention_counts if c >= mentions_min]

    data = Data(train_path=train_path,
                dev_path=dev_path,
                test_path=test_path,
                test_I_path=test_I_path,
                test_II_path=test_II_path,
                clauses=clauses,
                evaluation_mode=evaluation_mode,
                mentions=mentions,
                _all_path=_all_path,
                input_type=input_type)

    neural_kb = NeuralKB(data=data,
                         entity_embedding_size=entity_embedding_size,
                         predicate_embedding_size=predicate_embedding_size,
                         symbol_embedding_size=symbol_embedding_size,
                         model_type=model_type,
                         initializer_name=initializer_name,
                         rule_embeddings_type=rule_embeddings_type)

    nms_index_params = {
        'num_threads': multiprocessing.cpu_count() if not is_debug else 1,
        'method': 'hnsw', 'space': nms_space, 'm': nms_m, 'efc': nms_efc, 'efs': nms_efs
    }

    faiss_index_params = {}
    try:
        import faiss
        faiss_index_params['resource'] = None
        faiss_index_params['cpu'] = True
        if index_type in {'faiss'} and hasattr(faiss, 'StandardGpuResources'):
            faiss_index_params['resource'] = faiss.StandardGpuResources()
            faiss_index_params['cpu'] = False
        faiss_index_params['kernel_name'] = kernel_name
    except ImportError:
        pass

    symbol_index_params = {
        'neural_kb': neural_kb, 'kernel': kernel
    }

    index_type_to_params = {
        'nmslib': nms_index_params, 'faiss': faiss_index_params, 'symbol': symbol_index_params, 'exact': {}
    }

    assert index_type in index_type_to_params
    index_store = gntp.lookup.LookupIndexStore(index_type=index_type,
                                               index_params=index_type_to_params[index_type])

    aux_model = gntp.models.get_model_by_name(aux_loss_model)

    ntp_model = gntp.models.NTP(kernel=kernel,
                                max_depth=max_depth,
                                k_max=k_max,
                                retrieve_k_facts=retrieve_k_facts,
                                retrieve_k_rules=retrieve_k_rules,
                                index_refresh_rate=index_refresh_rate,
                                index_store=index_store,
                                unification_type=unification_type,
                                facts_kb=neural_kb.facts_kb)

    moe_model = gntp.models.MoE(ntp_model=ntp_model,
                                aux_model=aux_model,
                                moe_mixer_type=moe_mixer_type,
                                model_type=model_type,
                                mixed_losses=mixed_losses,
                                evaluation_mode=evaluation_mode,
                                mixed_losses_aggregator_type=mixed_losses_aggregator_type,
                                is_no_ntp0=is_no_ntp0,
                                entity_embedding_size=entity_embedding_size,
                                kernel_parameters=kernel_parameters)

    saver = tfe.Saver(var_list=moe_model.get_trainable_variables(neural_kb))

    if load_path is not None:
        logger.info('Loading model ..')
        saver.restore(load_path)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    batcher = Batcher(data, batch_size, nb_epochs, random_state, nb_corrupted_pairs, is_all, nb_aux_epochs)
    batches_per_epoch = batcher.nb_batches / nb_epochs if nb_epochs > 0 else 0

    start_time = time.time()

    for batch_no, (batch_start, batch_end) in enumerate(batcher.batches):
        if is_explanation is not None:  # or load_model:
            logger.info("EXPLANATION MODE ON - turning training off!")
            break

        Xi_batch, Xp_batch, Xs_batch, Xo_batch, target_inputs = batcher.get_batch(batch_no, batch_start, batch_end)

        # goals should be [GE, GE, GE]
        with tf.GradientTape() as tape:
            # If in the pre-training phase, do not consider the NTP's loss, just auxiliary model's
            is_model = is_auxiliary = True

            train_rules_only = False
            train_facts_only = False
            train_entities_only = False
            train_ntp0_only = False

            if batcher.is_pretraining is True:
                is_model = False
            elif model_type == 'ntp':  # not pre-training, and it's NTP
                is_auxiliary = False

                # if 0 < BATCH_NO - NB_PRETRAINING_BATCHES < NB_ONLY_RULES_BATCHES,
                #   then train only rule embeddings
                nb_pretraining_batches = batcher.nb_aux_batches

                nb_only_rules_batches = batches_per_epoch * nb_only_rules_epochs
                nb_only_facts_batches = batches_per_epoch * nb_only_facts_epochs
                nb_only_entities_batches = batches_per_epoch * nb_only_entities_epochs
                nb_only_ntp0_batches = batches_per_epoch * nb_only_ntp0_epochs
                nb_only_rules_entities_batches = batches_per_epoch * nb_only_rules_entities_epochs

                train_rules_only = 0 <= batch_no - nb_pretraining_batches < nb_only_rules_batches
                train_facts_only = 0 <= batch_no - nb_pretraining_batches < nb_only_facts_batches
                train_entities_only = 0 <= batch_no - nb_pretraining_batches < nb_only_entities_batches
                train_ntp0_only = 0 <= batch_no - nb_pretraining_batches < nb_only_ntp0_batches
                train_rules_entities_only = 0 <= batch_no - nb_pretraining_batches < nb_only_rules_entities_batches

                if train_rules_only:
                    train_ntp0_only = False
                    train_entities_only = False
                    train_rules_entities_only = False

            elif model_type == 'nlp':  # not pre-training, and it's NLP
                is_model = False

            neural_kb.create_neural_kb()

            final_scores, _ = moe_model.predict(goal_predicates=Xp_batch,
                                                goal_subjects=Xs_batch,
                                                goal_objects=Xo_batch,
                                                neural_kb=neural_kb,
                                                target_inputs=target_inputs,
                                                mask_indices=Xi_batch,
                                                is_training=True,
                                                is_model=is_model,
                                                is_auxiliary=is_auxiliary,
                                                only_ntp0=train_ntp0_only)

            loss = moe_model.loss(target_inputs, final_scores)

            if is_debug is True:
                print(final_scores)

            # Entropy regularization
            if rule_embeddings_type in {'attention', 'sparse-attention'} and entropy_regularization_weight is not None:
                entropy_regularization_terms = entropy_regularizer(neural_kb.rules_kb)
                loss += entropy_regularization_weight * sum(entropy_regularization_terms)

            trainable_variables = moe_model.get_trainable_variables(neural_kb,
                                                                    is_rules_only=train_rules_only,
                                                                    is_facts_only=train_facts_only,
                                                                    is_entities_only=train_entities_only,
                                                                    is_rules_entities_only=train_rules_entities_only)

            if l2_weight:
                loss += l2_weight * tf.add_n([tf.nn.l2_loss(var) for var in trainable_variables])

        logger.info('Loss @ batch {} on {}: {}'.format(batch_no, batcher.nb_batches, loss))

        gradients = tape.gradient(loss, trainable_variables)
        grads_and_vars = [(tf.clip_by_value(grad, -clip_value, clip_value), var)
                          for grad, var in zip(gradients, trainable_variables)]

        optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                  global_step=tf.train.get_or_create_global_step())

        if batch_no == 99:
            print('Training first 100 batches took {} seconds'.format(time.time() - start_time))

        # <HACKY>
        if check_path is not None:
            if batch_no == 0 or (batch_no + 1) % check_interval == 0:
                check_triples = gntp.read_triples(check_path)

                def scoring_function(_Xp, _Xs, _Xo):
                    _is_model = _is_auxiliary = True
                    if model_type == 'ntp':
                        _is_auxiliary = False
                    elif model_type == 'nlp':
                        _is_model = False
                    _final_scores, _ = moe_model.predict(_Xp, _Xs, _Xo, neural_kb, is_model=_is_model,
                                                         is_auxiliary=_is_auxiliary, is_training=False)
                    return _final_scores.numpy()
                if evaluation_mode in {'ranking'}:
                    if len(check_triples) > 0:
                        check_name = 'Check_{}'.format(len(grads_and_vars))
                        _evaluate = evaluate_per_predicate if is_ranking_per_predicate is True else evaluate
                        res = _evaluate(check_triples, data.all_triples, data.entity_to_idx, data.predicate_to_idx,
                                        batcher.entity_indices, scoring_function, test_batch_size)
                        logger.info('{} set evaluation'.format(check_name))
                        if is_ranking_per_predicate is True:
                            for k, v in res.items():
                                predicate_name = data.idx_to_predicate[k]
                                for _k, _v in v.items():
                                    logger.info("{}: {} {:.3f}".format(predicate_name, _k, _v))
                        else:
                            for k, v in res.items():
                                logger.info("{}: {}".format(k, v))
                elif evaluation_mode in {'ntn'}:
                    cut_point = None
                    eval_lst = [
                        (data.dev_triples, data.dev_labels, 'Dev'),
                        (data.test_triples, data.test_labels, 'Test')
                    ]
                    for (e_triples, e_labels, e_name) in eval_lst:
                        if len(e_triples) > 0:
                            accuracy, cut_point = evaluate_classification(e_triples, e_labels,
                                                                          data.entity_to_idx, data.predicate_to_idx,
                                                                          scoring_function, test_batch_size,
                                                                          cut_point=cut_point)
                            print('Accuracy ({}, {}, {}) {}'.format(e_name, cut_point, len(grads_and_vars), accuracy))
        # </HACKY>

    if save_path is not None:
        logger.info('Saving model ..')
        saver.save(file_prefix=save_path)

    print('Training took {} seconds'.format(time.time() - start_time))

    logger.info('Starting evaluation ..')

    neural_kb.create_neural_kb()

    idx_to_relation = {idx: relation for relation, idx in data.relation_to_idx.items()}

    if has_decode:

        print('Decoding ..')

        for neural_rule in neural_kb.neural_rules_kb:
            gntp.decode(neural_rule, neural_kb.relation_embeddings, idx_to_relation,
                        kernel=kernel)

        print('Decoding (v2) ..')

        for neural_rule in neural_kb.neural_rules_kb:
            gntp.decode_v2(neural_rule, neural_kb.relation_embeddings, idx_to_relation, data.nb_predicates,
                           kernel=kernel)

    # explanations for the train set, just temporarily
    if is_explanation is not None:
        _is_model = _is_auxiliary = True

        if model_type == 'ntp':
            _is_auxiliary = False
        elif model_type == 'nlp':
            _is_model = False

        from gntp.util import make_batches

        which_triples = []
        if is_explanation == 'train':
            which_triples = data.train_triples
        elif is_explanation == 'dev':
            which_triples = data.dev_triples
        elif is_explanation == 'test':
            which_triples = data.test_triples

        _triples = [(data.entity_to_idx[s], data.predicate_to_idx[p], data.entity_to_idx[o])
                    for s, p, o in which_triples]

        batches = make_batches(len(_triples), batch_size)

        explanations_filename = 'explanations-{}-{}.txt'.format(load_path.replace('/', '_'), is_explanation)
        with open(explanations_filename, 'w') as fw:
            for neural_rule in neural_kb.neural_rules_kb:
                decoded_rules = gntp.decode(neural_rule, neural_kb.relation_embeddings, idx_to_relation,
                                            kernel=kernel)

                for decoded_rule in decoded_rules:
                    fw.write(decoded_rule + '\n')

            fw.write('--' * 50 + '\n')

            for start, end in batches:
                batch = np.array(_triples[start:end])
                Xs_batch, Xp_batch, Xo_batch = batch[:, 0], batch[:, 1], batch[:, 2]

                _res, proof_states = moe_model.predict(Xp_batch, Xs_batch, Xo_batch, neural_kb,
                                                       is_training=False,
                                                       is_model=_is_model,
                                                       is_auxiliary=_is_auxiliary,
                                                       support_explanations=is_explanation is not None)

                # path_indices = decode_per_path_type_proof_states_indices(proof_states)
                path_indices = decode_proof_states_indices(proof_states, top_k=3)
                decoded_paths = decode_paths(path_indices, neural_kb)

                _ps, _ss, _os = Xp_batch.tolist(), Xs_batch.tolist(), Xo_batch.tolist()
                __triples = [(data.idx_to_entity[s], data.idx_to_predicate[p], data.idx_to_entity[o])
                             for s, p, o in zip(_ss, _ps, _os)]

                _scores = _res.numpy().tolist()

                for i, (_triple, _score, decoded_path) in enumerate(zip(__triples, _scores, decoded_paths)):
                    _s, _p, _o = _triple
                    _triple_str = '{}({}, {})'.format(_p, _s, _o)

                    # print(_triple_str, _score, decoded_path)
                    fw.write("{}\t{}\t{}\n".format(_triple_str, _score, decoded_path))
        logging.info('DONE with explanation...quitting.')
        sys.exit(0)

    def scoring_function(_Xp, _Xs, _Xo):
        """
        Scoring function used for computing the evaluation metrics.
        """
        _is_model = _is_auxiliary = True

        if model_type == 'ntp':
            _is_auxiliary = False
        elif model_type == 'nlp':
            _is_model = False

        _final_scores, _ = moe_model.predict(_Xp, _Xs, _Xo, neural_kb,
                                             is_model=_is_model,
                                             is_auxiliary=_is_auxiliary,
                                             is_training=False)
        return _final_scores.numpy()

    if evaluation_mode in {'countries'}:
        dev_auc = gntp.evaluation.evaluate_on_countries('dev', data.entity_to_idx, data.predicate_to_idx,
                                                        scoring_function, verbose=True)
        test_auc = gntp.evaluation.evaluate_on_countries('test', data.entity_to_idx, data.predicate_to_idx,
                                                         scoring_function, verbose=True)
        print('Last AUC-PR (dev) {:.4f}'.format(dev_auc))
        print('Last AUC-PR (test) {:.4f}'.format(test_auc))

    elif evaluation_mode in {'ranking'}:
        eval_lst = [
            (data.dev_triples, 'Dev'),
            (data.test_triples, 'Test'),
            (data.test_I_triples, 'Test-I'),
            (data.test_II_triples, 'Test-II'),
        ]

        if is_only_dev:
            eval_lst = [
                (data.dev_triples, 'Dev')
            ]

        for (eval_triples, eval_name) in eval_lst:
            if len(eval_triples) > 0:
                _evaluate = evaluate_per_predicate if is_ranking_per_predicate is True else evaluate
                ranks_path = '{}_{}.tsv'.format(save_ranks_prefix, eval_name.lower()) if save_ranks_prefix else None

                res = _evaluate(eval_triples,
                                data.all_triples,
                                data.entity_to_idx,
                                data.predicate_to_idx,
                                batcher.entity_indices,
                                scoring_function,
                                test_batch_size,
                                save_ranks_path=ranks_path)
                logger.info('{} set evaluation'.format(eval_name))

                if is_ranking_per_predicate is True:
                    for k, v in res.items():
                        predicate_name = data.idx_to_predicate[k]
                        for _k, _v in v.items():
                            logger.info("{}: {} {:.3f}".format(predicate_name, _k, _v))
                else:
                    for k, v in res.items():
                        logger.info("{}: {}".format(k, v))

    elif evaluation_mode in {'ntn'}:
        cut_point = None
        eval_lst = [
            (data.dev_triples, data.dev_labels, 'Dev'),
            (data.test_triples, data.test_labels, 'Test')
        ]
        for (e_triples, e_labels, e_name) in eval_lst:
            if len(e_triples) > 0:
                accuracy, cut_point = evaluate_classification(e_triples, e_labels,
                                                              data.entity_to_idx, data.predicate_to_idx,
                                                              scoring_function, test_batch_size, cut_point=cut_point)
                print('Accuracy ({}, {}) {}'.format(e_name, cut_point, accuracy))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
