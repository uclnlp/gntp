#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Do not remove this line - it seems to fix the following error on the UCLCS cluster:
# ImportError: dlopen: cannot load any more object with static TLS
import nmslib


import os
import sys
import json
import pdb
import argparse
import pickle
from memory_profiler import memory_usage

import numpy as np

import tensorflow as tf

from gntp.evaluation.classification import evaluate_classification
from gntp.evaluation import evaluate

from gntp.explanation import explain
from gntp.explanation.base import decode_paths, decode_proof_states_indices, decode_per_path_type_proof_states_indices

import gntp

from gntp.neuralkb import NeuralKB
from gntp.training.data import Data
from gntp.training.batcher import Batcher

from tensorflow.python.eager import context
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import option_builder

import logging
import time

logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)

DEBUG = False


def checkpoint_load(_checkpoint_path, neural_kb, optimizer):
    logger.info('Loading model...')

    logger.info('   neural kb and optimizer')
    checkpoint_model_prefix = os.path.join(_checkpoint_path, "model/")
    model_saver_path = tf.train.latest_checkpoint(checkpoint_model_prefix)

    # old format compatibility
    if os.path.exists(os.path.join(_checkpoint_path, "optim/")):
        import tensorflow.contrib.eager as tfe
        checkpoint_optim_prefix = os.path.join(_checkpoint_path, "optim/")
        optim_checkpoint_path = tf.train.latest_checkpoint(checkpoint_optim_prefix)
        if optim_checkpoint_path is not None:
            optim_checkpoint = tfe.Checkpoint(optimizer=optimizer, optimizer_step=tf.train.get_or_create_global_step())
            optim_checkpoint.restore(optim_checkpoint_path)
            logger.info('   optimiser')
        else:
            logger.info("   ....couldn't find optim/, ignoring it (loading old model).")

        model_saver = tfe.Saver(neural_kb.variables)
        model_saver.restore(model_saver_path)

    else:
        model_saver = tf.train.Saver(neural_kb.variables +
                                     optimizer.variables() +
                                     [tf.train.get_or_create_global_step()])
        model_saver.restore(None, model_saver_path)

    logger.info('... loading done.')


def check_checkpoint_finished(_checkpoint_path):
    finished_path = os.path.join(_checkpoint_path, 'finished')
    if os.path.exists(finished_path):
        return True
    return False


def load_random_state(_checkpoint_path):
    random_state = None
    try:
        variables_path = os.path.join(_checkpoint_path, "random_state.pickle")
        random_state = pickle.load(open(variables_path, 'rb'))
        logger.info('   random state')
    except FileNotFoundError:
        logger.info("   ...couldn't find random_state.pickle, ignoring it (loading old model).")

    # logger.info('... loading done.')
    return random_state


def checkpoint_store(_checkpoint_path, neural_kb, optimizer, random_state, args):
    logger.info('Storing model...')

    _variables_path = os.path.join(_checkpoint_path, "random_state.pickle")
    _checkpoint_model_prefix = os.path.join(_checkpoint_path, "model/ckpt")
    # _checkpoint_optim_prefix = os.path.join(_checkpoint_path, "optim/ckpt")

    if _checkpoint_path is not None and not os.path.exists(_checkpoint_path):
        os.makedirs(_checkpoint_path)

    if not os.path.exists(_checkpoint_model_prefix):
        os.makedirs(_checkpoint_model_prefix)

    var_list = neural_kb.variables + optimizer.variables() + [tf.train.get_or_create_global_step()]
    _model_saver = tf.train.Saver(var_list=var_list)

    _model_saver.save(None,
                      save_path=_checkpoint_model_prefix,
                      global_step=tf.train.get_or_create_global_step())

    pickle.dump(random_state, open(_variables_path, 'wb'))

    arguments_path = os.path.join(_checkpoint_path, '..', 'arguments.json')
    with open(arguments_path, 'w') as _f:
        json.dump(vars(args), _f, indent=4, sort_keys=True)

    logger.info('... storing done.')


def do_eval(evaluation_mode, model, neural_kb, data, batcher,
            batch_size, index_type_to_params, is_no_ntp0, is_explanation,
            dev_only=False, tensorboard=False, verbose=True,
            exact_knn_evaluation=None, test_batch_size=None):

    if exact_knn_evaluation is not None:
        if exact_knn_evaluation == 'exact':
            logging.warning('Using EXACT evaluation with a brute-force kNN. Expect slowness.')
        elif exact_knn_evaluation == 'faiss':
            logging.info('Using EXACT evaluation with FAISS.')

        _index_type = exact_knn_evaluation
        eval_index_store = gntp.lookup.LookupIndexStore(index_type=_index_type,
                                                        index_params=index_type_to_params[_index_type])
        model.index_store = eval_index_store
    else:
        logging.warning('Using non-exact kNN for evaluation. This might influence the performance.')

    def scoring_function(_Xp, _Xs, _Xo):
        """
        Scoring function used for computing the evaluation metrics.
        """
        _p_emb = tf.nn.embedding_lookup(neural_kb.relation_embeddings, _Xp)
        _s_emb = tf.nn.embedding_lookup(neural_kb.entity_embeddings, _Xs)
        _o_emb = tf.nn.embedding_lookup(neural_kb.entity_embeddings, _Xo)

        _res, _ = model.predict(_p_emb, _s_emb, _o_emb,
                                neural_facts_kb=neural_kb.neural_facts_kb,
                                neural_rules_kb=neural_kb.neural_rules_kb,
                                is_training=False,
                                no_ntp0=is_no_ntp0)
        return _res.numpy()

    if evaluation_mode in {'countries'}:
        datasets = ['dev'] if dev_only else ['dev', 'test']
        for dataset in datasets:
            start = time.time()
            auc = gntp.evaluation.evaluate_on_countries(dataset, data.entity_to_idx, data.predicate_to_idx,
                                                        scoring_function, verbose=verbose)
            logger.info('Evaluated {}, it took {}'.format(dataset, time.time() - start))
            if verbose:
                print('Last AUC-PR ({}) {:.4f}'.format(dataset, auc))
            if tensorboard:
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('performance_{}/auc_pr'.format(dataset), auc)

    elif evaluation_mode in {'ranking'}:
        datasets = [(data.dev_triples, 'Dev')] if dev_only else [(data.dev_triples, 'Dev'), (data.test_triples, 'Test')]
        for (eval_triples, eval_name) in datasets:
            if len(eval_triples) > 0:
                if is_explanation:
                    explain(eval_triples, data.entity_to_idx, data.predicate_to_idx, scoring_function)
                else:
                    start = time.time()
                    res = evaluate(eval_triples, data.all_triples, data.entity_to_idx, data.predicate_to_idx,
                                   batcher.entity_indices, scoring_function,
                                   test_batch_size)
                    logger.info('Evaluated {}, it took {}'.format(eval_name, time.time() - start))

                    if verbose:
                        logger.info('{} set evaluation'.format(eval_name))
                    for k, v in res.items():
                        if verbose:
                            logger.info("{}: {}".format(k, v))
                        if tensorboard:
                            with tf.contrib.summary.always_record_summaries():
                                tf.contrib.summary.scalar('performance_{}/{}'.format(eval_name, k), v)

    elif evaluation_mode in {'ntn'}:
        cut_point = None
        all_datasets = [(data.dev_triples, data.dev_labels, 'Dev'), (data.test_triples, data.test_labels, 'Test')]
        datasets = [(data.dev_triples, data.dev_labels, 'Dev')] if dev_only else all_datasets
        for (e_triples, e_labels, e_name) in datasets:
            if len(e_triples) > 0:
                start = time.time()
                accuracy, cut_point = evaluate_classification(e_triples, e_labels, data.entity_to_idx,
                                                              data.predicate_to_idx, scoring_function,
                                                              batch_size, cut_point=cut_point)
                logger.info('Evaluated {}, it took {}'.format(e_name, time.time() - start))
                if verbose:
                    print('Accuracy ({}, {}) {}'.format(e_name, cut_point, accuracy))


def main(argv):
    argparser = argparse.ArgumentParser('NTP 2.0', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    # WARNING: for countries, it's not necessary to enter the dev/test set as the evaluation does so
    # TODO: fix this behavior - all datasets should have the same behavior
    argparser.add_argument('--train', action='store', type=str)
    argparser.add_argument('--dev', action='store', type=str, default=None)
    argparser.add_argument('--test', action='store', type=str, default=None)

    argparser.add_argument('--clauses', '-c', action='store', type=str, default=None)
    argparser.add_argument('--mentions', action='store', type=str, default=None)
    argparser.add_argument('--mentions-min', action='store', type=int, default=1)

    # model params
    argparser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    argparser.add_argument('--batch-size', '-b', action='store', type=int, default=10)
    # k-max for the new variable
    argparser.add_argument('--k-max', '-m', action='store', type=int, default=None)
    argparser.add_argument('--max-depth', '-M', action='store', type=int, default=1)

    # training params
    argparser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    argparser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.001)
    argparser.add_argument('--clip', action='store', type=float, default=1.0)
    argparser.add_argument('--l2', action='store', type=float, default=0.01)

    argparser.add_argument('--kernel', action='store', type=str, default='rbf',
                           choices=['linear', 'rbf'])

    argparser.add_argument('--auxiliary-loss-weight', '--auxiliary-weight', '--aux-weight',
                           action='store', type=float, default=None)
    argparser.add_argument('--auxiliary-loss-model', '--auxiliary-model', '--aux-model',
                           action='store', type=str, default='complex')
    argparser.add_argument('--auxiliary-epochs', '--aux-epochs', action='store', type=int, default=0)

    argparser.add_argument('--corrupted-pairs', '--corruptions', '-C', action='store', type=int, default=1)
    argparser.add_argument('--all', '-a', action='store_true')

    argparser.add_argument('--retrieve-k-facts', '-F', action='store', type=int, default=None)
    argparser.add_argument('--retrieve-k-rules', '-R', action='store', type=int, default=None)

    argparser.add_argument('--index-type', '-i', action='store', type=str, default='nmslib',
                           choices=['nmslib', 'faiss', 'faiss-cpu', 'random', 'exact'])

    argparser.add_argument('--index-refresh-rate', '-I', action='store', type=int, default=100)

    argparser.add_argument('--nms-m', action='store', type=int, default=15)
    argparser.add_argument('--nms-efc', action='store', type=int, default=100)
    argparser.add_argument('--nms-efs', action='store', type=int, default=100)

    argparser.add_argument('--evaluation-mode', '-E', action='store', type=str, default='ranking',
                           choices=['ranking', 'countries', 'ntn', 'none'])
    argparser.add_argument('--exact-knn-evaluation', action='store', type=str, default=None,
                           choices=[None, 'faiss', 'exact'])

    argparser.add_argument('--loss-aggregator', action='store', type=str, default='sum',
                           choices=['sum', 'mean'])

    argparser.add_argument('--decode', '-D', action='store_true')
    argparser.add_argument('--seed', action='store', type=int, default=0)

    argparser.add_argument('--keep-prob', action='store', type=float, default=1.0)
    argparser.add_argument('--initializer', action='store', type=str, default='uniform',
                           choices=['uniform', 'xavier'])

    argparser.add_argument('--mixed-losses', action='store_true')
    argparser.add_argument('--mixed-losses-aggregator', action='store', type=str, default='mean',
                           choices=['mean', 'sum'])

    argparser.add_argument('--rule-embeddings-type', '--rule-type', '-X', action='store', type=str,
                           default='standard', choices=['standard', 'attention', 'sparse-attention'])

    argparser.add_argument('--unification-type', '-U', action='store', type=str,
                           default='classic', choices=['classic', 'joint'])

    argparser.add_argument('--unification-aggregation-type', action='store', type=str,
                           default='min', choices=['min', 'mul', 'minmul'])

    argparser.add_argument('--epoch-based-batches', action='store_true')

    argparser.add_argument('--evaluate-per-epoch', action='store_true')

    argparser.add_argument('--no-ntp0', action='store_true')

    # checkpointing and regular model saving / loading - if checkpoint-path is not None - do checkpointing
    argparser.add_argument('--dump-path', type=str, default=None)
    argparser.add_argument('--checkpoint', action='store_true')
    argparser.add_argument('--checkpoint-frequency', type=int, default=1000)
    argparser.add_argument('--save', action='store_true')
    argparser.add_argument('--load', action='store_true')

    argparser.add_argument('--explanation', '--explain', action='store', type=str,
                           default=None, choices=['train', 'dev', 'test'])

    argparser.add_argument('--profile', action='store_true')
    argparser.add_argument('--tf-profiler', action='store_true')
    argparser.add_argument('--tensorboard', action='store_true')
    argparser.add_argument('--multimax', action='store_true')

    argparser.add_argument('--dev-only', action='store_true')

    argparser.add_argument('--only-rules-epochs', action='store', type=int, default=0)
    argparser.add_argument('--test-batch-size', action='store', type=int, default=None)

    argparser.add_argument('--input-type', action='store', type=str, default='standard',
                           choices=['standard', 'reciprocal'])

    argparser.add_argument('--use-concrete', action='store_true')

    args = argparser.parse_args(argv)

    checkpoint = args.checkpoint
    dump_path = args.dump_path
    save = args.save
    load = args.load

    is_explanation = args.explanation

    nb_epochs = args.epochs
    nb_aux_epochs = args.auxiliary_epochs

    arguments_filename = None
    checkpoint_path = None
    if load:
        logger.info("Loading arguments from the loaded model...")
        arguments_filename = os.path.join(dump_path, 'arguments.json')
        checkpoint_path = os.path.join(dump_path, 'final_model/')
        # load a model, if there's one to load
    elif checkpoint and not check_checkpoint_finished(os.path.join(dump_path, 'checkpoints/')):
        checkpoint_path = os.path.join(dump_path, 'checkpoints/')
        logger.info("Loading arguments from an unfinished checkpoint...")
        arguments_filename = os.path.join(dump_path, 'arguments.json')

    loading_type = None

    if arguments_filename is not None and os.path.exists(arguments_filename):
        with open(arguments_filename, 'r') as f:
            json_arguments = json.load(f)
        args = argparse.Namespace(**json_arguments)
        if load:
            loading_type = 'model'
        elif checkpoint and not check_checkpoint_finished(os.path.join(dump_path, 'checkpoints/')):
            loading_type = 'checkpoint'

        # Load arguments from json

        # args = argparse.Namespace(**json_arguments)

        # args = vars(args)
        # for k, v in json_arguments.items():
        #     if k in args and args[k] != v:
        #         logger.info("\t{}={} (overriding loaded model's value of {})".format(k, args[k], v))
        #     if k not in args:
        #         args[k] = v
        #         logger.info("\t{}={} (overriding loaded model's value of {})".format(k, args[k], v))

    import pprint
    pprint.pprint(vars(args))

    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    clauses_path = args.clauses
    mentions_path = args.mentions
    mentions_min = args.mentions_min

    input_type = args.input_type

    entity_embedding_size = predicate_embedding_size = args.embedding_size
    symbol_embedding_size = args.embedding_size

    batch_size = args.batch_size
    seed = args.seed

    learning_rate = args.learning_rate
    clip_value = args.clip
    l2_weight = args.l2
    kernel_name = args.kernel

    aux_loss_weight = 1.0
    if 'auxiliary_loss_weight' in args:
        aux_loss_weight = args.auxiliary_loss_weight

    aux_loss_model = args.auxiliary_loss_model

    nb_corrupted_pairs = args.corrupted_pairs
    is_all = args.all

    index_type = args.index_type
    index_refresh_rate = args.index_refresh_rate

    retrieve_k_facts = args.retrieve_k_facts
    retrieve_k_rules = args.retrieve_k_rules

    nms_m = args.nms_m
    nms_efc = args.nms_efc
    nms_efs = args.nms_efs

    k_max = args.k_max
    max_depth = args.max_depth

    evaluation_mode = args.evaluation_mode
    exact_knn_evaluation = args.exact_knn_evaluation

    loss_aggregator = args.loss_aggregator

    has_decode = args.decode

    keep_prob = 1.0
    if 'keep_prob' in args:
        keep_prob = args.keep_prob
    initializer_name = args.initializer

    mixed_losses = args.mixed_losses
    mixed_losses_aggregator_type = args.mixed_losses_aggregator

    rule_embeddings_type = args.rule_embeddings_type

    unification_type = args.unification_type
    unification_aggregation_type = args.unification_aggregation_type

    is_no_ntp0 = args.no_ntp0
    checkpoint_frequency = args.checkpoint_frequency

    profile = args.profile
    tf_profiler = args.tf_profiler
    tensorboard = args.tensorboard

    multimax = args.multimax
    dev_only = args.dev_only

    n_only_rules_epochs = args.only_rules_epochs

    test_batch_size = args.test_batch_size

    if test_batch_size is None:
        test_batch_size = batch_size * (1 + nb_corrupted_pairs * 2 * (2 if is_all else 1))

    # fire up eager
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.enable_eager_execution(config=config)

    # set the seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)

    epoch_based_batches = args.epoch_based_batches
    evaluate_per_epoch = args.evaluate_per_epoch
    use_concrete = args.use_concrete

    import multiprocessing

    nms_index_params = {
        'method': 'hnsw',
        'space': 'l2',
        'num_threads': multiprocessing.cpu_count(),
        'm': nms_m,
        'efc': nms_efc,
        'efs': nms_efs
    }

    faiss_index_params = {}
    faiss_index_params_cpu = {}
    try:
        import faiss
        faiss_index_params = {
            'resource': faiss.StandardGpuResources() if index_type in {'faiss'} else None
        }
        if faiss_index_params['resource'] is not None:
            faiss_index_params['resource'].noTempMemory()
        faiss_index_params_cpu = {
            'cpu': True
        }
    except ImportError:
        pass

    random_index_params = {
        'random_state': random_state,
    }

    index_type_to_params = {
        'nmslib': nms_index_params,
        'faiss-cpu': faiss_index_params_cpu,
        'faiss': faiss_index_params,
        'random': random_index_params,
        'exact': {},
    }

    kernel = gntp.kernels.get_kernel_by_name(kernel_name)

    clauses = []
    if clauses_path:
        with open(clauses_path, 'r') as f:
            clauses += [gntp.parse_clause(line.strip()) for line in f.readlines()]

    mention_counts = gntp.read_mentions(mentions_path) if mentions_path else []
    mentions = [(s, pattern, o) for s, pattern, o, c in mention_counts if c >= mentions_min]

    data = Data(train_path=train_path,
                dev_path=dev_path,
                test_path=test_path,
                clauses=clauses,
                evaluation_mode=evaluation_mode,
                mentions=mentions,
                input_type=input_type)

    index_store = gntp.lookup.LookupIndexStore(index_type=index_type,
                                               index_params=index_type_to_params[index_type])

    aux_model = gntp.models.get_model_by_name(aux_loss_model)

    model = gntp.models.NTP(kernel=kernel,
                            max_depth=max_depth,
                            k_max=k_max,
                            retrieve_k_facts=retrieve_k_facts,
                            retrieve_k_rules=retrieve_k_rules,
                            index_refresh_rate=index_refresh_rate,
                            index_store=index_store,
                            unification_type=unification_type)

    neural_kb = NeuralKB(data=data,
                         entity_embedding_size=entity_embedding_size,
                         predicate_embedding_size=predicate_embedding_size,
                         symbol_embedding_size=symbol_embedding_size,
                         model_type='ntp',
                         initializer_name=initializer_name,
                         rule_embeddings_type=rule_embeddings_type,
                         use_concrete=use_concrete)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    if loading_type == 'checkpoint':
        logger.info("********** Resuming from an unfinished checkpoint **********")
        # dirty hack, but this initializes optimizer's slots, so the loader can populate them
        optimizer._create_slots(neural_kb.variables)
        checkpoint_load(checkpoint_path, neural_kb, optimizer)

    elif loading_type == 'model':
        load_path = os.path.join(dump_path, 'final_model/')
        checkpoint_load(load_path, neural_kb, optimizer)

    # bather will always be ran with the starting random_state...
    batcher = Batcher(data, batch_size, nb_epochs, random_state, nb_corrupted_pairs, is_all, nb_aux_epochs,
                      epoch_based_batches=epoch_based_batches)

    batches_per_epoch = batcher.nb_batches / nb_epochs if nb_epochs > 0 else 0

    # ...and after that, if there's a random state to load, load it :)
    if loading_type is not None:
        checkpoint_rs = load_random_state(checkpoint_path)
        random_state.set_state(checkpoint_rs.get_state())

    batch_times = []
    logger.info('Starting training (for {} batches)..'.format(len(batcher.batches)))

    if tf.train.get_or_create_global_step().numpy() > 0:
        logger.info('...checkpoint restoration - resuming from batch no {}'.format(
            tf.train.get_or_create_global_step().numpy() + 1))

    if tensorboard:
        # TODO add changeable params too
        if not os.path.exists(dump_path):
            os.makedirs(dump_path)
        else:
            # this should never happen
            pass

        writer = tf.contrib.summary.create_file_writer(dump_path)
        writer.set_as_default()

    per_epoch_losses = []

    if tf_profiler:
        profiler = model_analyzer.Profiler()

    start_training_time = time.time()

    n_epochs_finished = 0

    if profile:
        manager = multiprocessing.Manager()
        gpu_memory_profiler_return = manager.list()

        def gpu_memory_profiler():
            import subprocess
            import os
            env = os.environ.copy()
            which_gpu = -1
            if 'CUDA_VISIBLE_DEVICES' in env:
                try:
                    which_gpu = int(env['CUDA_VISIBLE_DEVICES'])
                except:
                    pass
            del env['LD_LIBRARY_PATH']
            while True:
                time.sleep(0.1)
                cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv"]
                output = subprocess.check_output(cmd, env=env)
                output = output.decode('utf-8')
                output = output.split('\n')
                if len(output) == 3:  # there's only one gpu
                    which_gpu = 0
                output = output[1:-1]
                if which_gpu > -1:
                    gpu_memory_profiler_return.append(int(output[which_gpu].split()[0]))
                else:
                    gpu_memory_profiler_return.append(output)
            return

        gpu_memory_job = multiprocessing.Process(target=gpu_memory_profiler)
        gpu_memory_job.start()

    is_epoch_end = False
    with context.eager_mode():

        for batch_no, (batch_start, batch_end) in enumerate(batcher.batches):

            if tf_profiler:
                opts = (
                    option_builder.ProfileOptionBuilder(
                        option_builder.ProfileOptionBuilder.trainable_variables_parameter())
                        .with_max_depth(100000)
                        .with_step(batch_no)
                        .with_timeline_output('eager_profile')
                        .with_accounted_types(['.*'])
                        .build()
                )

                context.enable_run_metadata()

            # print(sum(random_state.get_state()[1]))

            # TODO fix this - this was here due to checkpointing but causes the first batch to be skipped
            # and will likely cause the test to fail?
            # if tf.train.get_or_create_global_step().numpy() + 1 > batch_no:
            #     continue
            if is_explanation is not None:  # or load_model:
                logger.info("EXPLANATION MODE ON - turning training off!")
                break

            start_time = time.time()

            is_epoch_start = is_epoch_end
            is_epoch_end = (batch_no + 1) - int((batch_no + 1) / batches_per_epoch) * batches_per_epoch < 1

            Xi_batch, Xp_batch, Xs_batch, Xo_batch, target_inputs = batcher.get_batch(batch_no, batch_start, batch_end)

            Xi_batch = tf.convert_to_tensor(Xi_batch, dtype=tf.int32)

            # goals should be [GE, GE, GE]
            with tf.GradientTape() as tape:

                if n_only_rules_epochs > n_epochs_finished:
                    is_rules_only = True
                else:
                    is_rules_only = False

                neural_kb.create_neural_kb(is_epoch_start, training=True)

                p_emb = tf.nn.embedding_lookup(neural_kb.relation_embeddings, Xp_batch)
                s_emb = tf.nn.embedding_lookup(neural_kb.entity_embeddings, Xs_batch)
                o_emb = tf.nn.embedding_lookup(neural_kb.entity_embeddings, Xo_batch)

                if keep_prob != 1.0:
                    p_emb = tf.nn.dropout(p_emb, keep_prob)
                    s_emb = tf.nn.dropout(s_emb, keep_prob)
                    o_emb = tf.nn.dropout(o_emb, keep_prob)

                if batcher.is_pretraining:
                    # PRE-TRAINING
                    aux_scores = aux_model.predict(p_emb, s_emb, o_emb)
                    loss = aux_model.loss(target_inputs, aux_scores, aggregator=loss_aggregator)
                else:

                    goal_scores, other = model.predict(p_emb, s_emb, o_emb,
                                                       neural_facts_kb=neural_kb.neural_facts_kb,
                                                       neural_rules_kb=neural_kb.neural_rules_kb,
                                                       mask_indices=Xi_batch,
                                                       is_training=True,
                                                       target_inputs=target_inputs,
                                                       mixed_losses=mixed_losses,
                                                       aggregator_type=mixed_losses_aggregator_type,
                                                       no_ntp0=is_no_ntp0,
                                                       support_explanations=is_explanation is not None,
                                                       unification_score_aggregation=unification_aggregation_type,
                                                       multimax=multimax,
                                                       tensorboard=tensorboard)

                    proof_states, new_target_inputs = other

                    if multimax:
                        target_inputs = new_target_inputs

                    model_loss = model.loss(target_inputs, goal_scores, aggregator=loss_aggregator)
                    loss = model_loss

                    if aux_loss_weight is not None and aux_loss_weight > 0.0:
                        aux_scores = aux_model.predict(p_emb, s_emb, o_emb)
                        loss_aux = aux_loss_weight * aux_model.loss(target_inputs, aux_scores, aggregator=loss_aggregator)
                        loss += loss_aux

                if l2_weight:
                    loss_l2_weight = l2_weight * tf.add_n([tf.nn.l2_loss(var) for var in neural_kb.variables])
                    if loss_aggregator == 'mean':
                        num_of_vars = tf.reduce_sum([tf.reduce_prod(var.shape) for var in neural_kb.variables])
                        loss_l2_weight /= tf.cast(num_of_vars, tf.float32)
                    loss += loss_l2_weight

            # if not is_epoch_end:
            per_epoch_losses.append(loss.numpy())

            logger.info('Loss @ batch {} on {}: {}'.format(batch_no, batcher.nb_batches, loss))

            model_variables = neural_kb.get_trainable_variables(is_rules_only=is_rules_only)
            gradients = tape.gradient(loss, model_variables)
            grads_and_vars = [(tf.clip_by_value(grad, -clip_value, clip_value), var)
                              for grad, var in zip(gradients, model_variables)]

            optimizer.apply_gradients(grads_and_vars=grads_and_vars,
                                      global_step=tf.train.get_or_create_global_step())

            if tensorboard:
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('loss_total', loss)
                    tf.contrib.summary.scalar('loss_ntp_model', model_loss)
                    if aux_loss_weight is not None and aux_loss_weight > 0.0:
                        tf.contrib.summary.scalar('loss_aux_model', loss_aux)
                    if l2_weight != 0.0:
                        tf.contrib.summary.scalar('loss_l2_weight', loss_l2_weight)
                    tf.contrib.summary.histogram('embeddings_relation', neural_kb.relation_embeddings)
                    tf.contrib.summary.histogram('embeddings_entity', neural_kb.entity_embeddings)

                with tf.contrib.summary.always_record_summaries():
                    for grad, var in grads_and_vars:
                        tf.contrib.summary.scalar('gradient_sparsity_{}'.format(var.name.replace(':', '__')),
                                                  tf.nn.zero_fraction(grad))
                        # if batch_end % data.nb_examples == 0 or batch_end % data.nb_examples == 1:
                        #     pdb.set_trace()
                        gradient_norm = tf.sqrt(tf.reduce_sum(tf.pow(grad, 2)))
                        tf.contrib.summary.scalar('gradient_norm_{}'.format(var.name.replace(':', '__')),
                                                  gradient_norm)
                        tf.contrib.summary.histogram('gradient_{}'.format(var.name.replace(':', '__')),
                                                  grad)
                        tf.contrib.summary.histogram('variable_{}'.format(var.name.replace(':', '__')),
                                                     var)
                        # gradient_values = tf.reduce_sum(tf.abs(grad))
                        # tf.contrib.summary.scalar('gradient_values/{}'.format(var.name.replace(':', '__')),
                        #                           gradient_values)

                    # grads = [g for g, _ in grads_and_vars]
                    # flattened_grads = tf.concat([tf.reshape(t, [-1]) for t in grads], axis=0)
                    # flattened_vars = tf.concat([tf.reshape(t, [-1]) for t in neural_kb.variables], axis=0)
                    # tf.contrib.summary.histogram('values_grad', flattened_grads)
                    # tf.contrib.summary.histogram('values_var', flattened_vars)
            if tensorboard:
                with tf.contrib.summary.always_record_summaries():
                    tf.contrib.summary.scalar('time_per_batch', time.time() - start_time)
            if tensorboard and is_epoch_end:
                with tf.contrib.summary.always_record_summaries():
                    tb_pel = sum(per_epoch_losses)
                    if loss_aggregator == 'mean':
                        tb_pel /= len(per_epoch_losses)
                    tf.contrib.summary.scalar('per_epoch_loss', tb_pel)

            if is_epoch_end:
                n_epochs_finished += 1
                per_epoch_losses = []

            # post-epoch whatever...
            if evaluate_per_epoch and is_epoch_end:
                index_type = 'faiss' if exact_knn_evaluation is None else exact_knn_evaluation
                tmp_exact_knn_eval = exact_knn_evaluation
                if exact_knn_evaluation is None and index_type == 'faiss':
                    tmp_exact_knn_eval = 'faiss'
                do_eval(evaluation_mode, model, neural_kb, data, batcher,
                        batch_size, index_type_to_params, is_no_ntp0, is_explanation,
                        dev_only=True, tensorboard=tensorboard, verbose=True, exact_knn_evaluation=tmp_exact_knn_eval,
                        test_batch_size=test_batch_size)

            # # checkpoint saving
            if checkpoint_path is not None and (batch_no + 1) % checkpoint_frequency == 0:
                checkpoint_store(checkpoint_path, neural_kb, optimizer, random_state, args)

            if profile:
                if batch_no != 0:  # skip the first one as it's significantly longer (warmup?)
                    batch_times.append(time.time() - start_time)
                if batch_no == 10:
                    break

            if tf_profiler:
                profiler.add_step(batch_no, context.export_run_metadata())
                context.disable_run_metadata()
                # profiler.profile_operations(opts)
                profiler.profile_graph(options=opts)

    end_time = time.time()

    if tf_profiler:
        profiler.advise(options=model_analyzer.ALL_ADVICE)

    if profile:
        gpu_memory_job.terminate()
        if len(gpu_memory_profiler_return) == 0:
            gpu_memory_profiler_return = [0]
        nb_negatives = nb_corrupted_pairs * 2 * (2 if is_all else 1)
        nb_triple_variants = 1 + nb_negatives
        examples_per_batch = nb_triple_variants * batch_size
        print('Examples per batch: {}'.format(examples_per_batch))
        print('Batch times: {}'.format(batch_times))
        time_per_batch = np.average(batch_times)
        print('Average time per batch: {}'.format(time_per_batch))
        print('examples per second: {}'.format(examples_per_batch / time_per_batch))
    else:
        if is_explanation is None:
            logger.info('Training took {} seconds'.format(end_time - start_training_time))

    # last checkpoint save
    if checkpoint_path is not None:
        checkpoint_store(checkpoint_path, neural_kb, optimizer, random_state, args)

    # and save the model, if you want to save it (it's better practice to have
    # the checkpoint_path different to save_path, as one can save checkpoints on scratch, and models permanently
    if save:
        save_path = os.path.join(dump_path, 'final_model/')
        checkpoint_store(save_path, neural_kb, optimizer, random_state, args)

    # TODO prettify profiling
    if profile:
        return max(gpu_memory_profiler_return)

    logger.info('Starting evaluation ..')

    neural_kb.create_neural_kb()

    idx_to_relation = {idx: relation for relation, idx in data.relation_to_idx.items()}

    if has_decode:
        for neural_rule in neural_kb.neural_rules_kb:
            gntp.decode(neural_rule, neural_kb.relation_embeddings, idx_to_relation, kernel=kernel)

    # explanations for the train set, just temporarily
    if is_explanation is not None:

        from gntp.util import make_batches

        which_triples = []
        if is_explanation == 'train':
            which_triples = data.train_triples
        elif is_explanation == 'dev':
            which_triples = data.dev_triples
        elif is_explanation == 'test':
            which_triples = data.test_triples

        _triples = [(data.entity_to_idx[s],
                     data.predicate_to_idx[p],
                     data.entity_to_idx[o])
                    for s, p, o in which_triples]

        batches = make_batches(len(_triples), batch_size)

        explanations_filename = 'explanations-{}-{}.txt'.format(checkpoint_path.replace('/', '_'), is_explanation)
        with open(explanations_filename, 'w') as fw:
            for neural_rule in neural_kb.neural_rules_kb:
                decoded_rules = gntp.decode(neural_rule, neural_kb.relation_embeddings, idx_to_relation, kernel=kernel)

                for decoded_rule in decoded_rules:
                    fw.write(decoded_rule + '\n')

            fw.write('--' * 50 + '\n')

            for start, end in batches:
                batch = np.array(_triples[start:end])
                Xs_batch, Xp_batch, Xo_batch = batch[:, 0], batch[:, 1], batch[:, 2]

                _p_emb = tf.nn.embedding_lookup(neural_kb.relation_embeddings, Xp_batch)
                _s_emb = tf.nn.embedding_lookup(neural_kb.entity_embeddings, Xs_batch)
                _o_emb = tf.nn.embedding_lookup(neural_kb.entity_embeddings, Xo_batch)

                _res, (proof_states, _) = model.predict(_p_emb, _s_emb, _o_emb,
                                                        neural_facts_kb=neural_kb.neural_facts_kb,
                                                        neural_rules_kb=neural_kb.neural_rules_kb,
                                                        is_training=False,
                                                        no_ntp0=is_no_ntp0,
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

    eval_start = time.time()
    do_eval(evaluation_mode, model, neural_kb, data, batcher,
            batch_size, index_type_to_params, is_no_ntp0, is_explanation,
            dev_only=dev_only,
            exact_knn_evaluation=exact_knn_evaluation,
            test_batch_size=test_batch_size)
    logging.info('Evaluation took {} seconds'.format(time.time() - eval_start))


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))

    if '--profile' in sys.argv:
        mem_use, ret = memory_usage((main, (sys.argv[1:],)), interval=0.1, max_usage=True, retval=True)
        print('Maximum CPU memory used: {}'.format(mem_use))
        print('Maximum GPU memory used: {}'.format(ret))
        print('Total maximum memory used: {}'.format(mem_use[0] + ret))
    else:
        main(sys.argv[1:])
