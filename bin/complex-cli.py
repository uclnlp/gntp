#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from gntp.evaluation import fast_evaluate

from gntp.models import ComplEx

from gntp.neuralkb import NeuralKB
from gntp.training.data import Data
from gntp.training.batcher import Batcher

import logging
import time


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)


def evaluate(model: ComplEx,
             data: Data,
             nkb: NeuralKB,
             batch_size: int,
             is_brief: bool = False,
             save_ranks_prefix: str = None):
    e_all_emb = nkb.entity_embeddings

    def scoring_function(e_xp, e_xs, e_xo):
        e_p_emb = tf.nn.embedding_lookup(nkb.predicate_embeddings, e_xp)
        e_s_emb = tf.nn.embedding_lookup(nkb.entity_embeddings, e_xs)
        e_o_emb = tf.nn.embedding_lookup(nkb.entity_embeddings, e_xo)
        # [B, N]
        e_x_sp = model.score_sp(e_p_emb, e_s_emb, e_all_emb)
        e_x_po = model.score_po(e_p_emb, e_all_emb, e_o_emb)
        return e_x_sp.numpy(), e_x_po.numpy()

    for (eval_triples, eval_name) in [(data.dev_triples, 'Dev'), (data.test_triples, 'Test')]:
        if len(eval_triples) > 0:
            save_ranks_path = '{}_{}.tsv'.format(save_ranks_prefix, eval_name.lower()) if save_ranks_prefix else None

            res = fast_evaluate(eval_triples,
                                data.all_triples,
                                data.entity_to_idx,
                                data.predicate_to_idx,
                                scoring_function,
                                batch_size,
                                save_ranks_path=save_ranks_path)
            if is_brief:
                print('EVAL {} {}'.format(eval_name, str(res)))
            else:
                logger.info('{} set evaluation'.format(eval_name))
                for k, v in res.items():
                    logger.info("{}: {}".format(k, v))
    return


def main(argv):
    argparser = argparse.ArgumentParser('ComplEx-N3', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('--train', action='store', type=str)
    argparser.add_argument('--dev', action='store', type=str, default=None)
    argparser.add_argument('--test', action='store', type=str, default=None)

    # model params
    argparser.add_argument('--embedding-size', '-k', action='store', type=int, default=100)
    argparser.add_argument('--batch-size', '-b', action='store', type=int, default=25)
    # training params
    argparser.add_argument('--epochs', '-e', action='store', type=int, default=100)
    argparser.add_argument('--learning-rate', '-l', action='store', type=float, default=0.001)
    argparser.add_argument('--L3', action='store', type=float, default=None)
    argparser.add_argument('--seed', action='store', type=int, default=0)
    argparser.add_argument('--initializer', action='store', type=str, default='xavier',
                           choices=['uniform', 'xavier'])
    argparser.add_argument('--validate-every', '-V', action='store', type=int, default=None)
    argparser.add_argument('--input-type', '-I', action='store', type=str, default='standard',
                           choices=['standard', 'reciprocal'])

    argparser.add_argument('--save', action='store', type=str, default=None)
    argparser.add_argument('--load', action='store', type=str, default=None)

    argparser.add_argument('--save-ranks-prefix', action='store', type=str, default=None)

    args = argparser.parse_args(argv)

    import pprint
    pprint.pprint(vars(args))

    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    entity_embedding_size = predicate_embedding_size = args.embedding_size
    batch_size = args.batch_size
    nb_epochs = args.epochs
    seed = args.seed
    learning_rate = args.learning_rate
    l3_weight = args.L3
    initializer_name = args.initializer
    validate_every_epochs = args.validate_every
    input_type = args.input_type

    save_path = args.save
    load_path = args.load

    save_ranks_prefix = args.save_ranks_prefix

    # fire up eager
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.enable_eager_execution(config=config)

    # set the seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)

    data = Data(train_path, dev_path, test_path, input_type=input_type)
    model = ComplEx()
    nkb = NeuralKB(data, entity_embedding_size, predicate_embedding_size, initializer_name=initializer_name)
    saver = tfe.Saver(var_list=nkb.variables)

    if load_path is not None:
        logger.info('Loading model ..')
        saver.restore(load_path)

    # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    batcher = Batcher(data, batch_size, nb_epochs, random_state, nb_corrupted_pairs=0, is_all=False)

    validate_every_batches = None
    if validate_every_epochs is not None and nb_epochs > 0:
        validate_every_batches = int((batcher.nb_batches * validate_every_epochs) / nb_epochs)

    start_time = time.time()

    for batch_no, (batch_start, batch_end) in enumerate(batcher.batches, 1):
        xi_batch, xp_batch, xs_batch, xo_batch, target_inputs = batcher.get_batch(batch_no, batch_start, batch_end)

        # goals should be [GE, GE, GE]
        with tf.GradientTape() as tape:
            # [B, E]
            p_emb = tf.nn.embedding_lookup(nkb.predicate_embeddings, xp_batch)
            # [B, E]
            s_emb = tf.nn.embedding_lookup(nkb.entity_embeddings, xs_batch)
            # [B, E]
            o_emb = tf.nn.embedding_lookup(nkb.entity_embeddings, xo_batch)
            # [N, E]
            all_emb = nkb.entity_embeddings
            # Scalar
            loss = model.multiclass_loss(p_emb, s_emb, o_emb, all_emb)
            if l3_weight is not None:
                l3_regularizer = model.L3(p_emb) + model.L3(s_emb) + model.L3(o_emb)
                loss += l3_weight * l3_regularizer

        logger.info('Loss @ batch {} on {}: {}'.format(batch_no, batcher.nb_batches, loss))

        gradients = tape.gradient(loss, nkb.variables)
        grads_and_vars = [(grad, var) for grad, var in zip(gradients, nkb.variables)]

        optimizer.apply_gradients(grads_and_vars=grads_and_vars)

        if batch_no == 100:
            print('Training first 100 batches took {} seconds'.format(time.time() - start_time))

        if validate_every_batches is not None and batch_no % validate_every_batches == 0:
            evaluate(model, data, nkb, batch_size, is_brief=True)

    end_time = time.time()

    logger.info('Training required {} seconds'.format(end_time - start_time))

    if save_path is not None:
        logger.info('Saving model ..')
        saver.save(file_prefix=save_path)

    logger.info('Starting evaluation ..')
    evaluate(model, data, nkb, batch_size,
             save_ranks_prefix=save_ranks_prefix)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
