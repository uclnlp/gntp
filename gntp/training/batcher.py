# -*- coding: utf-8 -*-

import numpy as np

import gntp
from gntp.training.data import Data

import math


class Batcher:
    def __init__(self,
                 data: Data,
                 batch_size: int,
                 nb_epochs: int,
                 random_state: np.random.RandomState,
                 nb_corrupted_pairs: int,
                 is_all: bool = True,
                 nb_aux_epochs: int = 0,
                 epoch_based_batches: bool = False):
        self.data = data
        self.batch_size = batch_size
        self.nb_corrupted_pairs = nb_corrupted_pairs
        self.is_all = is_all
        self.random_state = random_state
        self.is_pretraining = None

        nb_total_epochs = nb_epochs + nb_aux_epochs
        size = nb_total_epochs * data.nb_examples
        self.curriculum_Xi = np.zeros(size, dtype=np.int32)
        self.curriculum_Xs = np.zeros(size, dtype=np.int32)
        self.curriculum_Xp = np.zeros(size, dtype=np.int32)
        self.curriculum_Xo = np.zeros(size, dtype=np.int32)

        for epoch_no in range(nb_total_epochs):
            curriculum_order = self.random_state.permutation(data.nb_examples)
            start = epoch_no * data.nb_examples
            end = (epoch_no + 1) * data.nb_examples
            self.curriculum_Xi[start: end] = data.Xi[curriculum_order]
            self.curriculum_Xs[start: end] = data.Xs[curriculum_order]
            self.curriculum_Xp[start: end] = data.Xp[curriculum_order]
            self.curriculum_Xo[start: end] = data.Xo[curriculum_order]

        if epoch_based_batches:
            self.batches = gntp.make_epoch_batches(self.curriculum_Xi.shape[0], batch_size, data.nb_examples)
        else:
            self.batches = gntp.make_batches(self.curriculum_Xi.shape[0], batch_size)

        self.nb_batches = len(self.batches)

        nb_batches_per_epoch = self.nb_batches / nb_total_epochs if nb_total_epochs > 0 else 0
        self.nb_aux_batches = math.ceil(nb_aux_epochs * nb_batches_per_epoch)

        self.entity_indices = np.array(sorted({data.entity_to_idx[entity] for entity in data.entity_set}))

    def get_batch(self, batch_no, batch_start, batch_end):
        current_batch_size = batch_end - batch_start

        # Let's keep the batches like this:
        # Positive, Negative, Negative, Positive, Negative, Negative, ..
        nb_negatives = self.nb_corrupted_pairs * 2 * (2 if self.is_all else 1)
        nb_triple_variants = 1 + nb_negatives

        xi_batch = np.zeros(current_batch_size * nb_triple_variants, dtype=self.curriculum_Xi.dtype)
        xs_batch = np.zeros(current_batch_size * nb_triple_variants, dtype=self.curriculum_Xs.dtype)
        xp_batch = np.zeros(current_batch_size * nb_triple_variants, dtype=self.curriculum_Xp.dtype)
        xo_batch = np.zeros(current_batch_size * nb_triple_variants, dtype=self.curriculum_Xo.dtype)

        # Indexes of positive examples in the Neural KB
        xi_batch[0::nb_triple_variants] = self.curriculum_Xi[batch_start:batch_end]

        # Positive examples
        xs_batch[0::nb_triple_variants] = self.curriculum_Xs[batch_start:batch_end]
        xp_batch[0::nb_triple_variants] = self.curriculum_Xp[batch_start:batch_end]
        xo_batch[0::nb_triple_variants] = self.curriculum_Xo[batch_start:batch_end]

        def corrupt(**kwargs):
            res = gntp.corrupt_triples(self.random_state,
                                       self.curriculum_Xs[batch_start:batch_end],
                                       self.curriculum_Xp[batch_start:batch_end],
                                       self.curriculum_Xo[batch_start:batch_end],
                                       self.data.Xs, self.data.Xp, self.data.Xo, self.entity_indices, **kwargs)
            return res

        for c_index in range(0, self.nb_corrupted_pairs):
            c_i = c_index * 2 * (2 if self.is_all else 1) + 1

            # Those negative examples do not have an index in the Neural KB
            xi_batch[c_i::nb_triple_variants] = -1

            # Let's corrupt the subject of the triples
            xs_corr, xp_corr, xo_corr = corrupt(corrupt_subject=True)
            xs_batch[c_i::nb_triple_variants] = xs_corr
            xp_batch[c_i::nb_triple_variants] = xp_corr
            xo_batch[c_i::nb_triple_variants] = xo_corr

            # Those negative examples do not have an index in the Neural KB
            xi_batch[c_i + 1::nb_triple_variants] = -1

            # Let's corrupt the object of the triples
            xs_corr, xp_corr, xo_corr = corrupt(corrupt_object=True)
            xs_batch[c_i + 1::nb_triple_variants] = xs_corr
            xp_batch[c_i + 1::nb_triple_variants] = xp_corr
            xo_batch[c_i + 1::nb_triple_variants] = xo_corr

            if self.is_all:
                # Those negative examples do not have an index in the Neural KB
                xi_batch[c_i + 2::nb_triple_variants] = -1

                # Let's corrupt the subject of the triples
                xs_corr, xp_corr, xo_corr = corrupt(corrupt_subject=True, corrupt_object=True)
                xs_batch[c_i + 2::nb_triple_variants] = xs_corr
                xp_batch[c_i + 2::nb_triple_variants] = xp_corr
                xo_batch[c_i + 2::nb_triple_variants] = xo_corr

                # Those negative examples do not have an index in the Neural KB
                xi_batch[c_i + 3::nb_triple_variants] = -1

                # Let's corrupt the object of the triples
                xs_corr, xp_corr, xo_corr = corrupt(corrupt_subject=True, corrupt_object=True)
                xs_batch[c_i + 3::nb_triple_variants] = xs_corr
                xp_batch[c_i + 3::nb_triple_variants] = xp_corr
                xo_batch[c_i + 3::nb_triple_variants] = xo_corr

        target_inputs = np.array(([1] + ([0] * nb_negatives)) * current_batch_size, dtype='float32')

        self.is_pretraining = batch_no < self.nb_aux_batches

        return xi_batch, xp_batch, xs_batch, xo_batch, target_inputs
