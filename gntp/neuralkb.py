# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow.contrib.eager as tfe

import numpy as np

from typing import Optional

import gntp

from gntp.training.data import Data

layers = tf.keras.layers


class NeuralKB:
    def __init__(self,
                 data: Data,
                 entity_embedding_size: int,
                 predicate_embedding_size: int,
                 symbol_embedding_size: Optional[int] = None,
                 model_type: Optional[str] = None,
                 initializer_name: Optional[str] = None,
                 rule_embeddings_type: str = 'standard',
                 use_concrete: bool = False):
        self.neural_facts_kb = None
        self.neural_rules_kb = None
        self.relation_embeddings = None
        self.model_type = model_type

        self.reader = gntp.readers.AverageReader()
        self.rule_embeddings_type = rule_embeddings_type
        self.data = data
        self.temp = 2.0
        self.use_concrete = use_concrete

        uniform_initializer = tf.random_uniform_initializer(-1.0, 1.0)
        xavier_initializer = tf.contrib.layers.xavier_initializer()

        ntp_initializer = uniform_initializer if initializer_name == 'uniform' else xavier_initializer

        def variable(shape, name, initializer=ntp_initializer):
            return tfe.Variable(initializer(shape), name=name) if shape[0] > 0 else None

        self.entity_embeddings = variable([data.nb_entities, entity_embedding_size], 'entity_embeddings')
        self.predicate_embeddings = variable([data.nb_predicates, predicate_embedding_size], 'predicate_embeddings')
        self.symbol_embeddings = variable([data.nb_symbols, symbol_embedding_size], 'symbol_embeddings')

        self.variables = [self.entity_embeddings, self.predicate_embeddings]

        if self.symbol_embeddings is not None:
            self.variables += [self.symbol_embeddings]

        self.aux_entity_embeddings = self.aux_predicate_embeddings = None
        if self.model_type == 'moe3':
            self.aux_entity_embeddings = variable(shape=[data.nb_entities, entity_embedding_size],
                                                  name='aux_entity_embeddings',
                                                  initializer=xavier_initializer)

            self.aux_predicate_embeddings = variable(shape=[data.nb_predicates, predicate_embedding_size],
                                                     name='aux_predicate_embeddings',
                                                     initializer=xavier_initializer)

            self.variables += [self.aux_entity_embeddings, self.aux_predicate_embeddings]

        # aux_proj is a projection used for projecting entity and predicate embeddings
        # before feeding them to a Neural Link Predictor

        self.facts_kb = [data.Xp, data.Xs, data.Xo]
        self.rules_kb = []

        name_no = 0

        self.rule_variables = []

        for clause_idx, clause in enumerate(data.clauses):
            self.rule_kb, clause_weight = [], int(clause.weight)

            new_predicate_name_to_var = dict()
            for atom in [clause.head] + list(clause.body):
                predicate_name = atom.predicate.name
                arg1_name = '{}{}'.format(atom.arguments[0].name, clause_idx)
                arg2_name = '{}{}'.format(atom.arguments[1].name, clause_idx)

                if predicate_name in data.predicate_to_idx:
                    predicate_var = [data.predicate_to_idx[predicate_name]] * clause_weight
                else:
                    if self.rule_embeddings_type in {'standard'}:
                        if predicate_name not in new_predicate_name_to_var:
                            predicate_var = variable(shape=[clause_weight, predicate_embedding_size],
                                                     name='predicate_{}'.format(name_no))

                            new_predicate_name_to_var[predicate_name] = predicate_var
                            name_no += 1
                        else:
                            predicate_var = new_predicate_name_to_var[predicate_name]

                    elif self.rule_embeddings_type in {'attention', 'sparse-attention'}:
                        if predicate_name not in new_predicate_name_to_var:
                            predicate_var = variable(shape=[clause_weight, data.nb_relations],
                                                     name='predicate_{}'.format(name_no),
                                                     initializer=xavier_initializer)

                            new_predicate_name_to_var[predicate_name] = predicate_var
                            name_no += 1
                        else:
                            predicate_var = new_predicate_name_to_var[predicate_name]
                    else:
                        raise ValueError('Unknown rule_embeddings_type {}'.format(self.rule_embeddings_type))

                self.rule_kb += [[predicate_var, arg1_name, arg2_name]]

            self.rule_variables += [variable for name, variable in new_predicate_name_to_var.items()]
            self.rules_kb += [self.rule_kb]

        self.variables += self.rule_variables

    def get_trainable_variables(self,
                                is_rules_only: bool = False,
                                is_facts_only: bool = False,
                                is_entities_only: bool = False,
                                is_rules_entities_only: bool = False):
        variables = []

        if is_rules_only is False:
            variables += [self.entity_embeddings, self.predicate_embeddings]
            if self.symbol_embeddings is not None:
                variables += [self.symbol_embeddings]
            if self.model_type == 'moe3':
                variables += [self.aux_entity_embeddings, self.aux_predicate_embeddings]

        variables += self.rule_variables

        if is_facts_only:
            variables = [self.entity_embeddings, self.predicate_embeddings]

        if is_entities_only:
            variables = [self.entity_embeddings]

        if is_rules_entities_only:
            variables = [self.entity_embeddings] + self.rule_variables

        return variables

    def create_neural_kb(self, is_epoch_start=False, training=False):
        if self.use_concrete:
            if is_epoch_start:
                self.temp /= 1.05
                if self.temp < 0.5:
                    self.temp = 0.5
                # print('TEMP:', self.temp)
        self.relation_embeddings = self.predicate_embeddings

        if len(self.data.pattern_id_to_symbol_ids) > 0:
            self.relation_embeddings = self.create_relation_embeddings()

        neural_facts_kb = [
            tf.nn.embedding_lookup(self.relation_embeddings, self.facts_kb[0]),
            tf.nn.embedding_lookup(self.entity_embeddings, self.facts_kb[1]),
            tf.nn.embedding_lookup(self.entity_embeddings, self.facts_kb[2]),
        ]

        neural_rules_kb = []
        for rule in self.rules_kb:
            rule_graph = []
            for atom in rule:
                atom_graph = []
                for term in atom:
                    term_is_indices = isinstance(term, list) or isinstance(term, np.ndarray)
                    term_is_tensor = gntp.is_tensor(term)

                    term_graph = term
                    if term_is_indices is True:
                        term_graph = tf.nn.embedding_lookup(self.relation_embeddings, term)

                    elif term_is_tensor:
                        # if term_graph is an un-normalised attention mask and it is the predicate of the atom:
                        if self.rule_embeddings_type in {'attention'} and len(atom_graph) == 0:
                            if not self.use_concrete:
                                attention_mask = tf.nn.softmax(term_graph)
                            else:
                                if not training:
                                    attention_mask = tf.nn.softmax(term_graph/self.temp)
                                else:
                                    attention_mask = tf.contrib.distributions.RelaxedOneHotCategorical(
                                        self.temp, logits=term_graph).sample()

                            term_graph = tf.einsum('cr,re->ce', attention_mask, self.relation_embeddings)

                        if self.rule_embeddings_type in {'sparse-attention'} and len(atom_graph) == 0:
                            attention_mask = gntp.sparse_softmax(term_graph)
                            term_graph = tf.einsum('cr,re->ce', attention_mask, self.relation_embeddings)

                    atom_graph += [term_graph]
                rule_graph += [atom_graph]
            neural_rules_kb += [rule_graph]

        self.neural_facts_kb = neural_facts_kb
        self.neural_rules_kb = neural_rules_kb

        return neural_facts_kb, neural_rules_kb

    def create_relation_embeddings(self):
        symbol_seq_embeddings = tf.nn.embedding_lookup(self.symbol_embeddings, self.data.np_symbol_ids)
        pattern_embeddings = self.reader.call(symbol_seq_embeddings, self.data.np_symbol_ids_len)
        relation_embeddings = tf.concat([self.predicate_embeddings, pattern_embeddings], axis=0)
        return relation_embeddings
