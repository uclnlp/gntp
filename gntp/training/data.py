# -*- coding: utf-8 -*-

import gntp
import numpy as np

from typing import List, Optional, Tuple

from gntp.parse.clauses import Clause


class Data:
    def __init__(self,
                 train_path: str,
                 dev_path: Optional[str] = None,
                 test_path: Optional[str] = None,
                 clauses: Optional[List[Clause]] = None,
                 evaluation_mode: str = 'ranking',
                 mentions: Optional[List[Tuple[str, str, str]]] = None,
                 _all_path: Optional[str] = None,
                 test_I_path: Optional[str] = None,
                 test_II_path: Optional[str] = None,
                 input_type: str = 'standard'):

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path

        self.test_I_path = test_I_path
        self.test_II_path = test_II_path

        self.input_type = input_type
        assert self.input_type in {'standard', 'reciprocal'}

        self.clauses = clauses if clauses else []

        self.mentions = mentions if mentions else []
        self.evaluation_mode = evaluation_mode if evaluation_mode else 'ranking'

        self.Xi = self.Xs = self.Xp = self.Xo = None

        # Loading the dataset
        self.train_triples = gntp.read_triples(self.train_path)

        self.reciprocal_train_triples = None
        if self.input_type in {'reciprocal'}:
            self.reciprocal_train_triples = [(o, 'inverse_{}'.format(p), s) for (s, p, o) in self.train_triples]
            self.train_triples += self.reciprocal_train_triples

        self.dev_triples, self.dev_labels = [], None
        self.test_triples, self.test_labels = [], None

        _all_triples = None

        if self.evaluation_mode not in {'ntn'}:
            self.dev_triples = gntp.read_triples(self.dev_path) if self.dev_path else []
            self.test_triples = gntp.read_triples(self.test_path) if self.test_path else []

            self.test_I_triples = gntp.read_triples(self.test_I_path) if self.test_I_path else []
            self.test_II_triples = gntp.read_triples(self.test_II_path) if self.test_II_path else []

            _all_triples = gntp.read_triples(_all_path) if _all_path else None
        else:
            dev_labeled_triples = gntp.read_labeled_triples(self.dev_path) if self.dev_path else []
            test_labeled_triples = gntp.read_labeled_triples(self.test_path) if self.test_path else []

            self.dev_triples = [triple for (triple, label) in dev_labeled_triples]
            self.test_triples = [triple for (triple, label) in test_labeled_triples]

            self.dev_labels = [label for (triple, label) in dev_labeled_triples]
            self.test_labels = [label for (triple, label) in test_labeled_triples]

        self.all_triples = self.train_triples + self.dev_triples + self.test_triples

        if _all_triples is not None:
            self.all_triples += _all_triples
            self.all_triples = sorted(set(self.all_triples))

        self.entity_set = {s for (s, _, _) in self.all_triples} | {o for (_, _, o) in self.all_triples}
        self.entity_set |= {s for (s, _, _) in self.mentions} | {o for (_, _, o) in self.mentions}

        self.predicate_set = {p for (_, p, _) in self.all_triples}

        self.pattern_set = {pattern for (_, pattern, _) in self.mentions}
        self.symbol_set = {symbol for pattern in self.pattern_set for symbol in pattern.split(':')}

        self.nb_examples = len(self.train_triples)
        self.nb_mentions = len(self.mentions)

        self.entity_to_idx = {entity: idx for idx, entity in enumerate(sorted(self.entity_set))}
        self.nb_entities = max(self.entity_to_idx.values()) + 1
        self.idx_to_entity = {v: k for k, v in self.entity_to_idx.items()}

        self.predicate_to_idx = {predicate: idx for idx, predicate in enumerate(sorted(self.predicate_set))}
        self.nb_predicates = max(self.predicate_to_idx.values()) + 1
        self.idx_to_predicate = {v: k for k, v in self.predicate_to_idx.items()}

        self.pattern_to_idx = {pattern: idx for idx, pattern in enumerate(sorted(self.pattern_set),
                                                                          start=self.nb_predicates)}
        self.idx_to_pattern = {v: k for k, v in self.pattern_to_idx.items()}

        self.relation_to_idx = {**self.predicate_to_idx, **self.pattern_to_idx}
        self.nb_relations = max(self.relation_to_idx.values()) + 1
        self.idx_to_relation = {v: k for k, v in self.relation_to_idx.items()}

        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(sorted(self.symbol_set))}
        self.nb_symbols = max(self.symbol_to_idx.values()) + 1 if len(self.symbol_set) > 0 else 0
        self.idx_to_symbol = {v: k for k, v in self.symbol_to_idx.items()}

        self.pattern_id_to_symbol_ids = {
            pattern_id: [self.symbol_to_idx[symbol] for symbol in pattern.split(':')]
            for pattern, pattern_id in self.pattern_to_idx.items()
        }

        # Triples
        tri_xs, tri_xp, tri_xo = gntp.triples_to_vectors(self.train_triples, self.entity_to_idx, self.predicate_to_idx)
        men_xs, men_xp, men_xo = gntp.triples_to_vectors(self.mentions, self.entity_to_idx, self.pattern_to_idx)

        # Triple and mention indices
        self.Xi = np.arange(start=0, stop=self.nb_examples + self.nb_mentions, dtype=np.int32)

        self.Xs = np.concatenate((tri_xs, men_xs), axis=0)
        self.Xp = np.concatenate((tri_xp, men_xp), axis=0)
        self.Xo = np.concatenate((tri_xo, men_xo), axis=0)

        assert self.Xi.shape == self.Xs.shape == self.Xp.shape == self.Xo.shape

        if len(self.pattern_id_to_symbol_ids) > 0:
            symbol_ids_lst = [s_ids for _, s_ids in sorted(self.pattern_id_to_symbol_ids.items(), key=lambda kv: kv[0])]
            symbol_ids_len_lst = [len(s_ids) for s_ids in symbol_ids_lst]

            self.np_symbol_ids = gntp.pad_sequences(symbol_ids_lst)
            self.np_symbol_ids_len = np.array(symbol_ids_len_lst, dtype=np.int32)

        return
