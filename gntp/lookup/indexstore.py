# -*- coding: utf-8 -*-

import tensorflow as tf

import gntp

from typing import List, Union, Dict, Any

import logging

logger = logging.getLogger(__name__)


class LookupIndexStore:
    def __init__(self,
                 index_params: Dict[str, Any],
                 index_type: str = 'nmslib'):
        self.index_type = index_type
        self.index_class = gntp.lookup.get_index_class_by_name(self.index_type)
        self.index_params = index_params
        self.store = {}

    # atoms is e.g. [KE, KE, KE], goals is e.g. [GE, X, GE]
    def get_or_create(self,
                      atoms: List[Union[tf.Tensor, str]],
                      goals: List[Union[tf.Tensor, str]],
                      index_refresh_rate: int,
                      position: int,
                      is_training: bool = True):

        # First create an hash key corresponding to the structure of facts and goals
        def to_key(ae, ge):
            return 'T' if gntp.is_tensor(ae) and gntp.is_tensor(ge) else 'V'

        # In this case it is TVT
        key = '{}-{}'.format(position, ''.join([to_key(ae, ge) for ae, ge in zip(atoms, goals)]))

        if key not in self.store:
            if self.index_type == 'symbol':
                index = self.index_class(**self.index_params)
            else:
                ground_atoms = [ae for ae, ge in zip(atoms, goals) if gntp.is_tensor(ae) and gntp.is_tensor(ge)]
                atom_2d = tf.concat(ground_atoms, axis=1)
                index = self.index_class(data=atom_2d.numpy(), **self.index_params)

            self.store[key] = index

        index = self.store[key]

        if index_refresh_rate is not None:
            if index.times_queried > index_refresh_rate and is_training is True:
                if self.index_type == 'symbol':
                    index = self.index_class(**self.index_params)
                else:
                    ground_atoms = [ae for ae, ge in zip(atoms, goals) if gntp.is_tensor(ae) and gntp.is_tensor(ge)]
                    atom_2d = tf.concat(ground_atoms, axis=1)
                    index = self.index_class(data=atom_2d.numpy(), **self.index_params)

                self.store[key] = index

        return index
