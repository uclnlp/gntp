# -*- coding: utf-8 -*-

import tensorflow as tf


class AverageReader:
    def __init__(self):
        super(AverageReader, self).__init__()

    def call(self, sequence, sequence_len):
        batch_size = sequence.get_shape()[0]
        max_len = sequence.get_shape()[1]
        embedding_size = sequence.get_shape()[2]

        mask = tf.sequence_mask(sequence_len, maxlen=max_len)
        mask = tf.reshape(mask, [batch_size, max_len, 1])
        mask = tf.tile(mask, [1, 1, embedding_size])
        mask = tf.cast(mask, dtype=sequence.dtype)
        masked_seq_emb = sequence * mask

        res = tf.reduce_sum(masked_seq_emb, axis=1)
        res = res / tf.cast(tf.expand_dims(sequence_len, axis=1), dtype=sequence.dtype)
        return res


class LSTMReader:
    def __init__(self, input_dim=100, nb_layers=1):
        super(LSTMReader, self).__init__()
        self.cells = [tf.contrib.rnn.LSTMCell(input_dim) for _ in range(nb_layers)]
        self.multicell = tf.contrib.rnn.MultiRNNCell(self.cells)

    def call(self, sequence, sequence_len):
        outputs, state = tf.nn.dynamic_rnn(cell=self.multicell,
                                           inputs=sequence,
                                           sequence_length=sequence_len,
                                           dtype=sequence.dtype)
        return state[0].c


class CNNReader:
    def __init__(self, nb_layers=1, conv_width=3, channels=100, out_channels=100,
                 activation=tf.nn.relu, pooling=tf.reduce_max):
        self.filters = []
        for i in range(nb_layers):
            self.filters += [tf.get_variable('conv_{}_filter'.format(i), [conv_width, channels, out_channels])]
        self.activation = activation
        self.pooling = pooling

    def call(self, sequence, sequence_len):
        batch_size = sequence.get_shape()[0]
        max_len = sequence.get_shape()[1]
        embedding_size = sequence.get_shape()[2]

        mask = tf.sequence_mask(sequence_len, maxlen=max_len)
        mask = tf.reshape(mask, [batch_size, max_len, 1])
        mask = tf.tile(mask, [1, 1, embedding_size])
        mask = tf.cast(mask, dtype=sequence.dtype)
        output = sequence * mask
        for filter in self.filters:
            output = self.activation(tf.nn.conv1d(output, filter, 1, padding='SAME'))
        return self.pooling(output, 1)
