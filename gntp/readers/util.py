# -*- coding: utf-8 -*-

import tensorflow as tf


def fused_rnn_backward(fused_rnn, inputs, sequence_length,
                       initial_state=None, dtype=None, scope=None, time_major=True):
    if not time_major:
        inputs = tf.transpose(inputs, [1, 0, 2])
    # assumes that time dim is 0 and batch is 1
    rev_inputs = tf.reverse_sequence(inputs, sequence_length, 0, 1)
    rev_outputs, last_state = fused_rnn(rev_inputs, sequence_length=sequence_length, initial_state=initial_state,
                                        dtype=dtype, scope=scope)
    outputs = tf.reverse_sequence(rev_outputs, sequence_length, 0, 1)
    if not time_major:
        outputs = tf.transpose(outputs, [1, 0, 2])
    return outputs, last_state


def fused_birnn(fused_rnn, inputs, sequence_length,
                initial_state=(None, None), dtype=None, scope=None, time_major=False):
    with tf.variable_scope(scope or "BiRNN"):
        sequence_length = tf.cast(sequence_length, tf.int32)
        if not time_major:
            inputs = tf.transpose(inputs, [1, 0, 2])
        outputs_fw, state_fw = fused_rnn(inputs, sequence_length=sequence_length, initial_state=initial_state[0],
                                         dtype=dtype, scope="FW")
        outputs_bw, state_bw = fused_rnn_backward(fused_rnn, inputs, sequence_length, initial_state[1],
                                                  dtype, scope="BW")
        if not time_major:
            outputs_fw = tf.transpose(outputs_fw, [1, 0, 2])
            outputs_bw = tf.transpose(outputs_bw, [1, 0, 2])
    return (outputs_fw, outputs_bw), (state_fw, state_bw)


def _bi_rnn(fused_rnn, sequence, seq_length):
    output = fused_birnn(fused_rnn, sequence, seq_length, dtype=tf.float32, scope='rnn')[0]
    output = tf.concat(output, 2)
    return output


def bi_lstm(size, sequence, seq_length):
    fused_rnn = tf.contrib.rnn.LSTMBlockFusedCell(size)
    return _bi_rnn(fused_rnn, sequence, seq_length)


def convnet(repr_dim, inputs, num_layers, conv_width=3):
    output = inputs
    for i in range(num_layers):
        output = _convolutional_block(output, repr_dim, conv_width=conv_width, name="conv_%d" % i)
    return output


def _convolutional_block(inputs, out_channels, conv_width=3, name='conv', activation=tf.nn.relu):
    channels = inputs.get_shape()[2].value
    f = tf.get_variable(name + '_filter', [conv_width, channels, out_channels])
    output = tf.nn.conv1d(inputs, f, 1, padding='SAME', name=name)
    return activation(output)
