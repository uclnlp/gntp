#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import numpy as np

from nltk.corpus import wordnet

import logging


logger = logging.getLogger(os.path.basename(sys.argv[0]))
np.set_printoptions(linewidth=48, precision=5, suppress=True)


def main(argv):
    argparser = argparse.ArgumentParser('Decoder', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('path', action='store', type=str)

    args = argparser.parse_args(argv)

    syns = list(wordnet.all_synsets())
    offsets_list = [(s.offset(), s) for s in syns]
    offsets_dict = dict(offsets_list)

    explanations_filename = args.path  # 'explanations-models_wn18_-dev.raw.txt'

    rules = None
    buffer = []
    with open(explanations_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '--' * 50:
                rules = buffer[:]
                buffer = []
            else:
                buffer.append(line)
    explanations = buffer

    data = []

    def parse_line(line):
        data = (line.split('\t')[0], line.split('\t')[1], eval(line.split('\t')[2]))
        return data

    for line in explanations:
        data.append(parse_line(line))

    def parse_fact(line):
        return [line[0:line.index('(')]] + line[line.index('(')+1:-1].split(', ')

    def get_synset(elem):
        return offsets_dict[elem].name()

    def decode_fact(line):
        arg0 = parse_fact(line)[0]
        arg1 = get_synset(int(parse_fact(line)[1]))
        arg2 = get_synset(int(parse_fact(line)[2]))
        # res = arg0, arg1, arg2
        res = arg0, '{}/{}'.format(arg1, (parse_fact(line)[1])), '{}/{}'.format(arg2, (parse_fact(line)[2]))
        return res

    def pretty_decoded_fact(fact):
        return "{}({}, {})".format(fact[0], fact[1], fact[2])

    pretty_sorted_data = []

    for datum in data:
        goal = datum[0]
        one_point = dict()
        one_point['score'] = datum[2][0][0]
        one_point['fact'] = pretty_decoded_fact(decode_fact(goal))
        first_three = []
        for proof in datum[2]:
            path = proof[1]
            pretty_proof = ['\t', str(proof[0])]

            for i, elem in enumerate(path):
                if '[' not in elem:
                    stuff = pretty_decoded_fact(decode_fact(elem))
                else:
                    stuff = elem
                pretty_proof.append(stuff)
            first_three.append(' '.join(pretty_proof))
        one_point['first_three'] = first_three
        pretty_sorted_data.append(one_point)

    pretty_sorted_data = sorted(pretty_sorted_data, key=lambda k: -k['score'])

    # readable printout
    with open(explanations_filename + '.output', 'w') as fw:

        for rule in rules:
            fw.writelines(rule + '\n')

        fw.write('--' * 50 + '\n')

        for datum in pretty_sorted_data:
            fw.write('{0} - {1}\n'.format(datum['score'], datum['fact']))

            for elem in datum['first_three']:
                fw.write(elem + '\n')

    for rule in rules:
        print(rule)

    print('--' * 50)

    for datum in pretty_sorted_data:
        print('{0} - {1}'.format(datum['score'], datum['fact']))

        for elem in datum['first_three']:
            print(elem)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
