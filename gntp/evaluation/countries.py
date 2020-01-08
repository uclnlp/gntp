# -*- coding: utf-8 -*-

import re
import collections

import numpy as np
import sklearn

from tabulate import tabulate

BaseAtom = collections.namedtuple("Atom", ["predicate", "arguments"])


class Atom(BaseAtom):
    def __hash__(self):
        hash_value = hash(self.predicate)
        for a in self.arguments:
            hash_value *= hash(a)
        return hash_value

    def __eq__(self, other):
        return (self.predicate == other.predicate) and (self.arguments == other.arguments)


def trim(string: str) -> str:
    """
    :param string: an input string
    :return: the string without trailing whitespaces
    """
    return re.sub("\A\s+|\s+\Z", "", string)


def parse_rules(rules, delimiter="#####", rule_template=False):
    kb = []
    for rule in rules:
        if rule_template:
            splits = re.split("\A\n?([0-9]?[0-9]+)", rule)
            # fixme: should be 0 and 1 respectively
            num = int(splits[1])
            rule = splits[2]
        rule = re.sub(":-", delimiter, rule)
        rule = re.sub("\),", ")"+delimiter, rule)
        rule = [trim(x) for x in rule.split(delimiter)]
        rule = [x for x in rule if x != ""]
        if len(rule) > 0:
            atoms = []
            for atom in rule:
                splits = atom.split("(")
                predicate = splits[0]
                args = [x for x in re.split("\s?,\s?|\)", splits[1]) if x != ""]
                atoms.append(Atom(predicate, args))
            if rule_template:
                kb.append((atoms, num))
            else:
                kb.append(atoms)
    return kb


def load_from_file(path, rule_template=False):
    with open(path, "r") as f:
        text = f.readlines()
        text = [x for x in text if not x.startswith("%") and x.strip() != ""]
        text = "".join(text)
        rules = [x for x in re.split("\.\n|\.\Z", text) if x != "" and
                 x != "\n" and not x.startswith("%")]
        kb = parse_rules(rules, rule_template=rule_template)
        return kb


def evaluate_on_countries(test_set, entity_to_index, predicate_to_index,
                          scoring_function, verbose=False):
    test_countries = []
    with open("./data/countries/{}.txt".format(test_set), "r") as f:
        for line in f.readlines():
            test_countries.append(line[:-1])
    regions = []
    with open("./data/countries/regions.txt", "r") as f:
        for line in f.readlines():
            regions.append(line[:-1])

    ground_truth = load_from_file("./data/countries/countries.nl")

    country2region = {}
    for atom in ground_truth:
        atom = atom[0]
        if atom.predicate == "locatedIn":
            country, region = atom.arguments
            if region in regions:
                country2region[country] = region

    located_in_ids = [predicate_to_index['locatedIn']] * len(regions)
    region_ids = [entity_to_index[region] for region in regions]

    def predict(country):
        country_ids = [entity_to_index[country]] * len(regions)
        Xp = np.array(located_in_ids)
        Xs = np.array(country_ids)
        Xo = np.array(region_ids)

        scores = scoring_function(Xp, Xs, Xo)
        return scores

    table = []
    scores_all = []
    target_all = []

    for country in test_countries:
        known_kb = country2region[country]
        region_idx = regions.index(known_kb)
        scores = predict(country)

        table += [[country] + list(scores)]

        target = np.zeros(len(regions), np.int32)
        target[region_idx] = 1

        scores_all += list(scores)
        target_all += list(target)

    if verbose:
        print(tabulate(table, headers=["country"] + regions))

    auc_val = sklearn.metrics.average_precision_score(target_all, scores_all)

    return auc_val
