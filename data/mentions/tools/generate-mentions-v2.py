#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import argparse

import gzip
import bz2

import numpy as np

import logging

logger = logging.getLogger(os.path.basename(sys.argv[0]))


def get_predicate_to_patterns(dataset_name):

    countries_mentions = {
        'neighborOf': [
            "is adjacent to",
            "borders with",
            "is butted against",
            "neighbours",
            "is a neighbor of",
            "is a neighboring country of",
            "is a neighboring state to",

            "was adjacent to",
            "borders",
            "was butted against",
            "neighbours with",
            "was a neighbor of",
            "was a neighboring country of",
            "was a neighboring state to"
        ],
        'locatedIn': [
            "is located in",
            "is situated in",
            "is placed in",
            "is positioned in",
            "is sited in",
            "is currently in",
            "can be found in",
            "is still in",
            "is localized in",
            "is present in",
            "is contained in",
            "is found in",

            "was located in",
            "was situated in",
            "was placed in",
            "was positioned in",
            "was sited in",
            "was currently in",
            "used to be found in",
            "was still in",
            "was localized in",
            "was present in",
            "was contained in",
            "was found in",
        ]
    }

    # $ cat train.tsv | awk '{ print $2 }' | sort | uniq | awk '{ print "\"" $1 "\": [ \"" $1 "\" ]," }'

    umls_mentions = {
        "adjacent_to": ["is adjacent to", "is next to"],  # neighbors, is a neighbor to
        "affects": ["affects"],  # influences, has an affect on, has an impact on
        "analyzes": ["analyzes", "is used for analyzing"],  # examines, inspects
        "assesses_effect_of": ["assesses the effect of"],  # estimates the effect of
        "associated_with": ["is associated with"],  # is related to, is connected to
        "carries_out": ["carries out"],  # executes, does
        "causes": ["causes", "is a cause of"],  # is a source of
        "co-occurs_with": ["co-occurs with", "occurs with"],  # happens together with
        "complicates": ["complicates"],
        "conceptual_part_of": ["is a conceptual part of"],
        "conceptually_related_to": ["is conceptually related to"],
        "connected_to": ["is connected to", "is related to"],
        "consists_of": ["consists of"],  # is composed of, is made of, is formed from
        "contains": ["contains"],  # comprises, incorporates
        "degree_of": ["is a degree of"],  # is an amount of
        "derivative_of": ["is a derivative of", "derives from"],
        "developmental_form_of": ["is a developmental form of"],
        "diagnoses": ["diagnoses", "is used for diagnosing"],
        "disrupts": ["disrupts"],  # messes up
        "evaluation_of": ["is an evaluation of"],  # is an assessment of, is an estimate of
        "exhibits": ["exhibits", "shows"],
        "indicates": ["indicates", "is an indicator of"],
        "ingredient_of": ["is an ingredient of"],  # is a component of
        "interacts_with": ["interacts with"],
        "interconnects": ["interconnects"],  # links to
        "isa": ["is a"],
        "issue_in": ["is an issue in", "is a problem in"],
        "location_of": ["is the location of"],
        "manages": ["manages"],  # controls
        "manifestation_of": ["is a manifestation of"],  # is a display of
        "measurement_of": ["is a measurement of"],
        "measures": ["measures", "is used for measuring"],  # quantifies
        "method_of": ["is a method of"],  # is a procedure of/for
        "occurs_in": ["occurs in"],  # happens in
        "part_of": ["is a part of", "is a component of"],
        "performs": ["performs"],
        "practices": ["practices"],
        "precedes": ["precedes", "comes before"],
        "prevents": ["prevents", "is used for preventing"],
        "process_of": ["is a process of"],
        "produces": ["produces"],  # fabricates
        "property_of": ["is a property of"],  # is a quality of, is an attribute of
        "result_of": ["is a result of", "is an outcome of"],  # is a consequence of
        "surrounds": ["surrounds"],
        "treats": ["treats"],  #
        "uses": ["uses"]  # utilizes
    }

    # descriptions from https://www.icpsr.umich.edu/icpsrweb/ICPSR/studies/5409 page 160ish onwards
    nations_mentions = {
        "accusation": ["accused", "charged", "blamed"],  # A negative charge or allegation directed at B
        "aidenemy": ["helped the enemy of", "aided the enemy of", "helped the foe of", "aided the foe of"],  # Total frequency of aid to subversive groups or enemy
        "attackembassy": ["attacked the embassy of", "assaulted the embassy of"],  # Any public demonstration by the actor's citizens directed at the object's foreign mission.
        # makes no sense to mentionise
        "blockpositionindex": ["blockpositionindex"],  # Bloc position A-B measured as absolute difference of position on following scale:
        "booktranslations": ["translated books in the language of", "translated books of"],  # The number of translations by A from a language that is the major spoken language of B
        "boycottembargo": ["boycotted or embargoed", "banned trade with", "stopped trading with"],  # The number of boycotts or embargoes.
        "commonbloc0": ["has the same bloc membership as", "has a common bloc membership with"],  # A and B have comnon bloc membership - 2 (Blocs are Coasnunist,Western and neutral.)
        "commonbloc1": ["differs in bloc membership to", "has a different bloc membership to"],  # A and B have different bloc membership - 1
        "commonbloc2": ["has the opposite bloc membership as", "has an opposite bloc membership to"],  # A and B have opposing bloc membership - 2
        "conferences": ["participared in a conference with", "conferenced with", "attended a conference with"],  # Total number of co-participation in international conferences Of three or more nations
        "dependent": ["is dependent of", "depends on", "relies on"],  # DEPENDENT A of 8 A once a colony, territory or part of homeland of 8.
        # makes no sense to mentionise
        "duration": ["duration"],  # Total duration of war, continuous military action, discrete military action and clash.
        "economicaid": ["has given economic aid to", "has aided", "economically aided"],  # The amount of economic aid that A has given to B.
        "eemigrants": ["has emigrants in", "has people who emigrated in"],  # Total number of emigrants A->B
        "embassy": ["has an embassy in", "has a diplomatic presence in"],  # Rating: embassy or legation A->B - 1, none - 0.
        # makes no sense to mentionise
        "emigrants3": ["emigrants3"],  # Total number of emigrants of A toward B, divided by A's population
        "expeldiplomats": ["expelled diplomats of", "threw out diplomats of", "evicted diplomats of"],  # The number of any expulsion of ambassadors and other diplomatic officials fmm another country, or any recalling of such officials for other than administrative reasons.
        "exportbooks": ["exported books to", "exported printed material to"],  # Total value of exports of printed matter A->B
        # makes no sense to mentionise
        "exports3": ["exports3"],  # Principal export of A to B. divided by A's GNP
        "independence": ["is independent of"],  # Independence of A and B predates 1946 or not
        "intergovorgs": ["has membership in intergovernmental organisation with", "share intergovernmental organisation membership"],  # The number of intergovernmental organizations in which both countries (L&8) have common membership
        # makes no sense to mentionise
        "intergovorgs3": ["intergovorgs3"],  # IGO A->B/COMMON MEMBERSHIP OF A
        "lostterritory": ["lost territory to", "suffered a loss of territory to", "was deprived of territory by"],  # LOST TERRITORY A->B  A has lost, and not regained, territory to 8 since 1900 or not
        "militaryactions": ["was involved in military actions with", "participated in military actions with"],  # Total number of continuous military action, discrete military action, and clash.
        "militaryalliance": ["is in a military alliance with", "is a military ally of"],  # Two countries are in alliance if a mutual defense treaty exists between them. Rating: mutual defense treaty yes - 1, no - 0.
        "negativebehavior": ["is involved in negative behavior with", "has negative behavior issues with"],  # Any acts or actions that reflect strained, tense, unfriendly, or hostile feelings or relations 6etween nations A and B.
        "negativecomm": ["is involved in negative communication with", "is in a negative communication with"],  # Total frequency of written or oral coaanunicationhy officials of a political unit such as accusation, representative protest, warning, threat, ultimatum and denunciation.
        "ngo": ["has membership in non-governmental organisation with", "share non-governmental organisation membership"],  # The number of non-governmental organizations in which both countries (A<->B) have coarnonmembership.
        # makes no sense to mentionise
        "ngoorgs3": ["ngoorgs3"],  # # IGO A->B/COMMON MEMBERSHIP OF A
        "nonviolentbehavior": ["has non-violent issues with", "shares non-violent behavior with"],  # Negative acts that reflect strained, unfriendly! or hostile feelings of some of the actor's citizens against an object of its policies.
        "officialvisits": ["officially visited", "went for an official visit to"],  # Total number of official political visits A -> B.
        "pprotests": ["protested", "protested the acts of"],  # The nutier of any official diplomatic comnunication or governmental statement by the executive leaders by a country which has as its primary purpose to protest against the actions of another nation.
        # makes no sense to mentionise
        "relbooktranslations": ["relbooktranslations"],  # The RELATIVE number of translations by A from a language that is the major spoken language of B
        # makes no sense to mentionise
        "reldiplomacy": ["reldiplomacy"],  # The nun6er of embassies or legations that A has in B, divided by total nutier of etiassies or legations that A has in all other countries.
        # makes no sense to mentionise
        "releconomicaid": ["releconomicaid"],  # The RELATIVE amount of economic aid that A has given to B.
        # makes no sense to mentionise
        "relemigrants": ["relemigrants"],  # # Total RELATIVE number of emigrants A->B
        # makes no sense to mentionise
        "relexportbooks": ["relexportbooks"],  # Total RELATIVE value of exports of printed matter A->B
        # makes no sense to mentionise
        "relexports": ["relexports"],  # Total RELATIVE value of exports f.o.b. A->B.
        # makes no sense to mentionise
        "relintergovorgs": ["relintergovorgs"],  # The RELATIVE number of intergovernmental organizations in which both countries (L&8) have common membership
        # makes no sense to mentionise
        "relngo": ["relngo"],  # # The RELATIVE number of non-governmental organizations in which both countries (A<->B) have coarnonmembership.
        # makes no sense to mentionise
        "relstudents": ["relstudents"],  # A's total RELATIVE number of students in country B.
        # makes no sense to mentionise
        "reltourism": ["reltourism"],  # # Total RELATIVE number of tourists A -> B.
        # makes no sense to mentionise
        "reltreaties": ["reltreaties"],  # Total RELATIVE number of bilateral and multilateral treaties and agreements signed between A and B
        "severdiplomatic": ["severed diplomatic ties with", "stopped diplomatic relations with"],  # The interruption of formal diplomatic relations by A with B
        "students": ["has students in", "has people studying in"],  # A's total number of students in country B.
        "timesinceally": ["was an ally to", "was on the same side of war as", "allied with"],  # TIME SINCE ON SAME SIDES OF WAR A<->B
        "timesincewar": ["was an enemy to", "was on the opposite side of war as", "waged war with"],  # TIME SINCE OPPOSITE SIDES OF A WAR A<->B
        "tourism": ["has tourists in", "has people touristically visiting"],  # Total number of tourists A -> B.
        # makes no sense to mentionise
        "tourism3": ["tourism3"],  # Total number of tourists A -> B, divided by A's population
        "treaties": ["has treaties with", "has agreements with"],  # Total number of bilateral and multilateral treaties and agreements signed between A and B
        "unoffialacts": ["has acts of unofficial violence to", "unofficially shows violence to"],  # Total frequency of unofficial violence.
        # makes no sense to mentionise
        "unweightedunvote": ["unweightedunvote"],  # Factor score distance between Acrs on major rotated dimensions extracted from roll call voting statistics in the UN General Assembly Plenary Sessions.
        "violentactions": ["was involved in violent actions with", "participated in violent actions with"],  # Violent actions comprise war, continuous military action, discrete military action, or clash.
        "warning": ["warned", "was warning"],  # These are distinctively military in nature, but are nonviolent.
        # makes no sense to mentionise
        "weightedunvote": ["weightedunvote"],  # The only difference from Variahle 32 is that here the distances l on issue dimensions are weighted by their percentage contribution to the total variance.
        # MISSING:
        # WAR - Any military action for a particular country in which the number of its soldiers involved equal or exceed.Q2 percent of its population
        # EXPORTS - Total value of exports f.o.b. A->B.
    }

    d = {
        'countries': countries_mentions,
        'umls': umls_mentions,
        'nations': nations_mentions
    }

    if dataset_name not in d:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))

    return d[dataset_name]


def iopen(file, *args, **kwargs):
    _open = open
    if file.endswith('.gz'):
        _open = gzip.open
    elif file.endswith('.bz2'):
        _open = bz2.open
    return _open(file, *args, **kwargs)


def read_triples(path):
    triples = []
    with iopen(path, 'rt') as f:
        for line in f.readlines():
            s, p, o = line.split()
            triples += [(s.strip(), p.strip(), o.strip())]
    return triples


def main(argv):
    argparser = argparse.ArgumentParser('Mentions generator',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    argparser.add_argument('triples', type=str, default='/dev/stdin')

    argparser.add_argument('--output', '-o', type=argparse.FileType('w'), default='/dev/stdout')

    argparser.add_argument('--dataset-name', '--name', '-n', type=str, default='countries')
    argparser.add_argument('--simple', '-S', action='store_true')
    argparser.add_argument('--seed', '-s', type=int, default=0)

    args = argparser.parse_args(argv)

    triples_path = args.triples
    out_fd = args.output

    dataset_name = args.dataset_name
    is_simple = args.simple

    seed = args.seed

    triples = read_triples(triples_path)
    predicate_to_patterns = get_predicate_to_patterns(dataset_name)

    rs = np.random.RandomState(seed)

    mentions = []

    for s, p, o in triples:
        if p not in predicate_to_patterns:
            raise ValueError('Unknown predicate {}'.format(p))

        patterns = predicate_to_patterns[p]
        assert len(patterns) > 0

        pattern = patterns[0]
        if not is_simple:
            nb_patterns = len(patterns)
            pattern_idx = rs.randint(nb_patterns)
            pattern = patterns[pattern_idx]

        pattern = '[XXX]:{}:[YYY]'.format(pattern.replace(' ', ':'))
        mention = (s, pattern, o)

        mentions += [mention]

    out_fd.writelines(['\t'.join(symbol for symbol in mention) + '\t1\n' for mention in mentions])


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main(sys.argv[1:])
