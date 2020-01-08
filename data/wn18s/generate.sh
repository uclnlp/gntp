#!/usr/bin/env bash

cat ../wn18/train.tsv | grep -E "_hyponym|_hypernym|_part_of|_has_part" | grep -v instance > train.tsv
cat ../wn18/dev.tsv | grep -E "_hyponym|_hypernym|_part_of|_has_part" | grep -v instance > dev.tsv
cat ../wn18/test.tsv | grep -E "_hyponym|_hypernym|_part_of|_has_part" | grep -v instance > test.tsv

cat ../wn18/clauses.gold.pl | grep -E "_hyponym|_hypernym|_part_of|_has_part" | grep -v instance > clauses.gold.pl

cat train.tsv | grep "^1" > train.s1.tsv
cat dev.tsv | grep "^1" > dev.s1.tsv
cat test.tsv | grep "^1" > test.s1.tsv
