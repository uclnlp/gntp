#!/usr/bin/env bash

echo Generating ./umls_$1 ..

mkdir -p ./umls_$1

../tools/split.py ../../umls/train.tsv --kept umls_$1/train_kept.tsv --held-out umls_$1/train_held_out.tsv --held-out-size $1 --seed 0
../tools/generate-mentions.py umls_$1/train_held_out.tsv -n umls -s 0 -o umls_$1/train_mentions.tsv

echo Generating ./umls_$1_simple ..

mkdir -p ./umls_$1_simple

../tools/split.py ../../umls/train.tsv --kept umls_$1_simple/train_kept.tsv --held-out umls_$1_simple/train_held_out.tsv --held-out-size $1 --seed 0
../tools/generate-mentions.py umls_$1_simple/train_held_out.tsv -n umls -s 0 -S -o umls_$1_simple/train_mentions.tsv
