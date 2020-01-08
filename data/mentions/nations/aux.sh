#!/usr/bin/env bash

echo Generating ./nations_$1 ..

mkdir -p ./nations_$1

../tools/split.py ../../nations/train.tsv --kept nations_$1/train_kept.tsv --held-out nations_$1/train_held_out.tsv --held-out-size $1 --seed 0
../tools/generate-mentions.py nations_$1/train_held_out.tsv -n nations -s 0 -o nations_$1/train_mentions.tsv

echo Generating ./nations_$1_simple ..

mkdir -p ./nations_$1_simple

../tools/split.py ../../nations/train.tsv --kept nations_$1_simple/train_kept.tsv --held-out nations_$1_simple/train_held_out.tsv --held-out-size $1 --seed 0
../tools/generate-mentions.py nations_$1_simple/train_held_out.tsv -n nations -s 0 -S -o nations_$1_simple/train_mentions.tsv
