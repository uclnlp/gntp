#!/usr/bin/env bash

echo Generating ./countries_$1 ..

mkdir -p ./countries_$1

../tools/split.py ../../countries/countries_S1.tsv --kept countries_$1/countries_S1_kept.tsv --held-out countries_$1/countries_S1_held_out.tsv --held-out-size $1 --seed 0
../tools/split.py ../../countries/countries_S2.tsv --kept countries_$1/countries_S2_kept.tsv --held-out countries_$1/countries_S2_held_out.tsv --held-out-size $1 --seed 0
../tools/split.py ../../countries/countries_S3.tsv --kept countries_$1/countries_S3_kept.tsv --held-out countries_$1/countries_S3_held_out.tsv --held-out-size $1 --seed 0

../tools/generate-mentions-v2.py countries_$1/countries_S1_held_out.tsv -n countries -s 0 -o countries_$1/countries_S1_mentions.tsv
../tools/generate-mentions-v2.py countries_$1/countries_S2_held_out.tsv -n countries -s 0 -o countries_$1/countries_S2_mentions.tsv
../tools/generate-mentions-v2.py countries_$1/countries_S3_held_out.tsv -n countries -s 0 -o countries_$1/countries_S3_mentions.tsv

cat countries_$1/countries_S1_kept.tsv > countries_$1/countries_S1_merged.tsv
cat countries_$1/countries_S2_kept.tsv > countries_$1/countries_S2_merged.tsv
cat countries_$1/countries_S3_kept.tsv > countries_$1/countries_S3_merged.tsv

cat countries_$1/countries_S1_mentions.tsv | awk '{ print $1 "\t" $2 "\t" $3 }' >> countries_$1/countries_S1_merged.tsv
cat countries_$1/countries_S2_mentions.tsv | awk '{ print $1 "\t" $2 "\t" $3 }' >> countries_$1/countries_S2_merged.tsv
cat countries_$1/countries_S3_mentions.tsv | awk '{ print $1 "\t" $2 "\t" $3 }' >> countries_$1/countries_S3_merged.tsv
