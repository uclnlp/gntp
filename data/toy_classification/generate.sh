#!/usr/bin/env bash

yes | head -n 256 | awk '{ print "s" NR "\tp\to" NR "\n" "o" NR "\tq\ts" NR }' > train.tsv

yes | head -n 256 | awk '{ print "o" NR "\tq\ts" NR "\t" 1 "\n" "o" NR "\tq\ts" (NR - 1) "\t-1" }' | tail -n 128 > dev.tsv
