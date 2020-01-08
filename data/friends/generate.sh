#!/usr/bin/env bash

yes A | head -n 128 | awk '{print $0NR-1}' > A_members.tsv
yes B | head -n 128 | awk '{print $0NR-1}' > B_members.tsv

for i in `cat A_members.tsv | tr "\n" " "`; do for j in `cat A_members.tsv | tr "\n" " "`; do echo $i $j; done; done > A_pairs.tsv
for i in `cat B_members.tsv | tr "\n" " "`; do for j in `cat B_members.tsv | tr "\n" " "`; do echo $i $j; done; done > B_pairs.tsv

cat A_pairs.tsv | gshuf | head -n 8192 | awk '{ print $1 "\tfriendOf\t" $2 }' > A_graph.tsv
cat B_pairs.tsv | gshuf | head -n 8192 | awk '{ print $1 "\tfriendOf\t" $2 }' > B_graph.tsv

cat A_graph.tsv B_graph.tsv > graph.tsv