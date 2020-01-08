yes | head -n 128 | awk '{ print "s" NR "\tp\to" NR}' > toy1.tsv
yes | head -n 128 | awk '{ print "s" NR "\tthe:p:pattern\to" NR "\t1"}' > toy1_mentions.txt
