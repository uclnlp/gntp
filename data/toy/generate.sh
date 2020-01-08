yes | head -n 1024 | awk '{ print "s" NR "\tp\to" NR "\n" "o" NR "\tq\ts" NR}' > toy1.tsv
