cat ../../../data/countries/countries_S1.nl | tr "(" "\t" | tr ")" "\t" | tr "," "\t" | tr -d "." | awk '{ print $2 "\t" $1 "\t" $3 }' > countries_S1.tsv
cat ../../../data/countries/countries_S2.nl | tr "(" "\t" | tr ")" "\t" | tr "," "\t" | tr -d "." | awk '{ print $2 "\t" $1 "\t" $3 }' > countries_S2.tsv
cat ../../../data/countries/countries_S3.nl | tr "(" "\t" | tr ")" "\t" | tr "," "\t" | tr -d "." | awk '{ print $2 "\t" $1 "\t" $3 }' > countries_S3.tsv
