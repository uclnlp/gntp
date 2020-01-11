# Greedy Neural Theorem Provers

WordNet (WN18RR):

```
PYTHONPATH=. python3 ./bin/gntp-moe-cli.py
    --train data/wn18rr/train.tsv 
    --dev data/wn18rr/dev.tsv
    --test data/wn18rr/test.tsv
    -c data/wn18rr/clauses.v1.pl -E ranking --max-depth 1 -b 1000 --corrupted-pairs 1
    -l 0.005 --l2 0.001 --k-max 10 --all -F 5 -R 1 -I 100 --seed 0 --model-type ntp
    --initializer uniform --rule-type standard --decode --kernel rbf --unification-type joint
    -i faiss -e 100 --auxiliary-epochs 0 --test-batch-size 10000
    --only-rules-epochs 95 --only-rules-entities-epochs 0 --input-type standard
    --train-slope --check-path data/wn18rr/dev.256.tsv --check-interval 1000
[..]
INFO:ntp2-moe-cli.py:Dev set evaluation
INFO:ntp2-moe-cli.py:MRR: 0.42639193207554427
INFO:ntp2-moe-cli.py:hits@1: 0.40458141067897163
INFO:ntp2-moe-cli.py:hits@3: 0.4339156229400132
INFO:ntp2-moe-cli.py:hits@5: 0.4505603164139749
INFO:ntp2-moe-cli.py:hits@10: 0.47000659195781147


100%|██████████| 3134/3134 [7:35:46<00:00,  8.60s/it]INFO:ntp2-moe-cli.py:Test set evaluation
INFO:ntp2-moe-cli.py:MRR: 0.43375447755342356
INFO:ntp2-moe-cli.py:hits@1: 0.4101786853860881
INFO:ntp2-moe-cli.py:hits@3: 0.4419272495213784
INFO:ntp2-moe-cli.py:hits@5: 0.45788130185067005
INFO:ntp2-moe-cli.py:hits@10: 0.48388640714741543
```
