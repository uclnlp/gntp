# Greedy Neural Theorem Provers

### WordNet (WN18RR):

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

INFO:gntp-moe-cli.py:Dev set evaluation
INFO:gntp-moe-cli.py:MRR: 0.42639193207554427
INFO:gntp-moe-cli.py:hits@1: 0.40458141067897163
INFO:gntp-moe-cli.py:hits@3: 0.4339156229400132
INFO:gntp-moe-cli.py:hits@5: 0.4505603164139749
INFO:gntp-moe-cli.py:hits@10: 0.47000659195781147

100%|██████████| 3134/3134 [7:35:46<00:00,  8.60s/it]INFO:gntp-moe-cli.py:Test set evaluation
INFO:gntp-moe-cli.py:MRR: 0.43375447755342356
INFO:gntp-moe-cli.py:hits@1: 0.4101786853860881
INFO:gntp-moe-cli.py:hits@3: 0.4419272495213784
INFO:gntp-moe-cli.py:hits@5: 0.45788130185067005
INFO:gntp-moe-cli.py:hits@10: 0.48388640714741543
```

### Freebase (FB122)

```bash
PYTHONPATH=. ./bin/gntp-moe-cli.py
    --train data/guo-fb122/fb122_triples.train
    --dev data/guo-fb122/fb122_triples.valid
    --test data/guo-fb122/fb122_triples.test
    --test-I data/guo-fb122/FB122-testI.txt
    --test-II data/guo-fb122/FB122-testII.txt
    -c data/guo-fb122/clauses.v3.pl -E ranking -e 100
    --only-rules-epochs 97 --only-rules-entities-epochs 100
    -k 80 --max-depth 1 -b 1000 --corrupted-pairs 1
    -l 0.005 --l2 1e-05 --k-max 10 --all -F 10 -R 1 -I 100 -i faiss
    --seed 0 --model-type ntp --initializer uniform --rule-type standard
    --decode --kernel rbf --unification-type joint
    --check-path data/guo-fb122/dev.256.tsv --check-interval 1000

[..]

INFO:gntp-moe-cli.py:Dev set evaluation
INFO:gntp-moe-cli.py:MRR: 0.6802735929791658
INFO:gntp-moe-cli.py:hits@1: 0.6525794684731631
INFO:gntp-moe-cli.py:hits@3: 0.6934861907243356
INFO:gntp-moe-cli.py:hits@5: 0.7109953100573215
INFO:gntp-moe-cli.py:hits@10: 0.7300677436164669

100%|██████████| 11243/11243 [9:03:54<00:00,  2.93s/it]INFO:gntp-moe-cli.py:Test set evaluation
INFO:gntp-moe-cli.py:MRR: 0.6845337172004905
INFO:gntp-moe-cli.py:hits@1: 0.6581428444365383
INFO:gntp-moe-cli.py:hits@3: 0.6970114738059237
INFO:gntp-moe-cli.py:hits@5: 0.7133327403717868
INFO:gntp-moe-cli.py:hits@10: 0.7327225829404963

100%|██████████| 5057/5057 [3:59:52<00:00,  2.70s/it]INFO:gntp-moe-cli.py:Test-I set evaluation
INFO:gntp-moe-cli.py:MRR: 0.31415683551847046
INFO:gntp-moe-cli.py:hits@1: 0.25983784852679453
INFO:gntp-moe-cli.py:hits@3: 0.33834289104211984
INFO:gntp-moe-cli.py:hits@5: 0.3734427526201305
INFO:gntp-moe-cli.py:hits@10: 0.41516709511568123

100%|██████████| 6186/6186 [5:13:56<00:00,  3.08s/it]INFO:gntp-moe-cli.py:Test-II set evaluation
INFO:gntp-moe-cli.py:MRR: 0.9873135249382692
INFO:gntp-moe-cli.py:hits@1: 0.9837536372453928
INFO:gntp-moe-cli.py:hits@3: 0.9902198512770772
INFO:gntp-moe-cli.py:hits@5: 0.99118978338183
INFO:gntp-moe-cli.py:hits@10: 0.9923213708373747
```
