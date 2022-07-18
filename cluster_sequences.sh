#!/bin/bash

# create fasta files
echo "create fasta files"
awk -F "\t" '{id=NR-1; print ">train_"id"_score_"$2"\n"$1}' data/train_sequences.txt > data/train.fasta
awk -F "\t" '{id=NR-1; l=length($1); s=substr($1, 18, l-29); print ">train_"id"_score_"$2"\n"s}' data/train_sequences.txt > data/train.stripped.fasta
awk -F "\t" '{l=length($1); s=substr($1, 18, l-29); print s"\t"$2}' data/train_sequences.txt > data/train_sequences.stripped.txt
awk -F "\t" '{id=NR-1; print ">test_"id"\n"$1}' data/test_sequences.txt > data/test.fasta
awk -F "\t" '{id=NR-1; l=length($1); s=substr($1, 18, l-29); print ">test_"id"\n"s}' data/test_sequences.txt > data/test.stripped.fasta
awk -F "\t" '{l=length($1); s=substr($1, 18, l-29); print s}' data/test_sequences.txt > data/test_sequences.stripped.txt

# create a mini data
echo "create a mini data"
head -n 1024 data/train_sequences.txt > data/train_sequences_mini.txt
head -n 256 data/test_sequences.txt > data/test_sequences_mini.txt

# cluster at 30% identity
echo "cluster training sequences at 30% identity"
mkdir -p data/cluster
mmseqs easy-cluster data/train.fasta data/cluster/train.cluster30 data/tmp1 --min-seq-id 0.3 --cov-mode 1 -c 0.8 --threads 8 >/dev/null 2>&1
mmseqs easy-cluster data/train.stripped.fasta data/cluster/train.stripped.cluster30 data/tmp2 --min-seq-id 0.3 --cov-mode 1 -c 0.8 --threads 8 >/dev/null 2>&1

# output result
echo "For full length sequences"
echo $(cut -f 1 data/cluster/train.cluster30_cluster.tsv | uniq | wc -l)
echo $(wc -l data/cluster/train.cluster30_cluster.tsv)

echo "For stripped sequences"
echo $(cut -f 1 data/cluster/train.stripped.cluster30_cluster.tsv | uniq | wc -l)
echo $(wc -l data/cluster/train.stripped.cluster30_cluster.tsv)

# write cluster id back
python prep_cluster.py data/train_sequences.txt data/cluster/train.cluster30_cluster.tsv > data/train_sequences.cluster30.tsv
python prep_cluster.py data/train_sequences.stripped.txt data/cluster/train.stripped.cluster30_cluster.tsv > data/train_sequences.stripped.cluster30.tsv