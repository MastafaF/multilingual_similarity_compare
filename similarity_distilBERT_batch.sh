#!/bin/bash
# Copyright Mastafa Foufa
# mastafa.foufa@hotmail.com

#------------------------------
# Usage: sh similarity_distilBERT_batch.sh MAX_LEN POOLING_STRAT GPU

# @TODO: let user specify kind of experiment: With preprocessing prior or Without
odir="./output/multi_distilBERT"
sdir="./src"
max_len=$1
pooling_strat=$2
gpu=$3

# If output directory does not exist then create it
if [ ! -d ${odir} ] ; then
  echo " - creating directory ${odir}"
  mkdir -p ${odir}
fi

# @TODO: put this inside the if [ ! -d ${odir} ]
echo " - Getting sentence embeddings from multilingual distilBERT with batches..."
# Get sentence embeddings and store them in output directory
# cd src directory
cd ${sdir}
python get_sentence_embeddings_distilBERT_batch.py --max_len ${max_len} --pooling_strat ${pooling_strat} --gpu ${gpu}

# input dir for similarity search is where output of embedding process are stored
in_dir="../output/multi_distilBERT/"
echo " - Doing similarity search for multilingual distilBERT..."
# Do similarity search with Faiss and plot Confusion Matrix with errors
python similarity_search.py --model xlm --dim 768 --in_dir ${in_dir}