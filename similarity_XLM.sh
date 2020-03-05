#!/bin/bash
# Copyright Mastafa Foufa
# mastafa.foufa@hotmail.com

#------------------------------
# Usage: sh similarity_XLM.sh MAX_LEN

# @TODO: let user specify kind of experiment: With preprocessing prior or Without
odir="./output/XLM_MLM"
sdir="./src"
max_len=$1

# If output directory does not exist then create it
if [ ! -d ${odir} ] ; then
  echo " - creating directory ${odir}"
  mkdir -p ${odir}
fi

echo " - Getting sentence embeddings from XLM with MLM on 100 languages..."
# Get sentence embeddings and store them in output directory
# cd src directory
cd ${sdir}
python get_sentence_embeddings_old_XLM.py --max_len ${max_len}

# input dir for similarity search is where output of embedding process are stored
in_dir="../output/XLM_MLM/"
echo " - Doing similarity search for XLM with MLM on 100 languages..."
# Do similarity search with Faiss and plot Confusion Matrix with errors
python similarity_search.py --model xlm --dim 1280 --in_dir ${in_dir}