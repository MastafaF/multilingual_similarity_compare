#!/bin/bash
# Copyright Mastafa Foufa
# mastafa.foufa@hotmail.com
#------------------------------
# Usage: sh similarity_sentenceBERT.sh

# @TODO: let user specify kind of experiment: With preprocessing prior or Without
odir="./output/processed_sentence_multiBERT"
sdir="./src"


# If output directory does not exist then create it 
if [ ! -d ${odir} ] ; then
  echo " - creating directory ${odir}"
  mkdir -p ${odir}
fi

echo " - Getting sentence embeddings from sentence-transformers..."
# Get sentence embeddings and store them in output directory 
# cd src directory 
cd ${sdir}
python get_sentence_embeddings_sentenceBERT.py

# input dir for similarity search is where output of embedding process are stored
in_dir="../output/processed_sentence_multiBERT/"
echo " - Doing similarity search..."
# Do similarity search with Faiss and plot Confusion Matrix with errors
python similarity_search.py --model sentence-transformers --dim 512 --in_dir ${in_dir}