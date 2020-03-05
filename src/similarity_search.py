"""
The goal of this file is to Encode a document
We inspire ourselves from https://github.com/facebookresearch/LASER/blob/master/source/embed.py

The idea is to reproduce the Similarity Search task from LASER with XLM-R
XLM-R is the new SOTA in multilingual embeddings, hence it should outperform LASER in this task
"""


import re
import os
import tempfile
import sys
import time
import argparse
import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import sys
sys.path.append("./lib/") # Just to call indexing.py in here
from indexing import IndexCreate, IndexSearchMultiple, IndexPrintConfusionMatrix


parser = argparse.ArgumentParser(description='Multilingual encoding and cross-lingual similarity search with different models.')
parser.add_argument('--model', type=str, required=True,
    help='Type of model to train (sentence-transformers, XLM, XLM-R)')

# dimension for sentence-transformers is 512
# dim for XLM with MLM 100 languages is 1280
parser.add_argument('--dim', type=int, default=512,
    help='Dimension of embeddings')

# directory of the input files with embeddings
# shoudl look like this: "../output/processed_sentence_multiBERT/"
parser.add_argument('--in_dir', type=str, required=True,
    help='Directory path of input file with embeddings')


args = parser.parse_args()
model = args.model 
dim = args.dim
input_dir = args.in_dir


textual = False
print('\nProcessing:')
all_texts = None
all_data = []
all_index = []
lang_arr = ['cs', 'de', 'en', 'es', 'fr', 'ru']
# for sentence transformers, multilingual model does not support 'cs' yet. So we remove it for now
# lang_arr = ['de', 'en', 'es', 'fr', 'ru']
# filename_base = "./dev/newtest2012."
if textual:
    all_texts = []
    print(' - using textual comparision')
    for l in lang_arr:
        with open("../dev/newtest2012." + l,
                  encoding='utf-8', errors='surrogateescape') as f:
            texts = f.readlines()
            print(' -   {:s}: {:d} lines'.format("../dev/newtest2012." + l, len(texts)))
            all_texts.append(texts)


# filename_base_no_preproc= "../output/newstest2012.{}.embed.npy"
# filename_base_preproc = "../output/processed/newstest2012.{}.embed.npy"
# filename_base_no_prepro_XLM_R = "../output/processed_XLM-R_new/newstest2012.{}.embed.npy"
# filename_base_preproc_sBERT = "../output/processed_sentence_multiBERT/newstest2012.{}.embed.npy"
for l in lang_arr:
    input_file = input_dir + "newstest2012.{}.embed.npy".format(l)
    d, idx = IndexCreate(input_file,
                         'FlatL2',
                         verbose=True, save_index=False, dim = dim, model = model)
    all_data.append(d)
    all_index.append(idx)

# @TODO: add texts as well to check
err = IndexSearchMultiple(all_data, all_index, texts=all_texts,
                          verbose=False, print_errors=False)
IndexPrintConfusionMatrix(err, lang_arr)