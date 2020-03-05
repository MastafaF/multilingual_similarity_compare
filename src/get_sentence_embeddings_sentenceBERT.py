from sentence_transformers import models, losses
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSDataReader
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import numpy as np
from tqdm import tqdm


if __name__=='__main__':
    # Open file
    lang_arr = ['cs', 'de', 'en', 'es', 'fr', 'ru']
    multilingual_embedder = SentenceTransformer('distiluse-base-multilingual-cased')
    # lang = "ru"
    for lang in lang_arr:
        #input_file_name = "../data/processed/wmt2012/newstest2012.tok.{}".format(lang)
        input_file_name = "../dev/newstest2012.{}".format(lang)
        arr_embed = []
        arr_line = []
        with open(input_file_name, 'r') as file:
            N_lines = 3003
            with tqdm(total=N_lines) as pbar:
                for line in file:
                    line = line.strip("\n")
                    # For each line get embedding
                    arr_line.append(line)
                    pbar.update(1)

        embeddings = multilingual_embedder.encode(arr_line) # list: [embed_1, embed_2, ...., embed_N]
        #  pour tout i, embed_i.shape = (512,)
        # @TODO: go from list(embeddings) to array of shape (N_lines, dim_embed) = (3003, 512)
        embeddings_arr = [list(embed) for embed in embeddings] # list of lists of length dim_embed
        # Store embedding in an array
        np_embed = np.array(embeddings_arr) # shape = (N_lines, dim_embed
        # save numpy array in memory
        np.save(file = "../output/processed_sentence_multiBERT/newstest2012.{}.embed".format(lang), arr = np_embed)
