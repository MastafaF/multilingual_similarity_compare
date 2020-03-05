# Get sentence representations
"""
Sentences have to be in the BPE format, i.e. tokenized sentences on which you applied fastBPE.
Below you can see an example for English, French, Spanish, German, Arabic and Chinese sentences.
"""

########################################################
############ Testing on simple examples ################
#########################################################

from transformers import DistilBertTokenizer, DistilBertModel
import torch
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from tqdm import tqdm
import argparse
import gc

parser = argparse.ArgumentParser(description='Getting sentence embeddings with XLM. ')
parser.add_argument('--max_len', type=int, default=40,
    help='Maximum length of tokens: all sentences with less tokens will be padded with 0, else we will remove all tokens after max_len index')


parser.add_argument('--pooling_strat',type=str, default='cls',
                    help='Pooling strategy to use to get sentence embeddings for last hidden layer')


args = parser.parse_args()
MAX_LEN = args.max_len
POOL_STRAT = args.pooling_strat

if POOL_STRAT == 'mean':
    print('Using mean pooling strategy...')
if POOL_STRAT == 'cls':
    print('Using CLS pooling strategy...')

class distilBERT_model:
  """
  from here: https://github.com/huggingface/pytorch-transformers/blob/a2d4950f5c909f7bb4ea7c06afa6cdecde7e8750/pytorch_transformers/modeling_xlm.py

  We can see all the possible models existing for XLM.
  We focus on MLM+TLM which is the model that best performance on cross-lingual tasks.

  """
  DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
      "distilbert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-config.json",
      "distilbert-base-uncased-distilled-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-distilled-squad-config.json",
      "distilbert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-cased-config.json",
      "distilbert-base-cased-distilled-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-cased-distilled-squad-config.json",
      "distilbert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-german-cased-config.json",
      "distilbert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-multilingual-cased-config.json",
      "distilbert-base-uncased-finetuned-sst-2-english": "https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-uncased-finetuned-sst-2-english-config.json",
  }

  def __init__(self, model_name):
    self.model_name = model_name
    self.model = DistilBertModel.from_pretrained(model_name)
    self.model.eval() # just inference mode used here (no need of back propagation error)
    self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name, do_lower_case=True)

  def encode(self, sentence:str, max_len:int):
    tokenizer = self.tokenizer
    """

      from https://huggingface.co/transformers/_modules/transformers/tokenization_utils.html 

      truncation_strategy: string selected in the following options:
                    - 'longest_first' (default) Iteratively reduce the inputs sequence until the input is under max_length
                        starting from the longest one at each token (when there is a pair of input sequences)
                    - 'only_first': Only truncate the first sequence
                    - 'only_second': Only truncate the second sequence
                    - 'do_not_truncate': Does not truncate (raise an error if the input sequence is longer than max_length)
                pad_to_max_length: if set to True, the returned sequences will be padded according to the model's padding side and
                    padding index, up to their max length. If no max length is specified, the padding is done up to the model's max length.
                    The tokenizer padding sides are handled by the class attribute `padding_side` which can be set to the following strings:
                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
                    Defaults to False: no padding.
      """
    # @TODO: get ID of PADDING tokens
    # Actually the padding the Padding is done with the ID = 0 for distillBERT
    # So all tokens with ids = 1 are just paddings
    input_ids = torch.tensor(
        tokenizer.encode(sentence, add_special_tokens=True, max_length=max_len, truncation_strategy="longest_first",
                         pad_to_max_length='right')).unsqueeze(0)  # Batch size 1

    # @TODO: get this id automatically
    id_padding_token = 0 # after visualization
    print("input ids")
    print(input_ids)
    # outputs = self.model(input_ids)
    # embed = outputs[0]  # The last hidden-state is the first element of the output tuple

    ########### CREATE ATTENTION MASKS ###################
    # This is just to apply attention on the part where there are actual tokens
    # Tokens are all ids different from id = 1
    ones = torch.ones((1, input_ids.shape[1]))
    zeros = torch.zeros((1, input_ids.shape[1]))
    attention_masks = torch.where(input_ids == 0, zeros, ones)  # put zeros where paddings are, one otherwise

    ########## FORWARD TO GET EMBEDDINGS ##########

    embeddings_tuple = self.model(input_ids=input_ids, attention_mask=attention_masks)
    embeddings_ = embeddings_tuple[0]
    # print(embeddings_)
    # print(embeddings_.shape)
    # print(embeddings_[:,0,:].shape)
    if POOL_STRAT == 'cls':
        embeddings_first_token_only = embeddings_[:, 0, :]
        embeddings_arr = embeddings_first_token_only.cpu().detach().numpy()
        # print(embeddings_arr.shape)
        del embeddings_, embeddings_first_token_only, embeddings_tuple
    elif POOL_STRAT == 'mean':
        input_mask_expanded = attention_masks.unsqueeze(-1).expand(embeddings_.size()).float()
        print("Input masks")
        print(input_mask_expanded)
        print(input_mask_expanded.shape)
        print(embeddings_ * input_mask_expanded)
        sum_embeddings = torch.sum(embeddings_ * input_mask_expanded)
        sum_mask = input_mask_expanded.sum(1)  # number of tokens in txt sequence
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embeddings_mean = sum_embeddings / sum_mask
        embeddings_arr = embeddings_mean.cpu().detach().numpy()
        # print(embeddings_arr.shape)
        del embeddings_, embeddings_mean, embeddings_tuple

    # free up ram
    gc.collect()
    return embeddings_arr


if __name__=='__main__':
    # for length in newstest2012.tok.fr
    # mean value = 26.6
    # std value = 15.4
    # max value = 145
    # Load XLM model
    multi_distilBERT = distilBERT_model("distilbert-base-multilingual-cased")
    max_len = MAX_LEN
    # Open file
    lang_arr = ['cs', 'de', 'en', 'es', 'fr', 'ru']

    # lang = "ru"
    for lang in lang_arr:
        # input_file_name = "../data/processed/wmt2012/newstest2012.tok.{}".format(lang)
        input_file_name = "../dev/newstest2012.{}".format(lang)
        arr_embed = []
        with open(input_file_name, 'r') as file:
            N_lines = 3003
            with tqdm(total=N_lines) as pbar:
                for line in file:
                    line = line.strip("\n")
                    # For each line get embedding
                    embed = multi_distilBERT.encode(sentence = line, max_len = max_len)
                    arr_embed.append(embed)
                    pbar.update(1)

        # Store embedding in an array
        np_embed = np.array(arr_embed)
        # save numpy array in memory
        np.save(file = "../output/multi_distilBERT/newstest2012.{}.embed".format(lang), arr = np_embed)