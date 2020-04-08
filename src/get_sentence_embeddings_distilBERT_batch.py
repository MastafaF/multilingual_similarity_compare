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
import time

parser = argparse.ArgumentParser(description='Getting sentence embeddings with XLM. ')
parser.add_argument('--max_len', type=int, default=40,
    help='Maximum length of tokens: all sentences with less tokens will be padded with 0, else we will remove all tokens after max_len index')


parser.add_argument('--pooling_strat',type=str, default='cls',
                    help='Pooling strategy to use to get sentence embeddings for last hidden layer')

parser.add_argument('--gpu', type=bool, default=True,
                    help='Use GPU or not?')


args = parser.parse_args()
MAX_LEN = args.max_len
POOL_STRAT = args.pooling_strat
GPU = args.gpu



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

  def __init__(self, model_name, gpu=False):
    self.model_name = model_name
    self.model = DistilBertModel.from_pretrained(model_name)
    self.model.eval() # just inference mode used here (no need of back propagation error)
    self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name, do_lower_case=True)
    self.use_gpu = gpu
    is_gpu_support = torch.cuda.is_available()
    if self.use_gpu and is_gpu_support:
        print("Using GPU...")
        self.device = torch.device("cuda")
    else:
        print("Using CPU...")
        self.device = torch.device("cpu")

    self.model.to(self.device)

    # Only evaluation mode with forward pass: no backward pass
    self.model.eval()


  def encode(self, sentences:list, max_len:int):
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
    # @TODO: get ID of PADDING tokens automatically (we printed first to get it here)
    # Actually the padding the Padding is done with the ID = 0 for distillBERT
    # So all tokens with ids = 0 are just paddings
    input_ids_arr = []
    for sentence in sentences:
        input_ids = tokenizer.encode(sentence, add_special_tokens=True, max_length=max_len,
                                     truncation_strategy="longest_first",
                                     pad_to_max_length='right')
        input_ids_arr.append(input_ids)


    input_ids = torch.tensor(np.array(input_ids_arr))


    ########### CREATE ATTENTION MASKS ###################
    # This is just to apply attention on the part where there are actual tokens
    # Tokens are all ids different from id = 0
    ones = torch.ones(input_ids.shape)
    zeros = torch.zeros(input_ids.shape)
    attention_masks = torch.where(input_ids == 0, zeros, ones)  # put zeros where paddings are, one otherwise

    ########## FORWARD TO GET EMBEDDINGS ##########
    # Move every tensor to the device used
    input_ids = input_ids.to(self.device)
    attention_masks = attention_masks.to(self.device)

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





###### From LASER repo ############
def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip())
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []

    if len(buffer) > 0:
        yield buffer


def EncodeTime(t):
    t = int(time.time() - t)
    if t < 1000:
        print(' in {:d}s'.format(t))
    else:
        print(' in {:d}m{:d}s'.format(t // 60, t % 60))


# Encode sentences (existing file pointers)
def EncodeFilep(inp_file, out_file, buffer_size=10000, verbose=False):
    n = 0
    t = time.time()
    for sentences in buffered_read(inp_file, buffer_size):
        # print(sentences[0])
        # print(XLM_model.encode(sentences=sentences, max_len=max_len).shape)
        multi_distilBERT.encode(sentences=sentences, max_len=max_len).tofile(out_file)

        # Free up RAM
        gc.collect()
        # encoder.encode_sentences(sentences).tofile(out_file)
        n += len(sentences)
        if verbose and n % 10000 == 0:
            print('\r - Encoder: {:d} sentences'.format(n), end='')
    if verbose:
        print('\r - Encoder: {:d} sentences'.format(n), end='')
        EncodeTime(t)



if __name__ == '__main__':
    # for length in newstest2012.tok.fr
    # mean value = 26.6
    # std value = 15.4
    # max value = 145
    # Load XLM model
    multi_distilBERT = distilBERT_model("distilbert-base-multilingual-cased", gpu=GPU)
    max_len = MAX_LEN
    # Open file
    lang_arr = ['cs', 'de', 'en', 'es', 'fr']
    for lang in lang_arr:
      # input_file_name = "../data/processed/wmt2012/newstest2012.tok.{}".format(lang)
        input_file_name = "../dev/newstest2012.{}".format(lang)
        arr_embed = []

        in_file = open(input_file_name, 'r')
        in_file = [line.rstrip('\n') for line in in_file]
        print("LENGTH OF TOTAL DOCUMENT", len(in_file))
        out_fname = "../output/multi_distilBERT/newstest2012.{}.embed.npy".format(lang)
        fout = open(out_fname, mode='wb')
        # On Google Colab I get killed when using buffer_size = 32, so I use buffer_size = 24 in practice
        EncodeFilep(inp_file=in_file, out_file=fout, buffer_size=32,
                    verbose=True)
        fout.close()
        # Free up memory
        gc.collect()
