# Get sentence representations
"""
Sentences have to be in the BPE format, i.e. tokenized sentences on which you applied fastBPE.
Below you can see an example for English, French, Spanish, German, Arabic and Chinese sentences.
"""

########################################################
############ Testing on simple examples ################
#########################################################

from transformers import XLMRobertaModel
from transformers import XLMRobertaTokenizer
import torch
import numpy as np
from tqdm import tqdm
import argparse
import gc
import time

parser = argparse.ArgumentParser(description='Getting sentence embeddings with XLM. ')
parser.add_argument('--max_len', type=int, default=40,
                    help='Maximum length of tokens: all sentences with less tokens will be padded with 0, else we will remove all tokens after max_len index')

parser.add_argument('--pooling_strat', type=str, default='cls',
                    help='Pooling strategy to use to get sentence embeddings for last hidden layer')

parser.add_argument('--gpu', type=bool, default=True,
                    help='Use GPU or not?')

args = parser.parse_args()

MAX_LEN = args.max_len  # args.max_len
POOL_STRAT = args.pooling_strat  # args.pooling_strat
GPU = args.gpu

if POOL_STRAT == 'mean':
    print('Using mean pooling strategy...')
if POOL_STRAT == 'cls':
    print('Using CLS pooling strategy...')


class XLM_R_model:
    """
    from here: https://github.com/huggingface/pytorch-transformers/blob/a2d4950f5c909f7bb4ea7c06afa6cdecde7e8750/pytorch_transformers/modeling_xlm.py

    We can see all the possible models existing for XLM.
    We focus on MLM+TLM which is the model that best performance on cross-lingual tasks.

    """
    XLM_PRETRAINED_MODEL_ARCHIVE_MAP = {
        'xlm-mlm-en-2048': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-en-2048-pytorch_model.bin",
        'xlm-mlm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-ende-1024-pytorch_model.bin",
        'xlm-mlm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enfr-1024-pytorch_model.bin",
        'xlm-mlm-enro-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-enro-1024-pytorch_model.bin",
        'xlm-mlm-tlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-tlm-xnli15-1024-pytorch_model.bin",
        'xlm-mlm-xnli15-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-xnli15-1024-pytorch_model.bin",
        'xlm-clm-enfr-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-enfr-1024-pytorch_model.bin",
        'xlm-clm-ende-1024': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-clm-ende-1024-pytorch_model.bin",
        'xlm-mlm-17-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-17-1280-pytorch_model.bin",
        'xlm-mlm-100-1280': "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-mlm-100-1280-pytorch_model.bin",
    }

    def __init__(self, model_name, gpu=False):
        # model = XLMModel.from_pretrained("xlm-mlm-enfr-1024", causal = False)
        self.model_name = model_name
        self.model = XLMRobertaModel.from_pretrained(model_name)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.model_name, do_lower_case=True)
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

    def encode(self, sentences: list, max_len: int):
        ########## For 15 languages ########
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
        input_ids_arr = []
        for sentence in sentences:
            input_ids = tokenizer.encode(sentence, add_special_tokens=True, max_length=max_len,
                                         truncation_strategy="longest_first",
                                         pad_to_max_length='right')
            input_ids_arr.append(input_ids)

        # Actually the padding the Padding is done with the ID = 1
        # So all tokens with ids = 1 are just paddings
        input_ids = torch.tensor(np.array(input_ids_arr))

        # print("input ids")
        # print(input_ids)
        # outputs = self.model(input_ids)
        # embed = outputs[0]  # The last hidden-state is the first element of the output tuple

        ########### CREATE ATTENTION MASKS ###################
        # This is just to apply attention on the part where there are actual tokens
        # Tokens are all ids different from id = 1
        ones = torch.ones(input_ids.shape)
        zeros = torch.zeros(input_ids.shape)
        attention_masks = torch.where(input_ids == 1, zeros, ones)  # put zeros where paddings are, one otherwise
        # print("attention masks")
        # print(attention_masks)
        # print("Input Id shape and Attention Masks shape")
        # print(input_ids.shape)
        # print(attention_masks.shape)
        # print("=========")
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
            # print("Input masks")
            # print(input_mask_expanded)
            # print(input_mask_expanded.shape)
            # print(embeddings_*input_mask_expanded)
            sum_embeddings = torch.sum(embeddings_ * input_mask_expanded)
            sum_mask = input_mask_expanded.sum(1)  # number of tokens in txt sequence
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            embeddings_mean = sum_embeddings / sum_mask
            embeddings_arr = embeddings_mean.cpu().detach().numpy()
            # print(embeddings_arr.shape)
            del embeddings_, embeddings_mean, embeddings_tuple

        # Check impact of torch.cuda.empty_cache()
        if self.use_gpu:
            torch.cuda.empty_cache()
        # free up ram
        gc.collect()
        return embeddings_arr


###### From LASER repo ############
def buffered_read(fp, buffer_size):
    buffer = []
    for src_str in fp:
        buffer.append(src_str.strip("\n"))
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
        XLM_model.encode(sentences=sentences, max_len=max_len).tofile(out_file)
        # Free up RAM
        gc.collect()
        # encoder.encode_sentences(sentences).tofile(out_file)
        n += len(sentences)
        if verbose and n % 10000 == 0:
            print('\r - Encoder: {:d} sentences'.format(n), end='')
    if verbose:
        print('\r - Encoder: {:d} sentences'.format(n), end='')
        EncodeTime(t)


######==================From LASER Repo=================############

if __name__ == '__main__':
    # for length in newstest2012.tok.fr
    # mean value = 26.6
    # std value = 15.4
    # max value = 145
    # Load XLM model
    XLM_model = XLM_R_model("xlm-roberta-large", gpu=GPU)
    max_len = MAX_LEN
    # Open file
    lang_arr = ['cs', 'de', 'en', 'es', 'fr']

    # lang = "ru"
    for lang in lang_arr:
        # input_file_name = "../data/processed/wmt2012/newstest2012.tok.{}".format(lang)
        input_file_name = "../dev/newstest2012.{}".format(lang)
        arr_embed = []

        in_file = open(input_file_name, 'r')
        in_file = [line.rstrip('\n') for line in in_file]
        # On Google Colab I get killed when using buffer_size = 32, so I use buffer_size = 24 in practice
        EncodeFilep(inp_file=in_file, out_file="../output/XLM_R/newstest2012.{}.embed.npy".format(lang), buffer_size=32,
                    verbose=True)

        # Free up memory
        gc.collect()
        # # Store embedding in an array
        np_embed = np.array(arr_embed)
        # save numpy array in memory
        np.save(file = "../output/XLM_R/newstest2012.{}.embed".format(lang), arr = np_embed)