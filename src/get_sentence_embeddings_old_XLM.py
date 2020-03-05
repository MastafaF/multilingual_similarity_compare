from transformers import XLMModel
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import XLMTokenizer
import numpy as np
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Getting sentence embeddings with XLM. ')
parser.add_argument('--max_len', type=int, default=40,
    help='Maximum length of tokens: all sentences with less tokens will be padded with 0, else we will remove all tokens after max_len index')


args = parser.parse_args()
MAX_LEN = args.max_len

class XLM_model:
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

  def __init__(self, model_name):
    # @TODO: check what causal refers again
    # model = XLMModel.from_pretrained("xlm-mlm-enfr-1024", causal = False)
    self.model_name = model_name
    self.model = XLMModel.from_pretrained(model_name, causal = False)
    self.tokenizer = XLMTokenizer.from_pretrained(self.model_name, do_lower_case=True)

  def encode(self, sentences, max_len):
    ########## For 15 languages ########
    tokenizer = self.tokenizer

    # sentences = ['[CLS] '+sentence+' [SEP]' for sentence in sentences]
    # sentences = [sentence for sentence in sentences]
    # In model.encode(), special characters can be added on the fly
    # cf. https://huggingface.co/transformers/_modules/transformers/tokenization_utils.html#PreTrainedTokenizer.add_special_tokens
    """"
    def encode(
        self,
        text,
        text_pair=None,
        add_special_tokens=True,
        max_length=None,
        stride=0,
        truncation_strategy="longest_first",
        pad_to_max_length=False,
        return_tensors=None,
        **kwargs
    ):
    """
    # The special character to separate sentences is </s> for XLM which is encoded as 1
    # If in tokenizer.encode() you add add_special_tokens = True
    # Then, it is not necessary to add </s> in each side of your sentence
    sentences = ["</s> " + sentence + " </s>" for sentence in sentences]
    sentences = [tokenizer.tokenize(sent) for sent in sentences]

    # maximum length of each sentence ie number of tokens maximal for all sentences
    MAX_LEN = max_len # AVG around 10 so 64 is good enough
    arr_ids_post = [tokenizer.convert_tokens_to_ids(txt) for txt in sentences]
    input_ids_post = pad_sequences(arr_ids_post,
                              maxlen = MAX_LEN,
                              dtype = 'long',
                              truncating = 'post',
                              padding = 'post') # 'post' to add 0s after in padding for ex
    ########### CREATE ATTENTION MASKS ###################
    # This is just to apply attention on the part where there are actual tokens t
    # Not on the padding elements set to 0 before.
    attention_masks_post = []

    for seq in input_ids_post:
        mask = [float(i>0) for i in seq]
        attention_masks_post.append(mask)


    ########### CREATE SPECIAL CHARACTER MASKS FOR BUILDING EMBEDDINGS WITHOUT FOCUSING ON [CLS] and [SEP] ###################
    # For XLM, it seems like [CLS] and [SEP] are not used at inference time.
    # <s> encoded as 0
    # mask_special_characters_post = []
    # for seq in input_ids_post:
    # #     mask_spec = [1 if i not in [627, 615] else 0 for i in seq]
    #     mask_spec = [1 if i in [0] else 0 for i in seq]
    #     mask_special_characters_post.append(mask_spec)
    input_ids = torch.tensor(input_ids_post)
    attention_masks = torch.tensor(attention_masks_post)
    # special_masks = torch.tensor(mask_special_characters_post)
    # lang_ids = torch.tensor(lang_ids)

    ########## FORWARD TO GET EMBEDDINGS ##########
    input_ids = input_ids.type(torch.int64)
    # lang_ids = lang_ids.type(torch.int64)
    embeddings_tuple = self.model( input_ids = input_ids, attention_mask = attention_masks)
    embeddings_ = embeddings_tuple[0]
    # print(embeddings_)
    # print(embeddings_.shape)
    # print(embeddings_[:,0,:].shape)
    embeddings_first_token_only = embeddings_[:,0,:]
    embeddings_arr = embeddings_first_token_only.cpu().detach().numpy()
    # print(embeddings_arr.shape)
    del embeddings_, embeddings_first_token_only, embeddings_tuple
    return embeddings_arr





if __name__=='__main__':
    # for length in newstest2012.tok.fr
    # mean value = 26.6
    # std value = 15.4
    # max value = 145
    # Load XLM model
    XLM_model = XLM_model("xlm-mlm-100-1280")
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
                    embed = XLM_model.encode(sentences = [line], max_len = max_len)
                    arr_embed.append(embed)
                    pbar.update(1)

        # Store embedding in an array
        np_embed = np.array(arr_embed)
        # save numpy array in memory
        np.save(file = "../output/XLM_MLM/newstest2012.{}.embed".format(lang), arr = np_embed)