MODEL_DIR=$1


mkdir -p $DATA_PATH

CODES_PATH=$MODEL_DIR/codes
VOCAB_PATH=$MODEL_DIR/vocab

mkdir -p $CODES_PATH
mkdir -p $VOCAB_PATH


# download codes
if [ ! -d $CODES_PATH/codes_xnli_100 ]; then
  if [ ! -f $CODES_PATH/codes_xnli_100.zip ]; then
    wget -c https://dl.fbaipublicfiles.com/XLM/codes_xnli_100 -P $CODES_PATH
  fi
fi

# download vocab
if [ ! -d $VOCAB_PATH/vocab_xnli_100 ]; then
  if [ ! -f $VOCAB_PATH/c.zip ]; then
    wget -c https://dl.fbaipublicfiles.com/XLM/vocab_xnli_100 -P $VOCAB_PATH
  fi
fi