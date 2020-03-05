# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#
# Usage: ./get-data-xnli.sh
#

set -e

# data paths
MAIN_PATH=$PWD
OUTPATH=$PWD/data/wmt
WMT_PATH=$PWD/dev

# tools paths
TOOLS_PATH=$PWD/tools
TOKENIZER=$TOOLS_PATH/tokenize.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py

PROCESSED_PATH=$PWD/data/processed/wmt2012
CODES_PATH=$MAIN_PATH/codes/codes_xnli_100
VOCAB_PATH=$MAIN_PATH/codes/codes_xnli_100
FASTBPE=$TOOLS_PATH/fastBPE/fast


# install tools: done independently this time
#sh ./install-tools.sh

# create directories
mkdir -p $OUTPATH
mkdir -p $PROCESSED_PATH

# download data
#sh ./wmt.sh

# Tokenize files
# sed 's/^/</s>/' adds special XLM characters </s> at the beginning of each line after tokenization
# awk '{print $0"</s>"}' yourFile >> adds special character </s> at the end of each line
echo "*** Preparing WMT data ***"
for lg in cs de en es fr ru ; do
    awk -F '\n' '{ print $1}' $WMT_PATH/newstest2012.$lg \
    | awk '{gsub(/\"/,"")};1' \
    | python $LOWER_REMOVE_ACCENT \
    | sh $TOKENIZER $lg \
    > $PROCESSED_PATH/newstest2012.tok.$lg
done
echo 'Finished preparing data.'

for lg in cs de en es fr ru ; do
    echo "BPE-rizing $lg ...."
    $FASTBPE applybpe $PROCESSED_PATH/newstest2012.bpe.$lg $PROCESSED_PATH/newstest2012.tok.$lg $CODES_PATH
    python preprocess.py $VOCAB_PATH $PROCESSED_PATH/newstest2012.bpe.$lg
done
echo 'Finished BPE-rizing data.'

for lg in cs de en es fr ru ; do
    echo "Adding special characters to $lg ...."
    awk '$0="</s> "$0' $PROCESSED_PATH/newstest2012.bpe.$lg \
    | awk '{print $0" </s>"}' \
    > $PROCESSED_PATH/newstest2012_XLM.bpe.$lg
done
echo 'Finished Adding special characters to data'
