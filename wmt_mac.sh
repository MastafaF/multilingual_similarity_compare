#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# evaluate similarity search on WMT newstest2011


# brew install gnu-tar
if [ ! -d dev ] ; then
  echo " - Download WMT data"
  wget -q http://www.statmt.org/wmt13/dev.tgz
  gtar --wildcards -xf dev.tgz "dev/newstest2012.??"
  /bin/rm dev.tgz
fi
