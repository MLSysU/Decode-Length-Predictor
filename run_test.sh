#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH

DATASET="data/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_TYPE="shareGPT"
LLM="/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/zhb/QwQ-32B"
BERT="/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/zhb/google-bert--bert-base-multilingual-uncased"

set -x
python scripts/test.py \
    --dataset $DATASET \
    --dataset_type $DATASET_TYPE \
    --llm $LLM \
    --bert $BERT
