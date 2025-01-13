#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH

DATASET="data/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_TYPE="shareGPT"
LLM="meta-llama/Llama-2-7b-chat-hf"
BERT="google-bert/bert-base-multilingual-uncased"

set -x
python scripts/test.py \
    --dataset $DATASET \
    --dataset_type $DATASET_TYPE \
    --llm $LLM \
    --bert $BERT
