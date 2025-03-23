#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH

DATASET="data/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_TYPE="shareGPT"
LLM="/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/zhb/QwQ-32B"
BERT="/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/zhb/google-bert--bert-base-multilingual-uncased"

INFERENCE_ARGS="--max_tokens 40960 --tensor_parallel_size 8"
# SKIP_FLAG="--skip_inference"

set -x
python scripts/preprocess_shareGPT.py \
    --dataset $DATASET \
    --dataset_type $DATASET_TYPE \
    --llm $LLM \
    --bert $BERT \
    $INFERENCE_ARGS \
    $SKIP_FLAG
