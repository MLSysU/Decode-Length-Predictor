#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH

DATASET="data/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_TYPE="shareGPT"
LLM="meta-llama/Llama-2-7b-chat-hf"
BERT="google-bert/bert-base-multilingual-uncased"

INFERENCE_ARGS="--max_tokens 100000 --temperature 0.8 --top_p 0.95 --tensor_parallel_size 1"
SKIP_FLAG="--skip_inference"

set -x
python scripts/preprocess_shareGPT.py \
    --dataset $DATASET \
    --dataset_type $DATASET_TYPE \
    --llm $LLM \
    --bert $BERT \
    $INFERENCE_ARGS \
    $SKIP_FLAG
