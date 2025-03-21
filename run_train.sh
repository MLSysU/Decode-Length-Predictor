#!/bin/bash

export PYTHONPATH=$PWD:$PYTHONPATH

DATASET="data/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json"
DATASET_TYPE="shareGPT"
LLM="meta-llama/Llama-2-7b-chat-hf"
BERT="google-bert/bert-base-multilingual-uncased"

TRAIN_ARGS="--epoch 4 --train_bs 32 --validate_bs 32 --bert_lr 5e-5 --lr 1e-3"
MODEL_ARGS="--hidden_dim 128"

set -x
torchrun --nproc_per_node 2 scripts/train_ddp.py \
    --dataset $DATASET \
    --dataset_type $DATASET_TYPE \
    --llm $LLM \
    --bert $BERT \
    $TRAIN_ARGS \
    $MODEL_ARGS
