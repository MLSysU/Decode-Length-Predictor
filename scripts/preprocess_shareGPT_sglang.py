import os
import sys
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from argparse import ArgumentParser

import asyncio
import io
import os

from PIL import Image
import requests
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from utils import (
    PreprocessArgs,
    Preprocess,
    save_cls_thresholds,
    load_shareGPT_first_round,
    get_inference_dataset_path,
    get_dataset_path,
    get_cls_thresholds_path,
)


def inference(args: PreprocessArgs):
    inference_path = get_inference_dataset_path(root_path, args.dataset_type, args.llm)

    # skip the inference step
    if args.skip_inference:
        # load inference dataset
        inference_dataset = Dataset.load_from_disk(inference_path)
        return inference_dataset

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    # load shareGPT
    conversations = load_shareGPT_first_round(args.dataset)
    llm = sgl.Engine(model_path=args.llm, tp_size = args.tensor_parallel_size)
    sampling_params = {"max_tokens", args.max_tokens}


    # inference
    messages = [[{"role": "user", "content": conversation}] for conversation in conversations]
    formatted_prompts = [
        tokenizer.apply_chat_template(conversation=message, tokenize=False, add_generation_prompt=True) for message in messages
    ]
    outputs = llm.generate(formatted_prompts, sampling_params)
    print(outputs)
    responses = [output['text'] for output in outputs]
    print(responses)
    # save dataset
    df = pd.DataFrame(
        {
            "prompt": conversations,
            "response": responses,
        }
    )
    inference_dataset = Dataset.from_pandas(df)
    os.makedirs(inference_path, exist_ok=True)
    inference_dataset.save_to_disk(inference_path)
    return inference_dataset


def main():
    parser = ArgumentParser()
    args = PreprocessArgs.add_cli_args(parser).parse_args()
    preprocess_args = PreprocessArgs.from_cli_args(args)

    assert preprocess_args.dataset_type == "shareGPT", f"'{__file__}' Only support shareGPT dataset"

    # inference
    inference_dataset = inference(preprocess_args)

    # preprocess
    train_dataset, validation_dataset, test_dataset, cls_thresholds = Preprocess(
        preprocess_args.bert, preprocess_args.llm
    ).preprocess(inference_dataset)

    # save dataset
    dataset_path = get_dataset_path(root_path, args.dataset_type, preprocess_args.llm, preprocess_args.bert)
    print("Saving dataset to:", dataset_path)
    os.makedirs(dataset_path, exist_ok=True)
    train_dataset.set_format(type="torch")
    train_dataset.save_to_disk(os.path.join(dataset_path, "train"))
    validation_dataset.set_format(type="torch")
    validation_dataset.save_to_disk(os.path.join(dataset_path, "validation"))
    test_dataset.set_format(type="torch")
    test_dataset.save_to_disk(os.path.join(dataset_path, "test"))

    # save cls_thresholds
    cls_thresholds_path = get_cls_thresholds_path(root_path, args.dataset_type, preprocess_args.llm, preprocess_args.bert)
    save_cls_thresholds(cls_thresholds, cls_thresholds_path)


if __name__ == "__main__":
    main()
