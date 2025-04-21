import os
import sys
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from argparse import ArgumentParser

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
        inference_dataset = Dataset.from_json(inference_path)
        return inference_dataset

    # load shareGPT
    conversations = load_shareGPT_first_round(args.dataset)

    # load model
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    llm = LLM(
        args.llm,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # inference
    messages = [[{"role": "user", "content": conversation}] for conversation in conversations]
    formatted_prompts = [
        tokenizer.apply_chat_template(conversation=message, tokenize=False, add_generation_prompt=True) for message in messages
    ]
    outputs = llm.generate(formatted_prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]

    # save dataset
    df = pd.DataFrame(
        {
            "prompt": conversations,
            "response": responses,
        }
    )
    inference_dataset = Dataset.from_pandas(df)
    os.makedirs(os.path.dirname(inference_path), exist_ok=True)
    inference_dataset.to_json(inference_path)
    return inference_dataset


def main():
    parser = ArgumentParser()
    args = PreprocessArgs.add_cli_args(parser).parse_args()
    preprocess_args = PreprocessArgs.from_cli_args(args)

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
