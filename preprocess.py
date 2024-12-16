from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import pandas as pd
import os
from config import *
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np


def process_first_round_conversation(sample):
    conversation = sample["conversation"]

    user_content = ""
    for i in range(len(conversation)):
        sentence = conversation[i]
        if sentence["role"] == "user":
            if i > 0:
                user_content += "\n"
            user_content += sentence["content"]
        else:
            break

    assistant_content = ""
    for j in range(i, len(conversation)):
        sentence = conversation[j]
        if sentence["role"] == "assistant":
            if j > i:
                assistant_content += "\n"
            assistant_content += conversation[j]["content"]
        else:
            break

    label = 0
    encoded_response = llm_tokenizer(assistant_content, truncation=False)
    for i, thresh in enumerate(cls_thresholds):
        if len(encoded_response["input_ids"]) < thresh:
            label = i
            break

    new_sample = {
        "prompt": user_content,
        "labels": label,
        "num_tokens": len(encoded_response["input_ids"]),
    }
    return pd.DataFrame(new_sample, index=[0])


def exact_first_round_prompt(dataset, num_workers=None):
    df = dataset.to_pandas()

    if num_workers is None:
        num_workers = cpu_count()

    with Pool(num_workers) as pool:
        ans_df_list = list(
            tqdm(
                pool.imap_unordered(process_first_round_conversation, [df.iloc[i] for i in range(len(df))]),
                total=len(df),
                desc="Processing conversations",
            )
        )

    ans_df = pd.concat(ans_df_list, ignore_index=True)
    return Dataset.from_pandas(ans_df)


def process_multi_round_conversation(i_sample):
    conversation_id, sample = i_sample
    conversation = sample["conversation"]
    dialogue_so_far = ""
    new_samples = {"prompt": [], "labels": [], "num_tokens": [], "conversation_id": [], "turn_id": []}

    for i, sentence in enumerate(conversation):
        if sentence["role"] == "user":
            dialogue_so_far += "[USER]: " + sentence["content"] + "\n"
        else:
            assistant_content = sentence["content"]

            encoded_response = llm_tokenizer(assistant_content, truncation=False)
            # Drop abnormal samples that have empty responses or might have been truncated.
            if len(encoded_response["input_ids"]) <= 1 or len(encoded_response["input_ids"]) >= 512:
                break

            # Add a new prediction sample
            new_samples["prompt"].append(dialogue_so_far)
            new_samples["conversation_id"].append(conversation_id)  # Assuming DataFrame has an index named 'name'
            new_samples["turn_id"].append(i // 2)
            new_samples["num_tokens"].append(len(encoded_response["input_ids"]))

            for i, thresh in enumerate(cls_thresholds):
                if len(encoded_response["input_ids"]) < thresh:
                    new_samples["labels"].append(i)
                    break

            dialogue_so_far += "[ASSISTANT]: " + sentence["content"] + "\n"

    return pd.DataFrame(new_samples)


def exact_multi_round_prompt(dataset, num_workers=None):
    df = dataset.to_pandas()

    if num_workers is None:
        num_workers = cpu_count()

    with Pool(num_workers) as pool:
        ans_df_list = list(
            tqdm(
                pool.imap_unordered(process_multi_round_conversation, [(i, df.iloc[i]) for i in range(len(df))]),
                total=len(df),
                desc="Processing conversations",
            )
        )

    ans_df = pd.concat(ans_df_list, ignore_index=True)
    return Dataset.from_pandas(ans_df)


def tokenize_function(example):
    example = bert_tokenizer(example["prompt"], truncation=False)
    if len(example["input_ids"]) >= 512:
        example["input_ids"] = example["input_ids"][-512:]
        example["token_type_ids"] = example["token_type_ids"][-512:]
        example["attention_mask"] = example["attention_mask"][-512:]
    return example


def preprocess(dataset: Dataset):
    # only one llm model
    dataset = dataset.filter(lambda x: x["model"] == llm_model)
    print(len(dataset))
    dataset = dataset.shuffle(seed=1)
    dataset = dataset.remove_columns(["openai_moderation", "redacted", "language", "conversation_id", "turn"])
    dataset = dataset.add_column("prompt", [""] * len(dataset))
    dataset = dataset.add_column("labels", [0] * len(dataset))
    dataset = dataset.add_column("num_tokens", [0] * len(dataset))
    # multi round
    dataset = (
        exact_multi_round_prompt(dataset, num_workers=32) if is_multi_round else exact_first_round_prompt(dataset, num_workers=32)
    )
    num_tokens_list = dataset["num_tokens"]
    print(np.percentile(num_tokens_list, [0, 25, 50, 75, 99, 100]))
    # tokenizer
    dataset = dataset.map(tokenize_function, num_proc=32, remove_columns="prompt")
    return dataset


def split_first_round_dataset(dataset: Dataset):
    sep_train_val = int(len(dataset) * 0.6)
    sep_val_test = int(len(dataset) * 0.8)
    return (
        dataset.select(range(sep_train_val)).shuffle(seed=1),
        dataset.select(range(sep_train_val, sep_val_test)).shuffle(seed=1),
        dataset.select(range(sep_val_test, len(dataset))).shuffle(seed=1),
    )


def split_multi_round_dataset(dataset: Dataset):
    sep_train_val = int(len(dataset) * 0.6)
    sep_val_test = int(len(dataset) * 0.8)
    while (
        sep_train_val < sep_val_test
        and abs(dataset[sep_train_val]["conversation_id"] - dataset[sep_train_val - 1]["conversation_id"]) < 0.1
    ):
        sep_train_val += 1
    while (
        sep_val_test < len(dataset)
        and abs(dataset[sep_val_test]["conversation_id"] - dataset[sep_val_test - 1]["conversation_id"]) < 0.1
    ):
        sep_val_test += 1

    return (
        dataset.select(range(sep_train_val)).shuffle(seed=1),
        dataset.select(range(sep_train_val, sep_val_test)),
        dataset.select(range(sep_val_test, len(dataset))),
    )


if __name__ == "__main__":
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    dataset = load_dataset(lmsys_path, split="train")
    dataset = preprocess(dataset)
    train_dataset, validation_dataset, test_dataset = (
        split_multi_round_dataset(dataset) if is_multi_round else split_first_round_dataset(dataset)
    )
    train_dataset.set_format("torch"), validation_dataset.set_format("torch"), test_dataset.set_format("torch")
    os.makedirs(dataset_path, exist_ok=True)
    train_dataset.save_to_disk(os.path.join(dataset_path, "train"))
    validation_dataset.save_to_disk(os.path.join(dataset_path, "validation"))
    test_dataset.save_to_disk(os.path.join(dataset_path, "test"))
