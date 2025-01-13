import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer


class Preprocess:
    def __init__(self, bert, llm):
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert)
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm)
        self.cls_thresholds = None

    def input_tokenize_function(self, example):
        # input
        example = self.bert_tokenizer(example["prompt"], truncation=False)
        example["input_token_length"] = len(example["input_ids"])
        if example["input_token_length"] > 512:
            example["input_ids"] = example["input_ids"][-512:]
            example["input_ids"][0] = 101  # add [CLS]
            example["token_type_ids"] = example["token_type_ids"][-512:]
            example["attention_mask"] = example["attention_mask"][-512:]
        return example

    def output_tokenize_function(self, example):
        tokens = self.llm_tokenizer(example["response"], truncation=False)
        example["output_token_length"] = len(tokens["input_ids"])
        return example

    def set_label_function(self, example):
        for i, thresh in enumerate(self.cls_thresholds):
            if example["output_token_length"] <= thresh:
                example["labels"] = i
                break
        return example

    def set_cls_thresholds(self, dataset: Dataset):
        output_token_lens = dataset["output_token_length"]
        self.cls_thresholds = np.percentile(output_token_lens, [25, 50, 75, 99]).astype(int).tolist() + [float("inf")]

    def split_dataset(self, dataset: Dataset):
        # train : validation : test = 6 : 2 : 2
        sep_train_val = int(len(dataset) * 0.6)
        sep_val_test = int(len(dataset) * 0.8)
        train_dataset, validation_dataset, test_dataset = (
            dataset.select(range(sep_train_val)).shuffle(seed=1),
            dataset.select(range(sep_train_val, sep_val_test)).shuffle(seed=1),
            dataset.select(range(sep_val_test, len(dataset))).shuffle(seed=1),
        )
        train_dataset = train_dataset.remove_columns(["prompt", "response"])
        validation_dataset = validation_dataset.remove_columns(["prompt", "response"])
        test_dataset = test_dataset.rename_columns({"prompt": "input", "response": "output"})
        return train_dataset, validation_dataset, test_dataset

    def preprocess(self, dataset: Dataset):
        dataset = dataset.map(self.input_tokenize_function, num_proc=32)
        dataset = dataset.map(self.output_tokenize_function, num_proc=32)
        self.set_cls_thresholds(dataset)
        dataset = dataset.map(self.set_label_function, num_proc=32)
        dataset = dataset.shuffle(seed=1)
        train_dataset, validation_dataset, test_dataset = self.split_dataset(dataset)
        return train_dataset, validation_dataset, test_dataset, self.cls_thresholds


