import os
import json


def load_shareGPT_first_round(path: str):
    with open(path, "r") as file:
        shareGPT_raw = json.load(file)
    # filter the first round prompt
    conversations = []
    for conversation in shareGPT_raw:
        if conversation["id"].endswith("_0") and len(conversation["conversations"]) > 0:
            conversation = conversation["conversations"][0]
            if conversation["from"] == "human":
                conversations.append(conversation["value"])
    return conversations


def save_cls_thresholds(cls_thresholds: list, path: str):
    with open(path, "w") as file:
        json.dump(cls_thresholds, file)


def load_cls_thresholds(path: str):
    with open(path, "r") as file:
        cls_thresholds = json.load(file)
    return cls_thresholds


def get_inference_dataset_path(root_path: str, dataset_type: str, llm: str):
    return os.path.join(
        root_path,
        "data",
        dataset_type,
        llm.split("/")[-1],
        "raw.json",
    )


def get_dataset_path(root_path: str, dataset_type: str, llm: str, bert: str, sub_path: str = ""):
    return os.path.join(
        root_path,
        "data",
        dataset_type,
        llm.split("/")[-1],
        bert.split("/")[-1],
        sub_path,
    )


def get_cls_thresholds_path(root_path: str, dataset_type: str, llm: str, bert: str):
    return os.path.join(
        root_path,
        "data",
        dataset_type,
        llm.split("/")[-1],
        bert.split("/")[-1],
        "cls_thresholds.json",
    )


def get_model_path(root_path: str, dataset_type: str, llm: str, bert: str):
    return os.path.join(
        root_path,
        "models/pretrained",
        dataset_type,
        llm.split("/")[-1],
        bert.split("/")[-1],
    )


def get_result_path(root_path: str, dataset_type: str, llm: str, bert: str):
    return os.path.join(
        root_path,
        "results",
        dataset_type,
        llm.split("/")[-1],
        bert.split("/")[-1],
    )
