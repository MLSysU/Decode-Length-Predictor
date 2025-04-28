import os
import sys
import datasets
import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import torch
from torch import nn
from tqdm import tqdm, trange
from models import BertRegressionModel
from argparse import ArgumentParser

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from utils import (
    TestArgs,
    get_dataset_path,
    get_model_path,
    get_result_path,
)


def test(
    model: BertRegressionModel,
    test_dataset: datasets.Dataset,
    args: TestArgs,
):
    device = torch.device("cuda")
    model = model.to(device)
    model.eval()

    # warmup
    input_ids = test_dataset["input_ids"][0].unsqueeze(0).to(device)
    attention_mask = test_dataset["attention_mask"][0].unsqueeze(0).to(device)
    for _ in trange(10, desc="Warmup"):
        model(input_ids, attention_mask)

    # predict
    input_ids_list = test_dataset["input_ids"]
    attention_mask_list = test_dataset["attention_mask"]
    prediction_list = []
    latency_list = []
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        for i in trange(len(test_dataset), desc="Predicting"):
            input_ids = input_ids_list[i].unsqueeze(0).to(device)
            attention_mask = attention_mask_list[i].unsqueeze(0).to(device)
            start_event.record()
            prediction = model(input_ids, attention_mask)
            end_event.record()
            torch.cuda.synchronize()
            prediction_list.extend(prediction)
            elasped_time = start_event.elapsed_time(end_event)
            latency_list.append(elasped_time)

    print(
        f"latency: mean(ms): {np.mean(latency_list):.4f}, max(ms): {np.max(latency_list):.4f}, min(ms): {np.min(latency_list):.4f}"
    )

    predictions = torch.tensor(prediction_list)
    return predictions, test_dataset["labels"], latency_list


def calc_metrics(
    predictions: torch.Tensor,
    cls_predictions: torch.Tensor,
    labels: torch.Tensor,
    latency_list: list,
    result_path: str,
):
    l1 = nn.L1Loss()(predictions, labels.type_as(predictions))
    mse = nn.MSELoss()(predictions, labels.type_as(predictions))
    acc = accuracy_score(labels, cls_predictions)
    f1 = f1_score(labels, cls_predictions, average="macro", zero_division=0)
    precision = precision_score(labels, cls_predictions, average="macro", zero_division=0)
    recall = recall_score(labels, cls_predictions, average="macro", zero_division=0)

    # metrics data
    metrics_df = pd.DataFrame(
        {
            "Model": [result_path.split("/")[-1]],
            "L1": [l1.item()],
            "MSE": [mse.item()],
            "ACC": [acc],
            "F1": [f1],
            "Precision": [precision],
            "Recall": [recall],
            "Latency": np.mean(latency_list),
        }
    )
    metrics_df.to_csv(os.path.join(result_path, "metrics.csv"), index=False)
    print(f"model: {result_path.split('/')[-1]}")
    print("metrtcs:")
    print(
        f"\tL1:{l1.item():.4f}, MES:{mse.item():.4f}, ACC:{acc:.4f}, F1:{f1:.4f}, Precision:{precision:.4f}, Recall:{recall:.4f}"
    )


def save_output(predictions, cls_predictions, labels, latency_list, output_token_length, result_path):
    # prediciton output for visualization
    output_df = pd.DataFrame(
        {
            "label": labels,
            "cls_predict": cls_predictions,
            "predict": predictions,
            "output_token_length": output_token_length,
            "latency": latency_list,
        }
    )
    output_df.to_csv(os.path.join(result_path, "output.csv"), index=False)


def save_benchmark(test_dataset, predict_interval, predict_mean_len, result_path):
    predict_left, predict_right = predict_interval

    benchmark_df = pd.DataFrame(
        {
            "request_id": [f"{i}" for i in range(len(test_dataset))],
            "input": test_dataset["input"],
            "output": test_dataset["output"],
            "input_token_length": test_dataset["input_token_length"],
            "output_token_length": test_dataset["output_token_length"],
            "predict_left": predict_left,
            "predict_right": predict_right,
            "predict_mean_len": predict_mean_len,
        }
    )
    benchmark_df.to_json(os.path.join(result_path, "benchmark.json"), orient="records", lines=False, indent=4)


def main():
    parser = ArgumentParser()
    args = TestArgs.add_cli_args(parser).parse_args()
    test_args = TestArgs.from_cli_args(args)

    # load dataset
    test_dataset = datasets.load_from_disk(
        get_dataset_path(root_path, test_args.dataset_type, test_args.llm, test_args.bert, "test")
    )
    # load model
    model = BertRegressionModel.from_pretrained(get_model_path(root_path, test_args.dataset_type, test_args.llm, test_args.bert))

    # predict
    predictions, labels, latency_list = test(model, test_dataset, test_args)
    cls_predictions = model.get_cls(predictions)

    # get path
    result_path = get_result_path(root_path, test_args.dataset_type, test_args.llm, test_args.bert)
    os.makedirs(result_path, exist_ok=True)
    print("Saving results to:", result_path)

    # metrics
    calc_metrics(predictions, cls_predictions, labels, latency_list, result_path)

    # save output
    save_output(predictions, cls_predictions, labels, latency_list, test_dataset["output_token_length"], result_path)

    # save benchmark
    predict_interval = model.get_interval(predictions)
    predict_mean_len = model.get_mean_len(predictions)
    save_benchmark(test_dataset, predict_interval, predict_mean_len, result_path)


if __name__ == "__main__":
    main()
