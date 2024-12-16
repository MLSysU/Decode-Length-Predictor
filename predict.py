import os
import time
import datasets
import evaluate
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
from config import *

from model import BertRegressionModel

if __name__ == "__main__":
    dist.init_process_group(backend="gloo")
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = BertRegressionModel(bert_path, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    model.eval()

    test_dataset = datasets.load_from_disk(os.path.join(dataset_path, "test"))
    collect_fn = DataCollatorWithPadding(tokenizer=bert_tokenizer)
    test_sampler = DistributedSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collect_fn, sampler=test_sampler)

    if local_rank == 0:
        print("Test dataset size: ", len(test_dataset))

    test_dataloader.sampler.set_epoch(0)
    labels_local_list = []
    predict_local_list = []
    num_tokens_local_list = []
    latency_local_list = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_time = time.time()
            predictions = model(input_ids, attention_mask)
            end_time = time.time()
            labels = batch["labels"]
            num_tokens = batch["num_tokens"]

            labels_local_list.extend(labels)
            predict_local_list.extend(predictions)
            num_tokens_local_list.extend(num_tokens)
            latency_local_list.append(end_time - start_time)

    labels_local = torch.tensor(labels_local_list)
    predict_local = torch.tensor(predict_local_list)
    num_tokens_local = torch.tensor(num_tokens_local_list)
    latency_local = torch.tensor(latency_local_list)

    labels_global = [torch.empty_like(labels_local) for _ in range(world_size)] if local_rank == 0 else None
    predict_global = [torch.empty_like(predict_local) for _ in range(world_size)] if local_rank == 0 else None
    num_tokens_global = [torch.empty_like(num_tokens_local) for _ in range(world_size)] if local_rank == 0 else None
    latency_global = [torch.empty_like(latency_local) for _ in range(world_size)] if local_rank == 0 else None

    dist.gather(labels_local, labels_global, dst=0)
    dist.gather(predict_local, predict_global, dst=0)
    dist.gather(num_tokens_local, num_tokens_global, dst=0)
    dist.gather(latency_local, latency_global, dst=0)

    if local_rank == 0:
        label = torch.concat(labels_global)
        predict = torch.concat(predict_global)
        num_tokens = torch.concat(num_tokens_global)
        latency = torch.concat(latency_global)

        round_predict = torch.round(predict).type(torch.LongTensor)
        round_predict = torch.clip(round_predict, 0, num_classes - 1)
        l1 = nn.L1Loss()(predict, label.type_as(predict))
        mse = nn.MSELoss()(predict, label.type_as(predict))
        acc = evaluate.load("accuracy").compute(predictions=round_predict, references=label.type_as(round_predict))["accuracy"]
        f1 = evaluate.load("f1", average="macro").compute(
            predictions=round_predict, references=label.type_as(round_predict), average="macro"
        )["f1"]
        precision = evaluate.load("precision", average="macro").compute(
            predictions=round_predict, references=label.type_as(round_predict), average="macro"
        )["precision"]
        recall = evaluate.load("recall", average="macro").compute(
            predictions=round_predict, references=label.type_as(round_predict), average="macro"
        )["recall"]
        latency_mean = torch.mean(latency)
        metrics_df = pd.DataFrame(
            {
                "Model": [result_path.split("/")[-1]],
                "L1": [l1.item()],
                "MSE": [mse.item()],
                "ACC": [acc],
                "F1": [f1],
                "Precision": [precision],
                "Recall": [recall],
                "Latency": [latency_mean.item()],
            }
        )
        output_df = pd.DataFrame(
            {"label": label, "round_predict": round_predict, "predict": predict, "num_tokens": num_tokens, "latency": latency}
        )
        os.makedirs(result_path, exist_ok=True)
        metrics_df.to_csv(os.path.join(result_path, "metrics.csv"), index=False)
        output_df.to_csv(os.path.join(result_path, "output.csv"), index=False)
        print(metrics_df)
