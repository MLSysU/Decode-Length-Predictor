import os
import sys
import math
import torch
import torch.utils
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import datasets
from argparse import ArgumentParser
from tqdm import tqdm
from models import BertRegressionModel
from utils import (
    TrainArgs,
    Parallel,
    get_dataset_path,
    load_cls_thresholds,
    get_cls_thresholds_path,
    get_model_path,
)


root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)


def load_data(args: TrainArgs, parallel: Parallel):
    # check parameters
    assert (
        args.train_bs % parallel.world_size == 0
    ), f"Train batch size {args.train_bs} should be divisible by world size {parallel.world_size}"

    # tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # load dataset
    dataset_path = get_dataset_path(root_path, args.dataset_type, args.llm, args.bert)
    train_dataset = datasets.load_from_disk(os.path.join(dataset_path, "train"))
    valid_dataset = datasets.load_from_disk(os.path.join(dataset_path, "validation"))

    if parallel.is_first_rank():
        print("Trainning data size: ", len(train_dataset))
        print("Validation data size: ", len(valid_dataset))

    # sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)

    # dataloader
    data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)
    train_micro_bs = args.train_bs // parallel.world_size
    train_dataloader = DataLoader(train_dataset, batch_size=train_micro_bs, collate_fn=data_collator, sampler=train_sampler)
    validate_dataloader = DataLoader(valid_dataset, batch_size=args.validate_bs, collate_fn=data_collator, sampler=valid_sampler)
    return train_dataloader, validate_dataloader


def init_model(args: TrainArgs, parallel: Parallel):
    cls_thresholds = load_cls_thresholds(get_cls_thresholds_path(root_path, args.dataset_type, args.llm, args.bert))
    model = BertRegressionModel.from_bert(args.bert, cls_thresholds, args.hidden_dim).to(parallel.device)
    model = DDP(model, device_ids=[parallel.rank], output_device=parallel.rank, find_unused_parameters=True)
    return model


def validate(
    epoch: int,
    model: DDP,
    validate_dataloader: DataLoader,
    parallel: Parallel,
):
    device = parallel.device
    # validation
    model.eval()
    labels_local_list = []
    predict_local_list = []

    # validation metrics
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    with torch.no_grad():
        for batch in validate_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            predict = model(input_ids=input_ids, attention_mask=attention_mask)
            labels_local_list.extend(labels)
            predict_local_list.extend(predict)

    labels_local = torch.tensor(labels_local_list).to(device)
    predict_local = torch.tensor(predict_local_list).to(device)

    labels = parallel.gather(labels_local, dst=0, dim=0)
    predict = parallel.gather(predict_local, dst=0, dim=0)

    if parallel.is_first_rank():
        cls_predict = model.module.get_cls(predict)
        l1_loss = l1_loss(predict, labels.type_as(predict))
        mse_loss = mse_loss(predict, labels.type_as(predict))

        labels = labels.cpu()
        cls_predict = cls_predict.cpu()
        acc = accuracy_score(labels, cls_predict)
        f1 = f1_score(labels, cls_predict, average="macro", zero_division=0)
        precision = precision_score(labels, cls_predict, average="macro", zero_division=0)
        recall = recall_score(labels, cls_predict, average="macro", zero_division=0)
        tqdm.write(
            f"Validation metrics for epoch {epoch}:\n"
            f"\tL1 error: {l1_loss:.3f}\n"
            f"\tMSE: {mse_loss:.3f}\n"
            f"\tAccuracy: {acc:.3f}\n"
            f"\tF1: {f1:.3f}\n"
            f"\tPrecision: {precision:.3f}\n"
            f"\tRecall: {recall:.3f}\n"
        )


def train(
    model: DDP,
    train_dataloader: DataLoader,
    validate_dataloader: DataLoader,
    args: TrainArgs,
    parallel: Parallel,
):
    device = parallel.device
    criterion = nn.MSELoss().to(device)
    bert_params = list(model.module.bert.parameters())
    fc_params = list(model.module.cls.parameters()) + list(model.module.fc1.parameters()) + list(model.module.fc2.parameters())
    optimizer = torch.optim.AdamW([{"params": bert_params, "lr": args.bert_lr}, {"params": fc_params, "lr": args.lr}])

    # train
    num_steps = args.epoch * math.ceil(len(train_dataloader.dataset) / args.train_bs)
    lr_scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps,
    )

    for epoch in tqdm(range(args.epoch)) if parallel.is_first_rank() else range(args.epoch):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        if epoch == (args.epoch // 2):
            for param in model.module.bert.parameters():
                param.requires_grad = False

        trainning_loss_sum = torch.tensor(0.0).to(device)
        trainning_count = torch.tensor(0.0).to(device)
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(output, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            trainning_loss_sum += loss
            trainning_count += 1

        # aggregate loss
        parallel.reduce(trainning_loss_sum, dst=0, op=dist.ReduceOp.SUM)
        parallel.reduce(trainning_count, dst=0, op=dist.ReduceOp.SUM)
        if parallel.is_first_rank():
            avg_trainning_loss = trainning_loss_sum.item() / trainning_count.item()
            tqdm.write(f"Training loss for epoch {epoch}: {avg_trainning_loss}")

        validate(epoch, model, validate_dataloader, parallel)


def save_model(model: DDP, args: TrainArgs, parallel: Parallel):
    # save model
    if parallel.is_first_rank():
        model_path = get_model_path(root_path, args.dataset_type, args.llm, args.bert)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(args.bert)
        tokenizer.save_pretrained(model_path)
        model.module.save_pretrained(model_path)
        print(f"Saving model to: {model_path}")


def main():
    parallel = Parallel()
    parser = ArgumentParser()
    args = TrainArgs.add_cli_args(parser).parse_args()
    train_args = TrainArgs.from_cli_args(args)

    train_dataloader, validate_dataloader = load_data(train_args, parallel)

    model = init_model(train_args, parallel)
    train(model, train_dataloader, validate_dataloader, train_args, parallel)

    save_model(model, train_args, parallel)

    parallel.destroy()


if __name__ == "__main__":
    main()
