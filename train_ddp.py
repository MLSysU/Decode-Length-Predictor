import datasets
import torch.utils
import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding
import evaluate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
from config import *
from model import BertClassificationModel, BertRegressionModel


dist.init_process_group(backend="gloo")
local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device(f"cuda:{local_rank}")

num_epochs = 6
train_batch_size = 64
valid_batch_size = 64
train_data_len = 0
first_lr = 1e-5
second_lr = 1e-4

if __name__ == "__main__":
    # tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # dataset
    train_dataset = datasets.load_from_disk(os.path.join(dataset_path, "train"))
    valid_dataset = datasets.load_from_disk(os.path.join(dataset_path, "validation"))

    if local_rank == 0:
        print("Trainning data size: ", len(train_dataset))
        print("Validation data size: ", len(valid_dataset))

    train_data_len = len(train_dataset)

    # dataloader
    data_collator = DataCollatorWithPadding(tokenizer=bert_tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=data_collator, sampler=train_sampler)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, collate_fn=data_collator, sampler=valid_sampler)

    # model
    model = BertRegressionModel(bert_path, hidden_dim=128).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=first_lr)

    # train
    num_steps = num_epochs * math.ceil(train_data_len / (train_batch_size * world_size))
    lr_scheduler = transformers.get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps,
    )

    training_loss_list = []
    validation_loss_list = []

    if local_rank == 0:
        writer = SummaryWriter()

    for epoch in tqdm(range(num_epochs)) if local_rank == 0 else range(num_epochs):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        if epoch == (num_epochs // 2):
            for param in model.module.bert.parameters():
                param.requires_grad = False
            for param_group in optimizer.param_groups:
                param_group["lr"] = second_lr

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
        dist.reduce(trainning_loss_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(trainning_count, dst=0, op=dist.ReduceOp.SUM)
        if local_rank == 0:
            avg_trainning_loss = trainning_loss_sum.item() / trainning_count.item()
            writer.add_scalar("Loss/train", avg_trainning_loss, epoch)
            print(f"Training loss for epoch {epoch}: {avg_trainning_loss}")
            training_loss_list.append(avg_trainning_loss)

        # validation
        model.eval()
        l1_loss = nn.L1Loss()
        mse_loss = nn.MSELoss()
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1", average="macro")
        precision_metric = evaluate.load("precision", average="macro")
        recall_metric = evaluate.load("recall", average="macro")

        labels_local_list = []
        predict_local_list = []

        with torch.no_grad():
            for batch in valid_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                predict = model(input_ids=input_ids, attention_mask=attention_mask)
                labels_local_list.extend(labels)
                predict_local_list.extend(predict)

        labels_local = torch.tensor(labels_local_list)
        predict_local = torch.tensor(predict_local_list)

        labels_global = [torch.empty_like(labels_local) for _ in range(world_size)] if local_rank == 0 else None
        predict_global = [torch.empty_like(predict_local) for _ in range(world_size)] if local_rank == 0 else None

        dist.gather(labels_local, gather_list=labels_global, dst=0)
        dist.gather(predict_local, gather_list=predict_global, dst=0)
        if local_rank == 0:
            labels = torch.concat(labels_global, dim=0)
            predict = torch.concat(predict_global, dim=0)
            round_predict = torch.round(predict).type(torch.LongTensor)
            round_predict = torch.clip(round_predict, 0, num_classes - 1)
            l1_loss = l1_loss(predict, labels.type_as(predict))
            mse_loss = mse_loss(predict, labels.type_as(predict))
            acc = accuracy_metric.compute(predictions=round_predict, references=labels.type_as(round_predict))
            f1 = f1_metric.compute(predictions=round_predict, references=labels.type_as(round_predict), average="macro")
            precision = precision_metric.compute(
                predictions=round_predict, references=labels.type_as(round_predict), average="macro"
            )
            recall = recall_metric.compute(predictions=round_predict, references=labels.type_as(round_predict), average="macro")
            print(
                f"Validation metrics for epoch {epoch}: L1 error: {l1_loss:.4f} MSE: {mse_loss:.4f} Accuracy: {acc['accuracy']:.4f} F1: {f1['f1']:.4f} Precision: {precision['precision']:.4f} Recall: {recall['recall']:.4f}"
            )
            validation_loss_list.append(mse_loss)

    if local_rank == 0:
        writer.flush()
        writer.close()
        cur_dir = os.path.dirname(__file__)
        train_dir = os.path.join(cur_dir, "train")
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        date_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(os.path.join(train_dir, date_time + ".txt"), "w") as f:
            f.write("Training loss:\n")
            for loss in training_loss_list:
                f.write(str(loss) + "\t")
            f.write("\nValidation loss:\n")
            for loss in validation_loss_list:
                f.write(str(loss) + "\t")
            f.write("\n")

    # save model
    if local_rank == 0:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.module.state_dict(), model_path)
        print(f"Model saved. {model_path}")
