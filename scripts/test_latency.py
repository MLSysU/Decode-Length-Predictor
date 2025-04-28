import os
import sys
import random
import torch
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from models import BertRegressionModel
from transformers import AutoTokenizer
from argparse import ArgumentParser

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from utils import (
    TestArgs,
    get_model_path,
    get_result_path,
)


def generate_random_input_ids(tokenizer: AutoTokenizer, batch_size=1, seq_length=512):
    # 获取词表中所有token的ID列表(排除特殊token)
    vocab_ids = list(range(tokenizer.vocab_size))
    special_ids = set(tokenizer.all_special_ids)
    regular_ids = [id for id in vocab_ids if id not in special_ids]

    input_ids = torch.zeros((batch_size, seq_length), dtype=torch.long)

    for i in range(batch_size):
        # [CLS]
        input_ids[i, 0] = tokenizer.cls_token_id

        # random fill
        middle_ids = random.choices(regular_ids, k=seq_length - 2)
        input_ids[i, 1 : seq_length - 1] = torch.tensor(middle_ids)

        # [SEP]
        input_ids[i, seq_length - 1] = tokenizer.sep_token_id

    return input_ids


@torch.inference_mode()
def test_latency(model_path: str):
    device = torch.device("cuda")

    # load model
    model = BertRegressionModel.from_pretrained(model_path).half()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = model.to(device)
    model.eval()

    # warmup
    input_ids = generate_random_input_ids(tokenizer).to(device)
    for _ in trange(10, desc="Warmup"):
        model(input_ids)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    latency_dict = {bs: 0.0 for bs in batch_sizes}

    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for bs in tqdm(batch_sizes, desc="Test Latency"):
        test_num = 10
        accum_latency = 0.0
        for _ in range(test_num):
            input_ids = generate_random_input_ids(tokenizer, bs, 512).to(device)
            start_event.record()
            model(input_ids)
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_time = start_event.elapsed_time(end_event)
            accum_latency += elapsed_time
        latency_dict[bs] = accum_latency / test_num
        tqdm.write(f"Batch Size {bs}: {latency_dict[bs]:.3f} ms")

    return latency_dict


def save_latency_chart(latency_dict, result_path):
    chart_path = os.path.join(result_path, "latency.png")

    batch_sizes = list(latency_dict.keys())
    latencies = list(latency_dict.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, latencies, marker="o")
    plt.xscale("log", base=2)  # Set x-axis to log2 scale
    plt.xticks(batch_sizes, [str(bs) for bs in batch_sizes])    
    plt.xlabel("Batch Size (log2 scale)")
    plt.ylabel("Latency (ms)")
    plt.title("Predictor's Latency")
    plt.grid(True)
    plt.savefig(chart_path)
    plt.close()
    print('Save latency chart to', chart_path)


def main():
    parser = ArgumentParser()
    args = TestArgs.add_cli_args(parser).parse_args()
    test_args = TestArgs.from_cli_args(args)

    # predict
    latency_dict = test_latency(get_model_path(root_path, test_args.dataset_type, test_args.llm, test_args.bert))

    # get path
    result_path = get_result_path(root_path, test_args.dataset_type, test_args.llm, test_args.bert)

    # save output
    save_latency_chart(latency_dict, result_path)


if __name__ == "__main__":
    main()
