import os
import json
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from config import *


if __name__ == "__main__":
    with open("data/ShareGPT_V3_unfiltered_cleaned_split.json", "r") as file:
        shareGPT_raw = json.load(file)

    # filter the first round prompt
    conversations = []
    for conversation in shareGPT_raw:
        if conversation["id"].endswith("_0") and len(conversation["conversations"]) > 0:
            conversation = conversation["conversations"][0]
            if conversation["from"] == "human":
                conversations.append(conversation["value"])
    del shareGPT_raw

    # load model
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    sampling_params = SamplingParams(max_tokens=1e5)
    llm = LLM(llm_path, tensor_parallel_size=1)

    # inference
    responses = []
    batch_size = 8
    length = len(conversations)
    for i in tqdm(range(0, length, batch_size)):
        end = min(i + batch_size, length)
        messages = [[{"role": "user", "content": conversation}] for conversation in conversations[i:end]]
        formatted_prompts = [
            tokenizer.apply_chat_template(conversation=message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        outputs = llm.generate(formatted_prompts, sampling_params)
        for output in outputs:
            responses.append(output.outputs[0].text)

    # save dataset
    df = pd.DataFrame(
        {
            "prompt": conversations,
            "response": responses,
        }
    )
    dataset = Dataset.from_pandas(df)
    save_path = os.path.join(dataset_path, "raw")
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
