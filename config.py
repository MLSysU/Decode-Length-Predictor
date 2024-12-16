# bert_model = "bert-tiny"
bert_model = "bert-base-uncased"
llm_model = "llama-2-7b-chat"

# 多轮对话
is_multi_round = False

# shareGPT数据
is_shareGPT = True

# 均匀分布
is_uniform = True

lmsys_path = "/fastdata/zhengzy/dataset/lmsys-chat-1m"
bert_path = "/fastdata/zhengzy/model/" + bert_model

dataset_path = (
    "./data/"
    + llm_model
    + ("_multi" if is_multi_round else "_first")
    + ("_uniform" if is_uniform else "_nonuniform")
    + ("_shareGPT" if is_shareGPT else "")
)
model_path = (
    "./model/"
    + llm_model
    + ("_multi" if is_multi_round else "_first")
    + ("_uniform" if is_uniform else "_nonuniform")
    + ("_shareGPT" if is_shareGPT else "")
    + "/"
    + bert_model
    + ".pth"
)
result_path = (
    "./result/"
    + llm_model
    + ("_multi" if is_multi_round else "_first")
    + ("_uniform" if is_uniform else "_nonuniform")
    + ("_shareGPT" if is_shareGPT else "")
    + "_"
    + bert_model
)

model_dict = {
    "vicuna-13b": {
        "llm_path": "/fastdata/zhengzy/model/vicuna-13b-v1.3",
        "first_round_cls_thresholds": [53, 185, 366, 897, float("inf")],
        "multi_round_cls_thresholds": [58, 147, 280, 499, float("inf")],
    },
    "llama-13b": {
        "llm_path": "/fastdata/zhengzy/model/llama-13b",
        "first_round_cls_thresholds": [11, 28, 88, 516, float("inf")],
        "multi_round_cls_thresholds": [11, 22, 62, 454, float("inf")],
    },
    "llama-2-13b-chat": {
        "llm_path": "/fastdata/zhengzy/model/Llama-2-13b-chat-hf",
        "first_round_cls_thresholds": [42, 141, 294, 503, float("inf")],
        "multi_round_cls_thresholds": [58, 147, 280, 499, float("inf")],
    },
    "llama-2-7b-chat": {
        "llm_path": "/fastdata/zhengzy/model/Llama-2-7b-chat-hf",
        "first_round_cls_thresholds": [42, 141, 294, 503, float("inf")],
        "multi_round_cls_thresholds": [58, 147, 280, 499, float("inf")],
    },
    "chatglm-6b": {
        "llm_path": "/fastdata/zhengzy/model/chatglm-6b",
        "first_round_cls_thresholds": [42, 141, 294, 503, float("inf")],
        "multi_round_cls_thresholds": [58, 147, 280, 499, float("inf")],
    },
}
uniform_thresholds = [250, 500, 750, 1000, float("inf")]

llm_path = model_dict[llm_model]["llm_path"]
cls_thresholds = (
    uniform_thresholds
    if is_uniform
    else (
        model_dict[llm_model]["multi_round_cls_thresholds"]
        if is_multi_round
        else model_dict[llm_model]["first_round_cls_thresholds"]
    )
)
num_classes = len(cls_thresholds)

all_model = [
    "vicuna-13b",
    "wizardlm-13b",
    "palm-2",
    "llama-13b",
    "llama-2-13b-chat",
    "llama-2-7b-chat",
    "chatglm-6b",
    "koala-13b",
    "claude-instant-1",
    "oasst-pythia-12b",
    "alpaca-13b",
    "mpt-7b-chat",
    "vicuna-7b",
    "dolly-v2-12b",
    "mpt-30b-chat",
    "fastchat-t5-3b",
    "claude-1",
    "gpt-4",
    "vicuna-33b",
    "guanaco-33b",
    "RWKV-4-Raven-14B",
    "stablelm-tuned-alpha-7b",
    "gpt-3.5-turbo",
    "claude-2",
    "gpt4all-13b-snoozy",
]
