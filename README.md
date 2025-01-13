# Decode-Length-Predictor

**Decode-Length-Predictor** is a machine learning model designed to predict the length of decoded tokens based on an input prompt for a specific Large Language Model (LLM). This project is inspired by the research presented in [Power-aware Deep Learning Model Serving with μ-Serve](https://www.usenix.org/conference/atc24/presentation/qiu).

---

## Table of Contents
- [Decode-Length-Predictor](#decode-length-predictor)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
    - [Installation](#installation)
    - [Download Dataset](#download-dataset)
    - [Preprocessing](#preprocessing)
    - [Training](#training)
    - [Testing](#testing)
  - [Evaluation](#evaluation)

---
## Quick Start

### Installation
To set up the environment, install the required dependencies by running:
```bash
pip install -r requirements.txt
```

### Download Dataset
Download the dataset (e.g., ShareGPT) and store it in the appropriate directory:
```bash
mkdir -p data/shareGPT
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json -O data/shareGPT/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Preprocessing
Generate output sequences from the LLM based on the dataset, and preprocess the data for training, validation, and testing:
```bash
./run_preprocess.sh
```

### Training
Train the model using the preprocessed dataset:
```bash
./run_train.sh
```

### Testing
Evaluate the trained model to obtain results and performance metrics:
```bash
./run_test.sh
```

---

## Evaluation
Evaluation results will be available soon.

---
