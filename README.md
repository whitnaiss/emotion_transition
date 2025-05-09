# Emotion Transition Labeling for Enhanced Sentiment Classification in Large Language Models

This repository contains all the code, data, and instructions needed to reproduce experiments on sentiment classification enhanced with **emotion transition labels**, particularly in the context of mental health analysis on social media.

## Model Setup

Due to file size limitations, **you will need to manually download the following models** and place them into their respective directories:

- `EmoLLM-7B` — used for emotion transition labeling
- `bert-base-uncased`
- `llama-3.2-1b`
- `qwen2.5-1.5b`

Place each model into its corresponding folder inside this project (create the folders if needed).

---

## Folder Structure

### `early_work/`
Contains the original dataset and the **random sampling logic**. Due to GPU constraints, only **1,000 tweets** were sampled for training/testing. You are encouraged to use the full dataset if resources allow.

---

## Code Overview

### Data Preparation

- `data_create.py`  
  Prepares the initial data format and outputs:
  - `train_text_class.jsonl`
  - `test_text_class.jsonl`

### Emotion Labeling

- `emollm_label_infer.py`  
  Uses `EmoLLM-7B` to annotate the dataset with sentence-level **emotion transition labels**.  
  - Outputs: `test_text_class_new.jsonl`

---

### Fine-Tuning Scripts

#### LLaMA-3.2-1B

- `run_llama_sft_raw_lora.sh`  
  Fine-tunes on raw data (`train_text_class.jsonl`)  
  → saves model to `llama_raw_lora_model/`

- `run_llama_sft_emotion_lora.sh`  
  Fine-tunes on **emotion-labeled data**  
  → saves model to `llama_lora_emotion_model/`

#### Qwen2.5-1.5B

- `run_qwen_sft_raw_lora.sh`  
  Fine-tunes on raw data  
  → output in `qwen_raw_lora_model/`

- `run_qwen_sft_emotion_lora.sh`  
  Fine-tunes on emotion-enhanced data  
  → output in `qwen_lora_emotion_model/`

#### BERT

- `bert_raw.py`  
  Fine-tunes `bert-base-uncased` on raw data  
  → results in `bert_raw_results.jsonl`

- `bert_emotion.py`  
  Fine-tunes on emotion-enhanced data  
  → results in `bert_emotion_results.jsonl`

---

### Inference Scripts

#### LLaMA-3.2-1B

- `run_llama_raw_infer.sh`  
  Performs inference using the raw model

- `run_llama_emotion_infer.sh`  
  Inference using the emotion-labeled model

#### Qwen2.5-1.5B

- `run_qwen_raw_infer.sh`  
- `run_qwen_emotion_infer.sh`  

---

## Evaluation

- `res_analysis.py`  
  Computes and prints key evaluation metrics:
  - Accuracy
  - Recall
  - F1 Score

All results will be printed to the console. BERT-specific outputs are saved as:
- `bert_raw_results.jsonl`
- `bert_emotion_results.jsonl`

---

## Notes

- This project was designed with resource constraints in mind; all experiments were run with limited GPU memory.
- The use of **emotion transition annotations** significantly improves classification performance, especially for encoder-based models such as BERT.

---

## Citation & Acknowledgment

If you use this work or adapt it, please cite appropriately and acknowledge EmoLLM and the open-source LLMs used (LLaMA, Qwen, BERT).

