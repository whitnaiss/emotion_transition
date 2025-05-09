import json
import pandas as pd 
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


qwen_predict_raw = "/root/autodl-tmp/code/qwen_raw_lora_model/v0-20250504-102331/checkpoint-50/infer_result/20250504-102601.jsonl"
qwen_predict_emotion = "/root/autodl-tmp/code/qwen_lora_emotion_model/v0-20250504-102400/checkpoint-50/infer_result/20250504-102615.jsonl"

llama_predict_raw = "/root/autodl-tmp/code/llama_raw_lora_model/v11-20250504-170925/checkpoint-50/infer_result/20250504-171055.jsonl"
llama_predict_emtion = "/root/autodl-tmp/code/llama_lora_emotion_model/v0-20250504-102110/checkpoint-50/infer_result/20250504-102236.jsonl"

bert_predict_raw = "/root/autodl-tmp/code/bert_raw_results.jsonl"
bert_predict_emotion = "/root/autodl-tmp/code/bert_emotion_results.jsonl"

predict_list = [qwen_predict_raw, qwen_predict_emotion, llama_predict_raw, llama_predict_emtion, bert_predict_raw, bert_predict_emotion]
for index, file_path in  enumerate(predict_list):
    print("the mode predict is: ", file_path.split('/')[4])
    df = pd.read_json(file_path, lines=True)[['response', 'labels']]
    print("the f1 score is:", round(f1_score(df['labels'], df['response'], average='weighted'), 4))
    print("the precision score is:", round(precision_score(df['labels'], df['response'], average='weighted'), 4))
    print("the recall score is:", round(recall_score(df['labels'], df['response'], average='weighted'), 4))
    print("the accuracy score is:", round(accuracy_score(df['labels'], df['response']), 4))
    print("\n")