import json
import pandas as pd 
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


predict_file = "/root/autodl-tmp/code/lora_model/v1-20250503-170930/checkpoint-50/infer_result/20250503-171234.jsonl"
predict_emotion_file = "/root/autodl-tmp/code/lora_model/v2-20250503-174039/checkpoint-50/infer_result/20250503-174614.jsonl"

predict_list = [predict_file, predict_emotion_file]
for index, file_path in  enumerate(predict_list):
    if index == 0:
        print("the result of the raw text model predict result ")
    else:
        print("the result of the emotion model predict result ")
    df = pd.read_json(file_path, lines=True)[['response', 'labels']]
    print(accuracy_score(df['labels'], df['response']))
    print("the f1 score is:", f1_score(df['labels'], df['response'], average='weighted'))
    print("the precision score is:", precision_score(df['labels'], df['response'], average='weighted'))
    print("the recall score is:", recall_score(df['labels'], df['response'], average='weighted'))
    print("the accuracy score is:", accuracy_score(df['labels'], df['response']))