import openai
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dotenv import load_dotenv
from collections import defaultdict
import itertools

# 加载 OpenAI API 密钥
load_dotenv()
openai.api_key = os.getenv("sk-proj-aevLzNnmxNwG-J11Z_pzTNJ2-aH48e_nkRDPXuxU7JPaT3h8B4w0DAy__Gu8Rn1YLecZ-")

# 情感类别
EMOTION_CATEGORIES = ["Happiness", "Sadness", "Anger", "Fear", "Surprise", "Neutral", "Ambiguous"]

def split_text_into_sentences(text):
    """ 使用正则表达式拆分文本为句子 """
    sentences = re.split(r'[.!?\n]+', text)
    return [s.strip() for s in sentences if s.strip()]

def annotate_text(text, text_id):
    """ 调用 OpenAI API 进行情感分析 """
    sentences = split_text_into_sentences(text)
    prompt = f"""
    You are an expert in emotion detection. Analyze the following text sentence by sentence and assign an emotion from the given categories to each sentence.

    Emotion categories: {", ".join(EMOTION_CATEGORIES)}

    For the given text, respond in the format:
    ID: {text_id} | Label: [Emotion1][Emotion2][Emotion3]...

    Each emotion in the response should correspond to a sentence in the text in the same order the sentences appear.

    Text:
    {text}
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=100
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error processing ID {text_id}: {e}")
        return "Error"

def process_dataset(input_csv, output_csv):
    """ 处理数据集，调用 OpenAI API 并保存结果 """
    df = pd.read_csv(input_csv)

    if 'text' not in df.columns:
        print(f"Error: 'text' column not found in {input_csv}")
        return
    
    df['sentence_emotions'] = df.apply(lambda row: annotate_text(row['text'], row['id']), axis=1)
    
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Annotation complete. Results saved to {output_csv}")

# 处理 D1 和 D2
process_dataset("mixed_dataset/train/D1.csv", "mixed_dataset/train/annotated_D1.csv")
process_dataset("mixed_dataset/train/D2.csv", "mixed_dataset/train/annotated_D2.csv")
