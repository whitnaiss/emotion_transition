import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os
import time

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SentimentDataset(Dataset):
    def __init__(self, reviews, labels, tokenizer, max_length=128):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        label = self.labels[idx]
        # Map 'positive' to 1, 'negative' to 0
        if isinstance(label, str):
            label_mapped = 1 if label.lower() == 'positive' else 0
        else:
            label_mapped = int(label)
        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros(self.max_length, dtype=torch.long)).flatten(),
            'labels': torch.tensor(label_mapped, dtype=torch.long),
            'text': review
        }

def train_and_evaluate(model_name='/root/autodl-tmp/bert-base-uncased', num_labels=2, learning_rate=5e-6, epochs=1, batch_size=16):
    # Load dataset
    train_df = pd.read_json('train_text_class_new.jsonl', lines=True)   
    test_df = pd.read_json('test_text_class_new.jsonl', lines=True)

    train_df['response'] = train_df['response'].apply(lambda x: 1 if str(x).lower() == 'positive' else 0)
    test_df['response'] = test_df['response'].apply(lambda x: 1 if str(x).lower() == 'positive' else 0)

    train_text = train_df['query'].tolist()
    train_labels = train_df['response'].tolist()
    test_text = test_df['query'].tolist()
    test_labels = test_df['response'].tolist()


    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    
    train_dataset = SentimentDataset(train_text, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_text, test_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds, train_true = [], []
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(true)
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_true, train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(train_true, train_preds, average='binary', zero_division=0)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
    
    # Evaluation on test set
    model.eval()
    test_preds, test_true, test_texts_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            ).logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true = labels.cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(true)
            test_texts_list.extend(batch['text'])
    test_acc = accuracy_score(test_true, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_true, test_preds, average='binary', zero_division=0)
    print(f"Test - Acc: {test_acc:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
    
    # Save test results to JSONL
    label_map = {0: 'negative', 1: 'positive'}
    with open('bert_emotion_results.jsonl', 'w', encoding='utf-8') as f:
        for text, label, pred in zip(test_texts_list, test_true, test_preds):
            item = {
                'text': text,
                'labels': label_map[label],
                'response': label_map[pred]
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print('Test results saved to test_results.jsonl')

def main():
    train_and_evaluate()

if __name__ == "__main__":
    main() 