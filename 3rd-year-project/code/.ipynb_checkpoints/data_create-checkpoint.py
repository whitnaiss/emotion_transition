import pandas as pd 
from sklearn.model_selection import train_test_split


prompt_info = f'''You are a language expert in text processing. Please, based on the description of the text, identify whether the sentiment expressed in the text is positive or negative!'''


train_df = pd.read_csv('train_all.csv')
test_df = pd.read_csv('test.csv')


train_df['query'] = train_df['text']
train_df['system'] = prompt_info
train_df['response'] = train_df['label']

test_df['query'] = test_df['text']
test_df['system'] = prompt_info
test_df['response'] = test_df['label']


train_df[['system', 'query', 'response']].to_json('train_text_class.jsonl', orient='records', lines=True, force_ascii=False)
test_df[['system', 'query', 'response']].to_json('test_text_class.jsonl', orient='records', lines=True, force_ascii=False)
