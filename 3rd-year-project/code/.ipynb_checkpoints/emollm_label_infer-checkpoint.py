from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

model_name = "/root/autodl-tmp/Emollm-7b"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt_info = """
    You are an emotion classifier.
    Your task is to analyze the following text sentence by sentence.
    Each sentence must be labeled with EXACTLY ONE emotion from this list:
    [Aesthetic Experience, Anger, Anxiety, Compassion, Depression, Envy, Fright,
    Gratitude, Guilt, Happiness, Hope, Jealousy, Love, Pride, Relief, Sadness, Shame]
    If unclear, use 'Ambiguous'.
    The output must strictly adhere to excluding any other unrelated information.
"""


def get_emotion_result(data):
    """
    Get the emotion result for the given text.
    """
    res = []

    for sentence in data.split('.'):
        if len(sentence) < 3:
            continue
        messages = [
            {"role": "system", "content": prompt_info},
            {"role": "user", "content": sentence}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response.split('\n')[0].strip()
        # print(response)
        res.append(response)
    return "context: " + data + "  " + "Emotional Change:" + ';'.join(res)


train_df = pd.read_json('train_text_class.jsonl', lines=True)
train_df['query'] = train_df['query'].apply(lambda x: get_emotion_result(x))
train_df[['system', 'query', 'response']].to_json('train_text_class_new.jsonl', orient='records', lines=True, force_ascii=False)

test_df = pd.read_json('test_text_class.jsonl', lines=True)
test_df['query'] = test_df['query'].apply(lambda x: get_emotion_result(x))
test_df[['system', 'query', 'response']].to_json('test_text_class_new.jsonl', orient='records', lines=True, force_ascii=False)