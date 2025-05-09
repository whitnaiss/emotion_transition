lora_path="/root/autodl-tmp/code/raw_lora_model/v1-20250503-170930/checkpoint-50"
lora_path_emotion="/root/autodl-tmp/code/emotion_lora_model/v2-20250503-174039/checkpoint-50"
val_path="./test_text_class_new.jsonl"

swift infer \
    --adapter ${lora_path_emotion} \
    --val_dataset ${val_path} \
    --temperature 0 \
