lora_path_emotion="/root/autodl-tmp/code/llama_lora_emotion_model/v0-20250504-102110/checkpoint-50"
val_path="./test_text_class_new.jsonl"

swift infer \
    --adapter ${lora_path_emotion} \
    --val_dataset ${val_path} \
    --temperature 0 \
