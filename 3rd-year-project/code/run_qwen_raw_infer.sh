lora_path="/root/autodl-tmp/code/qwen_raw_lora_model/v0-20250504-102331/checkpoint-50"
val_path="./test_text_class.jsonl"

swift infer \
    --adapter ${lora_path} \
    --val_dataset ${val_path} \
    --temperature 0 \
