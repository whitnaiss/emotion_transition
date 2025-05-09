lora_path="/root/autodl-tmp/code/llama_raw_lora_model/v11-20250504-170925/checkpoint-50"
val_path="./test_text_class.jsonl"

swift infer \
    --adapter ${lora_path} \
    --val_dataset ${val_path} \
    --temperature 0 \
