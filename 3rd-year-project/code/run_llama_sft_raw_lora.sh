model_path="/root/autodl-tmp/llama-3.2-1b"
data_path="./train_text_class.jsonl"
val_path="./test_text_class.jsonl"
out_path="./llama_raw_lora_model"

swift sft \
    --model ${model_path} \
    --model_type llama3_2 \
    --dataset ${data_path} \
    --val_dataset ${val_path} \
    --num_train_epochs 1 \
    --save_only_model true \
    --save_total_limit 2 \
    --per_device_train_batch_size 2 \
    --dataloader_num_workers 2 \
    --per_device_eval_batch_size 2 \
    --train_type lora \
    --learning_rate 6e-5 \
    --output_dir ${out_path} \
    --lora_rank 4 \
    --lora_alpha 32 
