#!/bin/bash

# 训练Anonymizer模型
python src/train_anonymizer.py \
    --model_name_or_path "Qwen/Qwen3-1.7B" \
    --attacker_model_path "model/checkpoint-268" \
    --utility_model_path "BAAI/bge-m3" \
    --train_file "data/train.jsonl" \
    --output_dir "model/anonymizer" \
    --privacy_weight 1.0 \
    --utility_weight 1.0 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 4 \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --num_train_epochs 1 \
    --max_steps 500 \
    --logging_steps 1 \
    --save_steps 50 \
    --bf16 True \
    --report_to "wandb" \
    --remove_unused_columns False \
    --temperature 0.6 \
    --top_p 0.95 \
    --top_k 20
