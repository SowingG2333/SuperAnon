#!/bin/bash

# 训练Attacker模型
python src/train_attacker.py \
    --model_name_or_path "Qwen/Qwen3-1.7B" \
    --train_file "data/attacker_train.jsonl" \
    --output_dir "model/attacker" \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --bf16 True \
    --max_seq_length 1024
