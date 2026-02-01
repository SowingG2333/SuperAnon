#!/bin/bash

# 训练Anonymizer模型
python src/train_anonymizer.py \
    --model_name_or_path "/root/autodl-tmp/hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
    --attacker_model_path "model/attacker/checkpoint-402" \
    --utility_model_path "/root/autodl-tmp/hf_cache/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181" \
    --train_file "data/train.jsonl" \
    --output_dir "model/anonymizer" \
    --privacy_weight 1.0 \
    --utility_weight 1.0 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --num_generations 8 \
    --max_prompt_length 384 \
    --max_completion_length 384 \
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