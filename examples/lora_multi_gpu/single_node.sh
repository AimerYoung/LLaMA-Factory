#!/bin/bash

CUDA_VISIBLE_DEVICES=0,2 accelerate launch \
    --config_file ../accelerate/single_config.yaml \
    /home/zcl/LLaMA-Factory/src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path AI-ModelScope/phi-2 \
    --dataset math_problem \
    --dataset_dir /home/zcl/LLaMA-Factory/data \
    --template default \
    --finetuning_type lora \
    --lora_target Wqkv,fc1,fc2 \
    --output_dir /home/zcl/LLaMA-Factory/saves/phi-2-v2/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 1e-4 \
    --num_train_epochs 3.0 \
    --val_size 0.1 \
    --ddp_timeout 180000000 \
    --plot_loss \
    --fp16
