#!/bin/bash

# "--stage sft": 指定训练阶段为指令监督微调
# "--do_train": 执行训练操作
# "--model_name_or_path": 指定使用的预训练模型的名称或路径。
# "--dataset": 指定使用的数据集名称，多个数据集之间用逗号分隔。
# "--dataset_dir": 指定数据集文件的目录。
# "--template": 指定使用的模板名称
# "--finetuning_type"：指定微调类型(lora)
# "--lora_target"：指定LoRA的目标参数
# "--output_dir"：指定训练结果输出的目录。
# "--overwrite_cache"：覆盖缓存文件。
# --overwrite_output_dir：覆盖输出目录。
# --cutoff_len 1024：指定输入文本的截断长度。
# --preprocessing_num_workers 16：指定预处理的工作线程数。
# --per_device_train_batch_size 1：指定每个设备的训练批次大小。
# --per_device_eval_batch_size 1：指定每个设备的评估批次大小。
# --gradient_accumulation_steps 8：指定梯度累积的步数。
# --lr_scheduler_type cosine：指定学习率调度器类型为余弦退火。
# --logging_steps 10：指定记录日志的步数。
# --warmup_steps 20：指定学习率预热的步数。
# --save_steps 100：指定保存模型的步数。
# --eval_steps 100：指定评估模型的步数。
# --evaluation_strategy steps：指定评估策略为按步数进行评估。
# --load_best_model_at_end：在训练结束时加载最佳模型。
# --learning_rate 5e-5：指定学习率的初始值。
# --num_train_epochs 3.0：指定训练的轮数。
# --max_samples 3000：指定最大训练样本数。
# --val_size 0.1：指定验证集的比例。
# --plot_loss：绘制训练损失曲线。
# --fp16：使用混合精度训练（16位浮点数）。

# CUDA_VISIBLE_DEVICES=0 python ../../src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --dataset alpaca_gpt4_en,glaive_toolcall \
#     --dataset_dir ../../data \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir ../../saves/LLaMA2-7B/lora/sft \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 1024 \
#     --preprocessing_num_workers 16 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --warmup_steps 20 \
#     --save_steps 100 \
#     --eval_steps 100 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 5e-5 \
#     --num_train_epochs 3.0 \
#     --max_samples 3000 \
#     --val_size 0.1 \
#     --plot_loss \
#     --fp16


# export USE_MODELSCOPE_HUB=1
# CUDA_VISIBLE_DEVICES=0 python /home/zcl/LLaMA-Factory/src/train_bash.py \
#     --stage sft \
#     --do_train \
#     --model_name_or_path AI-ModelScope/phi-2 \
#     --dataset alpaca_gpt4_zh \
#     --dataset_dir /home/zcl/LLaMA-Factory/data \
#     --template default \
#     --finetuning_type lora \
#     --lora_target q_proj,v_proj \
#     --output_dir /home/zcl/LLaMA-Factory/saves/phi-2/lora/sft \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --lr_scheduler_type cosine \
#     --logging_steps 10 \
#     --warmup_steps 20 \
#     --save_steps 1000 \
#     --evaluation_strategy steps \
#     --load_best_model_at_end \
#     --learning_rate 2e-5 \
#     --num_train_epochs 2.0 \
#     --max_samples 3000 \
#     --val_size 0.1 \
#     --plot_loss \
#     --fp16



export USE_MODELSCOPE_HUB=1
CUDA_VISIBLE_DEVICES=1 python /home/zcl/LLaMA-Factory/src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path AI-ModelScope/phi-2 \
    --dataset math_problem \
    --dataset_dir /home/zcl/LLaMA-Factory/data \
    --template default \
    --finetuning_type lora \
    --lora_target Wqkv \
    --output_dir /home/zcl/LLaMA-Factory/saves/phi-2-new/lora/sft \
    --overwrite_cache \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --max_samples 3000 \
    --val_size 0.1 \
    --plot_loss \
    --fp16