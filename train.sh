#! /bin/bash


CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train.py \
    --model_name Qwen2.5Omni \
    --model_path /home/jinzr/nfs/projects/qwen-audio-ft/Qwen2.5-Omni-7B \
    --save_path ./exp/Qwen2.5Omni \
    --learning_rate 3e-5 \
    --evaluation_steps 50 \
    --logging_steps 10 \
    --warmup_step_frac 0.2 \
    --loss xent \
    --target_label severe_osa \
    --num_class 2 \
    --lora_dropout 0.2 \
    --num_train_epochs 10 