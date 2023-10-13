#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning'
export model_name="longformer-base-4096"
export code="model_inference.py"
export batch_size=8
export test_date="07_23"

# export CUDA_VISIBLE_DEVICES=0
# python ../$code \
# --model_name $model_name \
# --batch_size $batch_size \
# --fp16 \
# --device cuda \
# --model_max_length 4096 \
# --test_date $test_date \
# --customized_model \
# --val_min_recall 0.90 &

# export CUDA_VISIBLE_DEVICES=0
# python ../$code \
# --model_name $model_name \
# --batch_size $batch_size \
# --fp16 \
# --device cuda \
# --model_max_length 4096 \
# --test_date $test_date \
# --customized_model \
# --val_min_recall 0.91 &

# export CUDA_VISIBLE_DEVICES=1
# python ../$code \
# --model_name $model_name \
# --batch_size $batch_size \
# --fp16 \
# --device cuda \
# --model_max_length 4096 \
# --test_date $test_date \
# --customized_model \
# --val_min_recall 0.92 &

# export CUDA_VISIBLE_DEVICES=1
# python ../$code \
# --model_name $model_name \
# --batch_size $batch_size \
# --fp16 \
# --device cuda \
# --model_max_length 4096 \
# --test_date $test_date \
# --customized_model \
# --val_min_recall 0.93 &

# export CUDA_VISIBLE_DEVICES=2
# python ../$code \
# --model_name $model_name \
# --batch_size $batch_size \
# --fp16 \
# --device cuda \
# --model_max_length 4096 \
# --test_date $test_date \
# --customized_model \
# --val_min_recall 0.94 &

export CUDA_VISIBLE_DEVICES=0
python ../$code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--test_date $test_date \
--customized_model \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=0
python ../$code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--test_date $test_date \
--customized_model \
--val_min_recall 0.96 &

export CUDA_VISIBLE_DEVICES=1
python ../$code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--test_date $test_date \
--customized_model \
--val_min_recall 0.97 &

export CUDA_VISIBLE_DEVICES=1
python ../$code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--test_date $test_date \
--customized_model \
--val_min_recall 0.98 &

export CUDA_VISIBLE_DEVICES=2
python ../$code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--test_date $test_date \
--customized_model \
--val_min_recall 0.99 &


# Wait for all commands to finish
wait

