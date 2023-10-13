#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v3_new_email/Fine-Tuning'
export model_name="deberta-v2-xlarge"
export code="model_inference.py"
export batch_size=2

export CUDA_VISIBLE_DEVICES=0
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 512 \
--val_min_recall 0.94 &

export CUDA_VISIBLE_DEVICES=0
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 512 \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=5
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 512 \
--val_min_recall 0.96 &

export CUDA_VISIBLE_DEVICES=1
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 512 \
--val_min_recall 0.97 &

export CUDA_VISIBLE_DEVICES=4
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 512 \
--val_min_recall 0.97 &

export CUDA_VISIBLE_DEVICES=5
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 512 \
--val_min_recall 0.98 &

export CUDA_VISIBLE_DEVICES=4
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 512 \
--val_min_recall 0.99 &


# Wait for all commands to finish
wait
