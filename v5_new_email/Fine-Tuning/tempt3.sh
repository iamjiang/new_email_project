#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v3_new_email/Fine-Tuning'
export model_name="bigbird-roberta-large"
export code="model_inference.py"
export batch_size=4

export CUDA_VISIBLE_DEVICES=1
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.90&

export CUDA_VISIBLE_DEVICES=2
python $code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--val_min_recall 0.91 &


# Wait for all commands to finish
wait