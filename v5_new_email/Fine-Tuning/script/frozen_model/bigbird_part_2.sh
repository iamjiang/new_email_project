#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning'
export model_name="bigbird-roberta-large"
export code="model_inference_frozen.py"
export batch_size=4
export test_date="07_23"


export CUDA_VISIBLE_DEVICES=0
python ../../$code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--test_date $test_date \
--customized_model \
--frozen_model \
--val_min_recall 0.94 &

export CUDA_VISIBLE_DEVICES=1
python ../../$code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--test_date $test_date \
--customized_model \
--frozen_model \
--val_min_recall 0.95 &

export CUDA_VISIBLE_DEVICES=2
python ../../$code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--test_date $test_date \
--customized_model \
--frozen_model \
--val_min_recall 0.96 &

export CUDA_VISIBLE_DEVICES=3
python ../../$code \
--model_name $model_name \
--batch_size $batch_size \
--fp16 \
--device cuda \
--model_max_length 4096 \
--test_date $test_date \
--customized_model \
--frozen_model \
--val_min_recall 0.97 &

# Wait for all commands to finish
wait

