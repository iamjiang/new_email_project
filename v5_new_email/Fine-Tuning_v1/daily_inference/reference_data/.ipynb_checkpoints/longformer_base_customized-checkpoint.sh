#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/'
export path='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/'
export test_date="08_23"
export model_name="longformer-base-4096"
export model_max_length=4096
export batch_size=8
export data_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/split_data"
export root_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/results/09_23"

export CUDA_VISIBLE_DEVICES=0
python $path/daily_pred.py \
--batch_size $batch_size \
--fp16 \
--model_name $model_name \
--customized_model \
--model_max_length $model_max_length \
--test_date $test_date \
--root_dir $root_dir \
--data_dir $data_dir \
--data_name email_data_0.parquet &

export CUDA_VISIBLE_DEVICES=0
python $path/daily_pred.py \
--batch_size $batch_size \
--fp16 \
--model_name $model_name \
--customized_model \
--model_max_length $model_max_length \
--test_date $test_date \
--root_dir $root_dir \
--data_dir $data_dir \
--data_name email_data_1.parquet &

export CUDA_VISIBLE_DEVICES=1
python $path/daily_pred.py \
--batch_size $batch_size \
--fp16 \
--model_name $model_name \
--customized_model \
--model_max_length $model_max_length \
--test_date $test_date \
--root_dir $root_dir \
--data_dir $data_dir \
--data_name email_data_2.parquet &

export CUDA_VISIBLE_DEVICES=1
python $path/daily_pred.py \
--batch_size $batch_size \
--fp16 \
--model_name $model_name \
--customized_model \
--model_max_length $model_max_length \
--test_date $test_date \
--root_dir $root_dir \
--data_dir $data_dir \
--data_name email_data_3.parquet &

export CUDA_VISIBLE_DEVICES=2
python $path/daily_pred.py \
--batch_size $batch_size \
--fp16 \
--model_name $model_name \
--customized_model \
--model_max_length $model_max_length \
--test_date $test_date \
--root_dir $root_dir \
--data_dir $data_dir \
--data_name email_data_4.parquet &

export CUDA_VISIBLE_DEVICES=2
python $path/daily_pred.py \
--batch_size $batch_size \
--fp16 \
--model_name $model_name \
--customized_model \
--model_max_length $model_max_length \
--test_date $test_date \
--root_dir $root_dir \
--data_dir $data_dir \
--data_name email_data_5.parquet &


export CUDA_VISIBLE_DEVICES=3
python $path/daily_pred.py \
--batch_size $batch_size \
--fp16 \
--model_name $model_name \
--customized_model \
--model_max_length $model_max_length \
--test_date $test_date \
--root_dir $root_dir \
--data_dir $data_dir \
--data_name email_data_6.parquet &

export CUDA_VISIBLE_DEVICES=3
python $path/daily_pred.py \
--batch_size $batch_size \
--fp16 \
--model_name $model_name \
--customized_model \
--model_max_length $model_max_length \
--test_date $test_date \
--root_dir $root_dir \
--data_dir $data_dir \
--data_name email_data_7.parquet &

wait
