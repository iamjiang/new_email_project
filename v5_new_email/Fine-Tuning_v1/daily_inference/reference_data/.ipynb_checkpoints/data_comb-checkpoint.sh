#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/'
export path='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/'
export root_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/split_data/pred_output/"
export output_name="inf_0823"
export val_min_recall=0.98

export model_name="longformer-base-4096"
python $path/data_comb.py \
--root_dir $root_dir  \
--model_name $model_name \
--val_min_recall $val_min_recall  \
--customized_model \
--output_name  $output_name &


export model_name="longformer-base-4096"
python $path/data_comb.py \
--root_dir $root_dir  \
--model_name $model_name \
--val_min_recall $val_min_recall  \
--output_name  $output_name &

wait