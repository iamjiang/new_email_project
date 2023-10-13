#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/'
export path='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/'
export root_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/split_data/pred_output/"
export output_name="inf_0923"

export model_name="longformer-base-4096"
python $path/data_comb.py \
--root_dir $root_dir  \
--model_name $model_name \
--customized_model \
--output_name  $output_name &


export model_name="longformer-base-4096"
python $path/data_comb.py \
--root_dir $root_dir  \
--model_name $model_name \
--output_name  $output_name &

wait