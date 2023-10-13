#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/'
export path='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/'

python $path/data_split.py \
--data_name email_data_2023-08-01_to_2023-08-31.csv \
--root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/datasets/raw_data  \
--output_dir /opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/split_data 

