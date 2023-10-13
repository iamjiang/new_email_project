#!/bin/bash

export path='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/'
export test_date="09_23"
export start_date='2022-09-01'
export end_date='2023-09-30'
export num_epochs=10
export es_patience=3
export data_name="train_val_test_dedup_target"
export train_neg_pos_ratio=5
export val_neg_pos_ratio=5


export CUDA_VISIBLE_DEVICES=1
python $path/main.py --model_name longformer-base-4096   \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs $num_epochs \
--es_patience $es_patience \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--test_date $test_date \
--train_undersampling \
--train_negative_positive_ratio $train_neg_pos_ratio \
--val_undersampling \
--val_negative_positive_ratio $val_neg_pos_ratio \
--customized_model \
--max_token_length 4096  \
--feedback_as_complaint \
--start_date $start_date \
--end_date $end_date \
--data_name $data_name &


export CUDA_VISIBLE_DEVICES=1
python $path/main.py --model_name longformer-base-4096   \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs $num_epochs \
--es_patience $es_patience \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--test_date $test_date \
--train_undersampling \
--train_negative_positive_ratio $train_neg_pos_ratio \
--val_undersampling \
--val_negative_positive_ratio $val_neg_pos_ratio \
--max_token_length 4096  \
--feedback_as_complaint \
--start_date $start_date \
--end_date $end_date \
--data_name $data_name &


# Wait for all commands to finish
wait
