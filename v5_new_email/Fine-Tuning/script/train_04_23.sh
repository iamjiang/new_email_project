#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning'
export test_date="04_23"

export CUDA_VISIBLE_DEVICES=0
python ../main.py \
--model_name roberta-large \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--test_date $test_date \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_undersampling \
--val_negative_positive_ratio 5 \
--max_token_length 512  &

export CUDA_VISIBLE_DEVICES=0
python ../main.py \
--model_name deberta-v3-large \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--max_token_length 512 \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--test_date $test_date \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_undersampling \
--val_negative_positive_ratio 5 \
--max_token_length 512 &

export CUDA_VISIBLE_DEVICES=1
python ../main.py --model_name longformer-base-4096   \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--test_date $test_date \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_undersampling \
--val_negative_positive_ratio 5 \
--max_token_length 4096  &

export CUDA_VISIBLE_DEVICES=2
python ../main.py --model_name longformer-large-4096   \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--test_date $test_date \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_undersampling \
--val_negative_positive_ratio 5 \
--max_token_length 4096  &

export CUDA_VISIBLE_DEVICES=3
python ../main.py \
--model_name bigbird-roberta-large   \
--train_batch_size 2 \
--val_batch_size 2 \
--num_epochs 10 \
--es_patience 3 \
--gradient_accumulation_steps 16 \
--gradient_checkpointing \
--fp16 \
--lr 1e-5 \
--loss_weight \
--use_schedule \
--test_date $test_date \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_undersampling \
--val_negative_positive_ratio 5 \
--max_token_length 4096 &

# Wait for all commands to finish
wait
