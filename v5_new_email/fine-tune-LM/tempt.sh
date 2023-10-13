#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/fine-tune-LM'

export CUDA_VISIBLE_DEVICES=0
python run_language_model.py \
--model_name longformer-large-4096 \
--batch_size 4 \
--num_epochs 25 \
--es_patience 5 \
--lr 1e-5 \
--weight_decay 5e-5 \
--fp16 \
--gradient_accumulation_steps 8 \
--use_schedule \
--train_undersampling \
--val_undersampling \
--test_undersampling \
--train_negative_positive_ratio 50 \
--val_negative_positive_ratio 50 \
--test_negative_positive_ratio 100 \
--wwm_probability 0.15

