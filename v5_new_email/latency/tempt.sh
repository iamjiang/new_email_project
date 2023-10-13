#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v4_new_email/latency'

export CUDA_VISIBLE_DEVICES=1
python latency.py \
--model_name bigbird-roberta-large \
--batch_size 4  \
--fp16 \
--customized_model \
--model_max_length 4096 \
--closed_status \
--num_sample 6000 \
--device cuda


