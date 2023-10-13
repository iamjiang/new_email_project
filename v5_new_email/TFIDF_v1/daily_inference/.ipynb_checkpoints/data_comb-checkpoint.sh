#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/'
export path='/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/'

python $path/data_comb.py \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/dedup_target  \
--feedback_as_complaint \
--output_name  inf_0923_feedback &

# python $path/data_comb.py \
# --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/dedup_target  \
# --output_name  inf_0923_no_feedback &

python $path/data_comb.py \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/dedup_time  \
--feedback_as_complaint \
--output_name  inf_0923_feedback &

# python $path/data_comb.py \
# --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/dedup_time  \
# --output_name  inf_0923_no_feedback &

python $path/data_comb.py \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/no_dup  \
--feedback_as_complaint \
--output_name  inf_0923_feedback &

# python $path/data_comb.py \
# --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/no_dup  \
# --output_name  inf_0923_no_feedback &

wait

