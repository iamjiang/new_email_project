#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/reference_data/'
export path='/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/reference_data/'

python $path/data_comb.py \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/reference_data/split_data/pred_output/dedup_target  \
--feedback_as_complaint \
--val_min_recall 0.98  \
--output_name  inf_0823_feedback &

# python $path/data_comb.py \
# --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/reference_data/split_data/pred_output/dedup_target  \
# --val_min_recall 0.98  \
# --output_name  inf_0823_no_feedback &

python $path/data_comb.py \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/reference_data/split_data/pred_output/dedup_time  \
--feedback_as_complaint \
--val_min_recall 0.98  \
--output_name  inf_0823_feedback &

# python $path/data_comb.py \
# --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/reference_data/split_data/pred_output/dedup_time  \
# --val_min_recall 0.98  \
# --output_name  inf_0823_no_feedback &

python $path/data_comb.py \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/reference_data/split_data/pred_output/no_dup  \
--feedback_as_complaint \
--val_min_recall 0.98  \
--output_name  inf_0823_feedback &

# python $path/data_comb.py \
# --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/reference_data/split_data/pred_output/no_dup  \
# --val_min_recall 0.98  \
# --output_name  inf_0823_no_feedback &

wait

