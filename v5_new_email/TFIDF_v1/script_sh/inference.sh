#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/script_sh'
export path='/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/'
export model_name="lightgbm"

python $path/inference.py \
--model_name $model_name \
--val_min_recall 0.98 \
--max_feature_num 990 \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_target/   \
--feedback_as_complaint &


python $path/inference.py \
--model_name $model_name \
--val_min_recall 0.98 \
--max_feature_num 990 \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_time/   \
--feedback_as_complaint &

python $path/inference.py \
--model_name $model_name \
--val_min_recall 0.98 \
--max_feature_num 990 \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/no_dup/   \
--feedback_as_complaint &


wait

