#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/script_sh/'
export path='/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/'

# python $path/main.py \
# --model_name lightgbm \
# --max_feature_num 990 \
# --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_target &

python $path/main.py \
--model_name lightgbm \
--feedback_as_complaint \
--max_feature_num 990 \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_target &


# python $path/main.py \
# --model_name lightgbm \
# --max_feature_num 990 \
# --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_time &

python $path/main.py \
--model_name lightgbm \
--feedback_as_complaint \
--max_feature_num 990 \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_time &

# python $path/main.py \
# --model_name lightgbm \
# --max_feature_num 990 \
# --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/no_dup &

python $path/main.py \
--model_name lightgbm \
--feedback_as_complaint \
--max_feature_num 990 \
--data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/no_dup &

wait



