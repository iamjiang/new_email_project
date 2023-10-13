#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production/model_pickle_file/script_sh/'

python ../TFIDF_model_load.py \
--model_name lightgbm \
--max_feature_num 990 \
--output_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production/model_pickle_file/ \
--test_date 08_23 \
--reference_date 2023-08-01 \
--val_min_recall 0.95








