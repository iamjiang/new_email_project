#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production/model_pickle_file/script_sh/'
export model_name="lightgbm"
export test_date="08_23"

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--default_threshold &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.9 &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.91 &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.92 &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.93 &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.94 &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.95 &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.96 &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.97 &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.98 &

python ../inference.py \
--model_name $model_name \
--test_date $test_date \
--max_feature_num 990 \
--val_min_recall 0.99 &

wait
