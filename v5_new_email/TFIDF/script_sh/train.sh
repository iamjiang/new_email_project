#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production/model_pickle_file/script_sh/'

# python main.py \
# --train_undersampling \
# --train_negative_positive_ratio 5 \
# --val_undersampling \
# --val_negative_positive_ratio 5 \
# --model_name lightgbm \
# --feedback_as_complaint \
# --max_feature_num 990 \
# --test_date 05_23 &

# python main.py \
# --train_undersampling \
# --train_negative_positive_ratio 5 \
# --val_undersampling \
# --val_negative_positive_ratio 5 \
# --model_name lightgbm \
# --feedback_as_complaint \
# --max_feature_num 990 \
# --test_date 06_23 &

# python ../main.py \
# --train_undersampling \
# --train_negative_positive_ratio 5 \
# --val_undersampling \
# --val_negative_positive_ratio 5 \
# --model_name lightgbm \
# --feedback_as_complaint \
# --max_feature_num 990 \
# --test_date 07_23 &

python ../main.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_undersampling \
--val_negative_positive_ratio 5 \
--model_name lightgbm \
--feedback_as_complaint \
--max_feature_num 990 \
--test_date 08_23 &



wait



