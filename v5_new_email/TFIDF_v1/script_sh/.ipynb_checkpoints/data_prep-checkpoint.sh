#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/script_sh/'

export path='/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/'
export test_date="09_23"
export start_date='2022-09-01'
export end_date='2023-09-30'

python $path/tfidf_data_prep.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_undersampling \
--val_negative_positive_ratio 5 \
--max_feature_num 990 \
--validation_split 0.2 \
--root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/  \
--output_dir features \
--feedback_as_complaint \
--start_date $start_date \
--end_date $end_date \
--close_status \
--test_date $test_date &

# python $path/tfidf_data_prep.py \
# --train_undersampling \
# --train_negative_positive_ratio 5 \
# --val_undersampling \
# --val_negative_positive_ratio 5 \
# --max_feature_num 990 \
# --validation_split 0.2 \
# --root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/  \
# --output_dir features \
# --start_date $start_date \
# --end_date $end_date \
# --close_status \
# --test_date $test_date &

python $path/tfidf_data_prep.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_undersampling \
--val_negative_positive_ratio 5 \
--max_feature_num 990 \
--validation_split 0.2 \
--root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/  \
--output_dir features \
--feedback_as_complaint \
--remove_duplicate_thread_id \
--start_date $start_date \
--end_date $end_date \
--close_status \
--test_date $test_date &

# python $path/tfidf_data_prep.py \
# --train_undersampling \
# --train_negative_positive_ratio 5 \
# --val_undersampling \
# --val_negative_positive_ratio 5 \
# --max_feature_num 990 \
# --validation_split 0.2 \
# --root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/  \
# --output_dir features \
# --remove_duplicate_thread_id \
# --start_date $start_date \
# --end_date $end_date \
# --close_status \
# --test_date $test_date &


python $path/tfidf_data_prep.py \
--train_undersampling \
--train_negative_positive_ratio 5 \
--val_undersampling \
--val_negative_positive_ratio 5 \
--max_feature_num 990 \
--validation_split 0.2 \
--root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/  \
--output_dir features \
--feedback_as_complaint \
--remove_duplicate_thread_id \
--deduplicate_thread_by_time \
--start_date $start_date \
--end_date $end_date \
--close_status \
--test_date $test_date &

# python $path/tfidf_data_prep.py \
# --train_undersampling \
# --train_negative_positive_ratio 5 \
# --val_undersampling \
# --val_negative_positive_ratio 5 \
# --max_feature_num 990 \
# --validation_split 0.2 \
# --root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/  \
# --output_dir features \
# --remove_duplicate_thread_id \
# --deduplicate_thread_by_time \
# --start_date $start_date \
# --end_date $end_date \
# --close_status \
# --test_date $test_date &

wait
