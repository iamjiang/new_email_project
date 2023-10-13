#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production/model_pickle_file/script_sh/'

python ../tfidf_data_prep.py \
--max_feature_num 990 \
--validation_split 0.2 \
--output_dir outputs \
--feedback_as_complaint \
--test_date 08_23 &

wait
