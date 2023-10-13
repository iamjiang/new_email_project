#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/'
export path='/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/'
N=9
for i in $(/usr/bin/seq 0 $N)
do
    input_data="email_data_$(printf "%01d" $i).parquet"
    output_data_v0="inf_0923_feedback_$(printf "%01d" $i)"
    output_data_v1="inf_0923_no_feedback_$(printf "%01d" $i)"

    python $path/daily_pred.py \
    --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data  \
    --data_name $input_data \
    --root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_target/  \
    --output_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/dedup_target/  \
    --feedback_as_complaint \
    --output_name $output_data_v0 &

    #  python $path/daily_pred.py \
    # --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data  \
    # --data_name $input_data \
    # --root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_target/  \
    # --output_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/dedup_target/  \
    # --output_name $output_data_v1 &
    
    python $path/daily_pred.py \
    --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data  \
    --data_name $input_data \
    --root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_time/  \
    --output_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/dedup_time/  \
    --feedback_as_complaint \
    --output_name $output_data_v0 &

    #  python $path/daily_pred.py \
    # --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data  \
    # --data_name $input_data \
    # --root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_time/  \
    # --output_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/dedup_time/  \
    # --output_name $output_data_v1 &
    
    python $path/daily_pred.py \
    --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data  \
    --data_name $input_data \
    --root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_time/  \
    --output_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/no_dup/  \
    --feedback_as_complaint \
    --output_name $output_data_v0 &

    #  python $path/daily_pred.py \
    # --data_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data  \
    # --data_name $input_data \
    # --root_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/features/dedup_time/  \
    # --output_dir /opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/split_data/pred_output/no_dup/  \
    # --output_name $output_data_v1 &
    
done

wait
