#!/bin/bash

path_1='/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/script_sh/'
bash $path_1/data_prep.sh
bash $path_1/train.sh
bash $path_1/inference.sh

path_2='/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/reference_data/'
bash $path_2/data_split.sh
bash $path_2/daily_pred.sh
bash $path_2/data_comb.sh

path_3='/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v1/daily_inference/'
bash $path_3/data_split.sh
bash $path_3/daily_pred.sh
bash $path_3/data_comb.sh

