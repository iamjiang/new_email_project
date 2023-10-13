#!/bin/bash

path_1='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/script/'
bash $path_1/train_v0.sh

path_2='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/'
bash $path_2/data_split.sh
bash $path_2/roberta_large_customized.sh
# bash $path_2/longformer_base_customized.sh
bash $path_2/longformer_large_customized.sh

path_3='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/'
bash $path_3/data_split.sh
bash $path_3/roberta_large_customized.sh
# bash $path_3/longformer_base_customized.sh
bash $path_3/longformer_large_customized.sh


path_1='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/script/'
bash $path_1/train_v1.sh

path_2='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/reference_data/'
bash $path_2/data_split.sh
bash $path_2/roberta_large.sh
bash $path_2/deberta_v3_large.sh
# bash $path_2/longformer_base.sh
bash $path_2/longformer_large.sh

path_3='/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning_v1/daily_inference/'
bash $path_3/data_split.sh
bash $path_3/roberta_large.sh
bash $path_3/deberta_v3_large.sh
# bash $path_3/longformer_base.sh
bash $path_3/longformer_large.sh


bash $path_2/data_comb.sh
bash $path_3/data_comb.sh
