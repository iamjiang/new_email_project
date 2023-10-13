#!/bin/bash
export PYTHONPATH=$PYTHONPATH'/opt/omniai/work/instance1/jupyter/v5_new_email/datasets'

python data_preprocess_without_split.py --closed_status


