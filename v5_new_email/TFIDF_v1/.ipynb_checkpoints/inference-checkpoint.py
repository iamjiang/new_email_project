import sys
import csv
csv.field_size_limit(sys.maxsize)
import time
import os
import re
import pandas as pd
import numpy as np
import argparse
import datasets
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets,DatasetDict
from datasets import load_from_disk
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

import joblib

import utils

import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=DataConversionWarning, module='sklearn')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='TFIDF+Classifier')
   
    parser.add_argument('--model_name', type=str, default="xgboost")
    parser.add_argument("--val_min_recall", default=0.9, type=float, help="minimal recall for valiation dataset")
    parser.add_argument("--default_threshold", action="store_true", help="undersampling or not")    
    parser.add_argument("--feedback_as_complaint", action="store_true", help="treat feedback as complaint in training and validation ?")
    parser.add_argument('--max_feature_num', type=int, default=7000)
    parser.add_argument('--data_dir', type=str, default="/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v0/features/dedup_target/")
    args= parser.parse_args()
    
    # args.train_undersampling=True

    print()
    print(args)
    print()

    if args.feedback_as_complaint:
        df_train=pd.read_pickle(os.path.join(args.data_dir,"train_data_"+str(args.max_feature_num)))
        df_val=pd.read_pickle(os.path.join(args.data_dir,"val_data_"+str(args.max_feature_num)))
        # df_test=pd.read_pickle(os.path.join(args.data_dir,"test_data_"+str(args.max_feature_num)))
    else:
        df_train=pd.read_pickle(os.path.join(data_dir,"train_data_v0_"+str(args.max_feature_num)))
        df_val=pd.read_pickle(os.path.join(data_dir,"val_data_v0_"+str(args.max_feature_num)))
        # df_test=pd.read_pickle(os.path.join(data_dir,"test_data_v0_"+str(args.max_feature_num))) 
       
    y_val=df_val.loc[:,["target_variable"]]
    index_val=df_val.loc[:,["snapshot_id","gcid","thread_id","time_variable"]]
    x_val=df_val.drop(["target_variable","snapshot_id","gcid","thread_id","time_variable"],axis=1)

    # y_test=df_test.loc[:,["target_variable"]]
    # index_test=df_test.loc[:,["snapshot_id","gcid","thread_id","time_variable"]]
    # x_test=df_test.drop(["target_variable","snapshot_id","gcid","thread_id","time_variable"],axis=1)
        
    if args.feedback_as_complaint:
        model_dir=os.path.join(args.data_dir,"tfidf_model")
    else:
        model_dir=os.path.join(args.data_dir,"tfidf_model_v0")
     
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if args.model_name=="catboost":
        model = joblib.load(os.path.join(model_dir,'catboost_model.pkl'))
        val_pred=model.predict_proba(x_val)[:,1]
        # test_pred=model.predict_proba(x_test)[:,1]
        
    elif args.model_name=="lightgbm":
        model = joblib.load(os.path.join(model_dir,'lightgbm_model.pkl'))
        val_pred=model.predict(x_val)
        # test_pred=model.predict(x_test)   
        
    elif args.model_name=="xgboost":
        model = joblib.load(os.path.join(model_dir,'xgboost_model.pkl'))
        val_pred=model.predict(xgb.DMatrix(x_val))
        # test_pred=model.predict(xgb.DMatrix(x_test))
        
    elif args.model_name=="randomforest":
        model = joblib.load(os.path.join(model_dir,'random_forest_model.pkl'))
        val_pred=model.predict_proba(x_val)[:,1]
        # test_pred=model.predict_proba(x_test)[:,1]
        
    else:
        raise ValueError("Invalid model name. Only catboost, lightgbm , xgboost or randomforest are support")
        
    if args.default_threshold:
        best_threshold=0.5
    else:
        best_threshold=utils.find_optimal_threshold(y_val.squeeze(), val_pred.squeeze(), min_recall=args.val_min_recall, pos_label=False)
        
        
    output_name="best_threshold_val.txt"
    with open(os.path.join(model_dir,output_name),'w') as f:
        f.write(str(best_threshold))
        
    # with open(os.path.join(model_dir,output_name),'r') as f:
    #     x=float(f.readline())
    
