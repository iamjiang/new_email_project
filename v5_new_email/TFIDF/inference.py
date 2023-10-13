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
    parser.add_argument("--test_date", type=str, default="04_23", help="the month for test set")
    parser.add_argument("--default_threshold", action="store_true", help="undersampling or not")    
    # parser.add_argument("--feedback_as_complaint", action="store_true", help="treat feedback as complaint in training and validation ?")
    parser.add_argument('--max_feature_num', type=int, default=7000)
    args= parser.parse_args()
    
    # args.train_undersampling=True

    print()
    print(args)
    print()
    
    root_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production/model_pickle_file/"
    data_dir=os.path.join(root_dir, "outputs", args.test_date)

    df_train=pd.read_pickle(os.path.join(data_dir,"train_data_"+str(args.max_feature_num)))
    df_val=pd.read_pickle(os.path.join(data_dir,"val_data_"+str(args.max_feature_num)))
    df_test=pd.read_pickle(os.path.join(data_dir,"test_data_"+str(args.max_feature_num)))
       
        
    y_val=df_val.loc[:,["target_variable"]]
    index_val=df_val.loc[:,["snapshot_id","gcid","thread_id","time_variable"]]
    x_val=df_val.drop(["target_variable","snapshot_id","gcid","thread_id","time_variable"],axis=1)

    y_test=df_test.loc[:,["target_variable"]]
    index_test=df_test.loc[:,["snapshot_id","gcid","thread_id","time_variable"]]
    x_test=df_test.drop(["target_variable","snapshot_id","gcid","thread_id","time_variable"],axis=1)
    

    model_dir=os.path.join(root_dir,"tfidf_model", args.test_date, str(args.max_feature_num))
     
    if args.model_name=="catboost":
        model = joblib.load(os.path.join(model_dir,'catboost_model.pkl'))
        val_pred=model.predict_proba(x_val)[:,1]
        test_pred=model.predict_proba(x_test)[:,1]
        
    elif args.model_name=="lightgbm":
        model = joblib.load(os.path.join(model_dir,'lightgbm_model.pkl'))
        val_pred=model.predict(x_val)
        test_pred=model.predict(x_test)   
        
    elif args.model_name=="xgboost":
        model = joblib.load(os.path.join(model_dir,'xgboost_model.pkl'))
        val_pred=model.predict(xgb.DMatrix(x_val))
        test_pred=model.predict(xgb.DMatrix(x_test))
        
    elif args.model_name=="randomforest":
        model = joblib.load(os.path.join(model_dir,'random_forest_model.pkl'))
        val_pred=model.predict_proba(x_val)[:,1]
        test_pred=model.predict_proba(x_test)[:,1]
        
    else:
        raise ValueError("Invalid model name. Only catboost, lightgbm , xgboost or randomforest are support")
        
    if args.default_threshold:
        best_threshold=0.5
    else:
        best_threshold=utils.find_optimal_threshold(y_val.squeeze(), val_pred.squeeze(), min_recall=args.val_min_recall, pos_label=False)
    y_pred=[1 if x>best_threshold else 0 for x in test_pred]
    test_output=utils.model_evaluate(y_test.values.reshape(-1),test_pred.squeeze(),best_threshold)

    output_dir=os.path.join(root_dir,"tfidf_model", args.test_date,str(args.max_feature_num), args.model_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    snapshot_id=index_test["snapshot_id"].tolist()
    gcid=index_test["gcid"].tolist()
    thread_id=index_test["thread_id"].tolist()
    time=index_test["time_variable"].tolist()

    # fieldnames = ['True label', 'Predicted label', 'Predicted_prob']
    fieldnames = ['snapshot_id','gcid','thread_id','time','True_label', 'Predicted_label', 
                  'Predicted_prob','best_threshold']
    if args.default_threshold:
        file_name="predictions_default.csv"
    else:
        file_name="predictions_"+str(args.val_min_recall).split(".")[-1]+".csv"
    
    with open(os.path.join(output_dir , file_name), 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, f, j, k, m, n, p, q in zip(snapshot_id, gcid, thread_id, time, y_test.values.reshape(-1), 
                                           y_pred, test_pred, [best_threshold]*len(y_pred)):
            writer.writerow(
                {'snapshot_id':i,'gcid':f,'thread_id':j,'time':k, 'True_label': m , 'Predicted_label': n,
                 'Predicted_prob': p, 'best_threshold':q})
    
    # file_name=args.model_name+"_"+"metrics_test.txt"
    # output_dir=os.path.join(os.getcwd(),"tfidf+structure")
    # with open(os.path.join(output_dir,file_name),'a') as f:
    #     f.write(f'{args.model_name},{test_output["total positive"]},{test_output["false positive"]},{test_output["false_negative"]}, \
    #     {test_output["precision"]},{test_output["recall"]},{test_output["f1_score"]},{test_output["AUC"]},{test_output["pr_auc"]},{best_threshold}\n')  
        
    # output_name=args.model_name+"_"+"y_true_pred.txt"
    # with open(os.path.join(output_dir,output_name),'w') as f:
    #     for x,y,z in zip(y_test.target_variable.tolist(),y_pred,test_pred.tolist()):
    #         f.write(str(x)+","+str(y)+","+str(z)+ '\n')

    print("==> performance on test set \n")
    print("")
    print("total positive: {:,} | false positive: {:,} | false_negative: {:,} | precision: {:.2%} | recall: {:.2%} | F1_score: {:.2%} | ROC_AUC: {:.1%} | PR_AUC: {:.1%}".\
           format(test_output["total positive"], test_output["false positive"], test_output["false_negative"], \
                 test_output["precision"], test_output["recall"], test_output["f1_score"], test_output["AUC"], test_output["pr_auc"]))


    print()
    print(f"\n===========Test Set Performance===============\n")
    print()
    y_pred=[1 if x>best_threshold else 0 for x in test_pred]
    print(classification_report(y_test, y_pred))
    print()
    print(confusion_matrix(y_test, y_pred))  
    