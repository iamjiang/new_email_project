import argparse
import pickle
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment=None
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
from collections import Counter

import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import HTML
import re
import textwrap
import random

import string
import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
stopwords_file = open(nltk_data_dir + '/corpora/stopwords/english')
stopwords_list = stopwords_file.readlines()
STOPWORDS=[x.strip() for x in stopwords_list]
nltk.data.path.append(nltk_data_dir)

from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from sklearn.metrics import precision_recall_fscore_support 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc 
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

import joblib
import utils

def val_mask_creation(dataset,target_variable, validation_split):

    dataset.sort_values(by='time', ascending=False, axis=0, inplace = True)
    dataset=dataset.reset_index(drop=True)

    train_idx=[]
    val_idx=[]

    LABEL=dataset[target_variable].values.squeeze()
    IDX=np.arange(LABEL.shape[0])
    target_list=np.unique(LABEL).tolist()

    for i in range(len(target_list)):

        _idx=IDX[LABEL==target_list[i]]

        split=int(np.floor(validation_split*_idx.shape[0]))

        val_idx.extend(_idx[ : split])
        # print(len(_idx[ : split]))
        train_idx.extend(_idx[split:])        

    all_idx=np.arange(LABEL.shape[0])

    val_idx=np.array(val_idx)
    train_idx=np.array(train_idx)

    df_train=dataset.loc[train_idx,:]
    df_train=df_train.reset_index(drop=True)

    df_val=dataset.loc[val_idx,:]
    df_val["data_type"]=["val"]*val_idx.shape[0]
    df_val=df_val.reset_index(drop=True)

    return df_train, df_val

def bow_preprocess(text):
    # lemma = nltk.wordnet.WordNetLemmatizer()
    text = str(text) 
    ### Remove stop word
    text = [word for word in word_tokenize(text) if word.lower() not in STOPWORDS]
    text = " ".join(text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    #Remove punctuation
    table = str.maketrans('', '', string.punctuation)
    text = [w.translate(table) for w in text.split()]
    text=" ".join(text)
    return text  

def main(df_test,args): 
    

    index_test=df_test.loc[:,['snapshot_id','gcid','thread_id','time']]
    index_test.rename(columns={"time": "time_variable"},inplace=True) 
    y_test=df_test.loc[:,["target"]]
    y_test.rename(columns={"target": "target_variable"},inplace=True)
    
    
    model_dict_load =joblib.load(os.path.join(args.output_dir,"lightgbm.pkl"))
        
    best_threshold=model_dict_load["best_threshold"]
    predict_model=model_dict_load["model"]
    bow_vectorizer=model_dict_load["bow_vectorizer"]

    vocab = bow_vectorizer.get_feature_names_out()

    test_tfidf = bow_vectorizer.transform(df_test['preprocessed_email'])
    print()
    print(test_tfidf.shape)
    print()
    # test_tfidf = bow_vectorizer.transform(df["bag_of_word"])
    test_tfidf = test_tfidf.toarray()
    test_tfidf = pd.DataFrame(test_tfidf,columns=vocab)
    test_data=pd.concat([test_tfidf.reset_index(drop=True), y_test.reset_index(drop=True), index_test.reset_index(drop=True)],axis=1)
    
    test_pred=predict_model.predict(test_tfidf)   
    y_pred=[1 if x>best_threshold else 0 for x in test_pred]
    test_output=utils.model_evaluate(y_test.values.reshape(-1),test_pred.squeeze(),best_threshold)
    
    
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
    
if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_feature_num', type=int, default=990)
    argparser.add_argument('--output_dir', type=str, default="/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production/model_pickle_file/")
    argparser.add_argument('--validation_split', type=float, default=0.2) 
    argparser.add_argument("--test_date", type=str, default="08_23", help="the month for test set")
    argparser.add_argument("--feedback_as_complaint", action="store_true", help="treat feedback as complaint in training and validation ?")
    
    args,_ = argparser.parse_known_args()
    args.feedback_as_complaint=True
    print(args)
    
    
    root_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/datasets/raw_data"
    data_name=[x for x in os.listdir(root_dir) if x.split(".")[-1]=="csv"]
    # data_name = [x for x in data_name if x.split("_")[2]!="2023-08-01"]
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_csv(os.path.join(root_dir,data))
        x=x.dropna(subset=['email'])
        x=x[x.email.notna()]
        x=x[x.email.str.len()>0]
        df=pd.concat([df,x],axis=0,ignore_index=True)
        print("{:<20}{:<20,}".format(data.split("_")[2],x.shape[0]))
    
    df=df[df.state=="closed"]
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df.time.apply(lambda x: x.year)
    df['month'] = df.time.apply(lambda x: x.month)
    df['day'] = df.time.apply(lambda x: x.day)

    df=df[~((df.year==2022) & (df.month==8))]
    df.sort_values(by='time', inplace = True)

    #### remove duplicated emails based on thread id
    grouped_df=df.groupby('thread_id')
    sorted_groups=[group.sort_values("time",ascending=False).reset_index(drop=True) for _, group in grouped_df]
    df=pd.concat(sorted_groups).drop_duplicates(subset="thread_id", keep="first").reset_index(drop=True)
    
    address_pattern = r"\d+\s+[a-zA-Z0-9\s,]+\s+[a-zA-Z]+\s+\d{5}"
    url_pattern = "(?:https?:\\/\\/)?(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"

    us_phone_num_pattern = '\(?\d{3}\)?[.\-\s]?\d{3}[.\-\s]?\d{4}'
    email_id_pattern = "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)" 

    def clean_re(text):
        text = str(text).lower()
        text = re.sub('<[^>]*>', '', text)
        text = re.sub('_x000d_', '', text)
        #remove non-alphanumerc characters: '\u200c\xa0\u200c\xa0\u200c\xa0\n'
        text = re.sub(r"\u200c\xa0+",'',text)

        text = re.sub(r"[^A-Za-z\s\d+\/\d+\/\d+\.\:\-,;?'\"%$]", '', text) # replace non-alphanumeric with space
        #remove long text such as encrypted string
        text = " ".join(word for word in text.split() if len(word)<=20) 

        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\n{1,}", " ", text)
        text = re.sub(r"\t{1,}", " ", text)
        text = re.sub("_{2,}","",text)
        text = re.sub("\[\]{1,}","",text)
        text = re.sub(r"(\s\.){2,}", "",text) #convert pattern really. . . . . . .  gotcha  into really. gotcha 

        # Define regular expression pattern for address and signature
        address_pattern = r"\d+\s+[a-zA-Z0-9\s,]+\s+[a-zA-Z]+\s+\d{5}"
        # signature_pattern = r"^\s*[a-zA-Z0-9\s,]+\s*$"

        url_pattern = \
        "(?:https?:\\/\\/)?(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"

        us_phone_num_pattern = \
        '\(?\d{3}\)?[.\-\s]?\d{3}[.\-\s]?\d{4}'

        email_id_pattern = \
        "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        # Remove address and signature from email
        text = re.sub(address_pattern, "", text)
        # text = re.sub(signature_pattern, "", text)
        text = re.sub(url_pattern, "", text)
        text = re.sub(us_phone_num_pattern, "", text)
        text = re.sub(email_id_pattern, "", text)
        
        ##remove numerical values 
        text=re.sub(r"\d+","",text)
        tokens=[word for word in word_tokenize(text) if not word.isnumeric() ]
        ##remove stop word 
        tokens=[word for word in tokens if word not in STOPWORDS]
        text=" ".join(tokens)

        return text

    df['email'] = df['email'].astype(str)
    df['preprocessed_email'] = df['email'].progress_apply(clean_re)
    
    df["text_length"]=df['preprocessed_email'].progress_apply(lambda x : len(x.lower().strip().split()))
    keep_columns=['snapshot_id', 'email', 'gcid', 'thread_id', 'state', 'time','is_complaint', \
                  'is_feedback', 'supervised_groups', 'year', 'month','day', 'preprocessed_email','text_length']
    df=df.loc[:,keep_columns]


    df.sort_values(by='time', inplace = True) 
    if args.test_date=="08_23":
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2,3,4,5,6,7]) \
        else "test"
    elif args.test_date=="07_23":
        df=df[df['time']<'2023-08-01']
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2,3,4,5,6]) \
        else "test"
    elif args.test_date=="06_23":
        df=df[df['time']<'2023-07-01']
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in [9,10,11,12,1,2,3,4,5]) \
        else "test"
    else:
        raise ValueError("Invalid test_date. Only Aug, July and June data are supported currently")
        
    df["data_type"]=df.progress_apply(set_categories,axis=1)
    if args.feedback_as_complaint:
        df['target']=np.where((df['is_complaint']=="Y") | (df['is_feedback']=="Y"),1,0)
    else:
        df['target']=df['is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    
    df1=df[df.data_type=="train"]
    df1=df1.reset_index(drop=True)
    df_train,df_val=val_mask_creation(df1,'target', validation_split=args.validation_split)
    
    df_test=df[df.data_type=="test"]
    df_test=df_test.reset_index(drop=True)
    if args.feedback_as_complaint:
        ## overwrite the target with the ground true complaint label
        df_test['target']=df_test['is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    
    
    main(df_test,args)


    