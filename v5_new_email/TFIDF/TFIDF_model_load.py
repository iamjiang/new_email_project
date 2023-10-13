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

def model_evaluate(target, y_prob, best_threshold):
    
    # best_threshold=find_optimal_threshold(target, predicted)
    y_pred=[1 if x>best_threshold else 0 for x in y_prob]
    
    true_label_mask=[1 if (x-target[i])==0 else 0 for i,x in enumerate(y_pred)]
    nb_prediction=len(true_label_mask)
    true_prediction=sum(true_label_mask)
    false_prediction=nb_prediction-true_prediction
    accuracy=true_prediction/nb_prediction
    
    
    false_positive=np.sum([1 if v==1 and target[i]==0  else 0 for i,v in enumerate(y_pred)])
    false_negative=np.sum([1 if v==0 and target[i]==1  else 0 for i,v in enumerate(y_pred)])
    
    # precision, recall, fscore, support = precision_recall_fscore_support(target, predicted.argmax(axis=1))
    
    # precision, recall, thresholds = precision_recall_curve(target.ravel(), torch.sigmoid(torch.from_numpy(predicted))[:,1].numpy().ravel())
    precision, recall, thresholds = precision_recall_curve(target, y_prob, pos_label=1)
    pr_auc = auc(recall, precision)
    
    prec=precision_score(target,y_pred,pos_label=1)
    rec=recall_score(target,y_pred,pos_label=1)
    fscore = f1_score(target,y_pred,pos_label=1)
    roc_auc = roc_auc_score(target,y_prob)
    
    return {
        "total positive":sum(target),
        "false positive":false_positive,
        "false_negative":false_negative,
        "precision":prec, 
        "recall":rec, 
        "f1_score":fscore,
        "AUC":roc_auc,
        "pr_auc":pr_auc
    }


if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_feature_num', type=int, default=990)
    argparser.add_argument('--model_name', type=str, default="lightgbm")
    argparser.add_argument('--output_dir', type=str, default=None)
    argparser.add_argument("--test_date", type=str, default="08_23", help="the month for test set")
    argparser.add_argument("--reference_date", type=str, default="2023-08-01", help="the month used for reference data")
    argparser.add_argument("--val_min_recall", default=0.95, type=float, help="minimal recall for valiation dataset")
    args,_ = argparser.parse_known_args()

    # args.output_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/containerization/complaint-model/TFIDF_container/app/model"
        
    print(args)
    
    root_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/datasets/raw_data"
    data_name=[x for x in os.listdir(root_dir) if x.split(".")[-1]=="csv"]
    data_name = [x for x in data_name if x.split("_")[2]==args.reference_date]
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

    df=df[df.month==int(args.test_date.split("_")[0][1])]
    df.sort_values(by='time', inplace = True)

    #### remove duplicated emails based on thread id
    grouped_df=df.groupby('thread_id')
    sorted_groups=[group.sort_values("time",ascending=False).reset_index(drop=True) for _, group in grouped_df]
    df=pd.concat(sorted_groups).drop_duplicates(subset="thread_id", keep="first").reset_index(drop=True)

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

    df.sort_values(by='time', inplace = True) 
    df=df.reset_index(drop=True)
    
    df['email'] = df['email'].astype(str)
    df['preprocessed_email'] = df['email'].progress_apply(clean_re)
    
    keep_columns=['snapshot_id', 'gcid','thread_id', 'time', 'is_feedback', 'email', 'preprocessed_email','is_complaint']
    df=df.loc[:,keep_columns]
    df['target']=df['is_complaint'].progress_apply(lambda x : 1 if x=="Y" else 0)
    
    input_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production/model_pickle_file/"
    # bow_vectorizer = pickle.load(open(os.path.join(input_dir,"bow_vectorizer.pickle"), "rb"))
    bow_vectorizer =joblib.load(os.path.join(input_dir,"outputs","bow_vectorizer.pickle"))
    # vocab = bow_vectorizer.vocabulary_.keys()
    vocab = bow_vectorizer.get_feature_names_out()

    index_test=df.loc[:,['snapshot_id','gcid','thread_id','time']]
    index_test.rename(columns={"time": "time_variable"},inplace=True) # rename these words to distinguish them from the tfidf 
    y_test=df.loc[:,["target"]]
    y_test.rename(columns={"target": "target_variable"},inplace=True)
    
    test_tfidf = bow_vectorizer.transform(df['preprocessed_email'])
    # test_tfidf = bow_vectorizer.transform(df["bag_of_word"])
    test_tfidf = test_tfidf.toarray()
    test_tfidf = pd.DataFrame(test_tfidf,columns=vocab)
    test_data=pd.concat([test_tfidf.reset_index(drop=True), y_test.reset_index(drop=True), index_test.reset_index(drop=True)],axis=1)


    model_dir=os.path.join(input_dir,'tfidf_model', args.test_date, str(args.max_feature_num))
    model = joblib.load(os.path.join(model_dir,'lightgbm_model.pkl'))
    
    csv_file="predictions_"+str(args.val_min_recall).split(".")[1]+".csv"
    best_threshold=pd.read_csv(os.path.join(model_dir, args.model_name, csv_file)).best_threshold.unique()[0]
    
    print()
    print(f"Best Threshold Value : {best_threshold}")
    print()
    
    pickle_file=args.model_name+".pkl"
    
    model_dict={"bow_vectorizer": bow_vectorizer,"model":model, "best_threshold":best_threshold}
    
    with open(os.path.join(args.output_dir,pickle_file),"wb") as f:
        pickle.dump(model_dict,f)
        # joblib.dump(model_dict, os.path.join(args.output_dir,pickle_file))
    
    model_dict_load =joblib.load(os.path.join(args.output_dir,"lightgbm.pkl"))
        
    best_threshold=model_dict_load["best_threshold"]
    predict_model=model_dict_load["model"]
    bow_vectorizer=model_dict_load["bow_vectorizer"]
   
        
    test_pred=predict_model.predict(test_tfidf)   
    
    y_pred=[1 if x>best_threshold else 0 for x in test_pred]
    test_output=model_evaluate(y_test.values.reshape(-1),test_pred.squeeze(),best_threshold)
    
    
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
    
    