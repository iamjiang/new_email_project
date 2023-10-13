import sys
sys.path.append("/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF_revisit_v0")
sys.path=list(set(sys.path))

import argparse
import pickle
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment=None
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)
from collections import Counter
from fuzzywuzzy import fuzz

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

jpmc_statutory_msgs = \
['this message may include proprietary or protected information',
 'This message, and any attachments to it, may disclosure under applicable law',
 'action required - ',
 'please feel free to let me know if you have further questions.',
 'disclaimer',
 'the information contained in this transmission may contain privileged and confidential information.',
 'tell us how we are doing better together.',
 'notice:',
 'Have a great weekend',
 'if you have received this message in error, please send it back to us, and immediately and permanently delete it.',
 'Please keep me posted if I can help in any way.',
 'do not click on the links or attachments unless you recognize the sender and you know the content is safe.',
 'received: from icprdc',
 'if you have questions or need assistance or give me a call',
 'please let me know if you have any questions or concerns.',
 'If you no longer wish to receive these emails',
 'hi all.',
 'this email has been scanned for viruses',
 'This and any attachments are intended solely',
 'thank you.',
 'The information contained in this email message is considered confidential and proprietary to the sender and is intended solely for review and use by the named recipient',
 'My work hours may not be yours. Please do not feel obligated to respond outside of your normal work hours.',
 'tell us how we are doing',
 'this message is confidential',
 'caution: this email',
 'Any unauthorized review, use or distribution is strictly prohibited',
 'the information in this email and any attachments',
 'disclaimer this email and any attachments are confidential and for the sole use of the recipients.',
 'i hope you are well.',
 'use of the information in this may be a of the law',
 'the information contained in this communication',
 'i hope your day is going by well.',
 'If you received this fax in error',
 'important reminder: j.p.',
 'This communication may contain information that is proprietary',
 'If you have received this communication in error',
 'if you are not the intended recipient, please contact the sender by reply email and destroy all copies of the original message.',
 'do not use, copy or disclose the information contained in this message or in any attachment.',
 'this electronic transmission is',
 'this transmission may',
 'If you have received this message in error, please advise the sender by reply email and delete the message.',
 'attachments should be read and retained by intended',
 'Please feel free to reach out to me',
 'providing a safer and more useful place for your human generated data.',
 'If you are not the named addressee',
 'If you are not the intended recipient, please delete this message and notify the sender immediately.',
 'This and any attachments are intended solely for the or to whom they are addressed',
 'If you have received this e-mail and are not an intended recipient',
 'copyright 2015 jpmorgan chase co. all rights reserved',
 'your privacy is important to us.',
 'This e-mail, including any attachments that accompany it',
 'this is a system generated message.',
 'confidentiality notice: This e-mail and any documents',
 'if you received this transmission in error, please immediately contact the sender and destroy the material in its entirety, whether in electronic or hard copy',
 'If the reader of this message is not the prohibited',
 'we aim to exceed your expectations.',
 'i appreciate the patience, let me know if you have additional questions.',
 'happy friday.',
 'jpmorgan chase sent you a document to review and sign',
 'if necessary for your business with jpmc, please save the decrypted content of this email in a secure location for future reference.',
 'if you are not the intended recipient, you are hereby notified that any review, dissemination, distribution, or duplication of this communication is strictly prohibited.',
 'if there are questions / comments, pls let me know.',
 'the information contained in this transmission is confidential',
 'Thank you for getting back to me.',
 'received: from vsinthank you for choosing jpmorgan chase for your',
 'subject re revolver increase - amendment no.',
 'the information contained in this e-mail and any accompanying documents',
 'It is intended exclusively for the individual or entity',
 'this email is for the use of',
 'to find out more click here.',
 'view additional information on',
 'it is intended only for the use of the persons it is addressed to.',
 'this email has been scanned for viruses and malware, and may have been automatically archived by mimecast ltd, an innovator in software as a service saas for business.',
 'specializing in; security, archiving and compliance.',
 'this alert was sent according to your settings.',
 'If you have received this message in error',
 'this is a secure',
 'please do not reply to this email address.',
 'unless expressly stated',
 'all rights reserved',
 'If the reader of this message is not the intended recipient',
 'if you are not the intended recipient, you are hereby notified that any disclosure, copying, distribution, or use of the prohibited.',
'Any unauthorized use is strictly prohibited',
'It is intended solely for use by the recipient',
 'You are hereby notified that any disclosure, copying, distribution',
]

jpmc_statutory_msgs = [msg.lower() for msg in jpmc_statutory_msgs]

remove_layout_starting_with = ['from:', 'date:', 'sent:', 'to:', 'cc:',\
                               'subject:', 'importance:', "reply-to:",\
                              'mailto:']

address_pattern = r"\d+\s+[a-zA-Z0-9\s,]+\s+[a-zA-Z]+\s+\d{5}"
url_pattern = "(?:https?:\\/\\/)?(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"

us_phone_num_pattern = '\(?\d{3}\)?[.\-\s]?\d{3}[.\-\s]?\d{4}'
email_id_pattern = "([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"  

def clean_line(line):
    if line.startswith(">"):
        line = line[1:]
    return line.strip()

def remove_layout(text):
    text = re.sub('_x000D_', "", text)
    text = re.sub("\r", "", text)
    text = "\n".join([clean_line(line) for line in text.split("\n")])
    paragraphs = text.split("\n\n")
    # paragraphs.remove("")

    filtered_paragraph=[]

    for paragraph in paragraphs:
        keep_paragraph=True

        for pattern in remove_layout_starting_with:
            if re.search(pattern, paragraph.lower()):
                keep_paragraph=False
                break

        ## remove signature containing address
        if re.search(address_pattern, paragraph.lower()):
            keep_paragraph=False
        ## remove signature containing phone_num
        if re.search(us_phone_num_pattern, paragraph.lower()):
            keep_paragraph=False

        if keep_paragraph:
            filtered_paragraph.append(paragraph)
    # remove layout information (short-text split by '\n\n')           
    filter_text=[i for i in filtered_paragraph if len(i.split())>=10]                 
    return "\n\n".join(filter_text)

def remove_duplicates_freq(text):
    paragraphs=text.split("\n\n")
    unique_paragraphs=[]
    duplicated_paragraphs=set()
    for p in paragraphs:
        if p not in duplicated_paragraphs:
            unique_paragraphs.append(p)
            duplicated_paragraphs.add(p)
            
    return "\n\n".join(unique_paragraphs)

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
    tokens=[word for word in tokens if word not in stopwords_list]
    text=" ".join(tokens)

    return text

def fuzz_search_phrase(text):
    sent_list=[]
    sent_text = nltk.sent_tokenize(text)

    threshold=90

    for p in jpmc_statutory_msgs:
        for sent in sent_text:
            score=fuzz.token_set_ratio(sent, p.lower())
            if score>=threshold:
                sent_text.remove(sent)
    sent_list=[str(v).strip().strip("\n") for v in sent_text]
    return " ".join(sent_list)

def remove_certain_phrase(text):
    sent_list=[]
    sent_text=nltk.sent_tokenize(text)

    for p in jpmc_statutory_msgs:
        regex=re.compile(p,re.IGNORECASE)
        for sent in sent_text:
            if regex.search(sent):
                sent_text.remove(sent)

    return " ".join(sent_text)

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

def main(df_train,df_val,args): 
    
    index_train=df_train.loc[:,['snapshot_id','gcid','thread_id','time']]
    index_train.rename(columns={"time": "time_variable"},inplace=True) ## rename these words to distinguish them from the tfidf possible fetures: time
    y_train=df_train.loc[:,["target"]]
    y_train.rename(columns={"target": "target_variable"},inplace=True) ## rename these words to distinguish them from the tfidf possible fetures: target

    index_val=df_val.loc[:,['snapshot_id','gcid','thread_id','time']]
    index_val.rename(columns={"time": "time_variable"},inplace=True) 
    y_val=df_val.loc[:,["target"]]
    y_val.rename(columns={"target": "target_variable"},inplace=True)

    # index_test=df_test.loc[:,['snapshot_id','gcid','thread_id','time']]
    # index_test.rename(columns={"time": "time_variable"},inplace=True) 
    # y_test=df_test.loc[:,["target"]]
    # y_test.rename(columns={"target": "target_variable"},inplace=True)
    
    ################## TFIDF ######################
    
    bow_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.90, min_df=2, max_features=args.max_feature_num)
    # bag-of-words feature matrix
    bow = bow_vectorizer.fit_transform(df_train['preprocessed_email'])
    # bow = bow_vectorizer.fit_transform(df_train["bag_of_word"])
    train_tfidf = bow.toarray()
    # vocab = bow_vectorizer.vocabulary_.keys()
    vocab = bow_vectorizer.get_feature_names_out()
    vocab = list(vocab)
    train_tfidf = pd.DataFrame(train_tfidf,columns=vocab)
    
    if args.remove_duplicate_thread_id:
        if args.deduplicate_thread_by_time:
            data_dir=os.path.join(args.root_dir, args.output_dir, "dedup_time")
        else:
            data_dir=os.path.join(args.root_dir, args.output_dir, "dedup_target")
    else:
        data_dir=os.path.join(args.root_dir, args.output_dir, "no_dup")
        
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
                 
    ### save bow_vectorizer for further model inference

    joblib.dump(bow_vectorizer, os.path.join(data_dir,'bow_vectorizer.pickle'))
    # bow_vectorizer = pickle.load(open("bow_vectorizer.pickle", "rb"))
    # bow_vectorizer =joblib.load(os.path.join(input_dir,"bow_vectorizer.pickle"))
    # vocab = bow_vectorizer.vocabulary_.keys()

    val_tfidf = bow_vectorizer.transform(df_val['preprocessed_email'])
    # val_tfidf = bow_vectorizer.transform(df_val["bag_of_word"])
    val_tfidf = val_tfidf.toarray()
    val_tfidf = pd.DataFrame(val_tfidf,columns=vocab)

    # test_tfidf = bow_vectorizer.transform(df_test['preprocessed_email'])
    # # test_tfidf = bow_vectorizer.transform(df_test["bag_of_word"])
    # test_tfidf = test_tfidf.toarray()
    # test_tfidf = pd.DataFrame(test_tfidf,columns=vocab)
    
    train_data=pd.concat([train_tfidf.reset_index(drop=True), y_train.reset_index(drop=True), index_train.reset_index(drop=True)],axis=1)
    val_data=pd.concat([val_tfidf.reset_index(drop=True), y_val.reset_index(drop=True), index_val.reset_index(drop=True)],axis=1)
    # test_data=pd.concat([test_tfidf.reset_index(drop=True), y_test.reset_index(drop=True), index_test.reset_index(drop=True)],axis=1)    
    
    if args.feedback_as_complaint:
        train_data.to_pickle(os.path.join(data_dir,"train_data_"+str(args.max_feature_num)))
        val_data.to_pickle(os.path.join(data_dir,"val_data_"+str(args.max_feature_num)))
        # test_data.to_pickle(os.path.join(data_dir,"test_data_"+str(args.max_feature_num)))
    else:
        train_data.to_pickle(os.path.join(data_dir,"train_data_v0_"+str(args.max_feature_num)))
        val_data.to_pickle(os.path.join(data_dir,"val_data_v0_"+str(args.max_feature_num)))
        # test_data.to_pickle(os.path.join(data_dir,"test_data_v0_"+str(args.max_feature_num)))        
    
if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument("--train_undersampling", action="store_true", help="undersampling or not")
    argparser.add_argument("--val_undersampling", action="store_true", help="undersampling or not")

    argparser.add_argument("--train_negative_positive_ratio",  type=int,default=5,help="Undersampling negative vs position ratio in training")
    argparser.add_argument("--val_negative_positive_ratio",  type=int,default=20,help="Undersampling negative vs position ratio in test set")
    argparser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")
    
    argparser.add_argument('--max_feature_num', type=int, default=990)
    argparser.add_argument('--root_dir', type=str, default=None)
    argparser.add_argument('--output_dir', type=str, default="outputs")
    argparser.add_argument('--validation_split', type=float, default=0.2) 
    argparser.add_argument("--start_date", type=str, default='2022-09-01', help="the starting date of time window")
    argparser.add_argument("--end_date", type=str, default='2023-09-30', help="the ending date of time window")
    argparser.add_argument("--test_date", type=str, default="08_23", help="the month for test set")
    argparser.add_argument("--feedback_as_complaint", action="store_true", help="treat feedback as complaint in training and validation ?")
    argparser.add_argument("--close_status", action="store_true")
    argparser.add_argument("--deduplicate_thread_by_time", action="store_true")
    argparser.add_argument("--remove_duplicate_thread_id", action="store_true")
    args,_ = argparser.parse_known_args()
        
    print(args)
    
    
    data_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/datasets/raw_data"
    data_name=[x for x in os.listdir(data_dir) if x.split(".")[-1]=="csv"]
    data_name=sorted(data_name)
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_csv(os.path.join(data_dir,data))
        # x=x.dropna(subset=['email'])
        # x=x[x.email.notna()]
        # x=x[x.email.str.len()>0]
        df=pd.concat([df,x],axis=0,ignore_index=True)
        print("{:<20}{:<20,}".format(data.split("_")[2],x.shape[0]))
    
    if args.close_status:
        df=df[df.state=="closed"]
    df['time'] = pd.to_datetime(df['time'])
    df=df[(df['time']<=args.end_date) & (df['time']>=args.start_date)]

    df['year'] = df.time.apply(lambda x: x.year)
    df['month'] = df.time.apply(lambda x: x.month)
    df['day'] = df.time.apply(lambda x: x.day)
    df.sort_values(by='time', inplace = True)
    df=df.reset_index(drop=True)
    
    #### remove duplicated emails based on thread id
    if args.remove_duplicate_thread_id:
        grouped_df=df.groupby('thread_id')
        if args.deduplicate_thread_by_time:
            sorted_groups=[group.sort_values("time",ascending=False).reset_index(drop=True) for _, group in grouped_df]
        else:
            sorted_groups=[group.sort_values("is_complaint",ascending=False).reset_index(drop=True) for _, group in grouped_df]
        df=pd.concat(sorted_groups).drop_duplicates(subset="thread_id", keep="first").reset_index(drop=True)
    
    
    # df["text_length"]=df['preprocessed_email'].progress_apply(lambda x : len(x.lower().strip().split()))
    keep_columns=['snapshot_id', 'email', 'gcid', 'thread_id', 'state', 'time','is_complaint', \
                  'is_feedback',  'year', 'month','day']
    df=df.loc[:,keep_columns]


    df.sort_values(by='time', inplace = True) 
    if args.test_date=="09_23":
        set_categories=lambda row: "train" if ((row["year"]==2022 and row["month"] in [9,10,11,12]) or \
                                               (row["year"]==2023 and row["month"] in [1,2,3,4,5,6,7,8])
                                              ) else "test"
    elif args.test_date=="08_23":
        df=df[df['time']<'2023-09-01']
        set_categories=lambda row: "train" if ((row["year"]==2022 and row["month"] in [9,10,11,12]) or \
                                               (row["year"]==2023 and row["month"] in [1,2,3,4,5,6,7])
                                              ) else "test"
    else:
        raise ValueError("Invalid test_date. Only Aug and Sep data are supported currently")
        
    df["data_type"]=df.progress_apply(set_categories,axis=1)
    if args.feedback_as_complaint:
        df['target']=np.where((df['is_complaint']=="Y") | (df['is_feedback']=="Y"),1,0)
    else:
        df['target']=df['is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    
    df1=df[df.data_type=="train"]
    df1=df1.reset_index(drop=True)
    df_train,df_val=val_mask_creation(df1,'target', validation_split=args.validation_split)
    
    if args.train_undersampling:
        df_train=utils.under_sampling(df_train,"target",seed=args.seed, negative_positive_ratio=args.train_negative_positive_ratio)
    if args.val_undersampling:
        df_var=utils.under_sampling(df_val,"target",seed=args.seed, negative_positive_ratio=args.val_negative_positive_ratio)    
    
    def data_cleaning(df):
        df['email'] = df['email'].astype(str)
        df['email'] = df['email'].str.lower()
        df['email'] = df['email'].str.replace("`", "'")
        df['email'] = df['email'].str.replace("’", "'") 

        # df['email'] = df['email'].progress_apply(remove_layout)
        df['email'] = df['email'].progress_apply(remove_duplicates_freq)
        df['email'] = df['email'].progress_apply(remove_certain_phrase)
        # df['email'] = df['email'].progress_apply(fuzz_search_phrase)
        df['preprocessed_email'] = df['email'].progress_apply(clean_re)
        return df
    
    df_train=data_cleaning(df_train)
    df_val=data_cleaning(df_val)
    
    
    # df_test=df[df.data_type=="test"]
    # df_test=df_test.reset_index(drop=True)
    # if args.feedback_as_complaint:
    #     ## overwrite the target with the ground true complaint label
    #     df_test['target']=df_test['is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    
    main(df_train, df_val, args)


    