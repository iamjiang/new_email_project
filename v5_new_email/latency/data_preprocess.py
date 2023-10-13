import time
import pandas as pd
pd.options.mode.chained_assignment=None #default="warn"
import numpy as np
from numpy import savez_compressed, load
import itertools
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity

import re
import time
import os
import pickle

import datasets
from datasets import load_dataset, load_metric, Dataset, concatenate_datasets,DatasetDict
from datasets import load_from_disk
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

from fuzzywuzzy import fuzz

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
# Load the stopwords from the new directory
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
stopwords_file = open(nltk_data_dir + '/corpora/stopwords/english')
stopwords_list = stopwords_file.readlines()
nltk.data.path.append(nltk_data_dir)

from transformers import AutoTokenizer
from datasets import Dataset


split_num=15

root_dir="/opt/omniai/work/instance1/jupyter/v4_new_email/datasets/raw_data"
data_name='email_data_2022-10-01_to_2022-10-31.csv'
# data_name=[x for x in os.listdir(root_dir) if x.split(".")[-1]=="csv"]
# df=pd.DataFrame()
# for data in data_name:
#     x=pd.read_csv(os.path.join(root_dir,data))
#     ### remove all missing email text
#     x=x.dropna(subset=['email'])
#     x=x[x.email.notna()]
#     x=x[x.email.str.len()>0]
#     df=pd.concat([df,x],axis=0,ignore_index=True)
#     print("{:<20}{:<20,}".format(data.split("_")[2],x.shape[0]))

start=time.time()
df=pd.read_csv(os.path.join(root_dir, 'email_data_2022-10-01_to_2022-10-31.csv'))

df.dropna(subset=['email'],inplace=True)
df=df[df.email.notna()]
df=df[df.email.str.len()>0]

df['time'] = pd.to_datetime(df['time'])
df['year'] = df.time.apply(lambda x: x.year)
df['month'] = df.time.apply(lambda x: x.month)
df['day'] = df.time.apply(lambda x: x.day)
df.sort_values(by='time', inplace = True) 

df=df[df.month!=8] ## 2022 August only have 4 non-complaint emails

df['email'] = df['email'].str.replace("`", "'")
df['email'] = df['email'].str.replace("’", "'")  

# df= df.sample(5000)

grouped_df=df.groupby('thread_id')
sorted_groups=[group.sort_values("time",ascending=False).reset_index(drop=True) \
               for _, group in grouped_df]
df=pd.concat(sorted_groups).drop_duplicates(subset="thread_id", keep="first")\
.reset_index(drop=True)

df=df[df.state=="closed"]

end=time.time()
print("It takes {:.4f} seconds".format(end-start))
print(df.shape)
print(df['is_complaint'].value_counts())

# split_num=args.split_num

# output_dir=os.path.join(os.getcwd(), "split_data")

# dfs=np.array_split(df,split_num)
# for i,f in enumerate(dfs):
#     filename=f"email_data_{i}.csv"
#     f.to_csv(os.path.join(output_dir,filename),index=False)