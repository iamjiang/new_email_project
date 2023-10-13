import re
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

import traceback
from app.config import modellogger
from app.utils.model_exception import *
from app.utils.app_utils import get_nltk_data_dir


def get_stopwords():
    nltk_data_dir = get_nltk_data_dir()
    modellogger.info(f"nltk_data_dir : {nltk_data_dir}")
    nltk_file_name = f'{nltk_data_dir}/corpora/stopwords/english'
    stopwords_file = open(nltk_file_name)
    stopwords_list = stopwords_file.readlines()
    return [x.strip() for x in stopwords_list]

STOPWORDS = get_stopwords()


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
    text = re.sub(r"\d+","",text)
    tokens = [word for word in word_tokenize(text) if not word.isnumeric() ]
    ##remove stop word 
    tokens = [word for word in tokens if word not in STOPWORDS]
    text = " ".join(tokens)
    
    return text


def preprocess(rawData, myModels):

    try:

        df = pd.DataFrame.from_records(rawData)
        df['email'] = df['email'].astype(str)
        df['preprocessed_email'] = df['email'].progress_apply(clean_re)
        # keep_columns = ['snapshot_id',  'thread_id', 'time','email', 'preprocessed_email','is_complaint', 'gcid']
        # df = df.loc[:,keep_columns]
        # df['target'] = df['is_complaint'].progress_apply(lambda x: 1 if x == "Y" else 0)

        # index = df.loc[:,['snapshot_id','thread_id','time']]
        # index.rename(columns={"time": "time_variable"},inplace=True)
        # y_true = df.loc[:,["target"]]
        # y_true.rename(columns={"target": "target_variable"},inplace=True)

        bow_vectorizer = myModels.bow_vectorizer
        tfidf_feature = bow_vectorizer.transform(df['preprocessed_email'])
        tfidf_feature = tfidf_feature.toarray()
        # vocab = bow_vectorizer.vocabulary_.keys()
        vocab = bow_vectorizer.get_feature_names_out()
#        print(f"vocab: {vocab}")

        new_col = []
        for col in list(vocab):
            col = str(col)
            col2 = col.replace(" ","_")
            new_col.append(col2)

        tfidf_feature = pd.DataFrame(tfidf_feature,columns=new_col)

        return tfidf_feature

    except Exception as e:
        errMsg = traceback.format_exc()
        raise ModelPreProcessException(errMsg)