import sys
sys.path.append('/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning/')
sys.path=list(set(sys.path))

import utils

import argparse
import os
import re
import pandas as pd
pd.options.mode.chained_assignment=None
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=True)
import joblib

from fuzzywuzzy import fuzz

import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
stopwords_file = open(nltk_data_dir + '/corpora/stopwords/english')
stopwords_list = stopwords_file.readlines()
STOPWORDS=[x.strip() for x in stopwords_list]
nltk.data.path.append(nltk_data_dir)


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

class model_pred():
    def __init__(self,args, data):
        self.root_dir=args.root_dir
        self.data=data
    def model_load(self):
        bow_vectorizer =joblib.load(os.path.join(self.root_dir, "bow_vectorizer.pickle"))
        if args.feedback_as_complaint:
            predict_model=joblib.load(os.path.join(self.root_dir, "tfidf_model" ,"lightgbm_model.pkl"))
        else:
            predict_model=joblib.load(os.path.join(self.root_dir, "tfidf_model_v0" ,"lightgbm_model.pkl"))
        
        tfidf_feature = bow_vectorizer.transform(self.data['preprocessed_email'])
        tfidf_feature = tfidf_feature.toarray()
        vocab = bow_vectorizer.get_feature_names_out()

        new_col=[]
        for col in list(vocab):
            col=str(col)
            col2=col.replace(" ","_")
            new_col.append(col2)

        tfidf_feature = pd.DataFrame(tfidf_feature,columns=new_col)
        y_predicts = predict_model.predict(tfidf_feature)
        
        ### For those email with all TFIDF value being zero, assume it is not complaint email
        # y_predicts[tfidf_feature.iloc[:,0:990].sum(axis=1)==0]=0
        # print()
        # print(f"There are {tfidf_feature[tfidf_feature.iloc[:,0:990].sum(axis=1)==0].shape[0]} rows with all TFIDF values being zero")
        # print()  
        
        return y_predicts
    
    

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_dir', type=str, default=None)
    argparser.add_argument('--data_name', type=str, default=None)
    argparser.add_argument('--root_dir', type=str, default=None)
    argparser.add_argument("--feedback_as_complaint", action="store_true")
    argparser.add_argument('--output_dir', type=str, default=None)
    argparser.add_argument('--output_name', type=str, default=None)
    args,_ = argparser.parse_known_args()
        
    print(args)

    # df=pd.read_csv(os.path.join(args.data_dir,args.data_name))
    df=pd.read_parquet(os.path.join(args.data_dir,args.data_name), engine="pyarrow")

    df['email'] = df['email'].astype(str)
    df['email'] = df['email'].str.lower()
    df['email'] = df['email'].str.replace("`", "'")
    df['email'] = df['email'].str.replace("’", "'") 
    
    # df['email'] = df['email'].progress_apply(remove_layout)
    df['email'] = df['email'].progress_apply(remove_duplicates_freq)
    
    df['email'] = df['email'].progress_apply(remove_certain_phrase)
    # df['email'] = df['email'].progress_apply(fuzz_search_phrase)
    df['preprocessed_email'] = df['email'].progress_apply(clean_re)
    
    df1=df[df['preprocessed_email'].str.len()==0].reset_index(drop=True)
    df1["prob_pred"]=0
    print()
    print(f"There are {df1.shape[0]} rows with email textbody being empty")
    print()
    
    df2=df[df['preprocessed_email'].str.len()>0].reset_index(drop=True)
    model_class=model_pred(args, df2)
    y_predicts=model_class.model_load()
    df2["prob_pred"]=y_predicts
    
    # email_data=pd.concat([df1,df2],axis=0).reset_index(drop=True)
    email_data=df2.copy()

    # email_data['time'] = pd.to_datetime(email_data['time'])
    # email_data['year'] = email_data.time.apply(lambda x: x.year)
    # email_data['month'] = email_data.time.apply(lambda x: x.month)
    # email_data['day'] = email_data.time.apply(lambda x: x.day)
#     email_data['target']=email_data["is_complaint"].progress_apply(lambda x: 1 if x=="Y" else 0)
    
#     best_threshold=utils.find_optimal_threshold(email_data['target'].values.squeeze(), email_data["prob_pred"].values.squeeze(), \
#                                                 min_recall=args.val_min_recall, pos_label=False)
    
#     email_data["best_threshold"]=best_threshold

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    email_data.to_pickle(os.path.join(args.output_dir, args.output_name))
    
    