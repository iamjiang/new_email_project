import sys
sys.path.append('/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning/')
sys.path=list(set(sys.path))

import csv
import os
import re
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=True)
from collections import defaultdict
import argparse
import logging

import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk_data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/", "transformers-models","nltk_data")
stopwords_file = open(nltk_data_dir + '/corpora/stopwords/english')
stopwords_list = stopwords_file.readlines()
STOPWORDS=[x.strip() for x in stopwords_list]
nltk.data.path.append(nltk_data_dir)

import torch
print("torch version is {}".format(torch.__version__))
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler

import datasets
from datasets import load_dataset, load_metric, concatenate_datasets,DatasetDict,Dataset
from datasets import load_from_disk
from datasets import disable_caching, enable_caching
enable_caching()

import transformers
print("Transformers version is {}".format(transformers.__version__))

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from accelerate import Accelerator

import utils

from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
class Loader_Creation(torch.utils.data.Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 feature_name
                ):
        super().__init__()
        self.dataset=dataset
        self.tokenizer=tokenizer
        
        self.dataset=self.dataset.map(lambda x:tokenizer(x[feature_name],truncation=True,padding="max_length"), 
                                      batched=True)
        self.dataset.set_format(type="pandas")
        self.dataset=self.dataset[:]
    
    def __len__(self):
        return self.dataset.shape[0]
    
    def __getitem__(self,index):
        
        _ids = self.dataset.loc[index]["input_ids"].squeeze()
        _mask = self.dataset.loc[index]["attention_mask"].squeeze()
        if "target" in self.dataset.columns:
            _target = self.dataset.loc[index]["target"].squeeze()
        _snapshot_id=self.dataset.loc[index]["snapshot_id"].squeeze()
        _gcid=self.dataset.loc[index]["gcid"].squeeze()
        _thread_id=self.dataset.loc[index]["thread_id"].squeeze()
        _time=self.dataset.loc[index]["time"].squeeze()
        # _is_feedback=self.dataset.loc[index]["is_feedback"].squeeze()
        # _text_length=self.dataset.loc[index]["text_length"].squeeze()
        
        if "target" in self.dataset.columns:
            return dict(
                input_ids=_ids,
                attention_mask=_mask,
                labels=_target,
                snapshot_id=_snapshot_id,
                gcid=_gcid,
                thread_id=_thread_id,
                time=_time,
                # is_feedback=_is_feedback,
                # text_length=_text_length
            )
        else:
            return dict(
                input_ids=_ids,
                attention_mask=_mask,
                snapshot_id=_snapshot_id,
                gcid=_gcid,
                thread_id=_thread_id,
                time=_time,
                # is_feedback=_is_feedback,
                # text_length=_text_length
            )           

    
    def collate_fn(self,batch):
        input_ids=torch.stack([torch.tensor(x["input_ids"]) for x in batch])
        attention_mask=torch.stack([torch.tensor(x["attention_mask"]) for x in batch])
        if "target" in self.dataset.columns:
            labels=torch.stack([torch.tensor(x["labels"]) for x in batch])
        snapshot_id=torch.stack([torch.tensor(x["snapshot_id"]) for x in batch])
        gcid=torch.stack([torch.tensor(x["gcid"]) for x in batch])
        thread_id=torch.stack([torch.tensor(x["thread_id"]) for x in batch])
        time=torch.stack([torch.tensor(x["time"]) for x in batch])
        # is_feedback=torch.stack([torch.tensor(x["is_feedback"]) for x in batch])
        # text_length=torch.stack([torch.tensor(x["text_length"]) for x in batch])
        
        pad_token_id=self.tokenizer.pad_token_id
        keep_mask = input_ids.ne(pad_token_id).any(dim=0)
        
        input_ids=input_ids[:, keep_mask]
        attention_mask=attention_mask[:, keep_mask]
        
        if "target" in self.dataset.columns:
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                snapshot_id=snapshot_id,
                gcid=gcid,
                thread_id=thread_id,
                time=time,
                # is_feedback=is_feedback,
                # text_length=text_length
            )
        else:
            return dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                snapshot_id=snapshot_id,
                gcid=gcid,
                thread_id=thread_id,
                time=time,
                # is_feedback=is_feedback,
                # text_length=text_length
            )            

def main(args, df, snapshot_map, gcid_map, thread_map, time_map, device):

    df=Dataset.from_pandas(df)

    model_name=os.path.join(args.root_dir,  args.output_dir)
            
    config=AutoConfig.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name, model_max_length=args.model_max_length)
    if args.model_name=="bigbird-roberta-large":
        config.block_size=16
        config.num_random_blocks=2
        config.attention_type="original_full"
        model=AutoModelForSequenceClassification.from_pretrained(model_name,config=config)
    else:
        model=AutoModelForSequenceClassification.from_pretrained(model_name)
          
    print()
    print(f"The maximal # input tokens : {tokenizer.model_max_length:,}")
    print(f"Vocabulary size : {tokenizer.vocab_size:,}")
    print(f"The # of parameters to be updated : {sum([p.nelement() for p in model.parameters() if p.requires_grad==True]):,}")
    print()

    df_module=Loader_Creation(df, tokenizer,args.feature_name)
    df_dataloader=DataLoader(df_module,
                              shuffle=False,
                              batch_size=args.batch_size,
                              collate_fn=df_module.collate_fn
                              )
    

    print()
    print('{:<30}{:<10,} '.format("mini-batch",len(df_dataloader)))
    print()
    
    accelerator = Accelerator(fp16=args.fp16)
    acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}
    if accelerator.is_main_process:
        accelerator.print("")
        logger.info(f'Accelerator Config: {acc_state}')
        accelerator.print("")
        
    model,  df_dataloader = accelerator.prepare(model, df_dataloader)
    
    def eval_func(data_loader,model,device,num_classes=2):
        
        snapshot_id=[]
        gcid=[]
        thread_id=[]
        time=[]
        # is_feedback=[]
        # text_length=[]
        
        fin_targets=[]
        fin_outputs=[]
        losses=[]
        model.eval()
        if args.multiple_gpus:
            model=model.to(device[0])
            model=torch.nn.DataParallel(model,device_ids=device)
        else:
            model=model.to(device)

        batch_idx=0
        for batch in tqdm(data_loader, position=0, leave=True):
            if not args.multiple_gpus:
                batch={k:v.type(torch.LongTensor).to(device) for k,v in batch.items()}
            else:
                batch={k:v.type(torch.LongTensor).to(device[0]) for k,v in batch.items()}
                
            inputs={k:v  for k,v in batch.items() if k in ["input_ids","attention_mask"]}
            with torch.no_grad():
                outputs=model(**inputs)
            logits=outputs['logits']

            loss = F.cross_entropy(logits.view(-1, num_classes), batch["labels"])
   
            losses.append(loss.item())
            if "labels" in batch.keys():
                fin_targets.append(batch["labels"].cpu().detach().numpy())
            fin_outputs.append(torch.softmax(logits.view(-1, num_classes),dim=1).cpu().detach().numpy())  
            
            snapshot_id.append(batch["snapshot_id"].cpu().detach().numpy())
            gcid.append(batch["gcid"].cpu().detach().numpy())
            thread_id.append(batch["thread_id"].cpu().detach().numpy())
            time.append(batch["time"].cpu().detach().numpy())
            # is_feedback.append(batch["is_feedback"].cpu().detach().numpy())
            # text_length.append(batch["text_length"].cpu().detach().numpy())

            batch_idx+=1
            
        if "labels" in batch.keys():
            return np.concatenate(fin_outputs),np.concatenate(fin_targets),\
                   np.concatenate(snapshot_id),np.concatenate(gcid),np.concatenate(thread_id) ,np.concatenate(time) 
        else:
            return np.concatenate(fin_outputs),np.concatenate(snapshot_id),np.concatenate(gcid),np.concatenate(thread_id),np.concatenate(time) 
    
    if "labels"  in next(iter(df_dataloader)).keys():
        df_pred,df_target,df_snapshot_id,df_gcid,df_thread_id,df_time=eval_func(df_dataloader,model, device)
    else:
        df_pred,df_snapshot_id,df_gcid,df_thread_id,df_time=eval_func(df_dataloader,model, device)        
    
    output_dir=os.path.join(args.data_dir,"pred_output", args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    snapshot_inverse_map={v:k for k, v in snapshot_map.items()}
    gcid_inverse_map={v:k for k, v in gcid_map.items()}
    thread_inverse_map={v:k for k, v in thread_map.items()}
    time_inverse_map={v:k for k, v in time_map.items()}
    
    if "labels"  in next(iter(df_dataloader)).keys():
        
        result_df=pd.DataFrame({'snapshot_id' : df_snapshot_id.tolist(), 
                                 'gcid' : df_gcid.tolist(),
                                 'thread_id' : df_thread_id.tolist(),
                                 'time' : df_time.tolist(),
                                 'target' : df_target.tolist(),
                                 'Predicted_prob' : df_pred[:,1].tolist()})
    else:
        result_df=pd.DataFrame({'snapshot_id' : df_snapshot_id.tolist(), 
                                 'gcid' : df_gcid.tolist(),
                                 'thread_id' : df_thread_id.tolist(),
                                 'time' : df_time.tolist(),
                                 'Predicted_prob' : df_pred[:,1].tolist()})     
        
    
    result_df["snapshot_id"]=list(map(snapshot_inverse_map.get,result_df["snapshot_id"]))
    result_df["gcid"]=list(map(gcid_inverse_map.get,result_df["gcid"]))
    result_df["thread_id"]=list(map(thread_inverse_map.get,result_df["thread_id"]))
    result_df["time"]=list(map(time_inverse_map.get,result_df["time"]))
    
    return result_df

#     file_name="predictions_"+args.data_name.split(".")[0][-1]+".csv"
    
#     if 'labels' in next(iter(df_dataloader)).keys():
#         fieldnames = ['snapshot_id','gcid','thread_id','time','text_length','True_label', 'Predicted_prob']
        
#         with open(os.path.join(output_dir,file_name),'w') as csv_file:
#             writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
#             writer.writeheader()
#             for i, j, s, k, m, n, p in zip(df_snapshot_id, df_gcid,df_thread_id, df_time, df_text_length,df_target, df_pred[:,1]):
#                 writer.writerow( {'snapshot_id':snapshot_inverse_map[i],'gcid':gcid_inverse_map[j],'thread_id':thread_inverse_map[s],'time':time_inverse_map[k],
#                      'text_length': m, 'True_label': n, 'Predicted_prob': p})  
#     else:
#         fieldnames = ['snapshot_id','gcid','thread_id','time','text_length','Predicted_prob']

#         with open(os.path.join(output_dir,file_name),'w') as csv_file:
#             writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
#             writer.writeheader()
#             for i, j, s, k, m, p in zip(df_snapshot_id, df_gcid,df_thread_id, df_time, df_text_length, df_pred[:,1]):
#                 writer.writerow( {'snapshot_id':snapshot_inverse_map[i],'gcid':gcid_inverse_map[j],'thread_id':thread_inverse_map[s],'time':time_inverse_map[k],
#                      'text_length': m, 'Predicted_prob': p}) 
                

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Pretrained Language Model')
    parser.add_argument('--multiple_gpus', action="store_true", help="use multiple gpus or not")
    parser.add_argument('--gpus', type=int, default=[], nargs='+', help='used gpu')
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed",  type=int,default=101,
            help="random seed for np.random.seed, torch.manual_seed and torch.cuda.manual_seed.")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument('--fp16',  action="store_true")
    parser.add_argument('--model_name', type=str, required = True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--feature_name", default="preprocessed_email", type=str)
    
    parser.add_argument('--customized_model',  action="store_true")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--test_date", type=str, default="09_23", help="the month for test set")   
    parser.add_argument('--root_dir', type=str, default="/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning/results/09_23")
    parser.add_argument("--data_dir", type=str, default="/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning/daily_inference/data") 
    parser.add_argument("--data_name", type=str, default="email_data_0.parquet")
    
    args= parser.parse_args()

    if args.customized_model:
        if len(args.model_name.split("-"))>=3:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_" + args.model_name.split("-")[2] + "_customized"
        else:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1] + "_customized"
    else:
        if len(args.model_name.split("-"))>=3:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1]+ "_" + args.model_name.split("-")[2]
        else:
            args.output_dir=args.model_name.split("-")[0] + "_" + args.model_name.split("-")[1]
            
    seed_everything(args.seed)

    print()
    print(args)
    print()
    
    df=pd.read_parquet(os.path.join(args.data_dir,args.data_name), engine="pyarrow")
    # df=df.sample(500)
    
    df['email'] = df['email'].astype(str)
    df['email'] = df['email'].str.lower()
    df['email'] = df['email'].str.replace("`", "'")
    df['email'] = df['email'].str.replace("’", "'")
    # df['email'] = df['email'].progress_apply(remove_layout)
    df['email'] = df['email'].progress_apply(remove_duplicates_freq)
    df['email'] = df['email'].progress_apply(remove_certain_phrase)
    # df['email'] = df['email'].progress_apply(fuzz_search_phrase)
    df['preprocessed_email'] = df['email'].progress_apply(clean_re)      
    
    ### validation set use potential complaint :  is_complaint=Y or is_feedback=Y
    ### test set use ground true complaint: is_complaint=Y
    # df['target']=np.where((df['is_complaint']=="Y") | (df['is_feedback']==1),1,0)
    if "is_complaint" in df.columns:
        df['target']=df['is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)
    # df['is_feedback']=df['is_feedback'].progress_apply(lambda x: 1 if x=="Y" else 0)
    
    hf_data=Dataset.from_pandas(df)
    # def compute_lenth(example):
    #     return {"text_length":len(example["input_ids"])}
    
    model_name=os.path.join(args.root_dir,  args.output_dir)
    config=AutoConfig.from_pretrained(model_name)
    tokenizer=AutoTokenizer.from_pretrained(model_name, model_max_length=args.model_max_length)
    
    hf_data=hf_data.map(lambda x: tokenizer(x["email"]),batched=True)
    # hf_data=hf_data.map(compute_lenth)
    hf_data.set_format(type="pandas")
    df=hf_data[:]

    _dir=os.path.join(args.data_dir,"pred_output", args.output_dir)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
            
    remaining=df[df['preprocessed_email'].str.len()==0]
    if remaining.shape[0]>0:
        if "is_complaint" in df.columns:
            keep_columns=['snapshot_id', 'gcid', 'thread_id', 'time', 'target']
            remaining=remaining.loc[:,keep_columns]  
            remaining=remaining.reset_index(drop=True)
            remaining['Predicted_prob']=0
        else:
            keep_columns=['snapshot_id', 'gcid', 'thread_id', 'time']
            remaining=remaining.loc[:,keep_columns]
            remaining['Predicted_prob']=0
            
        df=df[df['preprocessed_email'].str.len()>0]
        
        # file_name="remaining_"+args.data_name.split(".")[0][-1]+".csv"
        # remaining.to_csv(os.path.join(_dir, file_name), quoting=csv.QUOTE_ALL, index=False)
        print()
        print("{:<30}{:<15,}".format("samples without email text",remaining.shape[0]))
        print()

    if args.multiple_gpus:
        device=[torch.device(f"cuda:{i}") for i in args.gpus]
    else:
        device=torch.device(args.device)
    
    unique_snapshot=df["snapshot_id"].unique()
    snapshot_map={v:idx for idx ,v in enumerate(unique_snapshot)}
    
    unique_gcid=df["gcid"].unique()
    gcid_map={v:idx for idx ,v in enumerate(unique_gcid)}
    
    unique_thread=df["thread_id"].unique()
    thread_map={v:idx for idx ,v in enumerate(unique_thread)}
    
    unique_time=df["time"].unique()
    time_map={v:idx for idx ,v in enumerate(unique_time)}
    
    df["snapshot_id"]=list(map(snapshot_map.get,df["snapshot_id"]))
    df["gcid"]=list(map(gcid_map.get,df["gcid"]))
    df["thread_id"]=list(map(thread_map.get,df["thread_id"]))
    df["time"]=list(map(time_map.get,df["time"]))
    
    keep_columns=['snapshot_id',  'gcid', 'thread_id', 'time','target', 'preprocessed_email']
    df=df.loc[:,keep_columns]
    
    result_df=main(args,df, snapshot_map, gcid_map, thread_map, time_map, device)
    
    
    email_data=pd.concat([remaining,result_df],axis=0).reset_index(drop=True)
    
    output_dir=os.path.join(args.data_dir,"pred_output", args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_name="inf_"+"".join(args.test_date.split("_"))+"_"+args.data_name.split(".")[0][-1]
    email_data.to_pickle(os.path.join(output_dir, output_name))
    