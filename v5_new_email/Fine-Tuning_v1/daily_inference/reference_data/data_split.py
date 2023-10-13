import argparse
import pandas as pd
import numpy as np
import os
import shutil
import torch
from tqdm import tqdm
tqdm.pandas(position=0,leave=True)

def main(data, args):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    split_num=2*torch.cuda.device_count()
    dfs=np.array_split(data,split_num)
    for i,f in enumerate(dfs):
        # filename=f"email_data_{i}.csv"
        filename=f"email_data_{i}.parquet"
        # f.to_csv(os.path.join(output_dir,filename),index=False)
        f.reset_index(drop=True).to_parquet(os.path.join(args.output_dir,filename), index=False, engine="pyarrow")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="email_data_2023-09-01_to_2023-09-29.csv") 
    parser.add_argument("--root_dir", type=str, default="/opt/omniai/work/instance1/jupyter/v5_new_email/datasets/raw_data")    
    parser.add_argument("--output_dir", type=str, default="/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning/daily_inference/data")
    args= parser.parse_args()
    print()
    print(args)
    print()

    df=pd.read_csv(os.path.join(args.root_dir,args.data_name))

    df['time'] = pd.to_datetime(df['time'])
    df[(df.time>=args.data_name.split("_")[2]) & (df.time<=args.data_name.split("_")[4].split(".")[0])]
    
    # df['year'] = df.time.apply(lambda x: x.year)
    # df['month'] = df.time.apply(lambda x: x.month)
    # df['day'] = df.time.apply(lambda x: x.day)
    df.sort_values(by='time', inplace = True) 
    df=df.reset_index(drop=True)
    
    if "is_complaint" in df.columns:
        df['target']=df["is_complaint"].progress_apply(lambda x: 1 if x=="Y" else 0)

    df['email'] = df['email'].str.replace("`", "'")
    df['email'] = df['email'].str.replace("’", "'")  
    
    main(df, args)
    