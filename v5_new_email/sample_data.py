import pandas as pd
pd.set_option('display.max_rows', None,'display.max_columns', None)
from tqdm.auto import tqdm
tqdm.pandas(position=0,leave=True)
import numpy as np
import argparse
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default="TFIDF")
    
    args= parser.parse_args()
    
    if args.model_type=="TFIDF":
        test_date="05_23"
        data_dir=os.path.join("/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production", \
                              "tfidf_data", test_date)
        df_train=pd.read_pickle(os.path.join(data_dir,"train_data_990"))
        df_val=pd.read_pickle(os.path.join(data_dir,"val_data_990"))
        df_test=pd.read_pickle(os.path.join(data_dir,"test_data_990"))
        model_dir=os.path.join("/opt/omniai/work/instance1/jupyter/v5_new_email/TFIDF/production", \
                               "tfidf_model", test_date, "995", "lightgbm")
        output_df=pd.read_csv(os.path.join(model_dir , "predictions_95.csv"))
        pred_complaint=output_df.loc[:,["snapshot_id","thread_id","time","Predicted_prob"]]
        pred_complaint.rename(columns={"time":"time_variable","Predicted_prob":"pred_complaint"},inplace=True)
        
        best_threshold=output_df["best_threshold"].unique()[0]
        
        y_test=df_test.loc[:,["target_variable"]]
        index_test=df_test.loc[:,["snapshot_id","thread_id","time_variable"]]
        x_test=df_test.drop(["target_variable","snapshot_id","thread_id","time_variable"],axis=1)
        
        reference_data=pd.concat([index_test,x_test,y_test],axis=1)
        reference_data['time_variable'] = pd.to_datetime(reference_data['time_variable'])
        pred_complaint['time_variable'] = pd.to_datetime(pred_complaint['time_variable'])
        reference_data=pd.merge(left=reference_data, right=pred_complaint, \
                                on=["snapshot_id","thread_id","time_variable"],how="inner")
        
        for col in reference_data.columns:
            col=str(col)
            col2=col.replace(" ","_")
            reference_data.rename(columns={col:col2},inplace=True)      
            
        reference_data.sample(n=20,random_state=101)
        reference_data.to_csv("TFIDF_sample_data.csv")        
        
    elif args.model_type=="language":
        data_path=os.path.join("/opt/omniai/work/instance1/jupyter/", "v5_new_email","datasets")
        df=pd.read_pickle(os.path.join(data_path,"train_val_test_pickle"))
        df['time'] = pd.to_datetime(df['time'])
        df.sort_values(by='time', inplace = True) 
        set_categories=lambda row: "train" if (row["year"] in [2022,2023] and row["month"] in 
                                               [9,10,11,12,1,2,3,4]) else "test"
        df["data_type"]=df.progress_apply(set_categories,axis=1)
        df['target']=np.where((df['is_complaint']=="Y") | (df['is_feedback']=="Y"),1,0)

        df_train=df[df.data_type=="train"]
        df_train=df_train.reset_index(drop=True)

        df_test=df[df.data_type=="test"]
        df_test=df_test.reset_index(drop=True)
        ## overwrite the target with the ground true complaint label
        df_test['target']=df_test['is_complaint'].progress_apply(lambda x: 1 if x=="Y" else 0)

        model_dir=os.path.join("/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning/results/05_23/", 
                               "longformer_base_4096_customized")
        output_df=pd.read_csv(os.path.join(model_dir , "predictions_95.csv"))
        pred_complaint=output_df.loc[:,["snapshot_id","thread_id","Predicted_prob"]]
        pred_complaint.rename(columns={"Predicted_prob":"pred_complaint"},inplace=True)
        reference_data=df_test.loc[:,["snapshot_id","thread_id","time","preprocessed_email","target"]]
        reference_data=pd.merge(left=reference_data, right=pred_complaint, \
                                on=["snapshot_id","thread_id"],how="inner")
        
        
        reference_data.sample(n=20,random_state=101)
        reference_data.to_csv("LM_sample_data.csv")
        