import os
import pandas as pd

root_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/datasets"

df=pd.read_pickle(os.path.join(root_dir,"train_val_test_pickle"))

def label_distribution(df,year,month):
    df=df[(df.year==year) & (df.month==month)]
    tempt1=pd.DataFrame(df["is_complaint"].value_counts(dropna=False)).reset_index().\
    rename(columns={'index':'is_complaint','is_complaint':'count'})
    tempt2=pd.DataFrame(df["is_complaint"].value_counts(dropna=False,normalize=True)).reset_index().\
    rename(columns={'index':'is_complaint','is_complaint':'percentage'})
    tempt3=tempt1.merge(tempt2, on="is_complaint", how="inner")
    tempt3['year']=year
    tempt3['month']=month
    tempt3=tempt3.loc[:,['year','month','is_complaint','count','percentage']]
    return tempt3

dist_df=pd.DataFrame()
# dist_df=pd.concat([dist_df,label_distribution(df,2022,8)])
dist_df=pd.concat([dist_df,label_distribution(df,2022,9)])
dist_df=pd.concat([dist_df,label_distribution(df,2022,10)])
dist_df=pd.concat([dist_df,label_distribution(df,2022,11)])
dist_df=pd.concat([dist_df,label_distribution(df,2022,12)])
dist_df=pd.concat([dist_df,label_distribution(df,2023,1)])
dist_df=pd.concat([dist_df,label_distribution(df,2023,2)])
dist_df=pd.concat([dist_df,label_distribution(df,2023,3)])
dist_df=pd.concat([dist_df,label_distribution(df,2023,4)])
dist_df=pd.concat([dist_df,label_distribution(df,2023,5)])

print(dist_df)

