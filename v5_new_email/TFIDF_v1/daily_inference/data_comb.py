import sys
sys.path.append('/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning/')
sys.path=list(set(sys.path))

import argparse
import os
import pandas as pd
import utils


if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--data_dir', type=str, default=None)
    argparser.add_argument("--feedback_as_complaint", action="store_true")
    argparser.add_argument('--output_name', type=str, default=None)
    args,_ = argparser.parse_known_args()
        
    print(args)
    if args.feedback_as_complaint:
        data_name=[x for x in os.listdir(args.data_dir) if x.split("_")[2]=="feedback"]
    else:
        data_name=[x for x in os.listdir(args.data_dir) if x.split("_")[2]=="no"]
        
    data_name=sorted(data_name)
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(args.data_dir,data))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        # print("{:<20}{:<20,}".format(data.split("_")[2],x.shape[0]))
        
    df=df.reset_index(drop=True)
#     best_threshold=utils.find_optimal_threshold(df['target'].values.squeeze(), df["prob_pred"].values.squeeze(), \
#                                                 min_recall=args.val_min_recall, pos_label=False)
    
#     print("{:<20}{:<20,}".format("best_threshold : ",best_threshold))
    
#     df["best_threshold"]=best_threshold
    
    
    df.to_pickle(os.path.join(args.data_dir, args.output_name))
    
    for file in data_name:
        try:
            os.remove(os.path.join(args.data_dir,file))
        except FileNotFoundError:
            print("File not found")
