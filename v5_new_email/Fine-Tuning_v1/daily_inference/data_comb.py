import sys
sys.path.append('/opt/omniai/work/instance1/jupyter/v5_new_email/Fine-Tuning/')
sys.path=list(set(sys.path))

import argparse
import os
import pandas as pd
import utils


if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--root_dir', type=str, default=None)
    argparser.add_argument('--model_name', type=str, required = True)
    argparser.add_argument('--customized_model',  action="store_true")
    argparser.add_argument('--output_name', type=str, default=None)
    args,_ = argparser.parse_known_args()
        
    print(args)
    
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
            
    data_dir=os.path.join(args.root_dir,args.output_dir)
    data_name=[x for x in os.listdir(data_dir) if x.split("_")[0]=="inf"]

    data_name=sorted(data_name)
    df=pd.DataFrame()
    for data in data_name:
        x=pd.read_pickle(os.path.join(data_dir,data))
        df=pd.concat([df,x],axis=0,ignore_index=True)
        # print("{:<20}{:<20,}".format(data.split("_")[2],x.shape[0]))
        
    df=df.reset_index(drop=True)
#     best_threshold=utils.find_optimal_threshold(df['target'].values.squeeze(), df["prob_pred"].values.squeeze(), \
#                                                 min_recall=args.val_min_recall, pos_label=False)
    
#     print("{:<20}{:<20,}".format("best_threshold : ",best_threshold))
    
#     df["best_threshold"]=best_threshold
    
    
    df.to_pickle(os.path.join(data_dir, args.output_name))
    
    for file in data_name:
        try:
            os.remove(os.path.join(data_dir,file))
        except FileNotFoundError:
            print("File not found")
