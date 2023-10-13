import argparse
import pickle
import os
import pandas as pd
import joblib

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_feature_num', type=int, default=990)
    argparser.add_argument('--model_name', type=str, default="lightgbm")
    argparser.add_argument('--output_dir', type=str, default=None)
    argparser.add_argument("--test_date", type=str, default="08_23", help="the month for test set")
    argparser.add_argument("--val_min_recall", default=0.95, type=float, help="minimal recall for valiation dataset")
    argparser.add_argument("--feedback_as_complaint", action="store_true", help="treat feedback as complaint in training and validation ?")
    args,_ = argparser.parse_known_args()

    # args.output_dir="/opt/omniai/work/instance1/jupyter/v5_new_email/containerization/complaint-model/TFIDF_container/app/model"
        
    print(args)
    
    bow_vectorizer =joblib.load(os.path.join(args.output_dir,"outputs","bow_vectorizer.pickle"))
    # vocab = bow_vectorizer.vocabulary_.keys()
    vocab = bow_vectorizer.get_feature_names_out()
    
    if args.feedback_as_complaint:
        model_dir=os.path.join(args.output_dir,'tfidf_model', args.test_date, str(args.max_feature_num))
    else:
        model_dir=os.path.join(args.output_dir,'tfidf_model_v0', args.test_date, str(args.max_feature_num))
        
    model = joblib.load(os.path.join(model_dir,'lightgbm_model.pkl'))
    
    csv_file="predictions_"+str(args.val_min_recall).split(".")[1]+".csv"
    best_threshold=pd.read_csv(os.path.join(model_dir, args.model_name, csv_file)).best_threshold.unique()[0]

    print()
    print(f"Best Threshold Value : {best_threshold}")
    print()
    
    if args.feedback_as_complaint:
        pickle_file=args.model_name+".pkl"
    else:
        pickle_file=args.model_name+"_v0.pkl"
        
    model_dict={"bow_vectorizer": bow_vectorizer,"model":model, "best_threshold":best_threshold}
    
    with open(os.path.join(args.output_dir,pickle_file),"wb") as f:
        pickle.dump(model_dict,f)
        
