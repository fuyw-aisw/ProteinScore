import random
import argparse
import os
from utils import log
import pickle as pkl
from network import Attention_subloc
from subloc_data import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

LOCALIZATION_CATEGORIES = ["Membrane","Cytoplasm","Nucleus","Extracellular","Cell membrane","Mitochondrion","Plastid","Endoplasmic reticulum","Lysosome/Vacuole","Golgi apparatus","Peroxisome"]
MEMBRANE_CATEGORIES = ["Peripheral","Transmembrane","LipidAnchor","Soluble"]

def test(args):
    if args.task == "sl":
        dir_input = "data/prot_loc/swissprot_location.csv"
        CATEGORIES = LOCALIZATION_CATEGORIES
    else:
        dir_input = "data/prot_loc/swissprot_membrane.csv"
        CATEGORIES = MEMBRANE_CATEGORIES
    data_df = pd.read_csv(dir_input)
    test_df = data_df[data_df["Fold"] == 5]
    test_set = list(zip(test_df["ACC"].tolist(),test_df["Sequence"].tolist(),test_df[CATEGORIES].values.tolist()))
    test_set = CustomDataSet(test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)
    k_fold = 5
    model = Attention_subloc(pretrain_model=args.model,task=args.task, device=args.device).to(args.device)
    for i in range(k_fold):
        model_pt= f"subloc_model/model_mf_{args.task}_64_0.001_{i}_fold.pt"
        model.load_state_dict(torch.load(model_pt,map_location=args.device))
        model.eval()
    
        y_pred_all = []
        y_true_all = []
        x_ACC_all = []
        g_feat_all = []
        pred_dict = {}
        target_dict = {}
        embed_dict = {}
        with torch.no_grad():
            for idx_batch, batch in enumerate(test_loader):
                #print(batch)
                x_proteins, y_true,x_ACC = batch[0].to(args.device), batch[1].to(args.device),batch[2]
                
                y_pred,g_feat = model(x_proteins)
                y_pred = torch.sigmoid(y_pred)
                y_pred_all.append(y_pred)
                y_true_all.append(y_true)
                x_ACC_all.extend(x_ACC)
                g_feat_all.append(g_feat)
            y_pred_all = torch.cat(y_pred_all, dim=0).float().cpu().numpy()
            y_true_all = torch.cat(y_true_all, dim=0).float().cpu().numpy()
            g_feat_all = torch.cat(g_feat_all, dim=0).float().cpu().numpy()
            for j in range(len(x_ACC_all)):
                pred_dict[x_ACC_all[j]] = y_pred_all[j]
                target_dict[x_ACC_all[j]] = y_true_all[j]
                embed_dict[x_ACC_all[j]] = g_feat_all[j]
            target_df = pd.DataFrame(target_dict.items(),columns=["ACC","targets"])
            pred_df = pd.DataFrame(pred_dict.items(), columns=['ACC', 'preds'])
            embed_df = pd.DataFrame(embed_dict.items(), columns=['ACC', 'embeds'])
            df = target_df.merge(pred_df).merge(embed_df)
        result_name = "test_results/test_model_mf_subloc_"+str(args.task)+'_'+str(i)+"_fold.pkl"
        with open(result_name, "wb") as f:
            pkl.dump(df, f)

    

if __name__ == "__main__":
    seed = 2222
    p = argparse.ArgumentParser()
    p.add_argument('--device_num', type=str, default='0', help='')
    p.add_argument('--batch_size', type=int, default=64, help='')
    #p.add_argument('--model', type=str, default='sortedmodel/model_mf_alpha_edge_esm1b_512.pt', help='') #default 
    p.add_argument('--model', type=str, default=None, help='')
    p.add_argument('--task',type=str,default="sl") #sl:subcellular localization;mp:membrane type prediction
    p.add_argument('--learning_rate',type=float,default=1e-4)
    #p.add_argument('--epoch_th',type=str,default="10")
    
    args = p.parse_args()
    print(args)
    if args.device_num != '':
        args.device = "cuda:" + args.device_num
    #args.model_save_path = "dti_model/model_"
    #args.model
    test(args)
