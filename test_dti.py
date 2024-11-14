import random
import argparse
import os
from utils import log
import pickle as pkl
from network import AttentionDTI
from DTI_data import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc


def test(args):
    dir_input = ('./data/DTI/{}.txt'.format(args.dataset))
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    protein_ids,smiles,proteins,labels,fold_num = [],[],[],[],[]
    for i,pair in enumerate(data_list):
        pair = pair.strip().split()
        protein_id,compoundstr,proteinstr,label,fold = pair[-5], pair[-4],pair[-3], pair[-2], pair[-1]
        protein_ids.append(protein_id)  
        smiles.append(compoundstr)
        proteins.append(proteinstr)
        labels.append(float(label))
        fold_num.append(float(fold))
    
    data_list = list(zip(protein_ids,smiles,proteins,labels,fold_num))
    test_set = [(item[0],item[1],item[2],item[3]) for item in data_list if float(item[-1])==5]
    test_set = CustomDataSet(test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)
    model = AttentionDTI(pretrain_model=args.model,device=args.device).to(args.device) #initialize by the pretained model
    k_fold = 5
    precision_list,recall_list,accuracy_list,auc_list,prc_list = [],[],[],[],[]
    for i in range(k_fold):
        model_pt= f"dti_model/model_mf_{args.dataset}_48_0.0001_{i}_fold.pt"
        model.load_state_dict(torch.load(model_pt,map_location=args.device))
        model.eval()
    
        y_pred_labels = []
        y_pred_scores = []
        y_true_all = []
        with torch.no_grad():
            for idx_batch, batch in enumerate(test_loader):
                x_smiles, x_proteins, y_true = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
                y_pred,_ = model(x_smiles, x_proteins)
                y_true = y_true.to('cpu').data.numpy()
                y_score = F.softmax(y_pred, 1).to('cpu').data.numpy()
                y_label = np.argmax(y_score, axis=1)
                y_score = y_score[:,1]
                y_pred_scores.extend(y_score)
                y_pred_labels.extend(y_label)
                y_true_all.extend(y_true)
                #protein_ids.extend(protein_id)
                #smiles.extend(smile)
                print(1)
        result_name = "test_results/test_model_"+str(args.dataset)+'_'+str(i)+"_fold.pkl"
        with open(result_name, "wb") as f:
            pkl.dump([y_true_all,y_pred_scores, y_pred_labels], f)
        precision = precision_score(y_true_all,y_pred_labels)
        recall = recall_score(y_true_all,y_pred_labels)
        accuracy = accuracy_score(y_true_all,y_pred_labels)
        roc_auc = roc_auc_score(y_true_all,y_pred_scores)
        tpr, fpr, _ = precision_recall_curve(y_true_all,y_pred_scores)
        prc = auc(fpr,tpr)
        precision_list.append(precision)
        recall_list.append(recall)
        accuracy_list.append(accuracy)
        auc_list.append(roc_auc)
        prc_list.append(prc)
        log(f"{i} test_dataset ||| accuracy: {round(float(accuracy),4)} |||precision: {round(float(precision),4)} |||recall: {round(float(recall),4)} ||| roc_auc: {round(float(roc_auc),4)} ||| prc: {round(float(prc),4)}")
    accuracy_mean, accuracy_std = np.mean(accuracy_list),np.std(accuracy_list)
    precision_mean, precision_std = np.mean(precision_list),np.std(precision_list)
    recall_mean, recall_std = np.mean(recall_list),np.std(recall_list)
    auc_mean, auc_std = np.mean(auc_list),np.std(auc_list)
    prc_mean, prc_std = np.mean(prc_list),np.std(prc_list)
    log(f"Final_evaluation ||| accuracy_mean: {round(float(accuracy_mean),4)} ||| accuracy_std: {round(float(accuracy_std),4)}|||  precision_mean: {round(float(precision_mean),4)}||| precision_std: {round(float(precision_std),4)} |||recall_mean: {round(float(recall_mean),4)} ||recall_std: {round(float(recall_std),4)}|||auc_mean: {round(float(auc_mean),4)} ||| auc_std: {round(float(auc_std),4)} ||| prc_mean: {round(float(prc_mean),4)}||| prc_std: {round(float(prc_std),4)}")
        
                    

    

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--device_num', type=str, default='0', help='')
    p.add_argument('--batch_size', type=int, default=64, help='')
    #p.add_argument('--model', type=str, default='trained_models/model_mf_alpha_edge_esm1b_512_8.pt', help='') #default 
    p.add_argument('--model', type=str, default=None, help='')
    p.add_argument('--dataset',type=str,default="DrugBank")
    p.add_argument('--learning_rate',type=float,default=1e-4)
    
    args = p.parse_args()
    print(args)
    if args.device_num != '':
        args.device = "cuda:" + args.device_num
    #rgs.model_save_path = "dti_model/model_"
    #rgs.model
    test(args)
