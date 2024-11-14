import random
import argparse
import os
from utils import log
import pickle as pkl
from network import AttentionDTI,EdgeGenerator, JointGenerator
from DTI_data import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from nt_xent import NT_Xent

def set_masks(mask: torch.Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, AttentionDTI):
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, AttentionDTI):
            module.__explain__ = False
            module.__edge_mask__ = None

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def train_joint_DTI(model, generators, optimizer_model, optimizer_generator, train_loader, beta, gamma, eta, device, ith_epoch,weight_CE):
    model.train()
    #regular = nn.Sigmoid()
    CE_loss = torch.nn.CrossEntropyLoss(weight_CE)
    
    for idx_batch, batch in enumerate(train_loader):

        x_smiles = batch[0].to(device)
        x_proteins = batch[1].to(device)
        y_true = batch[2].to(device)
        optimizer_generator.zero_grad()
        optimizer_model.zero_grad()
        # two generators
        generator_a,generator_b = generators[0],generators[1]
        # The first generator
        kld_loss_g1, node_mask, edge_mask = generator_a(x_proteins)
        set_masks(edge_mask, model)
        y_pred_g1,g_feat_g1 = model(x_smiles,x_proteins,node_mask)
        clear_masks(model)
        #div_logits_g1 = torch.sum(torch.log(1-y_pred_g1),dim=1).view(-1)

        # The second generator
        kld_loss_g2, node_mask, edge_mask = generator_b(x_proteins)
        set_masks(edge_mask, model)
        y_pred_g2,g_feat_g2 = model(x_smiles,x_proteins,node_mask)
        clear_masks(model)
        #div_logits_g2 = torch.sum(torch.log(1-y_pred_g2),dim=1).view(-1)
        
        # Full graph
        y_pred, g_feat = model(x_smiles,x_proteins)
        #print(y_pred.dtype)
        sup_loss = CE_loss(y_pred,y_true)
        criterion = NT_Xent(g_feat.shape[0], 0.1, 1)
        aug_loss = (criterion(g_feat,g_feat_g1)+criterion(g_feat,g_feat_g2))/2
        #energy_loss = (div_logits_g1-div_logits_g2).pow(2).mean()
        kld_loss = (kld_loss_g1+kld_loss_g2)/2
        
        norm_g1 = torch.norm(g_feat_g1, dim=1, keepdim=True)
        norm_g2 = torch.norm(g_feat_g2, dim=1, keepdim=True)
        cosine_mx = torch.mm(g_feat_g1, g_feat_g2.t())/(norm_g1 * norm_g2)
        on_diag = torch.diagonal(cosine_mx).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cosine_mx).pow_(2).sum()/2
        ccl_loss = on_diag + off_diag

        #loss = sup_loss + cl_weight*(cl_loss+aug_loss+kld_loss+ccl_loss) - dist_weight*energy_loss 
        loss = sup_loss + beta*aug_loss+gamma*kld_loss+ eta*ccl_loss
        log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)} ||| sup_loss: {round(float(sup_loss),3)} ||| aug_loss: {round(float(beta*aug_loss),3)} ||| kld_loss: {round(float(gamma*kld_loss),3)}||| ccl_loss: {round(float(eta*ccl_loss),3)}")
        loss.backward()
        optimizer_generator.step()
        optimizer_model.step()

    return loss

def train_edge_DTI(model, generators, optimizer_model, optimizer_generator, train_loader,beta, gamma, eta, device, ith_epoch,weight_CE):
    model.train()
    CE_loss = torch.nn.CrossEntropyLoss(weight_CE)
    for idx_batch, batch in enumerate(train_loader):

        x_smiles = batch[0].to(device)
        x_proteins = batch[1].to(device)
        y_true = batch[2].to(device)
        optimizer_generator.zero_grad()
        optimizer_model.zero_grad()
        # two generators
        generator_a,generator_b = generators[0],generators[1]
        # The first generator
        kld_loss_g1,edge_mask = generator_a(x_proteins)
        set_masks(edge_mask, model)
        y_pred_g1,g_feat_g1 = model(x_smiles,x_proteins)
        clear_masks(model)

        # The second generator
        kld_loss_g2, edge_mask = generator_b(x_proteins)
        set_masks(edge_mask, model)
        y_pred_g2,g_feat_g2 = model(x_smiles,x_proteins)
        clear_masks(model)
        # Full graph
        y_pred, g_feat = model(x_smiles,x_proteins)
        sup_loss = CE_loss(y_pred,y_true)
        criterion = NT_Xent(g_feat.shape[0], 0.1, 1)
        aug_loss = (criterion(g_feat,g_feat_g1)+criterion(g_feat,g_feat_g2))/2
        #energy_loss = (div_logits_g1-div_logits_g2).pow(2).mean()
        kld_loss = (kld_loss_g1+kld_loss_g2)/2
        
        norm_g1 = torch.norm(g_feat_g1, dim=1, keepdim=True)
        norm_g2 = torch.norm(g_feat_g2, dim=1, keepdim=True)
        cosine_mx = torch.mm(g_feat_g1, g_feat_g2.t())/(norm_g1 * norm_g2)
        on_diag = torch.diagonal(cosine_mx).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cosine_mx).pow_(2).sum()/2
        ccl_loss = on_diag + off_diag

        loss = sup_loss + beta*aug_loss+gamma*kld_loss+ eta*ccl_loss
        log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)} ||| sup_loss: {round(float(sup_loss),3)}||| aug_loss: {round(float(beta*aug_loss),3)} ||| kld_loss: {round(float(gamma*kld_loss),3)}||| ccl_loss: {round(float(eta*ccl_loss),3)}")
        loss.backward()
        optimizer_generator.step()
        optimizer_model.step()

    return loss

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def finetune_DTI(args,seed):
    #with open("data/DTI/DrugBank.pkl", "rb") as f:
    #    drug_ids,protein_ids,smiles,proteins,protein_graphs,labels = pkl.load(f)
    #if args.dataset == "DrugBank":
    #    dir_input = ('./data/DTI/{}.txt'.format(args.dataset))
    #elif args.dataset == "KIBA":
    #    dir_input = ('./data/DTI/{}.txt'.format(args.dataset))
    if args.dataset == "DrugBank":
        weight_CE = None
    elif args.dataset == "BindingDB":
        weight_CE = torch.FloatTensor([0.3,0.7]).to(args.device)
    elif args.dataset == "KIBA":
        weight_CE = torch.FloatTensor([0.2,0.8]).to(args.device)
    dir_input = './data/DTI/{}.txt'.format(args.dataset)
    with open(dir_input, "r") as f:
        data_list = f.read().strip().split('\n')
    protein_ids,smiles,proteins,labels,fold_num = [],[],[],[],[]
    for i,pair in enumerate(data_list):
        pair = pair.strip().split()
        protein_id,compoundstr,proteinstr,label,fold = pair[-5], pair[-4],pair[-3], pair[-2], pair[-1]
        protein_ids.append(protein_id)  
        smiles.append(compoundstr)
        proteins.append(proteinstr)
        labels.append(float(label)) # str into float
        fold_num.append(float(fold))# str into float
    
    data_list = list(zip(protein_ids,smiles,proteins,labels,fold_num))
    
    #random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    
    #data_set = shuffle_dataset(data_list, seed)
    data_set = shuffle_dataset(data_list,seed)
    k_fold = 5
    
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.CrossEntropyLoss(weight_CE)
    for i in range(k_fold):
        train_set,valid_set = [(item[0],item[1],item[2],item[3]) for item in data_list if float(item[-1])!=i and float(item[-1])!=5],[(item[0],item[1],item[2],item[3]) for item in data_list if float(item[-1])==i]
        train_set = CustomDataSet(train_set)
        valid_set = CustomDataSet(valid_set)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)
        model = AttentionDTI(pretrain_model=args.model,device=args.device).to(args.device) #initialize by the pretained model
        optimizer_model = optim.AdamW(params = model.parameters(), lr = args.learning_rate)
        generators = []
        for j in range(2):
            if args.joint:
                generators.append(JointGenerator(device=args.device).to(args.device))
            else:
                generators.append(EdgeGenerator(device=args.device).to(args.device))
        generators_params = []
        for generator in generators:
            generator.reset_parameters()
            generators_params.append(generator.parameters())
        optimizer_generator = torch.optim.AdamW(chain.from_iterable(generators_params), lr=1e-3)

        train_loss = []
        val_loss = []
        es = 0
        for ith_epoch in range(args.max_epochs):
            if args.joint:
                loss = train_joint_DTI(model, generators,optimizer_model, optimizer_generator, train_loader,args.beta, args.gamma, args.eta, args.device,ith_epoch,weight_CE)
            else:
                loss = train_edge_DTI(model, generators, optimizer_model, optimizer_generator, train_loader,args.beta, args.gamma, args.eta, args.device,ith_epoch,weight_CE)
            
            train_loss.append(loss.clone().detach().cpu().numpy())
            eval_loss = 0
            model.eval()
            y_pred_scores = []
            y_pred_labels = []
            y_true_all = []
            valid_loss_epoch = []
            with torch.no_grad():
                for idx_batch, batch in enumerate(valid_loader):
                    x_smiles,x_proteins,y_true = batch[0].to(args.device), batch[1].to(args.device),batch[2].to(args.device)
                    y_pred,_ = model(x_smiles, x_proteins)
                    loss = criterion(y_pred, y_true)
                    valid_loss_epoch.append(loss.item())
                    y_true = y_true.to('cpu').data.numpy()
                    y_score = F.softmax(y_pred, 1).to('cpu').data.numpy()
                    y_label = np.argmax(y_score, axis=1)
                    y_score = y_score[:,1]
                    
                    
                    
                    y_pred_scores.extend(y_score)
                    y_pred_labels.extend(y_label)
                    y_true_all.extend(y_true)
                    
                    
                
                precision = precision_score(y_true_all,y_pred_labels)
                recall = recall_score(y_true_all,y_pred_labels)
                accuracy = accuracy_score(y_true_all,y_pred_labels)
                roc_auc = roc_auc_score(y_true_all,y_pred_scores)
                tpr, fpr, _ = precision_recall_curve(y_true_all,y_pred_scores)
                prc = auc(fpr,tpr)
                eval_loss = np.mean(valid_loss_epoch)
                val_loss.append(eval_loss)
                log(f"{i}/{ith_epoch} val_epoch ||| Loss: {round(float(eval_loss),4)}||| accuracy: {round(float(accuracy),4)} ||| precision: {round(float(precision),4)} |||recall: {round(float(recall),4)}  ||| roc_auc: {round(float(roc_auc),4)} ||| prc: {round(float(prc),4)}")
                # early stopping
                if ith_epoch == 0:
                    best_eval_loss = eval_loss
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    es = 0
                else:
                    es += 1
                    print("Counter {} of 5".format(es))

                if es > 4 or ith_epoch+1==20:
                    torch.save(model.state_dict(), args.model_save_path +str(args.dataset)+'_'+str(args.batch_size)+'_'+str(args.learning_rate)+'_'+str(i)+"_fold.pt")
                    break
                    #if args.model=="trained_models/model_mf_alpha_edge_esm1b_512.pt":
                    #    torch.save(model.state_dict(), args.model_save_path + "mf_"+str(args.dataset)+'_'+str(args.batch_size)+'_'+str(args.learning_rate)+'_'+str(ith_epoch+1)+'_'+str(i)+"_fold.pt")
                    #    break
                    #if args.model=="trained_models/model_bp_alpha_edge_esm1b_512.pt":
                    #    torch.save(model.state_dict(), args.model_save_path + "bp_"+str(args.dataset)+'_'+str(args.batch_size)+'_'+str(args.learning_rate)+'_'+str(ith_epoch+1)+'_'+str(i)+"_fold.pt")
                    #    break
                    #if args.model=="trained_models/model_cc_alpha_edge_esm1b_512.pt":
                    #    torch.save(model.state_dict(), args.model_save_path + "cc_"+str(args.dataset)+'_'+str(args.batch_size)+'_'+str(args.learning_rate)+'_'+str(ith_epoch+1)+'_'+str(i)+"_fold.pt")
                    #    break
    
def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False
    
if __name__ == "__main__":
    seed = 1234
    p = argparse.ArgumentParser()
    p.add_argument('--device_num', type=str, default='0', help='')
    p.add_argument('--batch_size', type=int, default=64, help='')
    p.add_argument('--model', type=str, default='sortedmodel/model_mf_alpha_edge_esm1b.pt', help='') #default 
    p.add_argument('--max_epochs', type=int, default=20)
    p.add_argument('--dataset',type=str,default="DrugBank")
    p.add_argument('--learning_rate',type=float,default=1e-4)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=0.05)
    p.add_argument('--eta', type=float, default=0.001)
    p.add_argument('--joint', type=str2bool, default=False)
    
    args = p.parse_args()
    print(args)
    if args.device_num != '':
        args.device = "cuda:" + args.device_num
    args.model_save_path = "dti_model/model_"
    finetune_DTI(args,seed)