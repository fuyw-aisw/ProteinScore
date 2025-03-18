import random
import argparse
import os
from utils import log
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain
from network import Attention_subloc, EdgeGenerator, JointGenerator
from subloc_data import CustomDataSet, collate_fn
from torch.utils.data import DataLoader
from nt_xent import NT_Xent

pos_weights_subloc = torch.tensor([1.1,1,1,2.9,2.35,3.7,8.9,4.45,6.46,7.6,31.2])
pos_weights_mem = torch.tensor([8,2.5,21.3,1])
LOCALIZATION_CATEGORIES = ["Membrane","Cytoplasm","Nucleus","Extracellular","Cell membrane","Mitochondrion","Plastid","Endoplasmic reticulum","Lysosome/Vacuole","Golgi apparatus","Peroxisome"]
MEMBRANE_CATEGORIES = ["Peripheral","Transmembrane","LipidAnchor","Soluble"]

def set_masks(mask: torch.Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, Attention_subloc):
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, Attention_subloc):
            module.__explain__ = False
            module.__edge_mask__ = None

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def focal_loss(y_pred, y, task,gamma=1):
    y = y.float()
    if task == "sl":
        pos_weights_bce = pos_weights_subloc 
    else:
        pos_weights_bce = pos_weights_mem  
    bceloss = F.binary_cross_entropy_with_logits(y_pred, y, pos_weight=pos_weights_bce.to(y_pred.device), reduction="none")
    logpt = -F.binary_cross_entropy_with_logits(y_pred, y, reduction="none")
    pt = torch.exp(logpt)
    # compute the loss
    focal_loss = ( (1-pt) ** gamma ) * bceloss
    return focal_loss.mean()

def train_joint_subloc(model, generators, optimizer_model, optimizer_generator, train_loader,task, beta,gamma,eta,device,ith_epoch):
    model.train()
    #regular = nn.Sigmoid()
    
    for idx_batch, batch in enumerate(train_loader):

        x = batch[0].to(device)
        y_true = batch[1].to(device)
        optimizer_generator.zero_grad()
        optimizer_model.zero_grad()
        # two generators
        generator_a,generator_b = generators[0],generators[1]
        # The first generator
        kld_loss_g1, node_mask, edge_mask = generator_a(x)
        set_masks(edge_mask, model)
        y_pred_g1,g_feat_g1 = model(x,node_mask)
        clear_masks(model)
        #div_logits_g1 = torch.sum(torch.log(1-y_pred_g1),dim=1).view(-1)

        # The second generator
        kld_loss_g2, node_mask, edge_mask = generator_b(x)
        set_masks(edge_mask, model)
        y_pred_g2,g_feat_g2 = model(x,node_mask)
        clear_masks(model)
        #div_logits_g2 = torch.sum(torch.log(1-y_pred_g2),dim=1).view(-1)
        
        # Full graph
        y_pred, g_feat = model(x)
        #print(y_pred.dtype)
        sup_loss = focal_loss(y_pred,y_true,task)
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

def train_edge_subloc(model, generators, optimizer_model, optimizer_generator, train_loader,task, beta, gamma, eta, device, ith_epoch):
    model.train()
    for idx_batch, batch in enumerate(train_loader):

        x = batch[0].to(device)
        y_true = batch[1].to(device)
        optimizer_generator.zero_grad()
        optimizer_model.zero_grad()
        # two generators
        generator_a,generator_b = generators[0],generators[1]
        # The first generator
        kld_loss_g1,edge_mask = generator_a(x)
        set_masks(edge_mask, model)
        y_pred_g1,g_feat_g1 = model(x)
        clear_masks(model)

        # The second generator
        kld_loss_g2, edge_mask = generator_b(x)
        set_masks(edge_mask, model)
        y_pred_g2,g_feat_g2 = model(x)
        clear_masks(model)
        # Full graph
        y_pred, g_feat = model(x)
        sup_loss = focal_loss(y_pred,y_true,task)
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



def finetune_subloc(args,seed):
    if args.task == "sl":
        dir_input = "data/prot_loc/swissprot_location.csv"
        CATEGORIES = LOCALIZATION_CATEGORIES
    else:
        dir_input = "data/prot_loc/swissprot_membrane.csv"
        CATEGORIES = MEMBRANE_CATEGORIES
    data_df = pd.read_csv(dir_input)
    #protein_ids,proteins,labels,fold_num = data_df["ACC"],data_df["Sequence"],data_df[LOCALIZATION_CATEGORIES].values.tolist(),data_df["Fold"]
    
    #random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    
    data_df = data_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    k_fold = 5
    for i in range(k_fold):
        train_df = data_df[(data_df["Fold"] != i) & (data_df["Fold"] != 5)]
        valid_df = data_df[data_df["Fold"] == i]
        train_set = list(zip(train_df["ACC"].tolist(),train_df["Sequence"].tolist(),train_df[CATEGORIES].values.tolist()))
        valid_set = list(zip(train_df["ACC"].tolist(),train_df["Sequence"].tolist(),train_df[CATEGORIES].values.tolist()))
        train_set = CustomDataSet(train_set)
        valid_set = CustomDataSet(valid_set)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,collate_fn=collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn)
        model = Attention_subloc(pretrain_model=args.model,task=args.task, device=args.device).to(args.device) #initialize by the pretained model
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
                loss = train_joint_subloc(model, generators, optimizer_model, optimizer_generator, train_loader,args.task, args.beta, args.gamma, args.eta, args.device,ith_epoch)
            else:
                loss = train_edge_subloc(model, generators, optimizer_model, optimizer_generator, train_loader,args.task, args.beta, args.gamma, args.eta, args.device,ith_epoch)
            
            train_loss.append(loss.clone().detach().cpu().numpy())
            eval_loss = 0
            model.eval()
            y_pred_all = []
            y_true_all = []
            with torch.no_grad():
                for idx_batch, batch in enumerate(valid_loader):
                    x_proteins,y_true = batch[0].to(args.device), batch[1].to(args.device)
                    y_pred,_= model(x_proteins)
                    y_true_all.append(y_true)
                    y_pred_all.append(y_pred)
                y_pred_all = torch.cat(y_pred_all, dim=0)
                y_true_all = torch.cat(y_true_all, dim=0)
             
                eval_loss = focal_loss(y_pred_all,y_true_all,args.task)
                

                log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),3)}")
                val_loss.append(eval_loss.cpu().numpy())
                if ith_epoch == 0:
                    best_eval_loss = eval_loss
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    es = 0
                    torch.save(model.state_dict(), args.model_save_path +str(args.task)+'_'+str(args.batch_size)+'_'+str(args.learning_rate)+'_'+str(i)+"_fold.pt")
                    
                else:
                    es += 1
                    print("Counter {} of 5".format(es))
                if es > 4:
                    break

                #if (ith_epoch+1) % 10 == 0:
                #    if args.model=="sortedmodel/model_mf_alpha_edge_esm1b.pt":
                #        torch.save(model.state_dict(), args.model_save_path + "mf_"+str(args.task)+'_'+str(args.batch_size)+'_'+str(args.learning_rate)+'_'+str(ith_epoch+1)+'_'+str(i)+"_fold.pt")
                #    if args.model=="sortedmodel/model_bp_alpha_edge_esm1b.pt":
                #        torch.save(model.state_dict(), args.model_save_path + "bp_"+str(args.task)+'_'+str(args.batch_size)+'_'+str(args.learning_rate)+'_'+str(ith_epoch+1)+'_'+str(i)+"_fold.pt")
                #    if args.model=="sortedmodel/model_cc_alpha_edge_esm1b.pt":
                #        torch.save(model.state_dict(), args.model_save_path + "cc_"+str(args.task)+'_'+str(args.batch_size)+'_'+str(args.learning_rate)+'_'+str(ith_epoch+1)+'_'+str(i)+"_fold.pt")
                        

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False

if __name__ == "__main__":
    seed = 2222
    p = argparse.ArgumentParser()
    p.add_argument('--device_num', type=str, default='0', help='')
    p.add_argument('--batch_size', type=int, default=64, help='')
    p.add_argument('--model', type=str, default=None, help='') #default 
    p.add_argument('--max_epochs', type=int, default=100)
    p.add_argument('--task',type=str,default="sl") #sl:subcellular localization;mp:membrane type prediction ['sl','mp']
    p.add_argument('--learning_rate',type=float,default=1e-4)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=0.05)
    p.add_argument('--eta', type=float, default=0.001)
    p.add_argument('--joint', type=str2bool, default=False)
    args = p.parse_args()
    print(args)
    if args.device_num != '':
        args.device = "cuda:" + args.device_num
    args.model_save_path = "subloc_model/model_"
    finetune_subloc(args,seed)
