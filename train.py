from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network import CL_protNET, MLP, EdgeGenerator, JointGenerator
from nt_xent import NT_Xent
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from itertools import chain
from sklearn import metrics
from utils import log
import argparse
from config import get_config
import numpy as np
import random
import time
import warnings
warnings.filterwarnings("ignore")
import pickle as pkl


def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, CL_protNET):
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, CL_protNET):
            module.__explain__ = False
            module.__edge_mask__ = None

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def train_joint(model, generators, optimizer_model, optimizer_generator, train_loader, alpha, beta ,gamma, eta, device, child_mx, ith_epoch):
    model.train()
    bce_loss = torch.nn.BCELoss(reduce=False)
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
        y_pred_g1,g_feat_g1,_= model(x,y_true,node_mask,child_mx)
        clear_masks(model)
        #div_logits_g1 = torch.sum(torch.log(1-y_pred_g1),dim=1).view(-1)

        # The second generator
        kld_loss_g2, node_mask, edge_mask = generator_b(x)
        set_masks(edge_mask, model)
        y_pred_g2,g_feat_g2,_= model(x,y_true,node_mask,child_mx)
        clear_masks(model)
        #div_logits_g2 = torch.sum(torch.log(1-y_pred_g2),dim=1).view(-1)
        
        # Full graph
        y_pred, g_feat, g_feat_label = model(x,y_true,None,child_mx)
        #print(y_pred.dtype)
        sup_loss = bce_loss(y_pred,y_true).mean()
        criterion = NT_Xent(g_feat.shape[0], 0.1, 1)
        cl_loss = criterion(g_feat,g_feat_label)
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
        loss = sup_loss + alpha*cl_loss+beta*aug_loss+gamma*kld_loss+ eta*ccl_loss
        log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)} ||| sup_loss: {round(float(sup_loss),3)}||| cl_loss: {round(float(alpha*cl_loss),3)} ||| aug_loss: {round(float(beta*aug_loss),3)} ||| kld_loss: {round(float(gamma*kld_loss),3)}||| ccl_loss: {round(float(eta*ccl_loss),3)}")
        loss.backward()
        optimizer_generator.step()
        optimizer_model.step()

    return loss

def train_edge(model, generators, optimizer_model, optimizer_generator, train_loader, alpha, beta, gamma, eta, device, child_mx, ith_epoch):
    model.train()
    bce_loss = torch.nn.BCELoss(reduce=False)
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
        y_pred_g1,g_feat_g1,_= model(x,y_true,None,child_mx)
        clear_masks(model)
        #div_logits_g1 = torch.sum(torch.log(1-y_pred_g1),dim=1).view(-1)

        # The second generator
        kld_loss_g2, edge_mask = generator_b(x)
        set_masks(edge_mask, model)
        y_pred_g2,g_feat_g2,_= model(x,y_true,None,child_mx)
        clear_masks(model)
        #div_logits_g2 = torch.sum(torch.log(1-y_pred_g2),dim=1).view(-1)
        # Full graph
        y_pred, g_feat, g_feat_label = model(x,y_true,None,child_mx)
        sup_loss = bce_loss(y_pred,y_true).mean()
        criterion = NT_Xent(g_feat.shape[0], 0.1, 1)
        cl_loss = criterion(g_feat,g_feat_label)
        aug_loss = (criterion(g_feat,g_feat_g1)+criterion(g_feat,g_feat_g2))/2
        #energy_loss = (div_logits_g1-div_logits_g2).pow(2).mean()
        kld_loss = (kld_loss_g1+kld_loss_g2)/2
        
        norm_g1 = torch.norm(g_feat_g1, dim=1, keepdim=True)
        norm_g2 = torch.norm(g_feat_g2, dim=1, keepdim=True)
        cosine_mx = torch.mm(g_feat_g1, g_feat_g2.t())/(norm_g1 * norm_g2)
        on_diag = torch.diagonal(cosine_mx).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(cosine_mx).pow_(2).sum()/2
        ccl_loss = on_diag + off_diag

        loss = sup_loss + alpha*cl_loss+beta*aug_loss+gamma*kld_loss+ eta*ccl_loss
        log(f"{idx_batch}/{ith_epoch} train_epoch ||| Loss: {round(float(loss),3)} ||| sup_loss: {round(float(sup_loss),3)}||| cl_loss: {round(float(alpha*cl_loss),3)} ||| aug_loss: {round(float(beta*aug_loss),3)} ||| kld_loss: {round(float(gamma*kld_loss),3)}||| ccl_loss: {round(float(eta*ccl_loss),3)}")
        loss.backward()
        optimizer_generator.step()
        optimizer_model.step()

    return loss



def train(config, task, suffix):
    t1 = time.time()
    train_set = GoTermDataset("train", task, config.AF2model,config.esm1b)
    valid_set = GoTermDataset("val", task, config.AF2model,config.esm1b)
    t2 = time.time()
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn,drop_last=True)
    val_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    t3 = time.time()
    print("set",t2-t1)
    print("loader",t3-t2)
    output_dim = valid_set.y_true.shape[-1] #number of class
    with open("./data/child_mx.pkl", "rb") as f:
        mx = pkl.load(f)
    child_mx = torch.tensor(mx[task],dtype=torch.float32).to(config.device)
    model = CL_protNET(output_dim, config.esmembed, config.pooling, config.hierarchical).to(config.device)

    optimizer_model = torch.optim.Adam(
        params = model.parameters(), 
        **config.optimizer,
        )
    bce_loss = torch.nn.BCELoss(reduce=False)
    
    train_loss = []
    val_loss = []
    val_aupr = []
    val_Fmax = []
    es = 0
    y_true_all = valid_set.y_true.float().reshape(-1)
    generators = []
    for i in range(2):
        if config.joint:
            generators.append(JointGenerator(device=config.device).to(config.device))
        else:
            generators.append(EdgeGenerator(device=config.device).to(config.device))
    generators_params = []
    for generator in generators:
        generator.reset_parameters()
        generators_params.append(generator.parameters())
    optimizer_generator = torch.optim.AdamW(chain.from_iterable(generators_params), lr=1e-3)
    
    for ith_epoch in range(config.max_epochs):
        if config.joint:
            loss = train_joint(model, generators, optimizer_model, optimizer_generator, train_loader, config.alpha, config.beta, config.gamma, config.eta, config.device, child_mx, ith_epoch)
        else:
            loss = train_edge(model, generators, optimizer_model, optimizer_generator, train_loader, config.alpha, config.beta, config.gamma, config.eta, config.device, child_mx, ith_epoch)
            
        train_loss.append(loss.clone().detach().cpu().numpy())
        eval_loss = 0
        model.eval()
        y_pred_all = []
        
        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                y_pred,_= model(data=batch[0].to(config.device),child_mx=child_mx)
                y_pred_all.append(y_pred)
            y_pred_all = torch.cat(y_pred_all, dim=0).cpu().reshape(-1)
            eval_loss = bce_loss(y_pred_all, y_true_all).mean()
                
            aupr = metrics.average_precision_score(y_true_all.numpy(), y_pred_all.numpy(), average="samples")
            val_aupr.append(aupr)
            log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),3)} ||| aupr: {round(float(aupr),3)}")
            val_loss.append(eval_loss.numpy())
            if ith_epoch == 0:
                best_eval_loss = eval_loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                es = 0
            else:
                es += 1
                print("Counter {} of 5".format(es))

            if es > 4 :
                torch.save(model.state_dict(), config.model_save_path + task + f"{suffix}.pt")
                torch.save(
                    {
                        "train_bce": train_loss,
                        "val_bce": val_loss,
                        "val_aupr": val_aupr,
                    }, config.loss_save_path + task + f"{suffix}.pt"
                )

                break
                

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=lambda s: [item for item in s.split(",")], default=['mf','bp','cc'], help="list of func to predict.")
    p.add_argument('--suffix', type=str, default='', help='')
    p.add_argument('--device', type=str, default='', help='')
    p.add_argument('--esmembed', default=True, type=str2bool, help='')
    p.add_argument('--pooling', default='MTP', type=str, choices=['MTP','GMP'], help='Multi-set transformer pooling or Global max pooling')
    p.add_argument('--AF2model', default=True, type=str2bool, help='whether to use AF2model for training')
    p.add_argument('--batch_size', type=int, default=64, help='')
    p.add_argument('--joint', type=str2bool, default=False, help='')
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--beta', type=float, default=0.05)
    p.add_argument('--gamma', type=float, default=0.05)
    p.add_argument('--eta', type=float, default=0.001)
    p.add_argument('--hierarchical', type=str2bool,default=True)
    p.add_argument('--esm1b',type=str2bool,default=True)
    args = p.parse_args()
    print(args)
    config = get_config()
    config.optimizer['lr'] = 1e-4
    config.batch_size = args.batch_size
    config.joint = args.joint
    config.max_epochs = 100
    if args.device != '':
        config.device = "cuda:" + args.device
    config.esmembed = args.esmembed
    config.pooling = args.pooling
    config.AF2model = args.AF2model
    config.alpha = args.alpha
    config.beta = args.beta
    config.gamma = args.gamma
    config.eta = args.eta
    config.hierarchical = args.hierarchical
    config.esm1b = args.esm1b
    for task in args.task:
        print("##############################################################")
        print("training for "+str(task)+" task")
        train(config, task, args.suffix)

