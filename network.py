import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from pool import GraphMultisetTransformer
from torch_geometric.nn import global_max_pool as gmp
import numpy as np
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119 # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 5 # including aromatic and self-loop edge
num_bond_direction = 3

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=True), nn.ReLU(inplace=True),
                                 nn.Linear(hidden_dim, output_dim, bias=True), nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.mlp(x)
    
class GraphCNN(nn.Module):
    def __init__(self, channel_dims=[512,512,512], fc_dim=512, num_classes=256, pooling='MTP'):
        super(GraphCNN, self).__init__() #channel_dims=[256,256,256], fc_dim=256, num_classes=128

        # Define graph convolutional layers
        gcn_dims = [512] + channel_dims #[512]

        gcn_layers = [GCNConv(gcn_dims[i-1], gcn_dims[i], bias=True) for i in range(1, len(gcn_dims))]

        self.gcn = nn.ModuleList(gcn_layers)
        self.pooling = pooling
        if self.pooling == "MTP":
            self.pool = GraphMultisetTransformer(512,256,512,None,10000, 0.25, ['GMPool_G','GMPool_G'], num_heads=8, layer_norm=True) 
            #self.pool = GraphMultisetTransformer(512,256,512, GCNConv, 500, 0.25, ['GMPool_G','SelfAtt','GMPool_I'], num_heads=8, layer_norm=True) 
        else:
            self.pool = gmp
        # Define dropout
        self.drop1 = nn.Dropout(p=0.2) #0.2
    #grad_cam
    ###
    #def activations_hook(self,grad): 
    #    self.final_conv_grads = grad 
    ###
    

    def forward(self, x, data):
        #x = data.x
        # Compute graph convolutional part
        x = self.drop1(x)
        for idx, gcn_layer in enumerate(self.gcn):
            if idx == 0:
                x = F.relu(gcn_layer(x, data.edge_index.long()))
            ####
            #elif idx == 2:
            #    with torch.enable_grad():
            #        self.final_conv_acts = gcn_layer(x, data.edge_index.long())
            #    self.final_conv_acts.register_hook(self.activations_hook)
            #    x = x + F.relu(self.final_conv_acts)
            ###
            else:
                x = x + F.relu(gcn_layer(x, data.edge_index.long()))
        if self.pooling == 'MTP':
            # Apply GraphMultisetTransformer Pooling
            g_level_feat = self.pool(x, data.batch, data.edge_index.long())
        else:
            g_level_feat = self.pool(x, data.batch)

        n_level_feat = x


        return n_level_feat, g_level_feat


class CL_protNET(torch.nn.Module):
    def __init__(self, out_dim, esm_embed=False, pooling='MTP',hierarchical=True):
        super(CL_protNET,self).__init__()
        self.esm_embed = esm_embed
        #self.pertub = pertub
        self.out_dim = out_dim
        self.one_hot_embed = nn.Embedding(21, 96)
        self.proj_aa = nn.Linear(96,512)#256)
        self.pooling = pooling
        if esm_embed:
            self.proj_esm = nn.Linear(1280,512)#256)
            self.gcn = GraphCNN(pooling=pooling)
        else:
            self.gcn = GraphCNN(pooling=pooling)
        self.label_em = MLP(self.out_dim,1024,512)#512,256)
        #self.esm_g_proj = nn.Linear(1280, 512)
        self.readout = nn.Sequential(
                        nn.Linear(512,1024),#(256,512),
                        nn.ReLU(),
                        nn.Dropout(0.2), #0.2
                        nn.Linear(1024,out_dim),#512, out_dim),
                        nn.Sigmoid()
        )
        self.hierarchical = hierarchical

     
    def forward(self, data, y=None, node_mask=None, child_mx=None):

        x_aa = self.one_hot_embed(data.native_x.long())
        x_aa = self.proj_aa(x_aa)
        if self.esm_embed:
            x = data.x.float()
            if node_mask is not None:
                x *= node_mask
                #print(node_mask.shape,x.shape)
                x_esm = self.proj_esm(x)
                x = F.relu(x_aa + x_esm)
            else:
                x_esm = self.proj_esm(x)
                x = F.relu(x_aa + x_esm)
            
        else:
            x = F.relu(x_aa)
        gcn_n_feat, gcn_g_feat = self.gcn(x, data)
        y_pred = self.readout(gcn_g_feat)
        if child_mx != None:
            if y is not None and self.hierarchical:
                g_feat_label = self.label_em(y)
                y_pred_a = (1 - y) * torch.max(y_pred.unsqueeze(1) * child_mx.unsqueeze(0), dim = -1)[0]
                y_pred_b = y * torch.max(y_pred.unsqueeze(1) * (child_mx.unsqueeze(0) * y.unsqueeze(1)), dim = -1)[0]
                y_pred = y_pred_a + y_pred_b
                return y_pred, gcn_g_feat, g_feat_label
            elif y is not None and not self.hierarchical:
                g_feat_label = self.label_em(y)
                y_pred = torch.max(y_pred.unsqueeze(1) * child_mx.unsqueeze(0), dim = -1)[0]
                return y_pred, gcn_g_feat, g_feat_label
            else:
                y_pred = torch.max(y_pred.unsqueeze(1) * child_mx.unsqueeze(0), dim = -1)[0] 
                return y_pred, gcn_g_feat

        else:
            if y is not None:
                g_feat_label = self.label_em(y)
                return y_pred, gcn_g_feat, g_feat_label
            else:    
                return y_pred, gcn_g_feat
        
# Subgraph generator, including edge mask and node mask
# Edge and node
class JointGenerator(torch.nn.Module):
    def __init__(self, device):
        super(JointGenerator, self).__init__()

        self.device = device
        self.gcn1 = GCNConv(1280,512)#256)
        self.gcn2 = GCNConv(512,256)#256,128)
        self.lin = torch.nn.Linear(256,1)#128,1)
        self.relu = torch.nn.ReLU()
        self.rate = 0.7
        self.epsilon = 0.00000001
    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.lin.reset_parameters()

    def _kld(self, mask):
        pos = mask
        neg = 1 - mask
        kld_loss = torch.mean(pos * torch.log(pos/0.5 + self.epsilon) + neg * torch.log(neg/0.5 + self.epsilon))

        return kld_loss

    def _sample_graph(self, sampling_weights, temperature=1.0):
        eps = torch.rand(sampling_weights.size())
        gate_inputs = torch.log(eps) - torch.log(1-eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs += sampling_weights / temperature
        graph = torch.sigmoid(gate_inputs)
        return graph

    def edge_sample(self, node_mask, edge_idx):
        src_val = node_mask[edge_idx[0]]
        dst_val = node_mask[edge_idx[1]]
        edge_val = 0.5 * (src_val + dst_val)

        return edge_val


    def forward(self, data):
        edges = data.edge_index.long()
        #print(edges.shape)
        x_embeds = data.x.float()
        pre = self.relu(self.gcn1(x_embeds,edges))
        pre = self.relu(self.gcn2(pre,edges))
        pre = self.lin(pre)
        pre = torch.clamp(pre, min = -10, max = 10)
        node_mask = self._sample_graph(pre)
        kld_loss = self._kld(node_mask)
        edge_mask = self.edge_sample(node_mask, edges)
 
        return kld_loss, node_mask, edge_mask

# Edge
class EdgeGenerator(torch.nn.Module):
    def __init__(self, device):
        super(EdgeGenerator, self).__init__()

        self.input_size = 2 * 1280
        self.device = device
        self.hidden_size = 512#256
        self.gcn1 = GCNConv(self.input_size,self.hidden_size)
        self.gcn2 = GCNConv(512,256)#256,128)
        self.lin = torch.nn.Linear(256,1)#128, 1)  #weights for sampling
        self.relu = torch.nn.ReLU()

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.lin.reset_parameters()


    def _sample_graph(self, sampling_weights, temperature=1.0):
        eps = torch.rand(sampling_weights.size())
        gate_inputs = torch.log(eps) - torch.log(1-eps)
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs += sampling_weights / temperature
        graph = torch.sigmoid(gate_inputs)
        return graph

    def _kld(self, mask):
        pos = mask
        neg = 1 - mask
        kld_loss = torch.mean(pos * torch.log(pos/0.5 + 0.00000001) + neg * torch.log(neg/0.5 + 0.000000001))

        return kld_loss

    def _create_explainer_input(self, pair, embeds):
        rows = pair[0]
        cols = pair[1]
        row_embeds = embeds[rows]
        col_embeds = embeds[cols]
        input_expl = torch.cat([row_embeds, col_embeds], 1)
        return input_expl

    def forward(self, data):
        edges = data.edge_index.long()
        x_embeds = data.x.float()
        input_embs = self._create_explainer_input(edges, x_embeds)
        pre = self.relu(self.gcn1(input_embs,edges))
        pre = self.relu(self.gcn2(pre,edges))
        pre = self.lin(pre)
        mask = self._sample_graph(pre)
        kld_loss = self._kld(mask)

        return kld_loss, mask

class GINEConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GINEConv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim), 
            nn.ReLU(), 
            nn.Linear(2*emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )
    ####
    #def activation_hook(self,grad):
    #    self.final_conv_grads = grad
    ####
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            ####
            #if layer == self.num_layer - 1:
            #    with torch.enable_grad():
            #        self.final_conv_acts = h
            #        self.final_conv_acts.register_hook(self.activation_hook)
            ####
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:  
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        
        return h, out
# define DTI network
class AttentionDTI(nn.Module):
    def __init__(self,device,pretrain_model=None):
        super(AttentionDTI, self).__init__()
        if pretrain_model == "sortedmodel/model_mf_alpha_edge_esm1b.pt":
            self.protein_embed = CL_protNET(out_dim=489,esm_embed=True)
        elif pretrain_model == "sortedmodel/model_bp_alpha_edge_esm1b.pt":
            self.protein_embed = CL_protNET(out_dim=1943,esm_embed=True)
        elif pretrain_model == "sortedmodel/model_cc_alpha_edge_esm1b.pt":
            self.protein_embed = CL_protNET(out_dim=320,esm_embed=True)
        else:
            self.protein_embed = CL_protNET(out_dim=489,esm_embed=True)
        if pretrain_model is not None:
            self.protein_embed.load_state_dict(torch.load(pretrain_model,map_location=device))
        #self.protein_embed.load_state_dict(torch.load(pretrain_model,map_location=device)) #finetune based on model trained on mf task
        self.drug_embed = GINet()
        self.readout = nn.Sequential(
            nn.Dropout(0.1), 
            nn.Linear(768,512),#256+512
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,2)
        )
        #self.readout = nn.Sequential(
        #    nn.Linear(768,256),
        #    nn.Linear(256,256),
        #    nn.Linear(256,2))
            
    def forward(self,drug,protein,mask=None):
        drug_feat,_ = self.drug_embed(drug)
        _,protein_feat = self.protein_embed(protein,node_mask=mask)
        pair = torch.cat([protein_feat, drug_feat], dim=1) #256+512=768
        y_pred = self.readout(pair)
        return y_pred,protein_feat

#define subcellular localization network
class Attention_subloc(nn.Module):
    def __init__(self,device,task,pretrain_model=None):
        super(Attention_subloc, self).__init__()
        if pretrain_model == "sortedmodel/model_mf_alpha_edge_esm1b.pt":
            self.protein_embed = CL_protNET(out_dim=489,esm_embed=True)
        elif pretrain_model == "sortedmodel/model_mf_alpha_edge_esm1b.pt":
            self.protein_embed = CL_protNET(out_dim=1943,esm_embed=True)
        elif pretrain_model == "sortedmodel/model_mf_alpha_edge_esm1b.pt":
            self.protein_embed = CL_protNET(out_dim=320,esm_embed=True)
        else:
            self.protein_embed = CL_protNET(out_dim=489,esm_embed=True)
        if pretrain_model is not None:
            self.protein_embed.load_state_dict(torch.load(pretrain_model,map_location=device))
           
        if task == "sl":
            self.out_dim = 11
        else:
            self.out_dim = 4
        self.readout = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,self.out_dim))
        #self.readout = nn.Sequential(
        #    nn.Linear(512,256),
        #    nn.Linear(256,self.out_dim))
    def forward(self,protein,mask=None):
        _,g_feat = self.protein_embed(protein,node_mask=mask)
        y_pred = self.readout(g_feat)
        return y_pred,g_feat
        
'''       
# define Phase separation network
class AttentionPS(nn.Module):
    def __init__(self,device,pretrain_model='sortedmodel/model_mf_alpha_edge_esm1b.pt'):
        super(AttentionPS, self).__init__()
        if pretrain_model == "sortedmodel/model_mf_alpha_edge_esm1b.pt":
            self.protein_embed = CL_protNET(out_dim=489,esm_embed=True)
        elif pretrain_model == "sortedmodel/model_bp_alpha_edge_esm1b.pt":
            self.protein_embed = CL_protNET(out_dim=1943,esm_embed=True)
        elif pretrain_model == "sortedmodel/model_cc_alpha_edge_esm1b.pt":
            self.protein_embed = CL_protNET(out_dim=320,esm_embed=True)
        self.protein_embed.load_state_dict(torch.load(pretrain_model,map_location=device)) #finetune based on model trained on mf task
        self.readout = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256,2) 
        )
    def forward(self,protein):
        _,protein_feat = self.protein_embed(protein)
        y_pred = self.readout(protein_feat)
        return y_pred
'''    
    
