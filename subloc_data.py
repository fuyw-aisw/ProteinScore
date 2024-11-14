import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from torch_geometric.data import Data, Batch
import pickle as pkl
from utils import protein_graph

with open("data/prot_loc/protein_esm2.pkl", "rb") as f:
    embed = pkl.load(f)
with open("data/prot_loc/protein_edge.pkl", "rb") as f:
    edge = pkl.load(f)
    
def encode_proteins(protein_id,protein_sequence,protein_embed=embed,protein_edge=edge):
    sequence = protein_sequence
    edge = np.array(protein_edge[protein_id])
    esm_embed = np.array(protein_embed[protein_id])
    data = protein_graph(sequence, edge, esm_embed)
    return data

class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, idx):
        return self.pairs[idx]

    def __len__(self):
        return len(self.pairs)

def collate_fn(batch):
    protein_ids,proteins,labels = map(list,zip(*batch))
    protein_ids_new,protein_graphs,labels_new = [],[],[]
    for idx in range(len(batch)):
        protein_ids_new.append(protein_ids[idx])
        protein_graphs.append(encode_proteins(protein_ids[idx],proteins[idx]))
        labels_new.append(labels[idx])
    labels_new = torch.tensor(labels_new)
    #print(protein_graphs)
    return Batch.from_data_list(protein_graphs), labels_new,protein_ids_new
                  