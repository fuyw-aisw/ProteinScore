import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch
import pickle as pkl
from utils import protein_graph

ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]
# encode smiles
def encode_smiles(smile):
    mol = Chem.MolFromSmiles(smile)
    N = mol.GetNumAtoms()
    M = mol.GetNumBonds()
    type_idx = []
    chirality_idx = []
    atomic_number = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()== 0:
            print(smile)
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        atomic_number.append(atom.GetAtomicNum())
    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
    x = torch.cat([x1, x2], dim=-1)
    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
        edge_feat.append([
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir())
        ])
    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
# encode proteins
with open("data/DTI/protein_esm2.pkl", "rb") as f:
    embed = pkl.load(f)
with open("data/DTI/protein_edge.pkl", "rb") as f:
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
    protein_ids,smiles,proteins,labels = map(list,zip(*batch))
    protein_graphs = []
    smiles_graphs = []
    labels_new = torch.zeros(len(batch),dtype=torch.long) 
    for idx in range(len(batch)):
        smiles_graphs.append(encode_smiles(smiles[idx]))
        protein_graphs.append(encode_proteins(protein_ids[idx],proteins[idx]))
        label = np.int(float(labels[idx]))
        labels_new[idx] = label
    #print(protein_graphs)
    return Batch.from_data_list(smiles_graphs), Batch.from_data_list(protein_graphs), labels_new
                  
                          
        
        
    

