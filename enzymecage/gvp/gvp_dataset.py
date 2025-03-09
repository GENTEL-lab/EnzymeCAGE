import copy
import os
import pickle as pkl
import time
from multiprocessing import Pool

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Dataset, HeteroData
from rdkit import Chem
from dgllife.utils import CanonicalAtomFeaturizer, mol_to_bigraph

from ..base import UID_COL, RXN_COL, SEQ_COL


def rxn_to_graph(rxn_smiles):
    reaction_graphs = []
    for mol_smiles in rxn_smiles.split('>>'):
        mol = Chem.MolFromSmiles(mol_smiles)
        g = mol_to_bigraph(mol,node_featurizer=CanonicalAtomFeaturizer())
        reaction_graphs.append(g)
    return reaction_graphs


class GVPDataset(Dataset):
    def __init__(self, df_data, protein_data, rxn_feat_path, pocket_data=None, esm_feature_path=None, unimol_feature_path=None, load_rxn_graph=False):
        super(GVPDataset, self).__init__()

        self.load_rxn_graph = load_rxn_graph
        if isinstance(df_data, str) and os.path.exists(df_data):
            self.df_data = pd.read_csv(df_data)
        elif isinstance(df_data, pd.DataFrame):
            self.df_data = df_data
        
        if isinstance(protein_data, str) and os.path.exists(protein_data):
            print('Loading preprocessed protein data...')
            self.protein_dict = torch.load(protein_data)
        elif isinstance(protein_data, dict):
            self.protein_dict = protein_data
        else:
            raise ValueError('Invalid protein data input')

        self.esm_feat_dict = pkl.load(open(esm_feature_path, 'rb')) if esm_feature_path else None
        self.unimol_feat_dict = pkl.load(open(unimol_feature_path, 'rb')) if unimol_feature_path else None

        print('Loading reaction feature dict...')
        self.rxn_feat_dict = pkl.load(open(rxn_feat_path, 'rb'))

        self.uniprot_ids = self.df_data[UID_COL].tolist()
        self.rxns = self.df_data[RXN_COL].tolist()
        self.targets = self.df_data['Label'].tolist()
        self.seqs = self.df_data[SEQ_COL].tolist()

        if pocket_data:
            if isinstance(pocket_data, str) and os.path.exists(pocket_data):
                pocket_data = pd.read_csv(pocket_data)
            assert isinstance(pocket_data, pd.DataFrame)
            self.uid_to_pocket = dict(zip(pocket_data[UID_COL], pocket_data['pocket_residues']))

        if load_rxn_graph:
            rxns_unique = list(set(self.rxns))
            self.reaction_graphs_dict = {rxn: rxn_to_graph(rxn) for rxn in tqdm(rxns_unique)}

    def len(self):
        return len(self.df_data)

    def construct_pocket_onehot(self, num_residue, pocket_residues):
        pocket_one_hot = torch.zeros(num_residue)
        if isinstance(pocket_residues, str):
            pocket_residues = [int(residue_id)-1 for residue_id in pocket_residues.split(',')]
            pocket_one_hot[pocket_residues] = 1
        elif not pocket_residues or pd.isna(pocket_residues):
            pass
        else:
            raise ValueError(f'pocket_residues must be str or None, get: {type(pocket_residues)}, value: {pocket_residues}')
        
        return pocket_one_hot

    def get(self, idx):
        uniprot_id = self.uniprot_ids[idx]
        reaction = self.rxns[idx]
        
        protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[uniprot_id]
        data = HeteroData()
        data.node_xyz = protein_node_xyz
        if self.load_rxn_graph:
            data.reaction_graph = self.reaction_graphs_dict[reaction]
        data.y = torch.tensor(self.targets[idx], dtype=torch.float)
        data.rxn = reaction
        data.seq = protein_seq
        data.reaction_feature = torch.tensor(self.rxn_feat_dict[reaction], dtype=torch.float)
        data.esm_feature = self.get_esm_feat(idx)

        data['protein'].node_s = protein_node_s
        data['protein'].node_v = protein_node_v
        data['protein', 'p2p', 'protein'].edge_index = protein_edge_index
        data['protein', 'p2p', 'protein'].edge_s = protein_edge_s
        data['protein', 'p2p', 'protein'].edge_v = protein_edge_v
        
        return data
    
    # def get(self, idx):
    #     uniprot_id = self.uniprot_ids[idx]
    #     reaction = self.rxns[idx]
        
    #     protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[uniprot_id]
    #     data_dict = {
    #         '_global_store': {
    #             'node_xyz': protein_node_xyz,
    #             'y': torch.tensor(self.targets[idx], dtype=torch.float),
    #             'rxn': reaction,
    #             'seq': protein_seq,
    #             'reaction_feature': torch.tensor(self.rxn_feat_dict[reaction], dtype=torch.float),
    #             'esm_feature': self.get_esm_feat(idx),
    #         },
    #         'protein': {
    #             'node_s': protein_node_s,
    #             'node_v': protein_node_v,
    #             'edge_index': protein_edge_index.transpose(0, 1),
    #             'edge_s': protein_edge_s,
    #             'edge_v': protein_edge_v
    #         }
    #     }
    #     if self.load_rxn_graph:
    #         data_dict['reaction_graph'] = self.reaction_graphs_dict[reaction]
        
    #     data = HeteroData.from_dict(data_dict)
        
    #     return data
    
    def get_esm_feat(self, idx):
        if self.esm_feat_dict and self.seqs[idx] in self.esm_feat_dict:
            return torch.tensor(self.esm_feat_dict[self.seqs[idx]], dtype=torch.float)
        else:
            # print('ESM feature not found for sequence idx: ', idx)
            return torch.zeros(1280)

    def get_unimol_feat(self, idx):
        if self.unimol_feat_dict and self.uniprot_ids[idx] in self.unimol_feat_dict:
            return torch.tensor(self.unimol_feat_dict[self.uniprot_ids[idx]], dtype=torch.float)
        else:
            return torch.zeros(512)


class GVPGINDataset(Dataset):
    def __init__(self, df_data, protein_data, substrate_data, rxn_feat_path, esm_feature_path=None):
        super(GVPGINDataset, self).__init__()

        if isinstance(df_data, str) and os.path.exists(df_data):
            self.df_data = pd.read_csv(df_data)
        elif isinstance(df_data, pd.DataFrame):
            self.df_data = df_data
        
        if isinstance(protein_data, str) and os.path.exists(protein_data):
            print('Loading preprocessed protein data...')
            self.protein_dict = torch.load(protein_data)
        elif isinstance(protein_data, dict):
            self.protein_dict = protein_data
        else:
            raise ValueError('Invalid protein data input')

        if isinstance(substrate_data, str) and os.path.exists(substrate_data):
            print('Loading preprocessed substrate data...')
            self.substrate_dict = torch.load(substrate_data)
        elif isinstance(substrate_data, dict):
            self.substrate_dict = substrate_data
        else:
            raise ValueError('Invalid substrate data input')

        self.esm_feat_dict = pkl.load(open(esm_feature_path, 'rb')) if esm_feature_path else None

        print('Loading reaction feature dict...')
        self.rxn_feat_dict = pkl.load(open(rxn_feat_path, 'rb'))

        self.uniprot_ids = self.df_data[UID_COL].tolist()
        self.rxns = self.df_data[RXN_COL].tolist()
        self.substrate_list = [rxn.split('>')[0] for rxn in self.rxns]
        self.targets = self.df_data['Label'].tolist()
        self.seqs = self.df_data[SEQ_COL].tolist()


    def len(self):
        return len(self.df_data)

    def get(self, idx):
        uniprot_id = self.uniprot_ids[idx]
        reaction = self.rxns[idx]
        substrate = self.substrate_list[idx]
        
        protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[uniprot_id]
        compound_node_features, input_atom_edge_list, input_atom_edge_attr_list = self.substrate_dict[substrate]

        data = HeteroData()
        data.node_xyz = protein_node_xyz
        data.y = torch.tensor(self.targets[idx], dtype=torch.float)
        data.rxn = reaction
        data.seq = protein_seq
        data.reaction_feature = torch.tensor(self.rxn_feat_dict[reaction], dtype=torch.float)
        data.esm_feature = self.get_esm_feat(idx)

        data['protein'].node_s = protein_node_s
        data['protein'].node_v = protein_node_v
        data['protein', 'p2p', 'protein'].edge_index = protein_edge_index
        data['protein', 'p2p', 'protein'].edge_s = protein_edge_s
        data['protein', 'p2p', 'protein'].edge_v = protein_edge_v

        data['compound'].x = compound_node_features
        data['compound', 'c2c', 'compound'].edge_index = input_atom_edge_list[:,:2].long().t().contiguous()
        data['compound', 'c2c', 'compound'].edge_weight = torch.ones(input_atom_edge_list.shape[0])
        data['compound', 'c2c', 'compound'].edge_attr = input_atom_edge_attr_list
        
        return data
    
    def get_esm_feat(self, idx):
        if self.esm_feat_dict and self.seqs[idx] in self.esm_feat_dict:
            return torch.tensor(self.esm_feat_dict[self.seqs[idx]], dtype=torch.float)
        else:
            # print('ESM feature not found for sequence idx: ', idx)
            return torch.zeros(1280)



def get_data_path(model_conf, data_type='train'):
    return os.path.join(model_conf.data_dir, f'split_{model_conf.data_split_by}', 'csv', f"{data_type}.csv")


def load_single_dataset(data_path, protein_dict, rxn_feat_path, esm_feature_path, pocket_data=None, unimol_feature_path=None, use_gat_encoder=False):
    # choose intersection of uniprotIDs from data and protein_dict
    df_data = pd.read_csv(data_path)
    proteins = set(df_data[UID_COL])

    intersect_proteins = set(protein_dict.keys()) & proteins
    protein_dict_train = {k: v for k, v in protein_dict.items() if k in intersect_proteins}
    df_data = df_data[df_data[UID_COL].isin(intersect_proteins)]
    gvp_dataset = GVPDataset(df_data, protein_dict_train, rxn_feat_path, pocket_data, esm_feature_path, unimol_feature_path, load_rxn_graph=use_gat_encoder)
    return gvp_dataset


def create_gvp_dataset(model_conf, protein_data_path, rxn_feat_path, pocket_data, esm_feature_path, unimol_feature_path=None, use_gat_encoder=False):
    if model_conf.auto_load_data:
        train_path = get_data_path(model_conf, 'train')
        valid_path = get_data_path(model_conf, 'valid')
        test_path = get_data_path(model_conf, 'test')
    else:
        train_path = model_conf.train_path
        valid_path = model_conf.valid_path
        test_path = model_conf.test_path

    # train_path, valid_path, test_path, protein_data_path, rxn_feat_path
    print('Loading preprocessed protein data from: ', protein_data_path)
    t1 = time.time()
    protein_dict = torch.load(protein_data_path)
    print(f'Time taken to load protein data: {round(time.time() - t1, 2)} s')

    train_set = load_single_dataset(train_path, protein_dict, rxn_feat_path, esm_feature_path, pocket_data, unimol_feature_path, use_gat_encoder)
    valid_set = load_single_dataset(valid_path, protein_dict, rxn_feat_path, esm_feature_path, pocket_data, unimol_feature_path, use_gat_encoder)
    test_set = load_single_dataset(test_path, protein_dict, rxn_feat_path, esm_feature_path, pocket_data, unimol_feature_path, use_gat_encoder)
    
    return train_set, valid_set, test_set


def load_gvpgin_dataset(data_path, protein_dict, substrate_dict, rxn_feat_path, esm_feature_path):
    # choose intersection of uniprotIDs from data and protein_dict
    df_data = pd.read_csv(data_path)
    proteins = set(df_data[UID_COL])

    intersect_proteins = set(protein_dict.keys()) & proteins
    protein_dict_train = {k: v for k, v in protein_dict.items() if k in intersect_proteins}
    df_data = df_data[df_data[UID_COL].isin(intersect_proteins)]
    gvp_dataset = GVPGINDataset(df_data, protein_dict_train, substrate_dict, rxn_feat_path, esm_feature_path)
    return gvp_dataset

def create_gvpgin_dataset(model_conf, protein_data_path, substrate_data_path, rxn_feat_path, esm_feature_path):
    if model_conf.auto_load_data:
        train_path = get_data_path(model_conf, 'train')
        valid_path = get_data_path(model_conf, 'valid')
        test_path = get_data_path(model_conf, 'test')
    else:
        train_path = model_conf.train_path
        valid_path = model_conf.valid_path
        test_path = model_conf.test_path
    
    print('Loading preprocessed protein data from: ', protein_data_path)
    t1 = time.time()
    protein_dict = torch.load(protein_data_path)
    print(f'Time taken to load protein data: {round(time.time() - t1, 2)} s')

    substrate_feat_dict = torch.load(substrate_data_path)

    train_set = load_gvpgin_dataset(train_path, protein_dict, substrate_feat_dict, rxn_feat_path, esm_feature_path)
    valid_set = load_gvpgin_dataset(valid_path, protein_dict, substrate_feat_dict, rxn_feat_path, esm_feature_path)
    test_set = load_gvpgin_dataset(test_path, protein_dict, substrate_feat_dict, rxn_feat_path, esm_feature_path)
    
    return train_set, valid_set, test_set
