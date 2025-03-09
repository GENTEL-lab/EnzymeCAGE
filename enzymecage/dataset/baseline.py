import os
import pickle as pkl

import pandas as pd
import torch
from torch_geometric.data import Dataset, HeteroData

from ..base import UID_COL, RXN_COL, SEQ_COL


class BaselineDataset(Dataset):
    def __init__(self, df_data, rxn_feat_path, esm_feature_path):
        super(BaselineDataset, self).__init__()

        if isinstance(df_data, str) and os.path.exists(df_data):
            self.df_data = pd.read_csv(df_data)
        elif isinstance(df_data, pd.DataFrame):
            self.df_data = df_data

        self.esm_feat_dict = pkl.load(open(esm_feature_path, 'rb')) if esm_feature_path else None
        
        self.esm_dim = list(self.esm_feat_dict.values())[0].shape[0]

        print('Loading reaction feature dict...')
        self.rxn_feat_dict = pkl.load(open(rxn_feat_path, 'rb'))

        self.uniprot_ids = self.df_data[UID_COL].tolist()
        self.rxns = self.df_data[RXN_COL].tolist()
        self.targets = self.df_data['Label'].tolist()
        self.seqs = self.df_data[SEQ_COL].tolist()

    def len(self):
        return len(self.df_data)

    def get(self, idx):
        uniprot_id = self.uniprot_ids[idx]
        reaction = self.rxns[idx]
        sequence = self.seqs[idx]
        
        data = HeteroData()
        data.y = torch.tensor(self.targets[idx], dtype=torch.float)
        data.rxn = reaction
        data.seq = sequence
        data.uid = uniprot_id
        data.reaction_feature = torch.tensor(self.rxn_feat_dict[reaction], dtype=torch.float)
        data.esm_feature = self.get_esm_feat(idx)
        
        return data
    
    def get_esm_feat(self, idx):
        if self.esm_feat_dict and self.seqs[idx] in self.esm_feat_dict:
            return torch.tensor(self.esm_feat_dict[self.seqs[idx]], dtype=torch.float)
        else:
            # print('ESM feature not found for sequence idx: ', idx)
            return torch.zeros(self.esm_dim)


def load_baseline_dataset(data_path, rxn_feat_path, esm_feature_path):
    if isinstance(data_path, str) and os.path.exists(data_path):
        df_data = pd.read_csv(data_path)
    elif isinstance(data_path, pd.DataFrame):
        df_data = data_path
    else:
        raise ValueError(f'Invalid data_path: {data_path}')
    
    gvp_dataset = BaselineDataset(df_data, rxn_feat_path, esm_feature_path)
    return gvp_dataset


def create_baseline_dataset(
        train_path, 
        valid_path,
        test_path,
        rxn_fp_path, 
        esm_mean_feature_path
    ):
    train_set = load_baseline_dataset(train_path, rxn_fp_path, esm_mean_feature_path)
    valid_set = load_baseline_dataset(valid_path, rxn_fp_path, esm_mean_feature_path)
    test_set = load_baseline_dataset(test_path, rxn_fp_path, esm_mean_feature_path)
    
    return train_set, valid_set, test_set