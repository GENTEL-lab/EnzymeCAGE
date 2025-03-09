import os
import random

import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolops import PatternFingerprint


def check_files(model_conf):
    if hasattr(model_conf, "train_path"):
        assert os.path.exists(model_conf.train_path)
    if hasattr(model_conf, "valid_path"):
        assert os.path.exists(model_conf.valid_path)
    if hasattr(model_conf, "test_path"):
        assert os.path.exists(model_conf.test_path)
    if hasattr(model_conf, "data_path"):
        assert os.path.exists(model_conf.data_path)
    if hasattr(model_conf, "rxn_fp"):
        assert os.path.exists(model_conf.rxn_fp)
    if hasattr(model_conf, "protein_gvp_feat"):
        assert os.path.exists(model_conf.protein_gvp_feat)
    if hasattr(model_conf, "esm_mean_feature"):
        assert os.path.exists(model_conf.esm_mean_feature)
    if hasattr(model_conf, "esm_node_feature"):
        assert os.path.exists(model_conf.esm_node_feature)
    if hasattr(model_conf, "mol_conformation"):
        assert os.path.exists(model_conf.mol_conformation)
    if hasattr(model_conf, "reaction_center"):
        assert os.path.exists(model_conf.reaction_center)


def tranverse_folder(folder):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def get_rdkit_mol(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    return mol


def remove_stereo(mol):
    mol = get_rdkit_mol(mol)
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol)


def cano_smiles(smiles, remove_stereo=False):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
      
    if remove_stereo:
        Chem.RemoveStereochemistry(mol)

    return Chem.MolToSmiles(mol)


def cano_rxn(rxn, exchange_pos=False, remove_stereo=False):
    data = rxn.split('>')
    reactants = data[0].split('.')
    reactants = [cano_smiles(each, remove_stereo) for each in reactants]
    products = data[-1].split('.')
    products = [cano_smiles(each, remove_stereo) for each in products]
    reactants = sorted(reactants)
    products = sorted(products)
    if exchange_pos:
        new_rxn = f"{'.'.join(products)}>>{'.'.join(reactants)}"
    else:
        new_rxn = f"{'.'.join(reactants)}>>{'.'.join(products)}"
    return new_rxn


def load_feature(model_conf):
    if model_conf.auto_load_data:
        feat_dir = os.path.join(model_conf.data_dir, f'split_{model_conf.data_split_by}', 'npy_feature', f'{model_conf.rxn_feat_type}_{model_conf.enz_feat_type}')
        train_feat_path = os.path.join(feat_dir, 'train_feature.npy')
        valid_feat_path = os.path.join(feat_dir, 'valid_feature.npy')
        test_feat_path = os.path.join(feat_dir, 'test_feature.npy')

        train_feat = np.load(train_feat_path)
        valid_feat = np.load(valid_feat_path)
        test_feat = np.load(test_feat_path)
    else:
        train_feat = np.load(model_conf.train_feat)
        valid_feat = np.load(model_conf.valid_feat)
        test_feat = np.load(model_conf.test_feat)

    return train_feat, valid_feat, test_feat


def get_data_path(model_conf, data_type='train'):
    return os.path.join(model_conf.data_dir, f'split_{model_conf.data_split_by}', 'csv', f"{data_type}.csv")


def load_origin_data(model_conf):
    if model_conf.auto_load_data:
        train_path = get_data_path(model_conf, 'train')
        valid_path = get_data_path(model_conf, 'valid')
        test_path = get_data_path(model_conf, 'test')

        origin_train_data = pd.read_csv(train_path)
        origin_valid_data = pd.read_csv(valid_path)
        origin_test_data = pd.read_csv(test_path)
    else:
        origin_train_data = pd.read_csv(model_conf.train_path)
        origin_valid_data = pd.read_csv(model_conf.valid_path)
        origin_test_data = pd.read_csv(model_conf.test_path)

    return origin_train_data, origin_valid_data, origin_test_data


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(5)    
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calc_rxn_center_fp(rxn_center):
    # rxn_center就是template
    prod_c = rxn_center.split('>>')[-1]
    reac_c = rxn_center.split('>>')[0]
    prod_mol = Chem.MolFromSmarts(prod_c)
    reac_mol = Chem.MolFromSmarts(reac_c)
    prod_fp = np.array(PatternFingerprint(prod_mol, 512))
    reac_fp = np.array(PatternFingerprint(reac_mol, 512))
    return reac_fp, prod_fp


def remove_nan(data):
    if isinstance(data, set):
        data = {each for each in data if not pd.isna(each)}
    elif isinstance(data, list):
        data = [each for each in data if not pd.isna(each)]
    return data


def check_dir(path):
    folder = os.path.dirname(path) if '.' in path.split('/')[-1] else path
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f'Make new directory: {folder}')


def get_substrate_from_rxn(rxn):
    rcts = rxn.split('>>')[0].split('.')
    rcts = sorted(rcts, key=lambda x: len(x), reverse=True)
    return rcts[0]
