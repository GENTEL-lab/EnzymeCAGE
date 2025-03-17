from collections import defaultdict
from copy import deepcopy
import os
import pickle as pkl
import time

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Dataset, HeteroData, Data
from torch_geometric.nn import radius_graph
from rdkit import Chem

from ..base import UID_COL, RXN_COL, SEQ_COL


# note this is different from the 2D case
allowable_features = {
    # atom maps in {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9; 'P': 15, 'S': 16, 'CL': 17}
    "possible_atomic_num_list": [1, 6, 7, 8, 9, 15, 16, 17, "unknown"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ],
    "possible_hybridization_list": [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED,
    ],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "possible_bond_dirs": [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE,
    ],
}

SKIP_MOL = ['[*H2]']


# shorten the sentence
feats = allowable_features



def mol_to_graph_data_obj_simple_3D(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric. Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr"""
    # atoms
    atom_features_list = []
    atom_count = defaultdict(int)
    for atom in mol.GetAtoms():
        atomic_number = atom.GetAtomicNum()
        atom_count[atomic_number] += 1
        if atomic_number not in feats["possible_atomic_num_list"]:
            atomic_number = "unknown"

        atom_feature = [feats["possible_atomic_num_list"].index(atomic_number)]
        if atom.GetChiralTag() in feats["possible_chirality_list"]:
            chirality_type = feats["possible_chirality_list"].index(atom.GetChiralTag())
        else:
            chirality_type = feats["possible_chirality_list"].index(Chem.rdchem.ChiralType.CHI_OTHER)
        atom_feature.append(chirality_type)
        
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
    N = len(mol.GetAtoms())

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    conformer = mol.GetConformers()[0]
    positions = conformer.GetPositions()
    positions = torch.Tensor(positions)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_feats_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_dir = bond.GetBondDir()
            if bond_dir not in feats["possible_bond_dirs"]:
                bond_dir = Chem.rdchem.BondDir.NONE
            edge_feature = \
                [feats["possible_bonds"].index(bond_type)] + \
                [feats["possible_bond_dirs"].index(bond_dir)]
            edges_list.append((i, j))
            edge_feats_list.append(edge_feature)
            edges_list.append((j, i))
            edge_feats_list.append(edge_feature)

        # Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_feats_list), dtype=torch.long)
        
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
        positions = None

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        positions=positions,
    )
    return data, atom_count


class GeometricDataset(Dataset):
    def __init__(self, df_data, protein_data, rxn_feat_path, mol_sdf_dir, pocket_node_feature, esm_feature_path=None, reacting_center_path=None, weight_col=None):
        super(GeometricDataset, self).__init__()

        if isinstance(df_data, str) and os.path.exists(df_data):
            self.df_data = pd.read_csv(df_data)
        elif isinstance(df_data, pd.DataFrame):
            self.df_data = df_data

        if isinstance(pocket_node_feature, str) and os.path.exists(pocket_node_feature):
            print('Loading esm2 protein node feature...')
            self.pocket_node_feature = torch.load(pocket_node_feature)
        elif isinstance(pocket_node_feature, dict):
            self.pocket_node_feature = pocket_node_feature
            
        self.esm_dim = list(pocket_node_feature.values())[0].shape[1]
        
        if isinstance(protein_data, str) and os.path.exists(protein_data):
            print('Loading preprocessed protein data...')
            self.protein_dict = torch.load(protein_data)
        elif isinstance(protein_data, dict):
            self.protein_dict = protein_data
        else:
            raise ValueError('Invalid protein data input')

        self.esm_feat_dict = pkl.load(open(esm_feature_path, 'rb')) if esm_feature_path else None

        print('Loading reaction feature dict...')
        self.rxn_feat_dict = pkl.load(open(rxn_feat_path, 'rb'))

        self.uniprot_ids = self.df_data[UID_COL].tolist()
        self.rxns = self.df_data[RXN_COL].tolist()
        self.targets = self.df_data['Label'].tolist()
        self.seqs = self.df_data[SEQ_COL].tolist()
        
        if weight_col is not None:
            if weight_col in self.df_data.columns:
                self.weights = self.df_data[weight_col].tolist()
            else:
                raise ValueError(f'Invalid weight column name: {weight_col}')
        else:
            self.weights = torch.ones(len(self.df_data))

        mol_to_index = pd.read_csv(os.path.join(mol_sdf_dir, 'mol2id.csv'))
        self.mol_to_index = dict(zip(mol_to_index['SMILES'], mol_to_index['ID']))
        self.mol_to_data = self.load_mol_feat(mol_sdf_dir)
        
        self.reacting_center_map = None
        if reacting_center_path:
            self.reacting_center_map = pkl.load(open(reacting_center_path, 'rb'))

    def len(self):
        return len(self.df_data)
    
    def load_mol_feat(self, mol_sdf_dir, use_cache=True):
        cache_path = os.path.join(mol_sdf_dir, 'mol_graph_dict.pt')
        
        if os.path.exists(cache_path) and use_cache:
            print('Loading cached mol graph data...')
            mol_data_dict = torch.load(cache_path)
            if len(self.mol_to_index.keys() - mol_data_dict.keys()) == 0:
                # Should be the same, or means there are data updates, need update cache
                return mol_data_dict
        print('Data updates detected, reload graph data...')
        mol_data_dict = {}
        for smiles, index in tqdm(self.mol_to_index.items()):
            try:
                sdf_path = os.path.join(mol_sdf_dir, f'{index}.sdf')
                mol = Chem.SDMolSupplier(sdf_path)[0]
                data, _ = mol_to_graph_data_obj_simple_3D(mol)
                if not hasattr(data, 'positions'):
                    data.positions = torch.Tensor([[0, 0, 0] * len(data.x)])
                data.radius_edge_index = radius_graph(data.positions, r=3, loop=False)
                mol_data_dict[smiles] = data
            except Exception as e:
                print(e)
                print(f'Conformation feature loading failed: {smiles}')
                pass
        torch.save(mol_data_dict, cache_path)
        
        return mol_data_dict

    def merge_data(self, data_list):
        data_merged = None
        start_index = 0
        mol_index = []
        for i, data in enumerate(data_list):
            now_index = [i] * len(data.x)
            mol_index.extend(now_index)
            
            if data_merged is None:
                data_merged = data
                start_index = len(data.x)
                continue
            
            data.edge_index += start_index
            data.radius_edge_index += start_index
            
            data_merged.x = torch.cat([data_merged.x, data.x])
            data_merged.edge_index = torch.cat([data_merged.edge_index, data.edge_index], dim=1)
            data_merged.radius_edge_index = torch.cat([data_merged.radius_edge_index, data.radius_edge_index], dim=1)
            data_merged.positions = torch.cat([data_merged.positions, data.positions])
            data_merged.edge_attr = torch.cat([data_merged.edge_attr, data.edge_attr])
            start_index += len(data.x)
        
        assert len(mol_index) == len(data_merged.x)
        
        # For the same Reaction, the substrates or products belong to the same molecule
        # and mol_index is used to mark which nodes belong to the same molecule
        data_merged.mol_index = torch.tensor(mol_index)
        
        return data_merged

    def get_rxn_graph_data(self, rxn):
        substrates = [smi.replace('*', 'C') for smi in rxn.split('>>')[0].split('.') if smi not in SKIP_MOL]
        products = [smi.replace('*', 'C') for smi in rxn.split('>>')[1].split('.') if smi not in SKIP_MOL]

        try:
            substrate_data_list = [deepcopy(self.mol_to_data[mol]) for mol in substrates]
            product_data_list = [deepcopy(self.mol_to_data[mol]) for mol in products]
        except Exception as e:
            print(e)
            print(rxn)

        substrates_data = self.merge_data(substrate_data_list)
        products_data = self.merge_data(product_data_list)
        return substrates_data, products_data

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
    
    def _generate_reacting_center(self, reaction, num_node_subs, num_node_prods):
        try:
            substrate_reacting_center = torch.zeros(num_node_subs, dtype=torch.float)
            product_reacting_center = torch.zeros(num_node_prods, dtype=torch.float)
            
            reacting_center_idx = self.reacting_center_map.get(reaction)
            if reacting_center_idx:
                subs_center_idx, prods_center_idx = reacting_center_idx
                if subs_center_idx:
                    substrate_reacting_center[subs_center_idx] = 1
                if prods_center_idx:
                    product_reacting_center[prods_center_idx] = 1
        except Exception as e:
            print(e)
            print(reaction)
            print(f'num_node_subs: {num_node_subs}, num_node_prods: {num_node_prods}')
            print(sss)
            
        return substrate_reacting_center, product_reacting_center

    def get(self, idx):
        
        uniprot_id = self.uniprot_ids[idx]
        reaction = self.rxns[idx]
        
        protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[uniprot_id]
        # single_data = self.protein_dict[uniprot_id]
        
        data = HeteroData()
        data.node_xyz = protein_node_xyz
        data.y = torch.tensor(self.targets[idx], dtype=torch.float)
        data.rxn = reaction
        data.uid = uniprot_id
        data.seq = protein_seq
        data.reaction_feature = torch.tensor(self.rxn_feat_dict[reaction], dtype=torch.float)
        data.esm_feature = self.get_esm_feat(idx)
        data.weight = torch.tensor(self.weights[idx], dtype=torch.float)

        substrates_data, products_data = self.get_rxn_graph_data(reaction)
        data['substrates'].x = substrates_data.x
        data['substrates'].mol_index = substrates_data.mol_index
        data['substrates'].positions = substrates_data.positions
        data['substrates', 's2s', 'substrates'].radius_edge_index = substrates_data.radius_edge_index
        
        data['products'].x = products_data.x
        data['products'].mol_index = products_data.mol_index
        data['products'].positions = products_data.positions
        data['products', 'p2p', 'products'].radius_edge_index = products_data.radius_edge_index
        
        substrate_reacting_center, product_reacting_center = self._generate_reacting_center(reaction, len(substrates_data.x), len(products_data.x))
        data['substrates'].reacting_center = substrate_reacting_center
        data['products'].reacting_center = product_reacting_center
        
        if hasattr(self, 'uid_to_pocket'):
            data.pocket = self.construct_pocket_onehot(len(protein_seq), self.uid_to_pocket[uniprot_id])

        data['protein'].esm_node_feature = torch.tensor(self.pocket_node_feature[uniprot_id], dtype=torch.float)
        
        n_nodes_esm = len(data['protein'].esm_node_feature)
        n_nodes_gvp = len(protein_node_xyz)
        assert n_nodes_esm == n_nodes_gvp, f"nodes of esm feature: {n_nodes_esm}, nodes of gvp feature: {n_nodes_gvp}"
        
        data['protein'].node_s = protein_node_s
        data['protein'].node_v = protein_node_v
        data['protein', 'p2p', 'protein'].edge_index = protein_edge_index
        data['protein', 'p2p', 'protein'].edge_s = protein_edge_s
        data['protein', 'p2p', 'protein'].edge_v = protein_edge_v
        
        return data
    
    def get_esm_feat(self, idx):
        if self.esm_feat_dict and self.seqs[idx] in self.esm_feat_dict:
            return torch.tensor(self.esm_feat_dict[self.seqs[idx]], dtype=torch.float)
            # return self.esm_feat_dict[self.seqs[idx]]
        else:
            print('ESM feature not found')
            return torch.zeros(self.esm_dim)
    

def load_geometric_dataset(data_path, protein_gvp_feat, rxn_fp_path, mol_sdf_dir, esm_node_feature, esm_mean_feature_path, reaction_center_path, weight_col=None):
    # choose intersection of uniprotIDs from data and protein_dict
    if isinstance(data_path, str) and os.path.exists(data_path):
        df_data = pd.read_csv(data_path)
    elif isinstance(data_path, pd.DataFrame):
        df_data = data_path
    else:
        raise ValueError(f'Invalid data_path: {data_path}')
    
    proteins = set(df_data[UID_COL])
    intersect_proteins = set(protein_gvp_feat.keys()) & proteins & set(esm_node_feature.keys())
    protein_dict_train = {k: v for k, v in protein_gvp_feat.items() if k in intersect_proteins}
    df_data = df_data[df_data[UID_COL].isin(intersect_proteins)]
    print(f'size of df_data: {len(df_data)}')
    gvp_dataset = GeometricDataset(df_data, protein_dict_train, rxn_fp_path, mol_sdf_dir, esm_node_feature, esm_mean_feature_path, reaction_center_path, weight_col)
    return gvp_dataset


def create_geometric_dataset(
        train_path, 
        valid_path,
        test_path,
        protein_gvp_feat, 
        rxn_fp_path, 
        mol_sdf_dir, 
        esm_node_feature_path, 
        esm_mean_feature_path, 
        reaction_center_path,
        weight_col=None,
    ):
    
    print('Loading preprocessed protein data from: ', protein_gvp_feat)
    t1 = time.time()
    protein_dict = torch.load(protein_gvp_feat)
    print(f'Time taken to load protein data: {round(time.time() - t1, 2)} s')

    esm_node_feature = torch.load(esm_node_feature_path)
    
    # only training data needs sample weight
    train_set = load_geometric_dataset(train_path, protein_dict, rxn_fp_path, mol_sdf_dir, esm_node_feature, esm_mean_feature_path, reaction_center_path, weight_col)
    valid_set = load_geometric_dataset(valid_path, protein_dict, rxn_fp_path, mol_sdf_dir, esm_node_feature, esm_mean_feature_path, reaction_center_path)
    test_set = load_geometric_dataset(test_path, protein_dict, rxn_fp_path, mol_sdf_dir, esm_node_feature, esm_mean_feature_path, reaction_center_path)
    
    return train_set, valid_set, test_set