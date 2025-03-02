import os
import sys

import mlcrate as mlc
from Bio.PDB import PDBParser
import torch
from tqdm import tqdm
import pandas as pd
torch.set_num_threads(1)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(ROOT_DIR, 'enzymecage')
sys.path.append(model_dir)
from gvp.data import ProteinGraphDataset
from utils import UID_COL


three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_protein_feature(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = ProteinGraphDataset([structure])
    protein = dataset[0]
    x = (protein.x, protein.seq, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v)
    return x


def batch_run(x):
    protein_dict = {}
    pdb, proteinFile, toFile = x
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb, proteinFile)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    
    protein_dict[pdb] = get_protein_feature(res_list)
    # if 'phosphatase_63' in proteinFile:
    #     print('num residues: ', len(protein_dict[pdb][0]))
    torch.save(protein_dict, toFile)


def calc_gvp_feature(data_path, pdb_dir, save_path):
    print('\n', '#' * 20, 'Calculating GVP Feature', '#' * 20, '\n')
    protein_embedding_folder = os.path.join(os.path.dirname(save_path), 'tmp')
    os.makedirs(protein_embedding_folder, exist_ok=True)

    df_data = pd.read_csv(data_path)
    uids = set(df_data[UID_COL])
    pdbfiles = [os.path.join(pdb_dir, f'{uid}.pdb') for uid in uids]

    input_ = []
    uniprot_id_list = []
    for filepath in pdbfiles:
        if not filepath.endswith('pdb') or not os.path.exists(filepath):
            continue
        
        uniprot_id = os.path.basename(filepath).replace('.pdb', '')
        uniprot_id_list.append(uniprot_id)
        toFile = f"{protein_embedding_folder}/{uniprot_id}.pt"
        x = (uniprot_id, filepath, toFile)
        input_.append(x)
    
    
    pool = mlc.SuperPool(10)
    pool.pool.restart()
    _ = pool.map(batch_run, input_)
    pool.exit()

    protein_dict = {}
    for uniprot_id in tqdm(uniprot_id_list):
        protein_dict.update(torch.load(f"{protein_embedding_folder}/{uniprot_id}.pt"))

    torch.save(protein_dict, save_path)
    print(f'Save gvp feature to {save_path}\n')


def run_single():
    uid = 'H9A1V3'
    pdb_file = f'/mnt/nas/ai-algorithm-data/liuyong/dataset/SynBio/enzyme-reaction-pairs/from_zt/3D-structures/active_site_new_no_ion/pdb_8A/{uid}.pdb'
    save_path = f'/mnt/nas/ai-algorithm-data/liuyong/code/SynBio/enzyme-rxn-prediction/dataset/gvp/gvp_protein_embedding_pocket/{uid}.pt'

    input_ = (uid, pdb_file, save_path)
    batch_run(input_)

    print(f'{uid} done')


if __name__ == '__main__':
    # run_single()
    main()



