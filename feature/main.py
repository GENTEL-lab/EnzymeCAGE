import argparse
import os
import sys
import pickle as pkl
from multiprocessing import Pool
from functools import partial
sys.path.append('../')
sys.path.append('./pkgs/')

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import esm
from Bio import PDB

from enzymecage.base import UID_COL, RXN_COL, SEQ_COL
from utils import check_dir, cano_rxn, tranverse_folder
from gvp_torchdrug_feature import calc_gvp_feature
from extract_pocket import get_pocket_info, extract_fix_num_residues, get_pocket_info_batch_new
from extract_reacting_center import extract_reacting_center, calc_aam


def calc_morgan_fp(data_path, save_path, append=True):
    print('\n', '#' * 20, 'Calculating morgan fp', '#' * 20, '\n')
    df_data = pd.read_csv(data_path)
    rxn_list = list(set(df_data[RXN_COL]))

    if os.path.exists(save_path) and append:
        rxn_to_fp = pkl.load(open(save_path, 'rb'))
    else:
        rxn_to_fp = {}
        
    for rxn in tqdm(rxn_list):
        rcts, prods = rxn.split('>>')
        mol_rcts = Chem.MolFromSmiles(rcts)
        mol_prods = Chem.MolFromSmiles(prods)
        rcts_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_rcts, radius=2, nBits=1024))
        prods_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol_prods, radius=2, nBits=1024))
        rxnfp = np.concatenate([rcts_fp, prods_fp])
        rxn_to_fp[rxn] = rxnfp
    print(f'Number of reactions: {len(rxn_to_fp)}')
    
    check_dir(os.path.dirname(save_path))
    pkl.dump(rxn_to_fp, open(save_path, 'wb'))
    print(f'Save morgan fp to {save_path}')


def calc_drfp(data_path, save_path, append=True):
    print('\n', '=' * 20, 'Calculating DRFP', '=' * 20, '\n')
    from drfp import DrfpEncoder

    df_data = pd.read_csv(data_path)
    rxn_list = list(set(df_data[RXN_COL]))

    if os.path.exists(save_path) and append:
        rxn_to_fp = pkl.load(open(save_path, 'rb'))
    else:
        rxn_to_fp = {}

    input_list = []
    for rxn in rxn_list:
        input_list.append([rxn])
        rxn_cano = cano_rxn(rxn, remove_stereo=True)
        if rxn_cano != rxn:
            input_list.append([rxn_cano])

    drfp_results = []
    with Pool(20) as p:
        for fp in tqdm(p.imap(DrfpEncoder.encode, input_list), total=len(input_list)):
            drfp_results.append(fp[0])

    for rxn, fp in zip(input_list, drfp_results):
        rxn_to_fp[rxn[0]] = fp
    print(f'Number of reactions: {len(rxn_to_fp)}')
    
    check_dir(os.path.dirname(save_path))
    pkl.dump(rxn_to_fp, open(save_path, 'wb'))
    print(f'Save drfp to {save_path}')


def batch_generator(sequence_list, batch_size):
    for i in range(0, len(sequence_list), batch_size):
        yield sequence_list[i:i + batch_size]


def calc_seq_esm_C_feature(data_path, esm_node_feat_dir, esm_mean_feat_path):
    print('\n', '#' * 20, 'Calculating ESM-C feature', '#' * 20, '\n')
    
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    
    # Load ESM-C model
    device = 'cuda'
    model = ESMC.from_pretrained("esmc_600m").to(device) # or "cpu"
    
    df_data = pd.read_csv(data_path)
    uid_to_seq = dict(zip(df_data[UID_COL], df_data[SEQ_COL]))
    os.makedirs(esm_node_feat_dir, exist_ok=True)
    os.makedirs(os.path.dirname(esm_mean_feat_path), exist_ok=True)

    # To ensure the order
    df_data = df_data.drop_duplicates(UID_COL)

    uid_list = []
    for uid in df_data[UID_COL].tolist():
        seq = uid_to_seq.get(uid)
        save_path = os.path.join(esm_node_feat_dir, f'{uid}.npz')

        # skip if already exists
        if os.path.exists(save_path):
            continue
        uid_list.append(uid)
    print(f"\n{len(uid_list)} proteins to calculate features...")

    cnt_fail = 0
    cnt_all = 0
    failed_seqs = []
    failed_uids = []
    
    if os.path.exists(esm_mean_feat_path):
        seq_to_feature = pkl.load(open(esm_mean_feat_path, 'rb'))
    else:
        seq_to_feature = {}

    for uid in tqdm(uid_list):
        seq = uid_to_seq.get(uid)

        save_path = os.path.join(esm_node_feat_dir, f'{uid}.npz')
        if os.path.exists(save_path):
            continue
        
        protein = ESMProtein(sequence=seq)

        # Extract per-residue representations
        try:
            with torch.no_grad():
                protein_tensor = model.encode(protein)
                logits_output = model.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
        except Exception as e:
            print(f'sequence length: {len(seq)}')
            print(e)
            cnt_fail += 1
            failed_seqs.append(seq)
            failed_uids.append(uid)
            continue
        
        # dim: (n,1152)
        node_feature = logits_output.embeddings[0].cpu()
        
        np.savez_compressed(save_path, node_feature=node_feature)
        
        seq_to_feature[seq] = node_feature.mean(axis=0)
        
        cnt_all += 1    
    
    with open(esm_mean_feat_path, 'wb') as f:
        pkl.dump(seq_to_feature, f)

    print(f'\ncnt_fail: {cnt_fail}')
    df_failed = pd.DataFrame({UID_COL: failed_uids, SEQ_COL: failed_seqs})
    failed_save_path = os.path.join(esm_node_feat_dir, 'failed_proteins.csv')
    df_failed.to_csv(failed_save_path, index=False)
    print(f'Save failed proteins to {failed_save_path}')
    
    

def calc_seq_esm_feature(data_path, esm_node_feat_dir, esm_mean_feat_path):
    print('\n', '#' * 20, 'Calculating ESM feature', '#' * 20, '\n')
    # Load ESM-2 model
    device = 'cuda'

    print('Loading ESM-2 model...')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)
    print('Molel loading done!')

    df_data = pd.read_csv(data_path)
    uid_to_seq = dict(zip(df_data[UID_COL], df_data[SEQ_COL]))
    os.makedirs(esm_node_feat_dir, exist_ok=True)
    os.makedirs(os.path.dirname(esm_mean_feat_path), exist_ok=True)

    # To ensure the order
    df_data = df_data.drop_duplicates(UID_COL)

    uid_list = []
    for uid in df_data[UID_COL].tolist():
        seq = uid_to_seq.get(uid)
        save_path = os.path.join(esm_node_feat_dir, f'{uid}.npz')

        # skip if already exists
        if os.path.exists(save_path):
            continue
        uid_list.append(uid)
    print(f"\n{len(uid_list)} proteins to calculate features...")

    cnt_fail = 0
    cnt_all = 0
    failed_seqs = []
    failed_uids = []
    
    if os.path.exists(esm_mean_feat_path):
        seq_to_feature = pkl.load(open(esm_mean_feat_path, 'rb'))
    else:
        seq_to_feature = {}

    for uid in tqdm(uid_list):
        seq = uid_to_seq.get(uid)

        save_path = os.path.join(esm_node_feat_dir, f'{uid}.npz')
        if os.path.exists(save_path):
            continue
        
        input_data = [(f'seq{cnt_all}', seq)]
        
        batch_labels, batch_strs, batch_tokens = batch_converter(input_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(device)

        # Extract per-residue representations
        try:
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
        except Exception as e:
            print(f'sequence length: {len(seq)}')
            print(e)
            cnt_fail += 1
            failed_seqs.append(seq)
            failed_uids.append(uid)
            continue

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(token_representations[i, 1 : tokens_len - 1].cpu().numpy())
        node_feature = sequence_representations[0]
        
        np.savez_compressed(save_path, node_feature=node_feature)
        
        seq_to_feature[seq] = node_feature.mean(axis=0)
        
        cnt_all += 1    
    
    with open(esm_mean_feat_path, 'wb') as f:
        pkl.dump(seq_to_feature, f)

    print(f'\ncnt_fail: {cnt_fail}')
    df_failed = pd.DataFrame({UID_COL: failed_uids, SEQ_COL: failed_seqs})
    failed_save_path = os.path.join(esm_node_feat_dir, 'failed_proteins.csv')
    df_failed.to_csv(failed_save_path, index=False)
    print(f'Save failed proteins to {failed_save_path}')


def generate_rdkit_conformation_v2(smiles, n_repeat=50):
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        # mol = Chem.RemoveAllHs(mol)
        # mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv2()
        # rid = AllChem.EmbedMolecule(mol, ps)
        for repeat in range(n_repeat):
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == 0:
                break
        if rid == -1:
            # print("rid", pdb, rid)
            ps.useRandomCoords = True
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == -1:
                mol.Compute2DCoords()
            else:
                AllChem.MMFFOptimizeMolecule(mol, confId=0)
        else:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
    except Exception as e:
        print(e)
        mol = None
    # mol = Chem.RemoveAllHs(mol)
    return mol


def generate_mol_conformation(data_path, save_dir):
    print('\n', '#' * 20, 'Generating Mol Conformation', '#' * 20, '\n')
    df_data = pd.read_csv(data_path)
    rxns = set(df_data[RXN_COL])
    
    all_smiles_list = []
    for rxn in rxns:
        smiles_list = rxn.split('>>')[0].split(".") + rxn.split('>>')[1].split(".")
        all_smiles_list.extend(smiles_list)
    all_smiles_list = [smi.replace('*', 'C') for smi in all_smiles_list]
    all_mols = list(set(all_smiles_list))
    
    mol_index_path = os.path.join(save_dir, 'mol2id.csv')
    if not os.path.exists(mol_index_path):
        df_all_mols = pd.DataFrame({'SMILES': all_mols, 'ID': range(len(all_mols))})
        df_all_mols.to_csv(mol_index_path, index=False)
    else:
        df_all_mols_old = pd.read_csv(mol_index_path)
        existing_mols = set(df_all_mols_old['SMILES'])
        # Incremental generation, not just re-generate all
        mols_to_generate = [smi for smi in all_mols if smi not in existing_mols]
        start_idx = df_all_mols_old['ID'].max() + 1
        # df_all_mols is mols to generate
        df_all_mols = pd.DataFrame({'SMILES': mols_to_generate, 'ID': range(start_idx, start_idx + len(mols_to_generate))})
        # df_to_save is all mols info
        df_to_save = pd.concat([df_all_mols_old, df_all_mols])
        df_to_save.to_csv(mol_index_path, index=False)
    
    failed_idx_list = []
    for smiles, idx in tqdm(df_all_mols[['SMILES', 'ID']].values):
        save_path = os.path.join(save_dir, f'{idx}.sdf')
        conformation = generate_rdkit_conformation_v2(smiles)
        if not conformation:
            failed_idx_list.append(idx)
            continue
        Chem.SDWriter(save_path).write(conformation)


def get_pocket_info_batch(input_dir, save_path, pocket_save_dir=None, max_residue_num=None):
    filelist = [path for path in tranverse_folder(input_dir) if path.endswith('.cif')]
    uid_list = [os.path.basename(each).replace('_transplant.cif', '') for each in filelist]
    
    if max_residue_num is None:
        running_func = partial(get_pocket_info, pocket_save_dir=pocket_save_dir)
    else:
        running_func = partial(extract_fix_num_residues, pocket_save_dir=pocket_save_dir, residue_num=max_residue_num)
        
    pocket_info_list = []
    with Pool(20) as pool:
        for res in tqdm(pool.imap(running_func, filelist), total=len(filelist)):
            pocket_info_list.append(res)
    df_pocket_info = pd.DataFrame({UID_COL: uid_list, 'pocket_residues': pocket_info_list})
    df_pocket_info.to_csv(save_path, index=False)
    print(f'\nSave pocket info to {save_path}\n')
    
    delete_empty(pocket_save_dir)


def get_esm_pocket_feature(pocket_info_path, esm_node_feat_dir, save_path):
    df_pocket_data = pd.read_csv(pocket_info_path)
    uid_to_pocket = dict(zip(df_pocket_data[UID_COL], df_pocket_data['pocket_residues']))
    esm_file_list = [each for each in tranverse_folder(esm_node_feat_dir) if each.endswith('.npz')]
    
    uid_to_pocket_node_feature = {}
    for filepath in tqdm(esm_file_list):
        uid = os.path.basename(filepath).replace('.npz', '')
        if uid in uid_to_pocket_node_feature:
            continue
        
        if not filepath.endswith('npz'):
            continue
        esm_node_feature = np.load(filepath)['node_feature']
        
        pocket_residue_ids = uid_to_pocket.get(uid)
        if not isinstance(pocket_residue_ids, str):
            # print('???')
            continue
    
        pocket_residue_ids = [int(i)-1 for i in pocket_residue_ids.split(',')]
        try:
            pocket_node_feature = esm_node_feature[pocket_residue_ids]
        except Exception as e:
            print(e)
            print(uid, ' Error')
            print(sss)
        
        uid_to_pocket_node_feature[uid] = pocket_node_feature
        
    torch.save(uid_to_pocket_node_feature, save_path)
    print(f'Save esm pocket feature to {save_path}\n')


def check_pocket_feature(gvp_feature_path, esm_feature_path, log_dir=None):
    gvp_feature = torch.load(gvp_feature_path)
    esm_feature = torch.load(esm_feature_path)
    
    assert len(gvp_feature) == len(esm_feature)
    bad_proteins = []
    for uid, gvp_node_feature in gvp_feature.items():
        esm_node_feature = esm_feature[uid]
        n_gvp_nodes = gvp_node_feature[0].shape[0]
        n_esm_nodes = esm_node_feature.shape[0]
        if n_gvp_nodes != n_esm_nodes:
            bad_proteins.append(uid)
    
    df_bad_pros = pd.DataFrame({UID_COL: bad_proteins})
    
    if len(df_bad_pros) > 0 and log_dir:
        print(f"Found {len(df_bad_pros)} bad proteins, which have different node numbers between gvp and esm features. These proteins should be removed from training data!")
        save_path = os.path.join(log_dir, 'bad_proteins.csv')
        df_bad_pros.to_csv(save_path, index=False)
        print(f"Save bad proteins to {save_path}\n")


def delete_empty(data_dir):
    file_list = tranverse_folder(data_dir)
    for filepath in tqdm(file_list):
        if os.path.getsize(filepath) < 1000:
            os.remove(filepath)
       

def calc_reacting_center(data_path, save_dir, append=True):
    print('\n', '#' * 20, 'Calculating Reaction Center', '#' * 20, '\n')
    
    calc_aam(data_path, save_dir, append)
    
    aam_path = os.path.join(save_dir, 'rxn2aam.pkl')
    rxn2aam = pkl.load(open(aam_path, 'rb'))
    
    reacting_center_path = os.path.join(save_dir, 'reacting_center.pkl')
    if os.path.exists(reacting_center_path) and append:
        cached_reacting_center_map = pkl.load(open(reacting_center_path, 'rb'))
    else:
        cached_reacting_center_map = {}
    
    df_data = pd.read_csv(data_path)
    rxns_to_run = df_data[RXN_COL].unique()
    rxns_to_run = [rxn for rxn in rxns_to_run if rxn not in cached_reacting_center_map]
    reacting_center_map = {}
    for rxn in tqdm(rxns_to_run):
        reacting_center_map[rxn] = extract_reacting_center(rxn, rxn2aam)
    
    if append:
        print(f'Append {len(reacting_center_map)} reacting center to {reacting_center_path}')    
    
    reacting_center_map.update(cached_reacting_center_map)
    pkl.dump(reacting_center_map, open(reacting_center_path, 'wb'))
    
    if not append:
        print(f'Calculate {len(reacting_center_map)} reacting center and save to {reacting_center_path}')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--alphafill_result_dir', type=str, default=None)
    parser.add_argument('--pocket_dir', type=str, help='If you already have pocket data, you can specify the directory here')
    parser.add_argument('--skip_calc_mol_conformation', action='store_true')
    args = parser.parse_args()
    
    # alphafill_result_dir = os.path.join(os.path.dirname(args.data_path), 'alphafill_results')
    alphafill_result_dir = args.alphafill_result_dir
    # if not os.path.exists(alphafill_result_dir):
    #     raise FileNotFoundError(f'alphafill result not found in {alphafill_result_dir}')
    
    if not args.pocket_dir:
        pocket_dir = os.path.join(os.path.dirname(args.data_path), 'pocket/alphafill_8A')
    else:
        pocket_dir = args.pocket_dir
        
    feature_dir = os.path.join(os.path.dirname(args.data_path), 'feature')
    
    esm_model = 'esm2_t33_650M_UR50D'
    # esm_model = 'ESM-C_600M'
    
    protein_feature_dir = os.path.join(feature_dir, 'protein')
    esm_node_feat_dir = os.path.join(protein_feature_dir, f'{esm_model}/node_level')
    esm_mean_feat_path = os.path.join(protein_feature_dir, f'{esm_model}/protein_level/seq2feature.pkl')
    esm_pocket_node_feature_path = os.path.join(protein_feature_dir, f'{esm_model}/pocket_node_feature/esm_node_feature.pt')
    gvp_feat_path = os.path.join(protein_feature_dir, 'gvp_feature/gvp_protein_feature.pt')
    pocket_info_save_path = os.path.join(os.path.dirname(pocket_dir), 'pocket_info.csv')
    
    reaction_feat_dir = os.path.join(feature_dir, 'reaction')
    morgan_save_path = os.path.join(reaction_feat_dir, 'morgan_fp/rxn2fp.pkl')
    drfp_save_path = os.path.join(reaction_feat_dir, 'drfp/rxn2fp.pkl')
    mol_conformation_dir = os.path.join(reaction_feat_dir, 'molecule_conformation')
    reacting_center_dir = os.path.join(reaction_feat_dir, 'reacting_center')

    os.makedirs(pocket_dir, exist_ok=True)
    os.makedirs(esm_node_feat_dir, exist_ok=True)
    os.makedirs(os.path.dirname(esm_mean_feat_path), exist_ok=True)
    os.makedirs(os.path.dirname(esm_pocket_node_feature_path), exist_ok=True)
    os.makedirs(os.path.dirname(gvp_feat_path), exist_ok=True)
    os.makedirs(os.path.dirname(morgan_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(drfp_save_path), exist_ok=True)
    os.makedirs(mol_conformation_dir, exist_ok=True)
    os.makedirs(reacting_center_dir, exist_ok=True)
    
    # Calculate reaction fingerprints
    calc_morgan_fp(args.data_path, morgan_save_path)
    calc_drfp(args.data_path, drfp_save_path)
    
    # Extract reaction center
    calc_reacting_center(args.data_path, reacting_center_dir)
    
    if not args.skip_calc_mol_conformation:
        # May be the most time consuming step
        # Generate molecular conformation by rdkit
        generate_mol_conformation(args.data_path, mol_conformation_dir)
    
    # Calculate GVP features of pockets
    calc_gvp_feature(args.data_path, pocket_dir, gvp_feat_path)
    
    # Calculate ESM features of the full sequence
    if esm_model == 'esm2_t33_650M_UR50D':
        calc_seq_esm_feature(args.data_path, esm_node_feat_dir, esm_mean_feat_path)
    elif esm_model == 'ESM-C_600M':
        calc_seq_esm_C_feature(args.data_path, esm_node_feat_dir, esm_mean_feat_path)
    
    # Extract esm feature of pocket nodes
    get_esm_pocket_feature(pocket_info_save_path, esm_node_feat_dir, esm_pocket_node_feature_path)
    
    # Extract pocket information if you don't specify the pocket directory
    if not args.pocket_dir:
        # Note: make sure the result file of alphafill ends with '_transplant.cif'
        get_pocket_info_batch(alphafill_result_dir, pocket_info_save_path, pocket_dir)
    else:
        if not os.path.exists(pocket_info_save_path):
            get_pocket_info_batch_new(args.pocket_dir, pocket_info_save_path)
    
    # Make sure the number of nodes of pocket is the same between GVP and ESM feature
    check_pocket_feature(gvp_feat_path, esm_pocket_node_feature_path, log_dir=os.path.dirname(args.data_path))
    
    print('\n ###### Feature calculation is finished! ######\n')
    

if __name__ == "__main__":
    main()
    
