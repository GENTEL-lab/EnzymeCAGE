import os
import argparse
import pickle as pkl
from copy import deepcopy

import pandas as pd
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


def get_rdkit_mol(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    return mol


def cano_smiles(smiles, remain_isomer=True):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=remain_isomer)


def calc_similarity(mol1, mol2):
    mol1 = get_rdkit_mol(mol1)
    mol2 = get_rdkit_mol(mol2)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
    return similarity


def remove_duplicate_mols(mols1, mols2):
    tmp1 = deepcopy(mols1)
    for mol1 in tmp1:
        flag = False
        for mol2 in mols2:
            simi = calc_similarity(mol1, mol2)
            if simi == 1:
                flag = True
                break
        if flag:
            mols1.remove(mol1)
            mols2.remove(mol2)
    return mols1, mols2


def neutralize_atoms(smi):
    try:
        mol = get_rdkit_mol(smi)
        pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
        at_matches = mol.GetSubstructMatches(pattern)
        at_matches_list = [y[0] for y in at_matches]
        if len(at_matches_list) > 0:
            for at_idx in at_matches_list:
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                hcount = atom.GetTotalNumHs()
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
                
        smi_uncharged = Chem.MolToSmiles(mol)
        if not get_rdkit_mol(smi_uncharged):
            smi_uncharged = smi
    except Exception as e:
        print(e)
        print(f'Cannot neutralize SMILES: {smi}, use original format')
        smi_uncharged = smi
    
    return smi_uncharged


def neutralize_rxn(rxn):
    if not rxn:
        return None
    
    rcts = [neutralize_atoms(smi) for smi in rxn.split('>>')[0].split('.')]
    prods = [neutralize_atoms(smi) for smi in rxn.split('>>')[1].split('.')]
    
    return '.'.join(rcts) + '>>' + '.'.join(prods)


def remove_redundent_from_rxn(rxn, items_to_remove=None):
    if not rxn:
        return None
    
    # 金属和卤素离子这些先保留
    if not items_to_remove:
        items_to_remove = ['*', '[H+]', '[H]*[H]']

    rcts = [smi for smi in rxn.split('>>')[0].split('.') if smi not in items_to_remove]
    prods = [smi for smi in rxn.split('>>')[1].split('.') if smi not in items_to_remove]
    
    rcts, prods = remove_duplicate_mols(rcts, prods)
    
    if not rcts or not prods:
        return None

    return '.'.join(rcts) + '>>' + '.'.join(prods)


def count_carbon_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')


def filter_unwanted_rxn(rxn):
    if not rxn:
        return None
    
    rct = rxn.split('>>')[0]
    prod = rxn.split('>>')[1]
    
    rct_mol = get_rdkit_mol(rct)
    prod_mol = get_rdkit_mol(prod)
    
    if len(rct_mol.GetAtoms()) > 150 or len(prod_mol.GetAtoms()) > 150:
        return None
    
    rcts = rct.split('.')
    pros = prod.split('.')
    
    n_rct_w_carbon = sum([1 for rct in rcts if count_carbon_atoms(rct) > 0])
    n_prod_w_carbon = sum([1 for pro in pros if count_carbon_atoms(pro) > 0])
    
    if n_rct_w_carbon > 3 or n_prod_w_carbon > 3:
        return None
    elif n_rct_w_carbon == 0 or n_prod_w_carbon == 0:
        return None
    else:
        return rxn


def cano_rxn(rxn, exchange_pos=False, remain_isomer=True):
    if not rxn:
        return None
    data = rxn.split('>')
    reactants = data[0].split('.')
    reactants = [cano_smiles(each, remain_isomer) for each in reactants]
    products = data[-1].split('.')
    products = [cano_smiles(each, remain_isomer) for each in products]
    reactants = sorted(reactants)
    products = sorted(products)
    
    if not products or not reactants:
        return None
    
    if exchange_pos:
        new_rxn = f"{'.'.join(products)}>>{'.'.join(reactants)}"
    else:
        new_rxn = f"{'.'.join(reactants)}>>{'.'.join(products)}"
    return new_rxn


def main(data_dir, uid2seq, rhea2template, save_dir):
    if not save_dir:
        save_dir = data_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    df_rhea2ec = pd.read_csv(f'{data_dir}/rhea2ec.tsv', sep='\t')
    df_rhea2uid = pd.read_csv(f'{data_dir}/rhea2uniprot_sprot.tsv', sep='\t')
    df_direction = pd.read_csv(f'{data_dir}/rhea-directions.tsv', sep='\t')
    df_rhea2smiles = pd.read_csv(f'{data_dir}/rhea-reaction-smiles.tsv', sep='\t', names=['RHEA_ID', 'SMILES'])
    
    rhea2smiles = dict(zip(df_rhea2smiles['RHEA_ID'], df_rhea2smiles['SMILES']))
    cnt_polymer = 0
    for id_master, id_lr, id_rl, id_bi in df_direction[['RHEA_ID_MASTER', 'RHEA_ID_LR', 'RHEA_ID_RL', 'RHEA_ID_BI']].values:
        rxn_smiles_lr = rhea2smiles.get(id_lr)
        if not rxn_smiles_lr:
            cnt_polymer += 1
            continue
        rhea2smiles[id_master] = rxn_smiles_lr
        rhea2smiles[id_bi] = rxn_smiles_lr
    
    print(f'Skip {cnt_polymer} polymer reactions without SMILES.\n')
    
    rhea2ec = dict(zip(df_rhea2ec['MASTER_ID'], df_rhea2ec['ID']))
    cnt_wo_ec = 0
    for id_master, id_lr, id_rl, id_bi in df_direction[['RHEA_ID_MASTER', 'RHEA_ID_LR', 'RHEA_ID_RL', 'RHEA_ID_BI']].values:
        ec = rhea2ec.get(id_master)
        if not ec:
            cnt_wo_ec += 1
            continue
        rhea2ec[id_lr] = ec
        rhea2ec[id_rl] = ec
        rhea2ec[id_bi] = ec
    
    df_rhea_full = df_rhea2uid.copy()
    df_rhea_full['SMILES'] = df_rhea_full['RHEA_ID'].apply(lambda x: rhea2smiles.get(x))
    df_rhea_full['EC number'] = df_rhea_full['RHEA_ID'].apply(lambda x: rhea2ec.get(x))
    df_rhea_full = df_rhea_full[df_rhea_full['SMILES'].notna()]
    
    rxns = df_rhea_full['SMILES'].unique()
    rxn2cano = {}
    for rxn in tqdm(rxns, desc='Canonicalizing reactions...'):
        try:
            rxn_ = remove_redundent_from_rxn(rxn)
            rxn_ = filter_unwanted_rxn(rxn_)
            rxn_ = neutralize_rxn(rxn_)
            rxn2cano[rxn] = cano_rxn(rxn_)
        except Exception as e:
            print(e)
            rxn2cano[rxn] = None
    
    df_rhea_full = df_rhea_full.rename(columns={'ID': 'UniprotID'})
    df_rhea_full['CANO_RXN_SMILES'] = df_rhea_full['SMILES'].map(rxn2cano)
    df_rhea_full['sequence'] = df_rhea_full['UniprotID'].map(uid2seq)
    df_rhea_full['reverse_template'] = df_rhea_full['RHEA_ID'].map(rhea2template)
    
    df_rhea_full['n_seq'] = df_rhea_full['sequence'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df_rhea_full = df_rhea_full[df_rhea_full['n_seq'] <= 1000]
    
    df_process_error = df_rhea_full[df_rhea_full['CANO_RXN_SMILES'].isna()].drop_duplicates('SMILES')
    df_process_error = df_process_error[['RHEA_ID', 'SMILES']]
    
    df_no_seq = df_rhea_full[df_rhea_full['sequence'].isna()].drop_duplicates('UniprotID')
    df_no_seq = df_no_seq[['UniprotID', 'RHEA_ID', 'SMILES']]
    
    df_rhea_full = df_rhea_full[df_rhea_full['CANO_RXN_SMILES'].notna() & df_rhea_full['sequence'].notna()].reset_index(drop=True)
    
    save_path = os.path.join(save_dir, 'rhea_rxn2uids.csv')
    df_rhea_full.to_csv(save_path, index=False)
    print(f'\nSaved to {save_path}\n')
    
    df_process_error.to_csv(os.path.join(save_dir, 'rhea_rxn_filtered.csv'), index=False)
    df_no_seq.to_csv(os.path.join(save_dir, 'rhea_no-seq_filtered.csv'), index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_dir", type=str, help="Path to Rhea raw tsv data folder")
    parser.add_argument("--template_path", type=str, help="Path to the template file, map RHEA ID to reaction template")
    parser.add_argument("--sequence_path", type=str, help="Path to the sequence file, map Uniprot ID to sequence")
    parser.add_argument("--save_dir", type=str, help="Path to save the cleaned data")
    args = parser.parse_args()
    
    rhea2template = {}
    if os.path.exists(args.template_path):
        rhea2template = pkl.load(open(args.template_path, 'rb'))
    
    uid2seq = {}
    if os.path.exists(args.sequence_path):
        uid2seq = pkl.load(open(args.sequence_path, 'rb'))
    
    main(args.tsv_dir, uid2seq, rhea2template, args.save_dir)
    
