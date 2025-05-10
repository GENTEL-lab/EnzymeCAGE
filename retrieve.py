import os
import math
import argparse
import pickle as pkl
from collections import defaultdict
from typing import *

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

from utils import get_rdkit_mol
from enzymecage.base import UID_COL, RXN_COL, SEQ_COL


def init_mapping_info(df_pos_pairs):
    if 'Label' in df_pos_pairs.columns:
        df_pos_pairs = df_pos_pairs[df_pos_pairs['Label'] == 1]
    
    uid_col = UID_COL if UID_COL in df_pos_pairs.columns else 'uniprotID'
    
    uid_to_seq = {}
    rxn_to_uid = defaultdict(set)
    for rxn, uid, seq in df_pos_pairs[[RXN_COL, uid_col, SEQ_COL]].values:
        uid_to_seq[uid] = seq
        rxn_to_uid[rxn].add(uid)
    
    return uid_to_seq, rxn_to_uid


def getRSim(rcts_smi_counter, pros_smi_counter, cand_rcts_molid_counter, cand_pros_molid_counter, sim):
    # rcts_smi_counter, pros_smi_counter, cand_rcts_molid_counter, cand_pros_molid_counter, sim
    info_dict = {'s1': rcts_smi_counter, 'p1': pros_smi_counter, 's2': cand_rcts_molid_counter, 'p2':cand_pros_molid_counter}
    ss = {} 
    simm = {}
    pairs = [('s1','s2'), ('s1', 'p2'), ('p1', 's2'), ('p1', 'p2')]
    compPairs = {}
    for pair_tuple in pairs:
        # pair_tuple -> ('s1','s2')
        pairings = set()
        simm[pair_tuple] = {}
        compPairs[pair_tuple]=[]

        for mol_x in info_dict[pair_tuple[0]].keys():
            simm[pair_tuple][mol_x] = (0.0, mol_x, None)
            if mol_x in sim:
                for mol_y in info_dict[pair_tuple[1]].keys():
                    if mol_y in sim[mol_x]:
                        pairings.add( (sim[mol_x][mol_y], mol_x, mol_y) )

        found = {'left': set(), 'right': set()}
        for v in sorted(pairings, key = lambda h: -h[0]):
            if v[1] not in found['left'] and v[2] not in found['right']:
                # if similarity is greater that zero
                if v[0] > simm[pair_tuple][v[1]][0]:
                    simm[pair_tuple][v[1]] = v
                    found['left'].add(v[1])
                    found['right'].add(v[2])
                    compPairs[pair_tuple].append([v[1], v[2]])
        s = []
        for mol_x in simm[pair_tuple]:
            s.append(simm[pair_tuple][mol_x][0])
        if len(s) > 0:
            ss[pair_tuple] = sum(s)/len(s)
        else:
            ss[pair_tuple] = 0.0
    S1 = math.sqrt(ss[pairs[0]]**2 + ss[pairs[3]]**2)/math.sqrt(2)
    S2 = math.sqrt(ss[pairs[1]]**2 + ss[pairs[2]]**2)/math.sqrt(2)

    return(S1, S2, compPairs)


def neutralize_atoms(smi):
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
    
    return smi_uncharged


def get_morgan_fp(mol): 
    mol = get_rdkit_mol(mol)
    Chem.SanitizeMol(mol)
    info1 = {}
    fpM = AllChem.GetMorganFingerprint(mol, 8, bitInfo=info1, invariants=AllChem.GetConnectivityInvariants(mol, includeRingMembership=False))
    return fpM, Chem.MolToSmiles(mol), mol.GetNumAtoms(), info1


def bulkTani(targetFp, fp, rxn_ids):
    similarity_list = DataStructs.BulkTanimotoSimilarity(targetFp, list(fp))
    dist={}
    for i in sorted(range(0, len(similarity_list))):
        dist[rxn_ids[i]] = similarity_list[i]
    return dist


def get_mol_simi_dict(test_rxns, all_cand_rxns):
    testset_smiles_list = []
    for rxn in test_rxns:
        smiles_list = rxn.split('>>')[0].split('.') + rxn.split('>>')[1].split('.')
        smiles_list = [smi for smi in smiles_list]
        testset_smiles_list.extend(smiles_list)
    test_smiles = list(set(testset_smiles_list))

    cand_smiles_list = []
    for rxn in all_cand_rxns:
        smiles_list = rxn.split('>>')[0].split('.') + rxn.split('>>')[1].split('.')
        cand_smiles_list.extend(smiles_list)
    cand_smiles = list(set(cand_smiles_list))

    cand_mol_ids = [f'MOL_{i}' for i, _ in enumerate(cand_smiles)]
    cand_mol_to_id_dict = dict(zip(cand_smiles, cand_mol_ids))
    cand_fp_list = []
    for smi in tqdm(cand_smiles):
        try:
            smi = neutralize_atoms(smi)
        except:
            print(f'Cannot neutralize SMILES: {smi}, use original format')

        try:
            fp = get_morgan_fp(smi)[0]
            cand_fp_list.append(fp)
        except:
            print(f'Error SMILES: ', smi)
            print(sss)

    cpd_simi_dict = {}
    for smi in tqdm(test_smiles):
        fp_target = get_morgan_fp(neutralize_atoms(smi))[0]
        cpd_simi_dict[smi] = bulkTani(fp_target, cand_fp_list, cand_mol_ids) 
    
    return cpd_simi_dict, cand_mol_to_id_dict


def run_retrieval(df_data, df_db, smiles_col, uid_to_proevi, uid_to_taxdis, topk=10, exclude_rxns=set()):
    """To retrieve candidate enzymes for each reaction in the df_data.

    Args:
        df_data (pd.DataFrame): Contains reactions to retrieve candidates.
        df_db (pd.DataFrame): Positive enzyme-reaction pairs(with about 320k pairs).
        smiles_col (str): Column name of reaction SMILES in df_data and df_db.
        uid_to_proevi (dict): Uniprot ID to protein evidence dictionary.
        uid_to_taxdis (dict): Uniprot ID to taxonomic distance dictionary.
        topk (int, optional): Top k similar reaction to consider. Defaults to 10.

    Returns:
        pd.DataFrame: data with candidate enzymes.
    """
        
    rxns_to_retrieve = {rxn for rxn in set(df_data[smiles_col]) if isinstance(rxn, str)}
    all_cand_rxns = set(df_db[smiles_col]) - exclude_rxns - rxns_to_retrieve
    
    UID_TO_SEQ, RXN_TO_UID = init_mapping_info(df_db)
    cpd_simi_dict, cand_mol_to_id_dict = get_mol_simi_dict(rxns_to_retrieve, all_cand_rxns)

    similar_rxns_map = {}
    for rxn_target in tqdm(rxns_to_retrieve, desc='Searching similar reactions'):
        rcts_smi_counter = Counter(rxn_target.split('>>')[0].split('.'))
        pros_smi_counter = Counter(rxn_target.split('>>')[1].split('.'))
        rxn_simi_list = []
        for cand_rxn in all_cand_rxns:
            if cand_rxn == rxn_target:
                rxn_simi_list.append(1)
                continue
            cand_rcts = [cand_mol_to_id_dict.get(smi) for smi in cand_rxn.split('>>')[0].split('.')]
            cand_pros = [cand_mol_to_id_dict.get(smi) for smi in cand_rxn.split('>>')[1].split('.')]
            cand_rcts_molid_counter, cand_pros_molid_counter = Counter(cand_rcts), Counter(cand_pros)
            S1, S2, _ = getRSim(rcts_smi_counter, pros_smi_counter, cand_rcts_molid_counter, cand_pros_molid_counter, cpd_simi_dict)
            max_simi = max(S1, S2)
            rxn_simi_list.append(max_simi)
        cand_rxn_info = [each for each in list(zip(all_cand_rxns, rxn_simi_list))]
        cand_rxn_info = sorted(cand_rxn_info, key=lambda x: x[1], reverse=True)
        similar_rxns_map[rxn_target] = cand_rxn_info[:topk]

    result_list = []
    for rxn, similar_rxns in tqdm(similar_rxns_map.items(), desc='Calculating scores'):
        df_list = []
        for cand_rxn, rxn_simi in similar_rxns:
            cand_enzs = RXN_TO_UID.get(cand_rxn, [])
            size = len(cand_enzs)
            tax_dis_list = [uid_to_taxdis.get(uid, 30) for uid in cand_enzs]
            pro_evi_list = [uid_to_proevi.get(uid, 5) for uid in cand_enzs]
            df = pd.DataFrame({'reaction': [rxn] * size, 'similar_rxn': [cand_rxn] * size, 'rxn_similarity': [rxn_simi] * size, 'enzyme': list(cand_enzs), 'tax_dis': tax_dis_list, 'pro_evi': pro_evi_list})
            df['Score'] = df['rxn_similarity'] * 100 - df['tax_dis'] - 0.1 * df['pro_evi']
            df_list.append(df)
        df_result = pd.concat(df_list)
        true_enzs = RXN_TO_UID.get(rxn, [])
        df_result['Label'] = df_result['enzyme'].apply(lambda x: 1 if x in true_enzs else 0)
        result_list.append(df_result)
    
    df_final_result = pd.concat(result_list).reset_index(drop=True)
    df_final_result[SEQ_COL] = df_final_result['enzyme'].map(UID_TO_SEQ)
    df_final_result[smiles_col] = df_final_result['reaction']
    df_final_result[UID_COL] = df_final_result['enzyme']
    
    return df_final_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--exclude_rxns', type=str, default='./dataset/orphan-rxns/rxns_exclude_in_retrieval.csv')
    parser.add_argument('--db_path', type=str, default='./dataset/db/enzyme-reaction-pairs.csv')
    parser.add_argument('--smiles_col', type=str, default='CANO_RXN_SMILES', help='column name of reaction SMILES in the data')
    parser.add_argument('--proevi_path', type=str, default='./dataset/others/uid_to_proevi.pkl')
    parser.add_argument('--taxdis_path', type=str, default='./dataset/others/uid_to_taxdis.pkl')
    args = parser.parse_args()
    
    save_path = args.data_path.replace('.csv', '_retrievel_cands.csv')
    
    assert os.path.exists(args.data_path), f'{args.data_path} does not exist!'
    assert os.path.exists(args.db_path), f'{args.db_path} does not exist!'
    
    if args.exclude_rxns and os.path.exists(args.exclude_rxns):
        df_exclude_rxns = pd.read_csv(args.exclude_rxns)
        exclude_rxns = set(df_exclude_rxns[args.smiles_col])
        print(f'Load {len(exclude_rxns)} rxns to exclude')
    else:
        exclude_rxns = set()
    
    df_data = pd.read_csv(args.data_path)
    df_db = pd.read_csv(args.db_path)
    uid_to_proevi = pkl.load(open(args.proevi_path, 'rb'))
    uid_to_taxdis = pkl.load(open(args.taxdis_path, 'rb'))
    
    df_retrievel_cands = run_retrieval(df_data, df_db, args.smiles_col, uid_to_proevi, uid_to_taxdis, exclude_rxns=exclude_rxns)
    df_retrievel_cands.to_csv(save_path, index=False)
    
    print(f'Retrieved candidates saved to: {save_path}')


if __name__ == "__main__":
    main()
