
import os
import sys
import argparse
import pickle as pkl
from collections import defaultdict
sys.path.append('../')

import pandas as pd
import numpy as np
from tqdm import tqdm

from enzymecage.rxn_similarity import calc_rxn_simi_matrix
from evaluate import eval_top_rank_result


df_rhea_v2502 = pd.read_csv('../dataset/RHEA/2023-07-12/rhea_rxn2uids.csv')
UNIPROT_TO_RXNS = defaultdict(set)
for rxn, uid in df_rhea_v2502[['CANO_RXN_SMILES', 'UniprotID']].values:
    UNIPROT_TO_RXNS[uid].add(rxn)


def save_to_fasta(save_path, name_list, seq_list):
    contents = []
    assert len(name_list) == len(seq_list)
    for name, seq in zip(name_list, seq_list):
        contents.append('>' + name)
        contents.append(seq)
    with open(save_path, 'w') as f:
        for line in contents:
            f.write(line + '\n')
            

def run_mmseqs(test_data_path, train_enzyme_fasta_path):
    mmseqs_folder = os.path.join(os.path.dirname(test_data_path), 'analysis/seq_similarity_to_train')
    os.makedirs(mmseqs_folder, exist_ok=True)
    
    df_test = pd.read_csv(test_data_path)
    df_uid = df_test.drop_duplicates('UniprotID')
    uids, seqs = df_uid['UniprotID'].values, df_uid['sequence'].values
    
    test_enzyme_fasta_path = os.path.join(mmseqs_folder, 'test_enzymes.fasta')
    save_to_fasta(test_enzyme_fasta_path, uids, seqs)

    search_result_path = os.path.join(mmseqs_folder, 'alnRes.m8')
    tmp_dir = os.path.join(mmseqs_folder, 'tmp')
    run_command = f'mmseqs easy-search {test_enzyme_fasta_path} {train_enzyme_fasta_path} {search_result_path} {tmp_dir}'
    print(f'Running command: {run_command}')
    os.system(run_command)
    
    return search_result_path
    

def prepare_for_corr_score(test_data_path, mmseqs_result_path):
    df_seq_idt = pd.read_csv(mmseqs_result_path, sep='\t', names=['query', 'target', 'similarity'] + [i for i in range(9)])
    df_seq_idt = df_seq_idt[['query', 'target', 'similarity']]
    save_path = mmseqs_result_path.replace('.m8', '.csv')
    df_seq_idt.to_csv(save_path, index=False)

    seq_idt_map = {}
    for q, t, s in df_seq_idt.values:
        seq_idt_map[f'{q}_{t}'] = s
        seq_idt_map[f'{t}_{q}'] = s
    
    uid_to_similar = {}
    for uid, df in df_seq_idt.groupby('query'):
        uid_to_similar[f'{uid}'] = set(df['target'])

    homologous_proteins = set(df_seq_idt['target'])
    related_rxns = set()
    for uid in homologous_proteins:
        rxns = UNIPROT_TO_RXNS.get(uid)
        if not rxns:
            print(f'pass {uid}')
            continue
        related_rxns.update(rxns)

    df_test = pd.read_csv(test_data_path)
    query_rxns = set(df_test['CANO_RXN_SMILES'])
    rxn_simi_map = calc_rxn_simi_matrix(query_rxns, related_rxns)

    return uid_to_similar, seq_idt_map, rxn_simi_map


def calc_correlation_score(df_data, uid_to_similar, seq_idt_map, rxn_simi_map):
    score_list = []
    for rxn, uid in tqdm(df_data[['CANO_RXN_SMILES', 'UniprotID']].values):
        similar_uids = uid_to_similar.get(uid, [])
        if not similar_uids:
            score = 0
        else:
            s_list = [0]
            for u in similar_uids:
                rxns = UNIPROT_TO_RXNS.get(u)
                if not rxns:
                    continue
                seq_idt = seq_idt_map.get(f'{uid}_{u}', 0.1)
                rxn_s_list = []
                for r in rxns:
                    s = rxn_simi_map[rxn][r]
                    rxn_s_list.append(s)
                s_list.append(np.mean(rxn_s_list) * seq_idt)
            score = max(s_list)
        score_list.append(score)
    df_data['corr_score'] = score_list
    return df_data


def init_pos_pairs(df_test):
    rxn2uid = {}
    df_pos = df_test[df_test['Label'] == 1]
    for rxn, df in df_pos.groupby('CANO_RXN_SMILES'):
        rxn2uid[rxn] = set(df['UniprotID'])
    return rxn2uid


def evaluate_external_test(pred_result_path, test_data_path, train_fasta_path):
    df_test = pd.read_csv(test_data_path)
    corr_score_path = os.path.join(os.path.dirname(test_data_path), 'corr_score_map.pkl')
    if not os.path.exists(corr_score_path):
        # Search homologs for protein in the test dataset
        mmseqs_result_path = run_mmseqs(test_data_path, train_fasta_path)
        
        # Prepare some necessary data mappings
        uid_to_similar, seq_idt_map, rxn_simi_map = prepare_for_corr_score(test_data_path, mmseqs_result_path)
        
        # Add a correlation score list to test dataframe to get the candidates
        df_test = calc_correlation_score(df_test, uid_to_similar, seq_idt_map, rxn_simi_map)
        
        df_test['key'] = df_test['CANO_RXN_SMILES']  + '_' + df_test['UniprotID']
        corr_score_map = dict(zip(df_test['key'], df_test['corr_score']))
        
        with open(corr_score_path, 'wb') as f:
            pkl.dump(corr_score_map, f)
    else:
        corr_score_map = pkl.load(open(corr_score_path, 'rb'))
    
    # Load prediction result
    df_pred = pd.read_csv(pred_result_path)
    df_pred['key'] = df_pred['CANO_RXN_SMILES'] + '_' + df_pred['UniprotID']
    df_pred['corr_score'] = df_pred['key'].map(corr_score_map)
    df_list = []
    for _, df in df_pred.groupby('CANO_RXN_SMILES'):
        df = df.sort_values('corr_score', ascending=False).reset_index(drop=True)
        # Only keep those with correlation score greater that median as candidate enzymes
        df.loc[df[len(df)//2:].index, 'pred'] = 0
        df_list.append(df)
    df_pred = pd.concat(df_list)
    
    rxn2uid = init_pos_pairs(df_test)
    top_percents = [0.01, 0.03, 0.05]
    test_rxns = set(df_pred['CANO_RXN_SMILES'])
    _, sr_dict = eval_top_rank_result(df_pred, test_rxns, true_enz_dict=rxn2uid, to_print=False, top_percent=top_percents)
    
    print('\n' + '#' * 10, 'Evaluation Result', '#' * 10)
    for percent in top_percents:
        key = f'Top {percent*100}%'
        result = sr_dict[key]
        print(f'{key} Success Rate: {round(result, 3)}')
    print('#' * 40)


def load_ensemble_result(pred_result_path):
    df_pred = pd.read_csv(pred_result_path)
    all_paths = pred_result_path.split('/')
    filename = all_paths[-1]
    seed_dir = all_paths[-2]
    assert seed_dir.startswith('seed_')
    
    df_pred['pred_ensemble'] = 0
    for i in range(40, 45):
        seed_dir = f'seed_{i}'
        path = '/'.join(all_paths[:-2] + [seed_dir, filename])
        df = pd.read_csv(path)
        df_pred[f'pred_ensemble'] += df['pred'].values / 5
    
    return df_pred
    

def evaluate_withanolide(pred_result_path, test_data_path, train_fasta_path):
    df_test = pd.read_csv(test_data_path)
    corr_score_path = os.path.join(os.path.dirname(test_data_path), 'corr_score_map.pkl')
    if not os.path.exists(corr_score_path):
        # Search homologs for protein in the test dataset
        mmseqs_result_path = run_mmseqs(test_data_path, train_fasta_path)
        
        # Prepare some necessary data mappings
        uid_to_similar, seq_idt_map, rxn_simi_map = prepare_for_corr_score(test_data_path, mmseqs_result_path)
        
        # Add a correlation score list to test dataframe to get the candidates
        df_test = calc_correlation_score(df_test, uid_to_similar, seq_idt_map, rxn_simi_map)
        
        df_test['key'] = df_test['CANO_RXN_SMILES']  + '_' + df_test['UniprotID']
        corr_score_map = dict(zip(df_test['key'], df_test['corr_score']))
        
        with open(corr_score_path, 'wb') as f:
            pkl.dump(corr_score_map, f)
    else:
        corr_score_map = pkl.load(open(corr_score_path, 'rb'))
    
    # Load prediction result and ensemble
    df_pred = load_ensemble_result(pred_result_path)
    df_pred['key'] = df_pred['CANO_RXN_SMILES'] + '_' + df_pred['UniprotID']
    df_pred['corr_score'] = df_pred['key'].map(corr_score_map)
    df_pred['pred_mix'] = df_pred['pred_ensemble'] * df_pred['corr_score']
    
    id2name = {
        'enzyme_11': 'CYP87G1',
        'enzyme_84': 'CYP88C7',
        'enzyme_64': 'CYP749B2'
    }
    
    print()
    print('#' * 10, 'Evaluation of withanolide biosynthesis', '#' * 10, '\n')
    for i, (_, df) in enumerate(df_pred.groupby('CANO_RXN_SMILES')):
        df = df.drop_duplicates('sequence')
        df = df.sort_values('pred_mix', ascending=False).reset_index(drop=True)
        df_pos = df[df['Label'] == 1]
        best_rank = df_pos.index[0]
        pos_enzyme = df_pos['UniprotID'].values[0]
        
        print(f'Reaction step: {i+1}, Best positive enzyme rank: {best_rank} ({id2name[pos_enzyme]})')
    
    print()
    print('#' * 60, '\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_result_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--train_fasta_path', type=str, default='../dataset/RHEA/2023-07-12/enzymes.fasta')
    parser.add_argument('--data_type', type=str, default='external-test')
    args = parser.parse_args()
    
    if args.data_type == 'external-test':
        evaluate_external_test(args.pred_result_path, args.test_data_path, args.train_fasta_path)
    elif args.data_type == 'withanolide':
        evaluate_withanolide(args.pred_result_path, args.test_data_path, args.train_fasta_path)
    else:
        raise ValueError(f'Invalid data type: {args.data_type}')
