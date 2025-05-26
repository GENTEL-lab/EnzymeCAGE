import argparse

import pandas as pd


def evaluate(rxns, df_pred, df_db):
    df_positive = df_db[df_db['CANO_RXN_SMILES'].isin(rxns)]
    rxn2uids_glu = {}
    for rxn, df in df_positive.groupby('CANO_RXN_SMILES'):
        rxn2uids_glu[rxn] = set(df['UniprotID'])

    print()
    print('#' * 10, 'Evaluation of glutarate biosynthesis', '#' * 10, '\n')
    for i, rxn in enumerate(rxns):
        df = df_pred[df_pred['CANO_RXN_SMILES'] == rxn]
        df = df.drop_duplicates('UniprotID').sort_values('pred', ascending=False).reset_index(drop=True)
        pos_uids = rxn2uids_glu[rxn]
        df['Label'] = df['UniprotID'].apply(lambda x: 1 if x in pos_uids else 0)
        df_pos = df[df['Label'] == 1]
        print(f'Reaction step: {i+1}, Best positive enzyme rank: {df_pos.index[0] + 1} ({df_pos["UniprotID"].values[0]})')
    print()
    print('#' * 60, '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rxn_path', type=str, default='./dataset/case-study/glutarate/rxns.csv')
    parser.add_argument('--pred_result_path', type=str, default='./checkpoints/pretrain/seed_42/rxns_retrievel_cands_epoch_19.csv')
    parser.add_argument('--db_path', type=str, default='./dataset/RHEA/2023-07-12/rhea_rxn2uids.csv')
    args = parser.parse_args()
    
    df_rxns = pd.read_csv(args.rxn_path)
    glutarate_rxns = df_rxns['CANO_RXN_SMILES'].values
    df_pred = pd.read_csv(args.pred_result_path)
    df_db = pd.read_csv(args.db_path)
    
    evaluate(glutarate_rxns, df_pred, df_db)