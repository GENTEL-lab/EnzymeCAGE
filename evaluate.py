import os
import argparse

import pandas as pd
import numpy as np

from retrieve import init_mapping_info
from enzymecage.base import UID_COL


def eval_top_rank_result(df_test_inference: pd.DataFrame, test_rxns: list, pred_col='pred', true_enz_dict=None, top_percent=None, to_print=True):
    """Evaluate the ranking result. 

    Args:
        df_test_inference (pd.DataFrame): testing data with prediction results.
    """
    rxn_col = 'CANO_RXN_SMILES' if 'CANO_RXN_SMILES' in df_test_inference.columns else 'reaction'
    enz_col = 'enzyme' if 'enzyme' in df_test_inference.columns else UID_COL

    inference_result = {}
    for rxn, df in df_test_inference.groupby(rxn_col):
        enzymes = df[enz_col].values
        preds = df[pred_col].values
        result = list(zip(enzymes, preds))
        result = sorted(result, key=lambda x: x[1], reverse=True)
        inference_result[rxn] = result
    
    best_rank_list = []
    for rxn, enzymes_ranked in inference_result.items():
        enzymes_ranked = list(map(lambda x: x[0], enzymes_ranked))
        true_enzymes = list(true_enz_dict.get(rxn))
        hit_ranks = []
        for enz in true_enzymes:
            if enz in enzymes_ranked:
                rank = enzymes_ranked.index(enz) + 1
                hit_ranks.append(rank)
            else:
                hit_ranks.append(-1)
    
        if max(hit_ranks) == -1:
            best_rank_list.append(-1)
        else:
            best_rank_list.append(min(filter(lambda x: x>0, hit_ranks)))
    
    eval_result = {}
    topk_list = [1, 3, 5, 10]
    for topk in topk_list:
        success_list = list(filter(lambda x: 0<x<=topk, best_rank_list))
        success_rate = len(success_list) / len(test_rxns)
        if to_print:
            print(f"=========== Top-{topk} success rate: {round(success_rate, 4)} ===========")
        eval_result[f'top{topk}'] = success_rate

    top_percent_result = {}
    if top_percent:
        if isinstance(top_percent, float):
            top_percent = [top_percent]

        for percent in top_percent:
            success_cnt = 0
            for rxn, enzymes_ranked in inference_result.items():
                topk = max(1, int(len(enzymes_ranked) * percent))
                enzymes_ranked = list(map(lambda x: x[0], enzymes_ranked))[:topk]
                true_enzymes = true_enz_dict.get(rxn, set())
                if len(true_enzymes & set(enzymes_ranked)) > 0:
                    success_cnt += 1
            success_rate = success_cnt / len(test_rxns)
            top_percent_result[f'Top {percent*100}%'] = success_rate
            if to_print:
                print(f'=========== Top Percent {percent*100}% SR: {round(success_rate, 4)}')

    if top_percent:
        return eval_result, top_percent_result
    else:
        return eval_result
    
    
def compute_dcg(rels, k=None):
    if k is None:
        k = len(rels)
    else:
        k = min(k, len(rels))
    
    # Calculate DCG
    dcg = 0
    for i in range(k):
        dcg += (2**rels[i] - 1) / np.log2(i + 2)
    
    return dcg


def calc_all_dcg(df_infer, k=None, rank_col='pred', label_col='Label'):
    dcg_list = []
    rxn_col = 'reaction' if 'reaction' in df_infer.columns else 'CANO_RXN_SMILES'
    for _, df in df_infer.groupby(rxn_col):
        df = df.sort_values(rank_col, ascending=False)
        rels = df[label_col].values
        dcg = compute_dcg(rels, k)
        dcg_list.append(dcg)
    return dcg_list


def calculate_enrichment_factor(df, active_column, score_column, topk=None, top_percent=None):
    sorted_df = df.sort_values(by=score_column, ascending=False)

    if not topk and top_percent:
        topk = max(int(top_percent * len(sorted_df)), 5)
    elif not top_percent and topk:
        top_percent = topk / len(sorted_df)
    else:
        assert False
    
    top_df = sorted_df.head(topk)    
    num_active_top = top_df[active_column].sum()    
    total_active = df[active_column].sum()
    if total_active == 0:
        return 0
    
    random_active_top = total_active * top_percent
    enrichment_factor = num_active_top / random_active_top
    
    return enrichment_factor


def cal_all_ef(df_score, label_col='Label', score_col='pred', topk=None, top_percent=None):
    rxn_col = 'reaction' if 'reaction' in df_score.columns else 'CANO_RXN_SMILES'
    ef_list = []
    for _, df in df_score.groupby(rxn_col):
        ef = calculate_enrichment_factor(df, label_col, score_col, topk, top_percent)
        ef_list.append(ef)
    return ef_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, required=True)
    parser.add_argument('--pos_pair_db_path', type=str, default=None)
    parser.add_argument('--pred_col', type=str, default='pred')
    args = parser.parse_args()
    
    if not args.pos_pair_db_path:
        args.pos_pair_db_path = args.result_path
        
    assert os.path.exists(args.result_path), f'result path not exists: {args.result_path}'
    assert os.path.exists(args.pos_pair_db_path), f'positive pair db path not exists: {args.pos_pair_db_path}'
    df_pred = pd.read_csv(args.result_path)
    df_pos_pairs = pd.read_csv(args.pos_pair_db_path)
    
    _, rxn_to_uid = init_mapping_info(df_pos_pairs)
    
    test_rxns = df_pred['CANO_RXN_SMILES'].unique()
    dcg_list = calc_all_dcg(df_pred, k=10)
    ef1_list = cal_all_ef(df_pred, top_percent=0.01)
    ef2_list = cal_all_ef(df_pred, top_percent=0.02)
    dcg, ef1, ef2 = np.mean(dcg_list), np.mean(ef1_list), np.mean(ef2_list)
    
    sr_dict = eval_top_rank_result(df_pred, test_rxns, true_enz_dict=rxn_to_uid, to_print=False, pred_col=args.pred_col)
    
    print('\n########### Evaluation Results ###########')
    print(f'Top-10 DCG: {dcg:.4f}')
    print(f'Top-1% EF : {ef1:.4f}')
    print(f'Top-2% EF : {ef2:.4f}')
    print(f'Top-1  SR : {sr_dict["top1"]*100:.2f}%')
    print(f'Top-3  SR : {sr_dict["top3"]*100:.2f}%')
    print(f'Top-5  SR : {sr_dict["top5"]*100:.2f}%')
    print(f'Top-10 SR : {sr_dict["top10"]*100:.2f}%')
    print('###########################################\n')


if __name__ == '__main__':
    main()
