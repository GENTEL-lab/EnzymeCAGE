import os
import argparse
import pickle as pkl

import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader

from config import Config
from enzymecage.model import EnzymeCAGE
from enzymecage.baseline import Baseline
from enzymecage.dataset.geometric import load_geometric_dataset
from enzymecage.dataset.baseline import load_baseline_dataset
from enzymecage.base import UID_COL, RXN_COL
from utils import seed_everything, check_files


def preprocess_infer_data(data_path):
    df_data = pd.read_csv(data_path)
    if UID_COL not in df_data.columns and 'enzyme' in df_data.columns:
        df_data[UID_COL] = df_data['enzyme']
    if RXN_COL not in df_data.columns and 'reaction' in df_data.columns:
        df_data[RXN_COL] = df_data['reaction']
    if 'Label' not in df_data.columns:
        df_data['Label'] = 0
    
    df_data.to_csv(data_path, index=False)


def reuse_identical_sequence_structure_features(df_data, protein_gvp_feat, esm_node_feature):
    seq_df = df_data[[UID_COL, 'sequence']].drop_duplicates()
    seq_to_source_uid = {}
    for _, row in seq_df.iterrows():
        uid = row[UID_COL]
        seq = row['sequence']
        if uid not in protein_gvp_feat or uid not in esm_node_feature:
            continue
        if len(protein_gvp_feat[uid][0]) != len(esm_node_feature[uid]):
            continue
        if seq not in seq_to_source_uid:
            seq_to_source_uid[seq] = uid

    reused_pairs = []
    for _, row in seq_df.iterrows():
        uid = row[UID_COL]
        seq = row['sequence']
        source_uid = seq_to_source_uid.get(seq)
        if source_uid is None or source_uid == uid:
            continue

        needs_reuse = uid not in protein_gvp_feat or uid not in esm_node_feature
        if not needs_reuse and uid in protein_gvp_feat and uid in esm_node_feature:
            needs_reuse = len(protein_gvp_feat[uid][0]) != len(esm_node_feature[uid])

        if needs_reuse:
            protein_gvp_feat[uid] = protein_gvp_feat[source_uid]
            esm_node_feature[uid] = esm_node_feature[source_uid]
            reused_pairs.append((uid, source_uid))

    return reused_pairs


def inference(model_conf):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}")

    df_data = pd.read_csv(model_conf.data_path)
    if UID_COL not in df_data.columns and 'enzyme' in df_data.columns:
        df_data[UID_COL] = df_data['enzyme']
    if RXN_COL not in df_data.columns and 'reaction' in df_data.columns:
        df_data[RXN_COL] = df_data['reaction']
    if 'Label' not in df_data.columns:
        df_data['Label'] = 0
    
    if model_conf.model == 'EnzymeCAGE':
        
        follow_batch = ['protein', 'reaction_feature', 'esm_feature', 'substrates', 'products']
        esm_model = None
        if hasattr(model_conf, 'esm_model'):
            esm_model = model_conf.esm_model
        model = EnzymeCAGE(
            use_esm=model_conf.use_esm,
            use_structure=model_conf.use_structure,
            use_drfp=model_conf.use_drfp,
            use_prods_info=model_conf.use_prods_info,
            esm_model=esm_model,
            interaction_method=model_conf.interaction_method,
            rxn_inner_interaction=model_conf.rxn_inner_interaction,
            pocket_inner_interaction=model_conf.pocket_inner_interaction,
            device=device
        )
                
        protein_gvp_feat = torch.load(model_conf.protein_gvp_feat, weights_only=False)
        if hasattr(model_conf, 'protein_gvp_feat_extra'):
            protein_gvp_feat_extra = torch.load(model_conf.protein_gvp_feat_extra, weights_only=False)
            protein_gvp_feat.update(protein_gvp_feat_extra)
            print(f'Loaded {len(protein_gvp_feat_extra)} extra GVP features.')

        esm_node_feature = torch.load(model_conf.esm_node_feature, weights_only=False)
        if hasattr(model_conf, 'esm_node_feature_extra'):
            esm_node_feature_extra = torch.load(model_conf.esm_node_feature_extra, weights_only=False)
            esm_node_feature.update(esm_node_feature_extra)
            print(f'Loaded {len(esm_node_feature_extra)} extra pocket ESM features.')

        esm_mean_feature = model_conf.esm_mean_feature
        if hasattr(model_conf, 'esm_mean_feature_extra'):
            esm_mean_feature = pkl.load(open(model_conf.esm_mean_feature, 'rb'))
            esm_mean_feature_extra = pkl.load(open(model_conf.esm_mean_feature_extra, 'rb'))
            esm_mean_feature.update(esm_mean_feature_extra)
            print(f'Loaded {len(esm_mean_feature_extra)} extra protein-level ESM features.')

        reused_pairs = reuse_identical_sequence_structure_features(df_data, protein_gvp_feat, esm_node_feature)
        if reused_pairs:
            print(f'Reused structure features for {len(reused_pairs)} UIDs based on identical sequences.')
        print('Loading protein dict...')
        valid_uids = set(protein_gvp_feat.keys()) & set(esm_node_feature.keys())
        df_data = df_data[df_data[UID_COL].isin(valid_uids)].reset_index(drop=True)

        infer_dataset = load_geometric_dataset(
            df_data,
            protein_gvp_feat,
            model_conf.rxn_fp,
            model_conf.mol_conformation,
            esm_node_feature,
            esm_mean_feature,
            model_conf.reaction_center,
        )

    elif model_conf.model == 'baseline':
        follow_batch = ['reaction_feature', 'esm_feature']
        esm_model = None
        if hasattr(model_conf, 'esm_model'):
            esm_model = model_conf.esm_model
        model = Baseline(device=device, esm_model=esm_model)        
        infer_dataset = load_baseline_dataset(df_data, model_conf.rxn_fp, model_conf.esm_mean_feature)

    print(f'len(infer_dataset): {len(infer_dataset)}')
    test_loader = DataLoader(infer_dataset, batch_size=model_conf.batch_size, shuffle=False, follow_batch=follow_batch)

    model_name_list = model_conf.model_list

    assert os.path.exists(model_conf.ckpt_dir), f'Model checkpoints dir not found: {model_conf.ckpt_dir}'
    
    for model_name in tqdm(model_name_list):
        ckpt_path = os.path.join(model_conf.ckpt_dir, model_name)
        if not os.path.exists(ckpt_path):
            print(f'Checkpoint file not found: {ckpt_path}')
            continue
        best_state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best_state_dict)
        model.to(device)
        
        filename = os.path.basename(model_conf.data_path).split('.')[0] + '_' + model_name.replace('.pth', '.csv')
        result_dir = model_conf.ckpt_dir
        if hasattr(model_conf, 'result_dir') and model_conf.result_dir:
            result_dir = model_conf.result_dir
        os.makedirs(result_dir, exist_ok=True)
        save_path = os.path.join(result_dir, filename)

        print('Start inference...')
        model.eval()
        preds, _ = model.evaluate(test_loader, show_progress=True)
        df_data['pred'] = preds.cpu()
        df_data.to_csv(save_path, index=False)
        print('Save pred result to: ', save_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    assert os.path.exists(args.config)
    model_conf = Config(args.config)
    
    seed = 42 if not hasattr(model_conf, 'seed') else model_conf.seed
    seed_everything(seed)
    
    check_files(model_conf)
    inference(model_conf)
