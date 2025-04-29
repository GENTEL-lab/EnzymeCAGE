import os
import argparse

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


def inference(model_conf):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
                
        protein_gvp_feat = torch.load(model_conf.protein_gvp_feat)
        esm_node_feature = torch.load(model_conf.esm_node_feature)
        
        infer_dataset = load_geometric_dataset(model_conf.data_path, 
                                               protein_gvp_feat, 
                                               model_conf.rxn_fp, 
                                               model_conf.mol_conformation, 
                                               esm_node_feature, 
                                               model_conf.esm_mean_feature, 
                                               model_conf.reaction_center)

    
    elif model_conf.model == 'baseline':
        follow_batch = ['reaction_feature', 'esm_feature']
        esm_model = None
        if hasattr(model_conf, 'esm_model'):
            esm_model = model_conf.esm_model
        model = Baseline(device=device, esm_model=esm_model)        
        infer_dataset = load_baseline_dataset(model_conf.data_path, model_conf.rxn_fp, model_conf.esm_mean_feature)

    
    df_data = pd.read_csv(model_conf.data_path)

    if 'Label' not in df_data.columns:
        df_data['Label'] = 0
    
    if model_conf.model == 'EnzymeCAGE':
        print(f'Loading protein dict...')
        df_data = df_data[df_data[UID_COL].isin(protein_gvp_feat.keys()) & df_data[UID_COL].isin(esm_node_feature.keys())]

    print(f'len(infer_dataset): {len(infer_dataset)}')
    test_loader = DataLoader(infer_dataset, batch_size=model_conf.batch_size, shuffle=False, follow_batch=follow_batch)
        

    if model_conf.predict_mode == 'only_best':
        model_name_list = ['best_model.pth']
    elif model_conf.predict_mode == 'all':
        model_name_list = [f'epoch_{i}.pth' for i in range(20)]
    else:
        assert False, 'predict_mode must be only_best or all'
        
    assert os.path.exists(model_conf.ckpt_dir), f'Model checkpoints dir not found: {model_conf.ckpt_dir}'
    
    for model_name in tqdm(model_name_list):
        ckpt_path = os.path.join(model_conf.ckpt_dir, model_name)
        if not os.path.exists(ckpt_path):
            continue
        best_state_dict = torch.load(ckpt_path)
        model.load_state_dict(best_state_dict)
        model.to(device)
        
        filename = os.path.basename(model_conf.data_path).split('.')[0] + '_' + model_name.replace('.pth', '.csv')
        save_path = os.path.join(model_conf.ckpt_dir, filename)

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
