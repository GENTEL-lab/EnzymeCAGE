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
from utils import seed_everything


def preprocess_infer_data(data_path):
    df_data = pd.read_csv(data_path)
    if 'uniprotID' not in df_data.columns and 'enzyme' in df_data.columns:
        df_data['uniprotID'] = df_data['enzyme']
    if 'CANO_RXN_SMILES' not in df_data.columns and 'reaction' in df_data.columns:
        df_data['CANO_RXN_SMILES'] = df_data['reaction']
    if 'Label' not in df_data.columns:
        df_data['Label'] = 0
    
    df_data.to_csv(data_path, index=False)


def inference(model_conf):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_conf.model == 'EnzymeCAGE':
        follow_batch = ['protein', 'reaction_feature', 'esm_feature', 'substrates', 'products']
        
        model = EnzymeCAGE(
            use_esm=model_conf.use_esm,
            use_structure=model_conf.use_structure,
            use_drfp=model_conf.use_drfp,
            use_prods_info=model_conf.use_prods_info,
            interaction_method=model_conf.interaction_method,
            rxn_inner_interaction=model_conf.rxn_inner_interaction,
            device=device
        )
        
        print('Model save dir: ', model_conf.ckpt_dir)
        
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
        model = Baseline(device=device)

        # model = Baseline(device=device)
        model_conf.ckpt_dir = model_conf.ckpt_dir
        print('Model save dir: ', model_conf.ckpt_dir)
        
        infer_dataset = load_baseline_dataset(model_conf.data_path, model_conf.rxn_fp, model_conf.esm_mean_feature)

    
    df_data = pd.read_csv(model_conf.data_path)
    # if 'uniprotID' not in df_data.columns and 'enzyme' in df_data.columns:
    #     df_data['uniprotID'] = df_data['enzyme']
    # if 'CANO_RXN_SMILES' not in df_data.columns and 'reaction' in df_data.columns:
    #     df_data['CANO_RXN_SMILES'] = df_data['reaction']
    # if 'sequence' not in df_data.columns:
    #     df_data['sequence'] = df_data['uniprotID'].map(UNIPROT_TO_SEQ)
    #     df_data = df_data[df_data['sequence'].notna()]

    # 只是为了不报错，实际上不会用到
    if 'Label' not in df_data.columns:
        df_data['Label'] = 0
    
    if model_conf.model == 'EnzymeCAGE':
        print(f'Loading protein dict...')
        df_data = df_data[df_data['uniprotID'].isin(protein_gvp_feat.keys()) & df_data['uniprotID'].isin(esm_node_feature.keys())]

    print(f'len(infer_dataset): {len(infer_dataset)}')
    test_loader = DataLoader(infer_dataset, batch_size=model_conf.batch_size, shuffle=False, follow_batch=follow_batch)
        

    if model_conf.predict_mode == 'only_best':
        model_name_list = ['best_model.pth']
    elif model_conf.predict_mode == 'all':
        model_name_list = [f'epoch_{i}.pth' for i in range(20)]
    else:
        assert False, 'predict_mode must be only_best or all'
    
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
        print(f'preds.shape: {preds.shape}')
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

    inference(model_conf)
