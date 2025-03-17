import os
import json
import argparse
import yaml

from tqdm import tqdm
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)

from config import Config
from enzymecage.model import EnzymeCAGE
from enzymecage.dataset.geometric import create_geometric_dataset
from enzymecage.dataset.baseline import create_baseline_dataset
from enzymecage.baseline import Baseline
from utils import seed_everything, check_files

TARGET_BATCH_SIZE = 256

WEIGHT_PRINT_FLAG = False


def get_accumulation_steps(batch_size):
    if batch_size >= TARGET_BATCH_SIZE // 2:
        accumulation_steps = 1
    else:
        accumulation_steps = TARGET_BATCH_SIZE // batch_size
    return accumulation_steps


def backup_config(config_path, save_dir=None):
    with open(config_path, "r", encoding="utf-8") as fp:
        conf_dict = yaml.load(fp, Loader=yaml.FullLoader)
        
    ckpt_dir = conf_dict["ckpt_dir"] if save_dir is None else save_dir
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        
    save_path = os.path.join(ckpt_dir, "config.yaml")
    with open(save_path, "w", encoding="utf-8") as fp:
        yaml.dump(conf_dict, fp)


def calculate_loss(loss_func, pred, target, batch_data):
        if hasattr(batch_data, 'weight'):
            weight = batch_data.weight
        else:
            weight = torch.ones_like(target)
        
        global WEIGHT_PRINT_FLAG
        if not WEIGHT_PRINT_FLAG:
            WEIGHT_PRINT_FLAG = True
            print(f'weights: {weight}')
            
        weighted_loss = loss_func(pred, target) * weight
        return torch.mean(weighted_loss)


def main(model_conf):
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
        
        if hasattr(model_conf, 'pretrain_model') and model_conf.pretrain_model:
            if os.path.exists(model_conf.pretrain_model):
                pretrain_model = torch.load(model_conf.pretrain_model, map_location=device)
                model.load_state_dict(pretrain_model)
                print(f'Load pretrained model from {model_conf.pretrain_model}')
            else:
                raise FileNotFoundError(f'Pretrain model not found: {model.pretrain_model}')
        
        print('Model save dir: ', model_conf.ckpt_dir)

        weight_col = model_conf.weight_col if hasattr(model_conf, 'weight_col') else None
        print(f'Sample weight column: {weight_col}')
        
        train_set, valid_set, test_set = create_geometric_dataset(train_path=model_conf.train_path,
                                                                  valid_path=model_conf.valid_path,
                                                                  test_path=model_conf.test_path,
                                                                  protein_gvp_feat=model_conf.protein_gvp_feat, 
                                                                  rxn_fp_path=model_conf.rxn_fp, 
                                                                  mol_sdf_dir=model_conf.mol_conformation, 
                                                                  esm_node_feature_path=model_conf.esm_node_feature, 
                                                                  esm_mean_feature_path=model_conf.esm_mean_feature, 
                                                                  reaction_center_path=model_conf.reaction_center,
                                                                  weight_col=weight_col)
    
    elif model_conf.model == 'baseline':        
        follow_batch = ['reaction_feature', 'esm_feature']
        model = Baseline(device=device)
        print('Model save dir: ', model_conf.ckpt_dir)
        train_set, valid_set, test_set = create_baseline_dataset(train_path=model_conf.train_path,
                                                                 valid_path=model_conf.valid_path,
                                                                 test_path=model_conf.test_path,
                                                                 rxn_fp_path=model_conf.rxn_fp,
                                                                 esm_mean_feature_path=model_conf.esm_mean_feature)
        
    else:
        raise ValueError('model type is not supported: {model_conf.model}')

    
    test_loader = DataLoader(test_set, batch_size=model_conf.batch_size, shuffle=False, follow_batch=follow_batch)
    valid_loader = DataLoader(valid_set, batch_size=model_conf.batch_size, shuffle=False, follow_batch=follow_batch)
    train_loader = DataLoader(train_set, batch_size=model_conf.batch_size, shuffle=True, follow_batch=follow_batch, drop_last=True)
    
    model.to(device)

    lr_init = model_conf.lr_init
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
    loss_func = nn.BCEWithLogitsLoss(reduction='none')

    os.makedirs(model_conf.ckpt_dir, exist_ok=True)
        
    best_metric = 0
    for epoch in range(model_conf.num_epochs):
        print(f'================= Epoch {epoch} =================')
        loss_total = 0
        model.train()
        target_list = []
        pred_list = []
        all_rxns = []
        
        for _, batch in enumerate(tqdm(train_loader)):
            batch.epoch = epoch
            target = batch.y.to(device)
            batch.to(device)

            pred = model(batch)
            # loss = loss_func(pred, target)
            loss = calculate_loss(loss_func, pred, target, batch)
            
            loss.backward()            
            optimizer.step()
            optimizer.zero_grad()
            
            loss_total += loss.item()
            target_list.append(target.detach())
            pred_list.append(pred.detach())
            all_rxns.extend(batch.rxn)
        
        torch.save(model.state_dict(), os.path.join(model_conf.ckpt_dir, "epoch_%d.pth" % epoch))

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        n_step = len(train_loader)
        print(f"binary loss: {round(loss_total / n_step, 6)}, lr: {round(lr, 6)}\n")
        
        target_list = torch.concat(target_list)
        pred_list = torch.concat(pred_list)
        
        if not model.sigmoid_readout:
            pred_list = torch.sigmoid(pred_list)
                        
        train_metric = model.calc_metric(pred_list, target_list, all_rxns)
        for k, v in train_metric.items():
            print(f'Train {k}: {round(v, 4)}')
        print()

        model.eval()
        _, valid_metric = model.evaluate(valid_loader)
        for k, v in valid_metric.items():
            print(f'validation {k}: {round(v, 4)}')
        
        print()
        _, test_metric = model.evaluate(test_loader)
        for k, v in test_metric.items():
            print(f'test {k}: {round(v, 4)}')
        
        early_stop_metric = valid_metric['AUC']

        if early_stop_metric > best_metric:
            best_metric = early_stop_metric
            best_model_path = os.path.join(model_conf.ckpt_dir, "best_model.pth")
            valid_metric_path = os.path.join(model_conf.ckpt_dir, "best_model_metrics_valid.json")
            test_metric_path = os.path.join(model_conf.ckpt_dir, "best_model_metrics_test.json")
            torch.save(model.state_dict(), best_model_path)
            print(f'Save best model to {best_model_path}')
            with open(valid_metric_path, 'w') as f:
                json.dump(valid_metric, f)
            with open(test_metric_path, 'w') as f:
                json.dump(test_metric, f)
        
        if scheduler:
            scheduler.step()

        print()
    
    best_state_dict = torch.load(os.path.join(model_conf.ckpt_dir, "best_model.pth"))
    model.load_state_dict(best_state_dict)
    
    model.eval()
    print('\n================= Evaluate on Test =================')
    test_preds, metric = model.evaluate(test_loader)
    for k, v in metric.items():
        print(f'test {k}: {round(v, 4)}')
        
    df_test = test_set.df_data
    df_test['pred'] = test_preds.cpu()
    df_test.to_csv(os.path.join(model_conf.ckpt_dir, f"test_result.csv"), index=False)
    
    print('\n================= Evaluate on Valid again =================')
    valid_preds, metric = model.evaluate(valid_loader)
    for k, v in metric.items():
        print(f'valid {k}: {round(v, 4)}')
        
    df_valid = valid_set.df_data
    df_valid['pred'] = valid_preds.cpu()
    df_valid.to_csv(os.path.join(model_conf.ckpt_dir, f"valid_result.csv"), index=False)
    
    print()
    print(df_valid.loc[0, 'pred'])
    print()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    assert os.path.exists(args.config)
    model_conf = Config(args.config)
    check_files(model_conf)
    
    seed = 42 if not hasattr(model_conf, 'seed') else model_conf.seed
    seed_everything(seed)
    backup_config(args.config)
    main(model_conf)
    