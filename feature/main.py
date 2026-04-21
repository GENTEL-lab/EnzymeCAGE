import argparse
import os
import sys
import pickle as pkl
from multiprocessing import Pool
from functools import partial
from pathlib import Path
sys.path.append('../')
sys.path.append('./pkgs/')

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from Bio import PDB

from enzymecage.base import UID_COL, RXN_COL, SEQ_COL
from utils import check_dir, cano_rxn, tranverse_folder
from gvp_torchdrug_feature import calc_gvp_feature, get_gvp_residue_ids_from_structure, resolve_structure_path
from extract_pocket import get_pocket_info, extract_fix_num_residues, get_pocket_info_batch_new
from extract_reacting_center import extract_reacting_center, calc_aam


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


def ensure_local_esmc_600m_weights():
    import shutil
    import zipfile
    import requests
    import esm.pretrained as esm_pretrained
    import esm.utils.constants.esm3 as esm_constants

    model_root = (Path(__file__).resolve().parents[1] / '.model_cache' / 'esmc-600m-2024-12').resolve()
    weight_path = model_root / 'data' / 'weights' / 'esmc_600m_2024_12_v0.pth'
    partial_path = weight_path.with_suffix(weight_path.suffix + '.part')
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_weight_paths = [weight_path]
    for snapshot_base in [
        Path('/data/liuyong/.cache/huggingface/hub/models--EvolutionaryScale--esmc-600m-2024-12/snapshots'),
        Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--EvolutionaryScale--esmc-600m-2024-12' / 'snapshots',
    ]:
        if snapshot_base.exists():
            candidate_weight_paths.extend(
                sorted(snapshot_base.glob('*/data/weights/esmc_600m_2024_12_v0.pth'))
            )

    resolved_model_root = model_root
    resolved_weight_path = None
    for candidate_weight_path in candidate_weight_paths:
        if candidate_weight_path.exists() and zipfile.is_zipfile(candidate_weight_path):
            resolved_weight_path = candidate_weight_path.resolve()
            resolved_model_root = resolved_weight_path.parents[2]
            print(f'Using existing ESM-C weights from {resolved_weight_path}')
            break

    if resolved_weight_path is None and partial_path.exists() and not zipfile.is_zipfile(partial_path):
        # A previous interrupted aria2 attempt may leave a preallocated but invalid
        # .part file, which would cause HTTP 416 on the next resume attempt.
        if partial_path.exists():
            partial_path.unlink()
        aria2_state = partial_path.with_suffix(partial_path.suffix + '.aria2')
        if aria2_state.exists():
            aria2_state.unlink()

    hf_partial_path = (
        Path.home()
        / '.cache'
        / 'huggingface'
        / 'hub'
        / 'models--EvolutionaryScale--esmc-600m-2024-12'
        / 'blobs'
        / '8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324.incomplete'
    )
    if resolved_weight_path is None and not partial_path.exists() and hf_partial_path.exists():
        shutil.copy2(hf_partial_path, partial_path)
        print(f'Reused partial ESM-C weights from {hf_partial_path}')

    if resolved_weight_path is None:
        url = 'https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12/resolve/main/data/weights/esmc_600m_2024_12_v0.pth'
        resume_size = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {'Range': f'bytes={resume_size}-'} if resume_size > 0 else {}
        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(60, 600),
            allow_redirects=True,
        )
        if response.status_code == 416 and partial_path.exists() and zipfile.is_zipfile(partial_path):
            os.replace(partial_path, weight_path)
            resolved_weight_path = weight_path.resolve()
        elif response.status_code not in (200, 206):
            raise RuntimeError(f'Failed to download ESM-C weights, status code: {response.status_code}')

        if resolved_weight_path is None and response.status_code == 200 and resume_size > 0:
            resume_size = 0

        if resolved_weight_path is None:
            total_size = None
            if response.headers.get('Content-Range') and '/' in response.headers['Content-Range']:
                total_size = int(response.headers['Content-Range'].rsplit('/', 1)[1])
            elif response.headers.get('Content-Length'):
                total_size = resume_size + int(response.headers['Content-Length'])

            mode = 'ab' if response.status_code == 206 and resume_size > 0 else 'wb'
            with open(partial_path, mode) as f, tqdm(
                total=total_size,
                initial=resume_size,
                unit='B',
                unit_scale=True,
                desc='Downloading ESM-C weights',
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

            os.replace(partial_path, weight_path)
            resolved_weight_path = weight_path.resolve()
            print(f'Downloaded ESM-C weights to {resolved_weight_path}')

    original_pretrained_data_root = esm_pretrained.data_root
    original_constants_data_root = esm_constants.data_root

    def _local_data_root(model):
        if model.startswith('esmc-600'):
            return resolved_model_root
        result = original_pretrained_data_root(model)
        if isinstance(result, str):
            return Path(result)
        return result

    esm_pretrained.data_root = _local_data_root
    esm_constants.data_root = _local_data_root
    return resolved_weight_path


def calc_seq_esm_C_feature(data_path, esm_node_feat_dir, esm_mean_feat_path):
    print('\n', '#' * 20, 'Calculating ESM-C feature', '#' * 20, '\n')
    
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    weight_path = ensure_local_esmc_600m_weights()
    print(f'Loading ESM-C model weights from {weight_path}')
    
    # Load ESM-C model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    import esm
    # Load ESM-2 model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def generate_gvp_compatible_pocket_info(data_path, pocket_dir, save_path, fallback_structure_dir=None):
    print('Generating GVP-compatible pocket residues information...')
    df_data = pd.read_csv(data_path)
    uid_list = list(dict.fromkeys(df_data[UID_COL].astype(str).tolist()))
    uid_to_sequence = dict(df_data[[UID_COL, SEQ_COL]].drop_duplicates().values)

    pocket_info_rows = []
    missing_structure_uids = []
    empty_residue_uids = []
    sequence_mismatch_uids = []
    for uid in tqdm(uid_list):
        structure_path = resolve_structure_path(uid, pocket_dir, fallback_structure_dir)
        if structure_path is None:
            missing_structure_uids.append(uid)
            continue

        try:
            residue_ids = get_gvp_residue_ids_from_structure(structure_path, uid_to_sequence.get(uid))
        except ValueError:
            sequence_mismatch_uids.append(uid)
            continue
        if len(residue_ids) == 0:
            empty_residue_uids.append(uid)
            continue

        pocket_info_rows.append({
            UID_COL: uid,
            'pocket_residues': ','.join(residue_ids),
            'structure_path': structure_path,
        })

    df_pocket_info = pd.DataFrame(pocket_info_rows)
    df_pocket_info.to_csv(save_path, index=False)
    print(f'Save GVP-compatible pocket info of {len(df_pocket_info)} proteins to {save_path}\n')

    base_dir = os.path.dirname(save_path)
    if missing_structure_uids:
        missing_path = os.path.join(base_dir, 'missing_structure_uids.csv')
        pd.DataFrame({UID_COL: missing_structure_uids}).to_csv(missing_path, index=False)
        print(f'Missing structure files for {len(missing_structure_uids)} proteins. Saved to {missing_path}')
    if empty_residue_uids:
        empty_path = os.path.join(base_dir, 'empty_gvp_residue_uids.csv')
        pd.DataFrame({UID_COL: empty_residue_uids}).to_csv(empty_path, index=False)
        print(f'No GVP-compatible residues for {len(empty_residue_uids)} proteins. Saved to {empty_path}')
    if sequence_mismatch_uids:
        mismatch_path = os.path.join(base_dir, 'sequence_mismatch_uids.csv')
        pd.DataFrame({UID_COL: sequence_mismatch_uids}).to_csv(mismatch_path, index=False)
        print(f'Structure/data sequence mismatch for {len(sequence_mismatch_uids)} proteins. Saved to {mismatch_path}')

    if len(df_pocket_info) == 0:
        raise ValueError('No usable structures found for feature generation. Please check pocket_dir / structure_dir inputs.')


def get_esm_pocket_feature(pocket_info_path, esm_node_feat_dir, save_path):
    df_pocket_data = pd.read_csv(pocket_info_path)
    uids = [str(each) for each in df_pocket_data[UID_COL].values]
    uid_to_pocket = dict(zip(uids, df_pocket_data['pocket_residues']))
    esm_file_list = [each for each in tranverse_folder(esm_node_feat_dir) if each.endswith('.npz')]
    
    uid_to_pocket_node_feature = {}
    failed_records = []
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
        
        pocket_residue_ids = [int(i) - 1 for i in pocket_residue_ids.split(',')]
        try:
            max_idx = max(pocket_residue_ids)
            if max_idx >= len(esm_node_feature):
                raise IndexError(
                    f'max pocket index {max_idx} >= esm node feature length {len(esm_node_feature)}'
                )
            pocket_node_feature = esm_node_feature[pocket_residue_ids]
        except Exception as e:
            print(e)
            print(uid, ' Error')
            failed_records.append({UID_COL: uid, 'error': str(e)})
            continue
        
        uid_to_pocket_node_feature[uid] = pocket_node_feature
        
    torch.save(uid_to_pocket_node_feature, save_path)
    print(f'Save esm feature of {len(uid_to_pocket_node_feature)} pockets to {save_path}\n')
    if failed_records:
        failed_save_path = os.path.join(os.path.dirname(save_path), 'failed_esm_pocket_uids.csv')
        pd.DataFrame(failed_records).to_csv(failed_save_path, index=False)
        print(f'Failed to calculate pocket ESM features for {len(failed_records)} proteins. Saved to {failed_save_path}')


def check_pocket_feature(gvp_feature_path, esm_feature_path, report_path=None):
    gvp_feature = torch.load(gvp_feature_path, weights_only=False)
    esm_feature = torch.load(esm_feature_path, weights_only=False)

    gvp_uids = set(gvp_feature.keys())
    esm_uids = set(esm_feature.keys())
    report_rows = []

    for uid in sorted(gvp_uids - esm_uids):
        report_rows.append({UID_COL: uid, 'issue': 'missing_esm_pocket_feature'})
    for uid in sorted(esm_uids - gvp_uids):
        report_rows.append({UID_COL: uid, 'issue': 'missing_gvp_feature'})

    for uid in sorted(gvp_uids & esm_uids):
        gvp_node_feature = gvp_feature[uid]
        esm_node_feature = esm_feature[uid]
        n_gvp_nodes = gvp_node_feature[0].shape[0]
        n_esm_nodes = esm_node_feature.shape[0]
        if n_gvp_nodes != n_esm_nodes:
            report_rows.append({
                UID_COL: uid,
                'issue': 'gvp_esm_node_mismatch',
                'n_gvp_nodes': n_gvp_nodes,
                'n_esm_nodes': n_esm_nodes,
            })

    if len(report_rows) == 0:
        print('Pocket feature check passed.')
        return

    df_report = pd.DataFrame(report_rows)
    if report_path:
        df_report.to_csv(report_path, index=False)
        print(f'Pocket feature inconsistency report saved to {report_path}')
    raise ValueError(f'Found {len(df_report)} inconsistent proteins between GVP and pocket ESM features.')
    

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
    parser.add_argument('--pocket_dir', type=str, help='If you already have pocket data, you can specify the directory here')
    parser.add_argument('--pocket_info_path', type=str, help='Optional precomputed pocket_info.csv, e.g. generated from P2Rank')
    parser.add_argument('--structure_dir', type=str, help='Fallback directory of full protein structures (.pdb/.cif), default: <data_dir>/af2_structures')
    parser.add_argument('--skip_rxn_feature', action='store_true')
    args = parser.parse_args()

    if args.pocket_info_path and not args.pocket_dir:
        raise ValueError('--pocket_info_path requires --pocket_dir to point to the extracted pocket structures')
    
    if args.pocket_dir and args.pocket_dir.endswith('/'):
        # remove '/' in the end of the path
        args.pocket_dir = args.pocket_dir[:-1]
    if args.structure_dir and args.structure_dir.endswith('/'):
        args.structure_dir = args.structure_dir[:-1]
    
    if not args.pocket_dir:
        pocket_dir = os.path.join(os.path.dirname(args.data_path), 'pocket/alphafill_8A')
    else:
        pocket_dir = args.pocket_dir

    if args.structure_dir:
        structure_dir = args.structure_dir
    else:
        structure_dir = os.path.join(os.path.dirname(args.data_path), 'af2_structures')
        if not os.path.exists(structure_dir):
            structure_dir = None
        
    feature_dir = os.path.join(os.path.dirname(args.data_path), 'feature')

    # esm_model = 'esm2_t33_650M_UR50D'
    esm_model = 'ESM-C_600M'
    
    protein_feature_dir = os.path.join(feature_dir, 'protein')
    esm_node_feat_dir = os.path.join(protein_feature_dir, f'{esm_model}/node_level')
    esm_mean_feat_path = os.path.join(protein_feature_dir, f'{esm_model}/protein_level/seq2feature.pkl')
    esm_pocket_node_feature_path = os.path.join(protein_feature_dir, f'{esm_model}/pocket_node_feature/esm_node_feature.pt')
    gvp_feat_path = os.path.join(protein_feature_dir, 'gvp_feature/gvp_protein_feature.pt')
    if args.pocket_info_path:
        pocket_info_save_path = args.pocket_info_path
    else:
        pocket_info_save_path = os.path.join(os.path.dirname(pocket_dir), 'pocket_info.csv')
    
    reaction_feat_dir = os.path.join(feature_dir, 'reaction')
    drfp_save_path = os.path.join(reaction_feat_dir, 'drfp/rxn2fp.pkl')
    mol_conformation_dir = os.path.join(reaction_feat_dir, 'molecule_conformation')
    reacting_center_dir = os.path.join(reaction_feat_dir, 'reacting_center')

    os.makedirs(pocket_dir, exist_ok=True)
    os.makedirs(esm_node_feat_dir, exist_ok=True)
    os.makedirs(os.path.dirname(esm_mean_feat_path), exist_ok=True)
    os.makedirs(os.path.dirname(esm_pocket_node_feature_path), exist_ok=True)
    os.makedirs(os.path.dirname(gvp_feat_path), exist_ok=True)
    os.makedirs(os.path.dirname(drfp_save_path), exist_ok=True)
    os.makedirs(mol_conformation_dir, exist_ok=True)
    os.makedirs(reacting_center_dir, exist_ok=True)
    
    if not args.skip_rxn_feature:
        # Calculate reaction fingerprints
        calc_drfp(args.data_path, drfp_save_path)
        
        # Extract reaction center
        calc_reacting_center(args.data_path, reacting_center_dir)
    
        # May be the most time consuming step
        # Generate molecular conformation by rdkit
        generate_mol_conformation(args.data_path, mol_conformation_dir)
    
    if args.pocket_info_path:
        if not os.path.exists(pocket_info_save_path):
            raise FileNotFoundError(f'Precomputed pocket info not found: {pocket_info_save_path}')
        print(f'Using precomputed pocket info from {pocket_info_save_path}')

        # Pocket PDBs already contain only the selected residues, so sequence
        # alignment to the full-length sequence would fail here.
        calc_gvp_feature(args.data_path, pocket_dir, gvp_feat_path, align_to_sequence=False)
    else:
        generate_gvp_compatible_pocket_info(args.data_path, pocket_dir, pocket_info_save_path, structure_dir)

        # Calculate GVP features of pockets
        calc_gvp_feature(args.data_path, pocket_dir, gvp_feat_path, structure_dir)
        
    # Calculate ESM features of the full sequence
    if esm_model == 'esm2_t33_650M_UR50D':
        calc_seq_esm_feature(args.data_path, esm_node_feat_dir, esm_mean_feat_path)
    elif esm_model == 'ESM-C_600M':
        calc_seq_esm_C_feature(args.data_path, esm_node_feat_dir, esm_mean_feat_path)
    
    # Extract esm feature of pocket nodes
    get_esm_pocket_feature(pocket_info_save_path, esm_node_feat_dir, esm_pocket_node_feature_path)
    
    feature_check_report_path = os.path.join(protein_feature_dir, 'pocket_feature_check_report.csv')
    check_pocket_feature(gvp_feat_path, esm_pocket_node_feature_path, feature_check_report_path)
    
    print('\n ###### Feature calculation is finished! ######\n')
    

if __name__ == "__main__":
    main()
    
