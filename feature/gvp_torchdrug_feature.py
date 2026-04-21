import os
import sys
import importlib.util

from Bio.PDB import MMCIFParser, PDBParser
import torch
from tqdm import tqdm
import pandas as pd
torch.set_num_threads(1)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(ROOT_DIR, 'enzymecage')
sys.path.append(model_dir)
gvp_data_path = os.path.join(model_dir, 'gvp', 'data.py')
gvp_data_spec = importlib.util.spec_from_file_location('enzymecage_gvp_data', gvp_data_path)
gvp_data_module = importlib.util.module_from_spec(gvp_data_spec)
gvp_data_spec.loader.exec_module(gvp_data_module)
ProteinGraphDataset = gvp_data_module.ProteinGraphDataset
from base import UID_COL


three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_gvp_compatible_residues(res_list, verbose=False):
    res_list = get_clean_res_list(res_list, verbose=verbose, ensure_ca_exist=True)
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    return res_list


def get_structure_sequence_and_clean_residues(structure_path):
    residues = load_structure_residues('protein', structure_path)
    clean_residues = get_clean_res_list(residues, verbose=False, ensure_ca_exist=True)
    structure_sequence = "".join([three_to_one.get(res.resname) for res in clean_residues])
    return structure_sequence, clean_residues


def get_structure_sequence_and_gvp_residues(structure_path):
    residues = load_structure_residues('protein', structure_path)
    clean_residues = get_clean_res_list(residues, verbose=False, ensure_ca_exist=True)
    gvp_residues = get_gvp_compatible_residues(clean_residues, verbose=False)
    structure_sequence = "".join([three_to_one.get(res.resname) for res in gvp_residues])
    return structure_sequence, gvp_residues


def align_clean_residues_to_sequence(structure_path, target_sequence=None):
    structure_sequence, clean_residues = get_structure_sequence_and_clean_residues(structure_path)
    if not target_sequence:
        return clean_residues, 'full_length_match'

    if structure_sequence == target_sequence:
        return clean_residues, 'exact_match'

    start_idx = structure_sequence.find(target_sequence)
    if start_idx >= 0:
        end_idx = start_idx + len(target_sequence)
        return clean_residues[start_idx:end_idx], 'substring_match'

    raise ValueError(
        f'Sequence mismatch between dataset ({len(target_sequence)}) and structure ({len(structure_sequence)})'
    )


def load_structure_residues(structure_name, structure_path):
    if structure_path.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    elif structure_path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f'Unsupported structure format: {structure_path}')
    structure = parser.get_structure(structure_name, structure_path)
    return list(structure.get_residues())


def get_gvp_residue_ids_from_structure(structure_path, target_sequence=None):
    clean_residues, _ = align_clean_residues_to_sequence(structure_path, target_sequence)
    compatible_position_ids = []
    for position_idx, residue in enumerate(clean_residues, start=1):
        if ('N' in residue) and ('CA' in residue) and ('C' in residue) and ('O' in residue):
            compatible_position_ids.append(str(position_idx))
    return compatible_position_ids


def resolve_structure_path(uid, structure_dir, fallback_structure_dir=None):
    for folder in [structure_dir, fallback_structure_dir]:
        if not folder:
            continue
        for ext in ['.pdb', '.cif']:
            filepath = os.path.join(folder, f'{uid}{ext}')
            if os.path.exists(filepath):
                return filepath
    return None


def get_protein_feature(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    res_list = get_gvp_compatible_residues(res_list)
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = ProteinGraphDataset([structure])
    protein = dataset[0]
    x = (protein.x, protein.seq, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v)
    return x


def batch_run(x):
    protein_dict = {}
    pdb, protein_file, to_file, target_sequence = x
    error_file = f'{to_file}.error'
    if os.path.exists(error_file):
        os.remove(error_file)
    try:
        res_list, _ = align_clean_residues_to_sequence(protein_file, target_sequence)
        res_list = get_gvp_compatible_residues(res_list, verbose=False)
        if len(res_list) == 0:
            with open(error_file, 'w') as f:
                f.write('No GVP-compatible residues after filtering')
            torch.save(protein_dict, to_file)
            return
        protein_dict[pdb] = get_protein_feature(res_list)
    except Exception as e:
        with open(error_file, 'w') as f:
            f.write(str(e))
    # if 'phosphatase_63' in proteinFile:
    #     print('num residues: ', len(protein_dict[pdb][0]))
    torch.save(protein_dict, to_file)


def calc_gvp_feature(data_path, pdb_dir, save_path, fallback_structure_dir=None, align_to_sequence=True):
    print('\n', '#' * 20, 'Calculating GVP Feature', '#' * 20, '\n')
    import mlcrate as mlc

    protein_embedding_folder = os.path.join(os.path.dirname(save_path), 'tmp')
    os.makedirs(protein_embedding_folder, exist_ok=True)

    df_data = pd.read_csv(data_path)
    uids = set(df_data[UID_COL])
    uid_to_sequence = dict(df_data[[UID_COL, 'sequence']].drop_duplicates().values)

    input_ = []
    uniprot_id_list = []
    missing_structure_uids = []
    for uid in uids:
        filepath = resolve_structure_path(uid, pdb_dir, fallback_structure_dir)
        if filepath is None:
            missing_structure_uids.append(uid)
            continue
        
        uniprot_id_list.append(uid)
        to_file = f"{protein_embedding_folder}/{uid}.pt"
        target_sequence = uid_to_sequence.get(uid) if align_to_sequence else None
        x = (uid, filepath, to_file, target_sequence)
        input_.append(x)
    
    if missing_structure_uids:
        missing_save_path = os.path.join(os.path.dirname(save_path), 'missing_structure_uids.csv')
        pd.DataFrame({UID_COL: sorted(missing_structure_uids)}).to_csv(missing_save_path, index=False)
        print(f'Missing structure files for {len(missing_structure_uids)} proteins. Saved to {missing_save_path}')
    
    pool = mlc.SuperPool(10)
    pool.pool.restart()
    _ = pool.map(batch_run, input_)
    pool.exit()

    protein_dict = {}
    failed_records = []
    for uniprot_id in tqdm(uniprot_id_list):
        tmp_path = f"{protein_embedding_folder}/{uniprot_id}.pt"
        protein_dict.update(torch.load(tmp_path))
        if uniprot_id not in protein_dict:
            error_path = f'{tmp_path}.error'
            error_msg = ''
            if os.path.exists(error_path):
                with open(error_path) as f:
                    error_msg = f.read().strip()
            failed_records.append({UID_COL: uniprot_id, 'error': error_msg})

    if failed_records:
        failed_save_path = os.path.join(os.path.dirname(save_path), 'failed_gvp_uids.csv')
        pd.DataFrame(failed_records).to_csv(failed_save_path, index=False)
        print(f'Failed to calculate GVP features for {len(failed_records)} proteins. Saved to {failed_save_path}')

    torch.save(protein_dict, save_path)
    print(f'Save gvp feature to {save_path}\n')


def run_single():
    uid = 'H9A1V3'
    pdb_file = f'/mnt/nas/ai-algorithm-data/liuyong/dataset/SynBio/enzyme-reaction-pairs/from_zt/3D-structures/active_site_new_no_ion/pdb_8A/{uid}.pdb'
    save_path = f'/mnt/nas/ai-algorithm-data/liuyong/code/SynBio/enzyme-rxn-prediction/dataset/gvp/gvp_protein_embedding_pocket/{uid}.pt'

    input_ = (uid, pdb_file, save_path)
    batch_run(input_)

    print(f'{uid} done')


if __name__ == '__main__':
    # run_single()
    main()
