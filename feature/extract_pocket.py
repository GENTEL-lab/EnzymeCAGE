import os
import json
import string
from multiprocessing import Pool
from functools import partial
from collections import defaultdict

from Bio import PDB
from Bio.PDB import MMCIFParser, PDBParser, PDBIO, NeighborSearch, MMCIFIO
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob


POCKET_RADIUS = 8.0
KEEP_LIGAND = True


def get_cpd_info(structure, related_chain_ids):
    related_chain_ids = set(related_chain_ids)
    chain_id_to_n_atoms = {}
    for model in structure:
        for chain in model:
            if chain.id in related_chain_ids:
                res_list = [each for each in chain.get_residues()]
                assert len(res_list) == 1
                res = res_list[0]
                n_atoms = len([_ for _ in res.get_atoms()])
                chain_id_to_n_atoms[res.resname] = n_atoms
    
    return chain_id_to_n_atoms
    

def get_best_ligand_chain(structure, trans_meta_info):
    for trans_result in trans_meta_info['hits']:
        cpd_counter = defaultdict(int)
        for trans_content in trans_result['transplants']:
            cpd_counter[trans_content['compound_id']] += 1

        cand_cpds = {cpd_name for cpd_name, cnt in cpd_counter.items() if cnt < 3}
        if not cand_cpds:
            continue

        concerned_data = []
        for each in trans_result['transplants']:
            if each['compound_id'] in cand_cpds:
                concerned_data.append([each['compound_id'], each['asym_id']])
        related_chains = list(map(lambda x: x[1], concerned_data))
        n_atoms_dict = get_cpd_info(structure, related_chains)
        # if len(n_atoms_dict) != len(related_chains):
        #     raise ValueError(f'Found chain not in the structure: {concerned_data}')
        
        # 选最大的那个Compound作为best fit
        sorted_result = sorted(n_atoms_dict.items(), key=lambda x: x[1], reverse=True)
        
        # best fit的原子数太少
        if sorted_result[0][1] <= 3:
            continue
            
        best_fit_cpd = sorted_result[0][0]
        chains_to_keep = [each[1] for each in concerned_data if each[0] == best_fit_cpd]
        
        return chains_to_keep
    
    return None


def remove_far_residues(input_path, save_dir=None, cutoff=8.0, save_format='cif', preserve_ligand=True, remove_ion=False):
    try:
        if save_dir is None:
            save_dir = os.path.dirname(input_path)
        meta_info_path = input_path.replace('.cif', '.json')
        meta_data = json.load(open(meta_info_path))

        filename = os.path.basename(input_path)
        if '_transplant.cif' in filename:
            mark = '_transplant.cif'
        elif '_final.cif' in filename:
            mark = '_final.cif'
        save_path = os.path.join(save_dir, filename.replace(mark, f'.{save_format}'))

        if input_path.endswith('.cif'):
            parser = MMCIFParser()
            structure = parser.get_structure("protein", input_path)
        elif input_path.endswith('.pdb'):
            parser = PDBParser()
            structure = parser.get_structure("protein", input_path)
        
        num_chains = len([each.get_id() for each in structure.get_chains()])
        if num_chains == 1:
            return

        chains_to_keep = get_best_ligand_chain(structure, meta_data)
        if chains_to_keep is None:
            # print(f'No ligand chain to keep: {filename}')
            return

        chains_to_remove = []
        for model in structure:
            for chain in model:
                res_list = [i for i in chain.get_residues()]
                if len(res_list) > 1: continue
                
                if chain.id not in chains_to_keep:
                    chains_to_remove.append(chain.id)

        for model in structure:
            for chain_id in chains_to_remove:
                model.detach_child(chain_id)

        # structure = filter_ligands(structure, remove_ion=remove_ion)

        # 获取所有原子对象
        atoms = list(structure.get_atoms())

        ligand_atoms = []
        for residue in structure.get_residues():
            if residue.id[0] != ' ':
                ligand_atoms.extend(residue.get_atoms())

        # 创建NeighborSearch对象
        ns = NeighborSearch(atoms)

        # 找出与符合条件的小分子原子距离小于cutoff的氨基酸
        residues_to_keep = set()
        for ligand_atom in ligand_atoms:
            nearby_residues = ns.search(ligand_atom.coord, cutoff, level='R')
            for res in nearby_residues:
                residues_to_keep.add(res)

        # 删除距离大于cutoff的氨基酸
        for model in structure:
            for chain in model:
                for res in list(chain.get_residues()):
                    if res not in residues_to_keep:
                        chain.detach_child(res.id)

        if not preserve_ligand:
            structure = filter_ligands(structure, all_ligands=True)

        structure = reset_chain_id(structure)

        # 保存修改后的结构
        if save_format == 'cif':
            io = MMCIFIO()
        elif save_format == 'pdb':
            io = PDBIO()
        else:
            raise ValueError(f"Save format not supported: {save_format}")
        io.set_structure(structure)
        io.save(save_path)
    except Exception as e:
        print(e)
        return


def reset_chain_id(structure):
    for model in structure:
        for i, chain in enumerate(model):
            chain.id = string.ascii_uppercase[i]
    return structure


def filter_ligands(structure, all_ligands=False, remove_ion=False):
    if not all_ligands:
        ligand_residues_list = []
        for chain in structure.get_chains():
            residues = [residue for residue in chain]
            if len(residues) == 1:
                # ligand通常以单一的residue来表征
                ligand_residues_list.append(residues)
        
        ligand_cnt_dict = defaultdict(int)
        for residues in ligand_residues_list:
            resname = residues[0].resname
            ligand_cnt_dict[resname] += 1
        
        ligand_to_remove = set(['HOH', 'H2O'])
        for ligand_name, cnt in ligand_cnt_dict.items():
            if cnt >= 3:
                ligand_to_remove.add(ligand_name)
        
        chain_id_to_delete = []
        for model in structure:
            for chain in model:
                residues = [res for res in chain]
                if len(residues) > 1:
                    continue
                elif len(residues) == 1:
                    if residues[0].resname in ligand_to_remove:
                        chain_id_to_delete.append(chain.id)
                    elif remove_ion:
                        if len([_ for _ in residues[0].get_atoms()]) == 1:
                            chain_id_to_delete.append(chain.id)
                else:
                    raise ValueError(f"Error of len(residues): {len(residues)}")
    else:
        chain_id_to_delete = []
        for model in structure:
            for chain in model:
                residues = [res for res in chain]
                if len(residues) == 1:
                    chain_id_to_delete.append(chain.id)
    
    chain_id_to_delete = set(chain_id_to_delete)
    for model in structure:
        for chain_id in chain_id_to_delete:
            model.detach_child(chain_id)
        
    return structure


def get_pocket_info(input_path, pocket_save_dir=None):
    try:
        # if save_dir is None:
        #     save_dir = os.path.dirname(input_path)
        meta_info_path = input_path.replace('.cif', '.json')
        meta_data = json.load(open(meta_info_path))

        if input_path.endswith('.cif'):
            parser = MMCIFParser()
            structure = parser.get_structure("protein", input_path)
        elif input_path.endswith('.pdb'):
            parser = PDBParser()
            structure = parser.get_structure("protein", input_path)
        
        num_chains = len([each.get_id() for each in structure.get_chains()])
        # print('num_chains: ', num_chains)
        if num_chains == 1:
            return

        chains_to_keep = get_best_ligand_chain(structure, meta_data)
        # print('chains_to_keep: ', chains_to_keep)
        if chains_to_keep is None:
            # print(f'No ligand chain to keep: {filename}')
            return

        chains_to_remove = []
        for model in structure:
            for chain in model:
                res_list = [i for i in chain.get_residues()]
                if len(res_list) > 1: continue
                
                if chain.id not in chains_to_keep:
                    chains_to_remove.append(chain.id)

        for model in structure:
            for chain_id in chains_to_remove:
                model.detach_child(chain_id)

        # structure = filter_ligands(structure, remove_ion=remove_ion)

        # 获取所有原子对象
        atoms = list(structure.get_atoms())

        ligand_atoms = []
        for residue in structure.get_residues():
            if residue.id[0] != ' ':
                ligand_atoms.extend(residue.get_atoms())

        # 创建NeighborSearch对象
        ns = NeighborSearch(atoms)

        # 找出与符合条件的小分子原子距离小于cutoff的氨基酸
        residues_to_keep = set()
        for ligand_atom in ligand_atoms:
            nearby_residues = ns.search(ligand_atom.coord, POCKET_RADIUS, level='R')
            for res in nearby_residues:
                residues_to_keep.add(res)

        # 删除距离大于cutoff的氨基酸
        for model in structure:
            for chain in model:
                for res in list(chain.get_residues()):
                    if res not in residues_to_keep:
                        chain.detach_child(res.id)

        structure = filter_ligands(structure, all_ligands=True)

        structure = reset_chain_id(structure)

        pocket_res_ids = []
        for chain in structure.get_chains():
            if chain.id != 'A':
                continue
            for residue in chain.get_residues():
                pocket_res_ids.append(str(residue.id[1]))
        # print(pocket_res_ids)
        pocket_info = ','.join(pocket_res_ids)
        
        if pocket_save_dir:
            os.makedirs(pocket_save_dir, exist_ok=True)
            filename = os.path.basename(input_path)
            if '_transplant.cif' in filename:
                mark = '_transplant.cif'
            elif '_final.cif' in filename:
                mark = '_final.cif'
            else:
                mark = '.cif'
            save_path = os.path.join(pocket_save_dir, filename.replace(mark, f'.pdb'))
            
            # 保存修改后的结构
            io = PDBIO()
            io.set_structure(structure)
            io.save(save_path)
        
        return pocket_info

    except Exception as e:
        print(e)
        return


def extract_fix_num_residues(input_path, pocket_save_dir=None, residue_num=32):
    
    try:
        # if save_dir is None:
        #     save_dir = os.path.dirname(input_path)
        meta_info_path = input_path.replace('.cif', '.json')
        meta_data = json.load(open(meta_info_path))

        if input_path.endswith('.cif'):
            parser = MMCIFParser()
            structure = parser.get_structure("protein", input_path)
        elif input_path.endswith('.pdb'):
            parser = PDBParser()
            structure = parser.get_structure("protein", input_path)
        
        num_chains = len([each.get_id() for each in structure.get_chains()])
        if num_chains == 1:
            return

        chains_to_keep = get_best_ligand_chain(structure, meta_data)
        if chains_to_keep is None:
            return

        chains_to_remove = []
        for model in structure:
            for chain in model:
                res_list = [i for i in chain.get_residues()]
                if len(res_list) > 1: continue
                
                if chain.id not in chains_to_keep:
                    chains_to_remove.append(chain.id)

        for model in structure:
            for chain_id in chains_to_remove:
                model.detach_child(chain_id)

        atoms = list(structure.get_atoms())

        ligand_atoms = []
        for residue in structure.get_residues():
            if residue.id[0] != ' ':
                ligand_atoms.extend(residue.get_atoms())

        ns = NeighborSearch(atoms)

        residues_to_keep = []
        for ligand_atom in ligand_atoms:
            nearby_residues = ns.search(ligand_atom.coord, 10.0, level='R')  # Set a high cutoff to include all candidate residues
            for res in nearby_residues:
                if res not in residues_to_keep and 'CA' in res:
                    residues_to_keep.append(res)

        # Sort residues by distance to ligand
        residues_to_keep.sort(key=lambda r: min(np.linalg.norm(ligand_atom.coord - r['CA'].coord) for ligand_atom in ligand_atoms))

        # Keep the closest num_residues residues
        residues_to_keep = residues_to_keep[:residue_num]

        residues_to_keep = set(residues_to_keep)

        for model in structure:
            for chain in model:
                for res in list(chain.get_residues()):
                    if res not in residues_to_keep:
                        chain.detach_child(res.id)

        structure = filter_ligands(structure, all_ligands=True)

        structure = reset_chain_id(structure)

        pocket_res_ids = []
        for chain in structure.get_chains():
            if chain.id != 'A':
                continue
            for residue in chain.get_residues():
                pocket_res_ids.append(str(residue.id[1]))
        # print(pocket_res_ids)
        pocket_info = ','.join(pocket_res_ids)
        
        if pocket_save_dir:
            os.makedirs(pocket_save_dir, exist_ok=True)
            filename = os.path.basename(input_path)
            if '_transplant.cif' in filename:
                mark = '_transplant.cif'
            elif '_final.cif' in filename:
                mark = '_final.cif'
            else:
                mark = '.cif'
            save_path = os.path.join(pocket_save_dir, filename.replace(mark, f'.pdb'))
            
            # 保存修改后的结构
            io = PDBIO()
            io.set_structure(structure)
            io.save(save_path)
        
        return pocket_info

    except Exception as e:
        print(e)
        return


def extract_pocket_by_resids(input_dir, save_dir, pocket_info_path):
    df_pocket_info = pd.read_csv(pocket_info_path)
    uid2pocket = dict(zip(df_pocket_info['uniprotID'], df_pocket_info['pocket_residues']))
    
    for uid, pocket_residues in tqdm(uid2pocket.items()):
        pdb_path = os.path.join(input_dir, f'{uid}.pdb')
        pocket_residues = [int(resid) for resid in pocket_residues.split(',')]
        save_path = os.path.join(save_dir, f'{uid}.pdb')
        
        # TODO 根据pocket_residues提取口袋，并将口袋氨基酸保存为pdb文件，存到save_path
        # Load the structure using Biopython's PDBParser
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(uid, pdb_path)
        
        # Initialize the PDBIO object to save selected residues
        io = PDB.PDBIO()
        
        # Define a selective class to save only the pocket residues
        class PocketSelect(PDB.Select):
            def __init__(self, pocket_residues):
                self.pocket_residues = pocket_residues
                
            def accept_residue(self, residue):
                # Only accept residues whose ID is in pocket_residues
                if residue.get_id()[1] in self.pocket_residues:
                    return True
                return False
        
        # Save the selected residues to a new PDB file
        io.set_structure(structure)
        io.save(save_path, PocketSelect(pocket_residues))
        
        # print(f'Pocket residues for {uid} saved to {save_path}')
        
        
def get_res_ids(pdb_file):
    parser = PDB.PDBParser(QUIET=True)    
    structure = parser.get_structure("protein", pdb_file)
    
    residue_id_list = []

    residues = [res for res in structure.get_residues()]
    
    for residue in residues:
        residue_id = residue.get_id()[1] 
        residue_id_list.append(str(residue_id))
    
    return residue_id_list


def get_pocket_info_batch_new(input_dir, save_path):
    print('Generating pocket residues information...')
    
    # Get a list of all PDB files in the input directory
    pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))

    # Initialize a list to store the pocket information for each PDB file
    pocket_info_list = []
    uid_list = []

    with Pool(20) as pool:
        for pocket_info in tqdm(pool.imap(get_res_ids, pdb_files), total=len(pdb_files)):
            pocket_info_list.append(','.join(pocket_info))
    
    uid_list = [os.path.basename(pdb_file).split('.')[0] for pdb_file in pdb_files]

    df_pocket_info = pd.DataFrame({'uniprotID': uid_list, 'pocket_residues': pocket_info_list})
    df_pocket_info.to_csv(save_path, index=False)
    print(f'\nSave pocket info to {save_path}\n')