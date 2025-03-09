import re
import sys
import pickle as pkl
from functools import partial
from collections import defaultdict

import numba
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdmolops import PatternFingerprint

sys.path.append('/home/liuy/code/Retrosynthesis/')
sys.path.append('/home/liuy/code/SynBio/enzyme-rxn-prediction/')
sys.path.append('/home/liuy/code/SynBio/EnzymeCAGE/feature')
from selenzyme_baseline.main import calc_rxn_simi_matrix
from rdchiral_local.rdchiral import template_extractor
from pkgs.rxnmapper import BatchedMapper


from localmapper import localmapper
mapper = localmapper(device='cpu')


RXN2AAM = pkl.load(open('/home/liuy/data/RHEA/previous_versions/2025-02-05/processed/rxn2aam.pkl', 'rb'))


def cano_smiles(smiles, remain_isomer=True):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=remain_isomer)


def cano_rxn(rxn, exchange_pos=False, remain_isomer=True):
    data = rxn.split('>')
    reactants = data[0].split('.')
    reactants = [cano_smiles(each, remain_isomer) for each in reactants]
    products = data[-1].split('.')
    products = [cano_smiles(each, remain_isomer) for each in products]
    reactants = sorted(reactants)
    products = sorted(products)
    if exchange_pos:
        new_rxn = f"{'.'.join(products)}>>{'.'.join(reactants)}"
    else:
        new_rxn = f"{'.'.join(reactants)}>>{'.'.join(products)}"
    return new_rxn


def remove_wildcard_atom(input_string):
    pattern = r'\*|' + r'\[\*\]|' + r'\[1\*\]|' + r'\[n\*\]|' + r'\[\d+\*\]'    
    result = re.sub(pattern, 'C', input_string)
    return result


def calc_template_fp(rxn_center):
    prod_c = rxn_center.split('>>')[-1]
    reac_c = rxn_center.split('>>')[0]
    prod_mol = Chem.MolFromSmarts(prod_c)
    reac_mol = Chem.MolFromSmarts(reac_c)
    prod_fp = np.array(PatternFingerprint(prod_mol, 512))
    reac_fp = np.array(PatternFingerprint(reac_mol, 512))
    return reac_fp, prod_fp




def get_template(rxn, radius=1):
    if isinstance(radius, int):
        res = {}
        try:
            reaction_dict = {
                'reactants': rxn.split('>')[0],
                'products': rxn.split('>')[-1],
                '_id': 0
            }
            res = template_extractor.extract_from_reaction(reaction_dict, radius)
        except Exception as e:
            print(e)
        
        template = None
        if res:
            template = res.get('reaction_smarts')
        
        return template

    elif isinstance(radius, list):
        template_list = []
        # try:
        reaction_dict = {
            'reactants': rxn.split('>')[0],
            'products': rxn.split('>')[-1],
            '_id': 0
        }
        template_list = template_extractor.extract_from_reaction_hier(reaction_dict, radius)
        # except Exception as e:
        #     print(e)
        if template_list:
            template_list = [each.get('reaction_smarts') for each in template_list]

        return template_list
    
def get_localmapper_template(rxn, radius=1, run_aam=True):
    rxn = remove_wildcard_atom(rxn)
    if run_aam:
        aam = mapper.get_atom_map(rxn)
    else:
        aam = rxn
    template = get_template(aam, radius)
    return template

@numba.jit(nopython=True, parallel=True)
def fast_cosine_matrix(u, M):
    # 这里多次计算可能导致结果不一致
    scores = np.zeros(M.shape[0])
    for i in numba.prange(M.shape[0]):
        v = M[i]
        m = u.shape[0]
        udotv = 0
        u_norm = 0
        v_norm = 0
        for j in range(m):
            if (np.isnan(u[j])) or (np.isnan(v[j])):
                continue

            udotv += u[j] * v[j]
            u_norm += u[j] * u[j]
            v_norm += v[j] * v[j]

        u_norm = np.sqrt(u_norm)
        v_norm = np.sqrt(v_norm)

        if (u_norm == 0) or (v_norm == 0):
            ratio = 0
        else:
            ratio = udotv / (u_norm * v_norm)
        scores[i] = ratio
    return scores


def calc_template_similarity(template, candidate_templates, tmpl_to_fp=None):
    all_tmpls = list(candidate_templates) + [template]
    
    if not tmpl_to_fp:
        tmpl_to_fp = {}

    for tmpl in all_tmpls:
        if tmpl not in tmpl_to_fp:
            tmpl_to_fp[tmpl] = calc_template_fp(tmpl)
    
    reac_fp, prod_fp = tmpl_to_fp[template]

    cand_reac_fps = np.array([tmpl_to_fp[tmpl][0] for tmpl in candidate_templates])
    cand_prod_fps = np.array([tmpl_to_fp[tmpl][1] for tmpl in candidate_templates])
    
    prod_sim_matrix = fast_cosine_matrix(prod_fp, cand_prod_fps)
    reac_sim_matrix = fast_cosine_matrix(reac_fp, cand_reac_fps)

    prod_sim_matrix2 = fast_cosine_matrix(prod_fp, cand_reac_fps)
    reac_sim_matrix2 = fast_cosine_matrix(reac_fp, cand_prod_fps)
    
    # template_sim_matrix = (prod_sim_matrix + reac_sim_matrix) / 2
    # template_sim_matrix2 = (prod_sim_matrix2 + reac_sim_matrix2) / 2

    template_sim_matrix = prod_sim_matrix * reac_sim_matrix
    template_sim_matrix2 = prod_sim_matrix2 * reac_sim_matrix2

    final_simis = []
    for a, b in zip(template_sim_matrix, template_sim_matrix2):
        final_simis.append(max(a, b))

    return np.array(final_simis)


def calc_center_cover_rate(template, rxn):
    template_num_atoms = sum([Chem.MolFromSmarts(smarts).GetNumAtoms() for smarts in template.split('>>')])
    rxn_num_atoms = sum([Chem.MolFromSmiles(smi).GetNumAtoms() for smi in rxn.split('>>')])
    return template_num_atoms/rxn_num_atoms


def generate_weight_list(similarity_list):
    n = len(similarity_list)
    if n == 0:
        return []
    initial_weights = [i*i for i in list(range(n, 0, -1))] 
    total = sum(initial_weights)
    weight_list = [w / total for w in initial_weights]
    return weight_list


from multiprocessing import Pool
def calc_rxnmapper_aam(rxns):
    assert isinstance(rxns, list)
    aam_list = []
    rxn_mapper = BatchedMapper(batch_size=128)
    rxns_to_run = [cano_rxn(remove_wildcard_atom(rxn), remain_isomer=False) for rxn in tqdm(rxns)]
    for i, results in enumerate(tqdm(rxn_mapper.map_reactions_with_info(rxns_to_run), total=len(rxns_to_run))):
        aam = results.get('mapped_rxn')
        aam_list.append(aam)
    return aam_list


def get_tmpls_for_all_radius(rxns, radius):
    aam_list = calc_rxnmapper_aam(rxns)
    for i in range(len(aam_list)):
        if not aam_list[i]:
            aam_list[i] = rxns[i]
    
    rxn_to_alltmpls = {}
    all_template_list = []
    func = partial(get_template, radius=radius)
    with Pool(20) as pool:
        for i, templates in enumerate(tqdm(pool.imap(func, aam_list), total=len(aam_list), desc='extracting templates')):
            if not templates or len(templates) < len(radius):
                # cannot have null template, if None, replace with the full reaction
                templates = [aam_list[i]] * len(radius)
            rxn_to_alltmpls[rxns[i]] = templates
            all_template_list.extend(templates)
    all_template_list = list(set(all_template_list))
    
    tmpl_to_fp = {}
    for template in tqdm(all_template_list, desc='calculating template fp'):
        if not isinstance(template, str):
            continue
        tmpl_to_fp[template] = calc_template_fp(template)

    return tmpl_to_fp, rxn_to_alltmpls


def calc_rxn_similarity_hier(r1, r2, verbose=False):
    # radius = [0, 1, 2, 3, 4, 5, 6]
    radius = [0, 1, 2, 3, 4, 6, 8, 10]
    r1 = cano_rxn(r1, remain_isomer=False)
    r2 = cano_rxn(r2, remain_isomer=False)
    t1_list = get_localmapper_template(r1, radius)
    t2_list = get_localmapper_template(r2, radius)
    simi_list = []
    for t1, t2 in zip(t1_list, t2_list):
        simi = calc_template_similarity(t1, [t2])[0]
        simi_list.append(simi)
        
    template_simi_list = simi_list[:2]
    chemical_env_simi_list = simi_list[2:]

    if template_simi_list[1] > 0.9 and template_simi_list[0] < 0.5:
        template_similarity = max(template_simi_list)
    else:
        w_list = generate_weight_list(template_simi_list)
        # template_similarity = sum([w*s for w, s in zip(w_list, template_simi_list)])
        template_similarity = np.mean(template_simi_list)

    w_env_list = generate_weight_list(chemical_env_simi_list)
    # chem_env_similarity = sum([w*s for w, s in zip(w_env_list, chemical_env_simi_list)])
    chem_env_similarity = np.mean(chemical_env_simi_list)
    mol_level_similarity = calc_rxn_simi_matrix([r1], [r2])[r1][r2]

    # center cover rate
    ccr1 = calc_center_cover_rate(t1_list[1], r1)
    ccr2 = calc_center_cover_rate(t2_list[1], r2)
    ccr = ((ccr1 + ccr2) / 2) ** 2
    
    # 这里可以根据不同EC-level的相似度分布来进行调整
    final_similarity = 0
    if template_similarity >= 0.75:
        final_similarity = 0.5
    elif 0.75 > template_similarity >= 0.5:
        final_similarity = 0.2
    else:
        final_similarity = 0.1
    # elif 0.5 > template_similarity >= 0.3:
    #     final_similarity = 0.1
    # else:
    #     final_similarity = 0.1

    final_similarity += (mol_level_similarity*(1-ccr) + chem_env_similarity * ccr) / 2
    
    if verbose:
        print(f'final_similarity: {round(final_similarity, 4)}, template_similarity: {round(template_similarity, 4)}, chem_env_similarity: {round(chem_env_similarity, 4)}')
        print(f'ccr: {ccr}')
        print(f'mol_level_similarity: {mol_level_similarity}')
        print(f'template_simi_list: {template_simi_list}')
        print(f'chem_env_similarity: {chemical_env_simi_list}')
        # print(f'EC weight: {weight}, r1_top2ec: {r1_top2ec}, r2_top2ec: {r2_top2ec}')
    
    return final_similarity


def calc_rxn_similarity_hier_batch(query_rxns, candidate_rxns, mid_save_path=None):
    radius_list = [0, 1, 2, 3, 4, 6, 8, 10]
    all_rxns = list(set(query_rxns) | set(candidate_rxns))
    tmpl_to_fp, rxn_to_alltmpls = get_tmpls_for_all_radius(all_rxns, radius_list)
    # rxn_to_alltmpls: {rxn1: [tmpl1, tmpl2, ...], rxn2: [tmpl1, tmpl2, ...], ...}
    # size of each value: len(radius_list)
    
    print('=' * 80)
    print(f'Calculating the molecule level reaction similarity...')
    print('=' * 80)
    backbone_simi_map = calc_rxn_simi_matrix(query_rxns, candidate_rxns)
    
    print('=' * 80)
    print(f'Calculating template similarity matrix...')
    print('=' * 80)
    radius_to_simi_matrix = {}
    radius_to_template_index = {}
    for radius_index, radius in tqdm(enumerate(radius_list), desc='calc_template_similarity'):
        template_list = [rxn_to_alltmpls[rxn][radius_index] for rxn in all_rxns]
        template_index_map = {template: index for index, template in enumerate(template_list)}
        radius_to_template_index[radius_index] = template_index_map
        
        tmpl_simi_matrix = []
        for tmpl in tqdm(template_list):
            simi_list = calc_template_similarity(tmpl, template_list, tmpl_to_fp)
            tmpl_simi_matrix.append(simi_list)
        tmpl_simi_matrix = np.stack(tmpl_simi_matrix)
        radius_to_simi_matrix[radius_index] = tmpl_simi_matrix
    
    if mid_save_path is not None:
        with open(mid_save_path, 'wb') as f:
            pkl.dump((radius_to_simi_matrix, radius_to_template_index), f)
            
    
    radius_index_to_calc_ccr = 1
            
    template_list = []
    for rxn in all_rxns:
        t_list = rxn_to_alltmpls[rxn]
        template_list.append(t_list[radius_index_to_calc_ccr])
    template_list = list(set(template_list))
    template_to_num_atoms = {}
    for template in template_list:
        template_to_num_atoms[template] = sum([Chem.MolFromSmarts(smarts).GetNumAtoms() for smarts in template.split('>>')])
    
    rxn_to_num_atoms = {}
    for rxn in all_rxns:
        rxn_to_num_atoms[rxn] = sum([Chem.MolFromSmiles(smi).GetNumAtoms() for smi in rxn.split('>>')])
    
    similarity_map = defaultdict(dict)
    for r1 in tqdm(query_rxns, desc='calc_overall_similarity'):
        for r2 in candidate_rxns:
            t1_list = rxn_to_alltmpls[r1]
            t2_list = rxn_to_alltmpls[r2]
            
            simi_list = []
            for radius_index, radius in enumerate(radius_list):
                template_index_map = radius_to_template_index[radius_index]
                t1 = t1_list[radius_index]
                t2 = t2_list[radius_index]
                t1_idx = template_index_map[t1]
                t2_idx = template_index_map[t2]
                simi = radius_to_simi_matrix[radius_index][t1_idx][t2_idx]
                simi_list.append(simi)
            
            # templates of radius 0 and 1 are used to calculate the template similarity(reaction center similarity)
            reaction_center_simi_list = simi_list[:2]
            # templates of radius 2 and above are used to calculate the chemical environment similarity
            chemical_env_simi_list = simi_list[2:]
            
            if reaction_center_simi_list[1] > 0.9 and reaction_center_simi_list[0] < 0.5:
                template_similarity = max(reaction_center_simi_list)
            else:
                template_similarity = np.mean(reaction_center_simi_list)
            
            chem_env_similarity = np.mean(chemical_env_simi_list)

            # ccr is center cover rate, which mean (number of atoms in the reaction center) / (number of atoms in the reaction)
            t1, t2 = t1_list[radius_index_to_calc_ccr], t2_list[radius_index_to_calc_ccr]
            ccr1 = template_to_num_atoms[t1] / rxn_to_num_atoms[r1]
            ccr2 = template_to_num_atoms[t2] / rxn_to_num_atoms[r2]
            ccr = ((ccr1 + ccr2) / 2) ** 2

            if template_similarity >= 0.75:
                overall_similarity = 0.5
            elif 0.75 > template_similarity >= 0.5:
                overall_similarity = 0.2
            else:
                overall_similarity = 0.1
            
            backbone_level_similarity = backbone_simi_map[r1][r2]

            # ccr high: the molecule is small, the reaction center will take a larger part of the molecule, so the chem_env_similarity will be more accurate
            # ccr low: the molecule is large, the reaction center will take a smaller part of the molecule, so the backbone_level_similarity will be slightly more accurate
            overall_similarity += (backbone_level_similarity*(1-ccr) + chem_env_similarity * ccr) / 2
            
            similarity_map[r1][r2] = overall_similarity

    return similarity_map


def main_rhea():
    df_rhea_v2502 = pd.read_csv('/home/liuy/data/RHEA/previous_versions/2025-02-05/processed/rhea_rxn2uids.csv')
    rxns = df_rhea_v2502['CANO_RXN_SMILES'].unique()
    
    mid_save_path = '/home/liuy/data/RHEA/previous_versions/2025-02-05/processed/all_tmpls_simi_matrix.pkl'
    
    similarity_map = calc_rxn_similarity_hier_batch(rxns, rxns, mid_save_path=mid_save_path)
    
    save_path = '/home/liuy/data/RHEA/previous_versions/2025-02-05/processed/rxn_hier_similarity_map.pkl'
    with open(save_path, 'wb') as f:
        pkl.dump(similarity_map, f)
    
    print(f'Save to {save_path}')
    

if __name__ == '__main__':
        
    main_rhea()