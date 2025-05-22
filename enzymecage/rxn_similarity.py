import math
from collections import defaultdict, Counter

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
import pandas as pd
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


def get_rdkit_mol(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    return mol


def get_morgan_fp(mol, inchi=0): 
    mol = get_rdkit_mol(mol)
    Chem.SanitizeMol(mol)
    info1 = {}
    fpM = AllChem.GetMorganFingerprint(mol, 8, bitInfo=info1, invariants=AllChem.GetConnectivityInvariants(mol, includeRingMembership=False))
    return fpM, Chem.MolToSmiles(mol), mol.GetNumAtoms(), info1


def bulkTani(targetFp, fp, rxn_ids):
    similarity_list = DataStructs.BulkTanimotoSimilarity(targetFp, list(fp))
    dist={}
    for i in sorted(range(0, len(similarity_list))):
        dist[rxn_ids[i]] = similarity_list[i]
    return dist


def getRSim(rcts_smi_counter, pros_smi_counter, cand_rcts_molid_counter, cand_pros_molid_counter, sim):
    # rcts_smi_counter, pros_smi_counter, cand_rcts_molid_counter, cand_pros_molid_counter, sim
    info_dict = {'s1': rcts_smi_counter, 'p1': pros_smi_counter, 's2': cand_rcts_molid_counter, 'p2':cand_pros_molid_counter}
    ss = {} 
    simm = {}
    pairs = [('s1','s2'), ('s1', 'p2'), ('p1', 's2'), ('p1', 'p2')]
    compPairs = {}
    for pair_tuple in pairs:
        # pair_tuple -> ('s1','s2')
        pairings = set()
        simm[pair_tuple] = {}
        compPairs[pair_tuple]=[]

        for mol_x in info_dict[pair_tuple[0]].keys():
            simm[pair_tuple][mol_x] = (0.0, mol_x, None)
            if mol_x in sim:
                for mol_y in info_dict[pair_tuple[1]].keys():
                    if mol_y in sim[mol_x]:
                        pairings.add( (sim[mol_x][mol_y], mol_x, mol_y) )

        found = {'left': set(), 'right': set()}
        for v in sorted(pairings, key = lambda h: -h[0]):
            if v[1] not in found['left'] and v[2] not in found['right']:
                # if similarity is greater that zero
                if v[0] > simm[pair_tuple][v[1]][0]:
                    simm[pair_tuple][v[1]] = v
                    found['left'].add(v[1])
                    found['right'].add(v[2])
                    compPairs[pair_tuple].append([v[1], v[2]])
        s = []
        for mol_x in simm[pair_tuple]:
            s.append(simm[pair_tuple][mol_x][0])
        if len(s) > 0:
            ss[pair_tuple] = sum(s)/len(s)
        else:
            ss[pair_tuple] = 0.0
    S1 = math.sqrt(ss[pairs[0]]**2 + ss[pairs[3]]**2)/math.sqrt(2)
    S2 = math.sqrt(ss[pairs[1]]**2 + ss[pairs[2]]**2)/math.sqrt(2)

    return(S1, S2, compPairs)


def neutralize_atoms(smi):
    mol = get_rdkit_mol(smi)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
            
    smi_uncharged = Chem.MolToSmiles(mol)
    if not get_rdkit_mol(smi_uncharged):
        smi_uncharged = smi
    
    return smi_uncharged


def get_mol_simi_dict(test_rxns, all_cand_rxns):
    testset_smiles_list = []
    for rxn in test_rxns:
        smiles_list = rxn.split('>>')[0].split('.') + rxn.split('>>')[1].split('.')
        smiles_list = [smi for smi in smiles_list]
        testset_smiles_list.extend(smiles_list)
    test_smiles = list(set(testset_smiles_list))

    cand_smiles_list = []
    for rxn in all_cand_rxns:
        smiles_list = rxn.split('>>')[0].split('.') + rxn.split('>>')[1].split('.')
        cand_smiles_list.extend(smiles_list)
    cand_smiles = list(set(cand_smiles_list))

    cand_mol_ids = [f'MOL_{i}' for i, _ in enumerate(cand_smiles)]
    cand_mol_to_id_dict = dict(zip(cand_smiles, cand_mol_ids))
    cand_fp_list = []
    for smi in tqdm(cand_smiles):
        try:
            smi = neutralize_atoms(smi)
        except:
            print(f'Cannot neutralize SMILES: {smi}, use original format')

        try:
            fp = get_morgan_fp(smi)[0]
            cand_fp_list.append(fp)
        except:
            print(f'Error SMILES: ', smi)
            print(sss)

    cpd_simi_dict = {}
    for smi in tqdm(test_smiles):
        fp_target = get_morgan_fp(neutralize_atoms(smi))[0]
        cpd_simi_dict[smi] = bulkTani(fp_target, cand_fp_list, cand_mol_ids) 
    
    return cpd_simi_dict, cand_mol_to_id_dict


def remove_nan(data):
    if isinstance(data, set):
        data = {each for each in data if not pd.isna(each)}
    elif isinstance(data, list):
        data = [each for each in data if not pd.isna(each)]
    return data


def calc_rxn_simi_matrix(test_rxns, cand_rxns):
    cpd_simi_dict, cand_mol_to_id_dict = get_mol_simi_dict(test_rxns, cand_rxns)
    simi_matrix = defaultdict(dict)
    for rxn_target in tqdm(test_rxns, desc='Calculating reaction similarity'):
        rcts_smi_counter = Counter(rxn_target.split('>>')[0].split('.'))
        pros_smi_counter = Counter(rxn_target.split('>>')[1].split('.'))
        for cand_rxn in cand_rxns:
            cand_rcts = [cand_mol_to_id_dict.get(smi) for smi in cand_rxn.split('>>')[0].split('.')]
            cand_pros = [cand_mol_to_id_dict.get(smi) for smi in cand_rxn.split('>>')[1].split('.')]
            cand_rcts_molid_counter, cand_pros_molid_counter = Counter(cand_rcts), Counter(cand_pros)
            S1, S2, _ = getRSim(rcts_smi_counter, pros_smi_counter, cand_rcts_molid_counter, cand_pros_molid_counter, cpd_simi_dict)
            max_simi = max(S1, S2)
            simi_matrix[rxn_target][cand_rxn] = max_simi
    return simi_matrix
