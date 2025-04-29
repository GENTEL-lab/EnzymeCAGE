import os
import re
import sys
import pickle as pkl
# sys.path.append('./pkgs/')

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType

from pkgs.rxnmapper import BatchedMapper

VERBOSE = False
USE_STEREOCHEMISTRY = True
MAXIMUM_NUMBER_UNMAPPED_PRODUCT_ATOMS = 5
INCLUDE_ALL_UNMAPPED_REACTANT_ATOMS = True


def bond_to_label(bond):
    '''This function takes an RDKit bond and creates a label describing
    the most important attributes'''
    a1_label = str(bond.GetBeginAtom().GetAtomicNum())
    a2_label = str(bond.GetEndAtom().GetAtomicNum())
    if bond.GetBeginAtom().HasProp('molAtomMapNumber'):
        a1_label += bond.GetBeginAtom().GetProp('molAtomMapNumber')
    if bond.GetEndAtom().HasProp('molAtomMapNumber'):
        a2_label += bond.GetEndAtom().GetProp('molAtomMapNumber')
    atoms = sorted([a1_label, a2_label])

    return '{}{}{}'.format(atoms[0], bond.GetSmarts(), atoms[1])


def mols_from_smiles_list(all_smiles):
    '''Given a list of smiles strings, this function creates rdkit
    molecules'''
    mols = []
    for smiles in all_smiles:
        if not smiles: continue
        mols.append(Chem.MolFromSmiles(smiles))
    return mols

def replace_deuterated(smi):
    return re.sub('\[2H\]', r'[H]', smi)

def clear_mapnum(mol):
    [a.ClearProp('molAtomMapNumber') for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
    return mol

def get_tagged_atoms_from_mols(mols):
    '''Takes a list of RDKit molecules and returns total list of
    atoms and their tags'''
    atoms = []
    atom_tags = []
    for mol in mols:
        new_atoms, new_atom_tags = get_tagged_atoms_from_mol(mol)
        atoms += new_atoms 
        atom_tags += new_atom_tags
    return atoms, atom_tags

def get_tagged_atoms_from_mol(mol):
    '''Takes an RDKit molecule and returns list of tagged atoms and their
    corresponding numbers'''
    atoms = []
    atom_tags = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atoms.append(atom)
            atom_tags.append(str(atom.GetProp('molAtomMapNumber')))
    return atoms, atom_tags

def atoms_are_different(atom1, atom2):
    '''Compares two RDKit atoms based on basic properties'''

    if atom1.GetAtomicNum() != atom2.GetAtomicNum(): return True # must be true for atom mapping
    if atom1.GetTotalNumHs() != atom2.GetTotalNumHs(): return True
    if atom1.GetFormalCharge() != atom2.GetFormalCharge(): return True
    if atom1.GetDegree() != atom2.GetDegree(): return True
    #if atom1.IsInRing() != atom2.IsInRing(): return True # do not want to check this!
    # e.g., in macrocycle formation, don't want the template to include the entire ring structure
    if atom1.GetNumRadicalElectrons() != atom2.GetNumRadicalElectrons(): return True
    if atom1.GetIsAromatic() != atom2.GetIsAromatic(): return True 

    # Check bonds and nearest neighbor identity
    bonds1 = sorted([bond_to_label(bond) for bond in atom1.GetBonds()]) 
    bonds2 = sorted([bond_to_label(bond) for bond in atom2.GetBonds()]) 
    if bonds1 != bonds2: return True

    return False

def find_map_num(mol, mapnum):
    return [(a.GetIdx(), a) for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber') 
         and a.GetProp('molAtomMapNumber') == str(mapnum)][0]

def get_tetrahedral_atoms(reactants, products):
    tetrahedral_atoms = []
    for reactant in reactants:
        for ar in reactant.GetAtoms():
            if not ar.HasProp('molAtomMapNumber'):
                continue
            atom_tag = ar.GetProp('molAtomMapNumber')
            ir = ar.GetIdx()
            for product in products:
                try:
                    (ip, ap) = find_map_num(product, atom_tag)
                    if ar.GetChiralTag() != ChiralType.CHI_UNSPECIFIED or\
                            ap.GetChiralTag() != ChiralType.CHI_UNSPECIFIED:
                        tetrahedral_atoms.append((atom_tag, ar, ap))
                except IndexError:
                    pass
    return tetrahedral_atoms

def set_isotope_to_equal_mapnum(mol):
    for a in mol.GetAtoms():
        if a.HasProp('molAtomMapNumber'):
            a.SetIsotope(int(a.GetProp('molAtomMapNumber')))
            
def get_frag_around_tetrahedral_center(mol, idx):
    '''Builds a MolFragment using neighbors of a tetrahedral atom,
    where the molecule has already been updated to include isotopes'''
    ids_to_include = [idx]
    for neighbor in mol.GetAtomWithIdx(idx).GetNeighbors():
        ids_to_include.append(neighbor.GetIdx())
    symbols = ['[{}{}]'.format(a.GetIsotope(), a.GetSymbol()) if a.GetIsotope() != 0\
               else '[#{}]'.format(a.GetAtomicNum()) for a in mol.GetAtoms()]
    return Chem.MolFragmentToSmiles(mol, ids_to_include, isomericSmiles=True,
                                   atomSymbols=symbols, allBondsExplicit=True,
                                   allHsExplicit=True)
            
def check_tetrahedral_centers_equivalent(atom1, atom2):
    '''Checks to see if tetrahedral centers are equivalent in
    chirality, ignoring the ChiralTag. Owning molecules of the
    input atoms must have been Isotope-mapped'''
    atom1_frag = get_frag_around_tetrahedral_center(atom1.GetOwningMol(), atom1.GetIdx())
    atom1_neighborhood = Chem.MolFromSmiles(atom1_frag, sanitize=False)
    for matched_ids in atom2.GetOwningMol().GetSubstructMatches(atom1_neighborhood, useChirality=True):
        if atom2.GetIdx() in matched_ids:
            return True
    return False

def clear_isotope(mol):
    [a.SetIsotope(0) for a in mol.GetAtoms()]

def get_missing_atoms(reactants, products):
    _, tags_rcts = get_tagged_atoms_from_mols(reactants)
    _, tags_prods = get_tagged_atoms_from_mols(products)
    tags_rcts, tags_prods = set(tags_rcts), set(tags_prods)
    missing_tags = (tags_rcts | tags_prods) - (tags_rcts & tags_prods)
    return missing_tags

def get_changed_atoms(reactants, products):
    '''Looks at mapped atoms in a reaction and determines which ones changed'''

    err = 0
    prod_atoms, prod_atom_tags = get_tagged_atoms_from_mols(products)

    if VERBOSE: print('Products contain {} tagged atoms'.format(len(prod_atoms)))
    if VERBOSE: print('Products contain {} unique atom numbers'.format(len(set(prod_atom_tags))))

    reac_atoms, reac_atom_tags = get_tagged_atoms_from_mols(reactants)
    if len(set(prod_atom_tags)) != len(set(reac_atom_tags)):
        if VERBOSE: print('warning: different atom tags appear in reactants and products')
        #err = 1 # okay for Reaxys, since Reaxys creates mass
    if len(prod_atoms) != len(reac_atoms):
        if VERBOSE: print('warning: total number of tagged atoms differ, stoichometry != 1?')
        #err = 1

    # Find differences 
    changed_atoms = [] # actual reactant atom species
    changed_atom_tags = [] # atom map numbers of those atoms

    # Product atoms that are different from reactant atom equivalent
    for i, prod_tag in enumerate(prod_atom_tags):

        for j, reac_tag in enumerate(reac_atom_tags):
            if reac_tag != prod_tag: continue
            if reac_tag not in changed_atom_tags: # don't bother comparing if we know this atom changes
                # If atom changed, add
                if atoms_are_different(prod_atoms[i], reac_atoms[j]):
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break
                # If reac_tag appears multiple times, add (need for stoichometry > 1)
                if prod_atom_tags.count(reac_tag) > 1:
                    changed_atoms.append(reac_atoms[j])
                    changed_atom_tags.append(reac_tag)
                    break

    # Reactant atoms that do not appear in product (tagged leaving groups)
    for j, reac_tag in enumerate(reac_atom_tags):
        if reac_tag not in changed_atom_tags:
            if reac_tag not in prod_atom_tags:
                changed_atoms.append(reac_atoms[j])
                changed_atom_tags.append(reac_tag)

    # Atoms that change CHIRALITY (just tetrahedral for now...)
    tetra_atoms = get_tetrahedral_atoms(reactants, products)
    if VERBOSE:
        print('Found {} atom-mapped tetrahedral atoms that have chirality specified at least partially'.format(len(tetra_atoms)))
    [set_isotope_to_equal_mapnum(reactant) for reactant in reactants]
    [set_isotope_to_equal_mapnum(product) for product in products]
    for (atom_tag, ar, ap) in tetra_atoms:
        if VERBOSE: 
            print('For atom tag {}'.format(atom_tag))
            print('    reactant: {}'.format(ar.GetChiralTag()))
            print('    product:  {}'.format(ap.GetChiralTag()))
        if atom_tag in changed_atom_tags:
            if VERBOSE:
                print('-> atoms have changed (by more than just chirality!)')
        else:
            unchanged = check_tetrahedral_centers_equivalent(ar, ap) and \
                    ChiralType.CHI_UNSPECIFIED not in [ar.GetChiralTag(), ap.GetChiralTag()]
            if unchanged:
                if VERBOSE: 
                    print('-> atoms confirmed to have same chirality, no change')
            else:
                if VERBOSE:
                    print('-> atom changed chirality!!')
                # Make sure chiral change is next to the reaction center and not
                # a random specifidation (must be CONNECTED to a changed atom)
                tetra_adj_to_rxn = False
                for neighbor in ap.GetNeighbors():
                    if neighbor.HasProp('molAtomMapNumber'):
                        if neighbor.GetProp('molAtomMapNumber') in changed_atom_tags:
                            tetra_adj_to_rxn = True
                            break
                if tetra_adj_to_rxn:
                    if VERBOSE:
                        print('-> atom adj to reaction center, now included')
                    changed_atom_tags.append(atom_tag)
                    changed_atoms.append(ar)
                else:
                    if VERBOSE:
                        print('-> adj far from reaction center, not including')
    [clear_isotope(reactant) for reactant in reactants]
    [clear_isotope(product) for product in products]


    if VERBOSE: 
        print('{} tagged atoms in reactants change 1-atom properties'.format(len(changed_atom_tags)))
        for smarts in [atom.GetSmarts() for atom in changed_atoms]:
            print('  {}'.format(smarts))

    missing_atom_tags = get_missing_atoms(reactants, products)
    changed_atom_tags = list(set(changed_atom_tags) | missing_atom_tags)
    
    return changed_atom_tags


def transform_tag_to_id(mol1, mol2, tags):
    idx_match = mol1.GetSubstructMatch(mol2)
    idx_mol2_to_mol1 = {mol2_idx: mol1_idx for mol2_idx, mol1_idx in enumerate(idx_match)}
    
    mol2_tag_to_idx = {}
    for atom in mol2.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            tag = str(atom.GetProp('molAtomMapNumber'))
            mol2_tag_to_idx[tag] = atom.GetIdx()

    mol1_idx_list = []
    for tag in tags:
        if tag in mol2_tag_to_idx:
            mol2_idx = mol2_tag_to_idx[tag]
            mol1_idx = idx_mol2_to_mol1[mol2_idx]
            mol1_idx_list.append(mol1_idx)

    return mol1_idx_list


def get_rdkit_mol(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    return mol


# def calc_aam(data_path, save_dir, append=True, rerun=False):
#     save_path = os.path.join(save_dir, 'rxn2aam.pkl')
#     if os.path.exists(save_path) and append and not rerun:
#         cached_rxn2aam = pkl.load(open(save_path, 'rb'))
#     else:
#         cached_rxn2aam = {}

#     rxn_mapper = BatchedMapper(batch_size=128)
    
#     df_data = pd.read_csv(data_path)
#     rxns_to_run = df_data['CANO_RXN_SMILES'].unique()
#     rxns_to_run = [rxn for rxn in rxns_to_run if rxn not in cached_rxn2aam]
    
#     result_list = []
#     for results in tqdm(rxn_mapper.map_reactions_with_info(rxns_to_run), total=len(rxns_to_run)):
#         result_list.append(results.get('mapped_rxn'))

#     rxn2aam = dict(zip(rxns_to_run, result_list))
#     rxn2aam.update(cached_rxn2aam)
    
#     with open(save_path, 'wb') as f:
#         pkl.dump(rxn2aam, f)



def calc_aam(data_path, save_dir, append=True, rerun=False):
    
    from localmapper import localmapper
    mapper = localmapper(device='cpu')

    save_path = os.path.join(save_dir, 'rxn2aam.pkl')
    if os.path.exists(save_path) and append and not rerun:
        cached_rxn2aam = pkl.load(open(save_path, 'rb'))
    else:
        cached_rxn2aam = {}
    
    df_data = pd.read_csv(data_path)
    rxns_to_run = df_data['CANO_RXN_SMILES'].unique()
    rxns_to_run = [rxn for rxn in rxns_to_run if rxn not in cached_rxn2aam]
    
    result_list = [mapper.get_atom_map(rxn) for rxn in tqdm(rxns_to_run)]

    rxn2aam = dict(zip(rxns_to_run, result_list))
    rxn2aam.update(cached_rxn2aam)
    
    with open(save_path, 'wb') as f:
        pkl.dump(rxn2aam, f)
        

def extract_reacting_center(rxn, rxn2aam):
    rxn_aam = rxn2aam.get(rxn)
    if isinstance(rxn_aam, list):
        rxn_aam = rxn_aam[0]
    if not rxn_aam:
        return [[], []]

    rcts_aam, prods_aam = rxn_aam.split('>>')
    rcts_aam_mol_list = [get_rdkit_mol(smi) for smi in rcts_aam.split('.')]
    prods_aam_mol_list = [get_rdkit_mol(smi) for smi in prods_aam.split('.')]

    rcts, prods = rxn.split('>>')
    rcts_mol_list = [get_rdkit_mol(smi) for smi in rcts.split('.')]
    prods_mol_list = [get_rdkit_mol(smi) for smi in prods.split('.')]

    changed_atom_tags = get_changed_atoms(rcts_aam_mol_list, prods_aam_mol_list)
    changed_atom_tags = set(changed_atom_tags)
    # print(f'Origin: {changed_atom_tags}')

    center_rcts = []
    start_idx = 0
    for rct_mol, rct_aam_mol in zip(rcts_mol_list, rcts_aam_mol_list):
        rct_center = transform_tag_to_id(rct_mol, rct_aam_mol, changed_atom_tags)
        rct_center = [idx + start_idx for idx in rct_center]
        center_rcts.extend(rct_center)
        start_idx += len(rct_mol.GetAtoms())
    
    center_prods = []
    start_idx = 0
    for prod_mol, prod_aam_mol in zip(prods_mol_list, prods_aam_mol_list):
        prod_center = transform_tag_to_id(prod_mol, prod_aam_mol, changed_atom_tags)
        prod_center = [idx + start_idx for idx in prod_center]
        center_prods.extend(prod_center)
        start_idx += len(prod_mol.GetAtoms())

    reaction_center = [center_rcts, center_prods]
    
    return reaction_center


if __name__=='__main__':
    data_path = '/home/liuy/data/RHEA/previous_versions/2025-02-05/tsv/rhea_rxn2uids.csv'
    save_dir = '/home/liuy/data/RHEA/previous_versions/2025-02-05/processed'
    calc_aam(data_path, save_dir)