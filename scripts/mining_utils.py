from Bio.PDB import MMCIFParser, PDBParser


THREE_TO_ONE = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'ILE': 'I',
    'LYS': 'K',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TRP': 'W',
    'TYR': 'Y',
}


def load_structure_residues(structure_name, structure_path):
    if structure_path.endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    elif structure_path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f'Unsupported structure format: {structure_path}')

    structure = parser.get_structure(structure_name, structure_path)
    return list(structure.get_residues())


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False):
    clean_res_list = []
    for residue in res_list:
        hetero, _, _ = residue.full_id[-1]
        if hetero != ' ':
            if verbose:
                print(residue, residue.full_id, 'is hetero')
            continue

        if residue.resname not in THREE_TO_ONE:
            if verbose:
                print(residue, 'has non-standard resname')
            continue

        if ensure_ca_exist and 'CA' not in residue:
            continue

        clean_res_list.append(residue)

    return clean_res_list


def get_gvp_compatible_residues(res_list, verbose=False):
    res_list = get_clean_res_list(res_list, verbose=verbose, ensure_ca_exist=True)
    return [residue for residue in res_list if ('N' in residue) and ('CA' in residue) and ('C' in residue) and ('O' in residue)]


def get_structure_sequence_and_gvp_residues(structure_path):
    residues = load_structure_residues('protein', structure_path)
    gvp_residues = get_gvp_compatible_residues(residues, verbose=False)
    structure_sequence = ''.join(THREE_TO_ONE[residue.resname] for residue in gvp_residues)
    return structure_sequence, gvp_residues
