import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / 'scripts'))

from mining_utils import get_structure_sequence_and_gvp_residues  # noqa: E402


SUPPORTED_EXTENSIONS = ('.pdb', '.cif')


class ResidueSelect(Select):
    def __init__(self, selected_keys):
        self.selected_keys = selected_keys

    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        return (chain_id, residue.id) in self.selected_keys


def detect_java_bin(java_home=None):
    if java_home:
        return str(Path(java_home) / 'bin' / 'java')
    return 'java'


def check_java_version(java_bin):
    result = subprocess.run([java_bin, '-version'], capture_output=True, text=True)
    output = (result.stderr or '') + (result.stdout or '')
    if result.returncode != 0:
        raise RuntimeError(f'Failed to run `{java_bin} -version`:\n{output}')

    match = re.search(r'version "(\d+)', output)
    if not match:
        raise RuntimeError(f'Failed to parse Java version from:\n{output}')

    version = int(match.group(1))
    if version < 17:
        raise RuntimeError(
            f'P2Rank 2.5.1 requires Java 17+, but `{java_bin}` is version {version}. '
            'Please switch JAVA_HOME or install a newer Java runtime.'
        )


def resolve_reaction_subset(input_csv):
    if not input_csv:
        return None

    df_input = pd.read_csv(input_csv)
    if 'UniprotID' not in df_input.columns:
        raise ValueError(f'UniprotID column not found in {input_csv}')
    return set(df_input['UniprotID'].dropna().astype(str).tolist())


def list_structure_files(structure_dir, allowed_uids=None):
    structure_dir = Path(structure_dir)
    structure_files = {}
    for ext in SUPPORTED_EXTENSIONS:
        for structure_path in sorted(structure_dir.glob(f'*{ext}')):
            uid = structure_path.stem
            if allowed_uids is not None and uid not in allowed_uids:
                continue
            structure_files[uid] = structure_path.resolve()

    if not structure_files:
        raise FileNotFoundError(f'No usable structure files found in {structure_dir}')

    return structure_files


def write_p2rank_dataset(structure_files, dataset_path):
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for structure_path in structure_files.values():
            f.write(f'{structure_path}\n')


def resolve_prank_executable(p2rank_home):
    prank_path = Path(p2rank_home) / 'prank'
    if not prank_path.exists():
        raise FileNotFoundError(f'P2Rank executable not found: {prank_path}')
    return prank_path.resolve()


def run_p2rank(structure_files, p2rank_home, output_dir, threads=4, java_home=None):
    prank_path = resolve_prank_executable(p2rank_home)
    dataset_path = output_dir / 'p2rank_input.ds'
    raw_output_dir = output_dir / 'raw'
    write_p2rank_dataset(structure_files, dataset_path)

    env = os.environ.copy()
    if java_home:
        env['JAVA_HOME'] = str(Path(java_home).resolve())

    cmd = [
        str(prank_path),
        'predict',
        '-threads',
        str(threads),
        '-c',
        'alphafold',
        '-visualizations',
        '0',
        '-o',
        str(raw_output_dir),
        str(dataset_path),
    ]
    subprocess.run(cmd, check=True, env=env)
    return raw_output_dir


def resolve_prediction_file(prediction_dir, uid, suffix):
    exact_path = prediction_dir / f'{uid}{suffix}'
    if exact_path.exists():
        return exact_path

    matches = sorted(prediction_dir.glob(f'{uid}*{suffix}'))
    if matches:
        return matches[0]

    raise FileNotFoundError(f'Failed to locate {uid}{suffix} under {prediction_dir}')


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def build_structure_mappings(structure_path):
    _, gvp_residues = get_structure_sequence_and_gvp_residues(str(structure_path))
    chain_residue_to_position = {}
    residue_label_to_positions = {}
    position_to_residue = {}

    for position, residue in enumerate(gvp_residues, start=1):
        chain_id = residue.get_parent().id
        residue_label = format_residue_label(residue.id)
        key = (str(chain_id).strip(), residue_label)
        chain_residue_to_position[key] = position
        residue_label_to_positions.setdefault(residue_label, []).append(position)
        position_to_residue[position] = residue

    return chain_residue_to_position, residue_label_to_positions, position_to_residue


def format_residue_label(residue_id):
    residue_number = str(residue_id[1])
    insertion_code = residue_id[2].strip()
    return f'{residue_number}{insertion_code}'


def resolve_position(chain_id, residue_label, chain_residue_to_position, residue_label_to_positions):
    key = (chain_id, residue_label)
    if key in chain_residue_to_position:
        return chain_residue_to_position[key]

    positions = residue_label_to_positions.get(residue_label, [])
    if len(positions) == 1:
        return positions[0]

    raise KeyError(f'Unable to map residue {chain_id}:{residue_label} to a unique sequence position')


def load_pocket_positions(residue_csv_path, structure_path, pocket_rank):
    df_residue = normalize_columns(pd.read_csv(residue_csv_path, skipinitialspace=True))
    required_columns = {'chain', 'residue_label', 'pocket'}
    if not required_columns.issubset(df_residue.columns):
        raise ValueError(
            f'{residue_csv_path} is missing required columns: {sorted(required_columns - set(df_residue.columns))}'
        )

    df_residue['pocket'] = pd.to_numeric(df_residue['pocket'], errors='coerce')
    pocket_rows = df_residue[df_residue['pocket'] == pocket_rank]
    if pocket_rows.empty:
        raise ValueError(f'No residues assigned to pocket rank {pocket_rank} in {residue_csv_path}')

    chain_residue_to_position, residue_label_to_positions, position_to_residue = build_structure_mappings(structure_path)
    selected_positions = []
    for _, row in pocket_rows.iterrows():
        chain_id = str(row['chain']).strip()
        residue_label = str(row['residue_label']).strip()
        selected_positions.append(
            resolve_position(chain_id, residue_label, chain_residue_to_position, residue_label_to_positions)
        )

    selected_positions = sorted(set(selected_positions))
    selected_keys = {
        (position_to_residue[position].get_parent().id, position_to_residue[position].id)
        for position in selected_positions
    }
    return selected_positions, selected_keys


def load_prediction_metadata(prediction_csv_path, pocket_rank):
    if not prediction_csv_path.exists():
        return {}

    df_prediction = normalize_columns(pd.read_csv(prediction_csv_path, skipinitialspace=True))
    if 'rank' not in df_prediction.columns:
        return {}

    df_prediction['rank'] = pd.to_numeric(df_prediction['rank'], errors='coerce')
    pocket_rows = df_prediction[df_prediction['rank'] == pocket_rank]
    if pocket_rows.empty:
        return {}

    row = pocket_rows.iloc[0].to_dict()
    metadata = {}
    for column in ['name', 'score', 'probability', 'center_x', 'center_y', 'center_z', 'residue_ids']:
        if column in row:
            metadata[column] = row[column]
    return metadata


def load_structure(structure_path):
    if structure_path.suffix == '.pdb':
        parser = PDBParser(QUIET=True)
    elif structure_path.suffix == '.cif':
        parser = MMCIFParser(QUIET=True)
    else:
        raise ValueError(f'Unsupported structure format: {structure_path}')
    return parser.get_structure(structure_path.stem, str(structure_path))


def save_pocket_structure(structure_path, selected_keys, output_path):
    structure = load_structure(structure_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), ResidueSelect(selected_keys))


def extract_pockets(structure_files, raw_output_dir, output_dir, pocket_rank):
    prediction_dir = raw_output_dir / 'predictions'
    if not prediction_dir.exists():
        prediction_dir = raw_output_dir
    pocket_dir = output_dir / 'pocket'
    pocket_info_rows = []
    failed_rows = []

    for uid, structure_path in tqdm(structure_files.items(), desc='Extracting P2Rank pockets'):
        try:
            residue_csv_path = resolve_prediction_file(prediction_dir, uid, '_residues.csv')
            prediction_csv_path = resolve_prediction_file(prediction_dir, uid, '_predictions.csv')
            selected_positions, selected_keys = load_pocket_positions(residue_csv_path, structure_path, pocket_rank)
            pocket_path = pocket_dir / f'{uid}.pdb'
            save_pocket_structure(structure_path, selected_keys, pocket_path)

            row = {
                'UniprotID': uid,
                'pocket_residues': ','.join(str(position) for position in selected_positions),
                'structure_path': str(structure_path),
                'pocket_path': str(pocket_path.resolve()),
                'pocket_rank': pocket_rank,
            }
            row.update(load_prediction_metadata(prediction_csv_path, pocket_rank))
            pocket_info_rows.append(row)
        except Exception as exc:
            failed_rows.append({'UniprotID': uid, 'error': str(exc)})

    pocket_info_path = output_dir / 'pocket_info.csv'
    pd.DataFrame(pocket_info_rows).to_csv(pocket_info_path, index=False)
    print(f'Saved pocket info for {len(pocket_info_rows)} proteins to {pocket_info_path}')

    if failed_rows:
        failed_path = output_dir / 'failed_p2rank_pockets.csv'
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
        print(f'Failed to extract pockets for {len(failed_rows)} proteins. Saved to {failed_path}')
    else:
        failed_path = output_dir / 'failed_p2rank_pockets.csv'
        if failed_path.exists():
            failed_path.unlink()

    return pocket_info_path


def main():
    parser = argparse.ArgumentParser(description='Run P2Rank on candidate structures and extract top-ranked pocket PDBs.')
    parser.add_argument('--structure_dir', required=True, help='Directory with candidate structures named as <UniprotID>.pdb/.cif')
    parser.add_argument('--output_dir', required=True, help='Output directory, e.g. dataset/demo/pocket/p2rank')
    parser.add_argument('--p2rank_home', required=True, help='Path to the extracted P2Rank directory')
    parser.add_argument('--input_csv', help='Optional EnzymeCAGE pair CSV. If set, only process UIDs appearing in this file')
    parser.add_argument('--threads', type=int, default=4, help='Number of P2Rank threads')
    parser.add_argument('--pocket_rank', type=int, default=1, help='Pocket rank to extract from P2Rank outputs')
    parser.add_argument('--java_home', help='Optional JAVA_HOME to use for running P2Rank')
    args = parser.parse_args()

    java_bin = detect_java_bin(args.java_home)
    check_java_version(java_bin)

    allowed_uids = resolve_reaction_subset(args.input_csv)
    structure_files = list_structure_files(args.structure_dir, allowed_uids=allowed_uids)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_output_dir = run_p2rank(
        structure_files,
        args.p2rank_home,
        output_dir,
        threads=args.threads,
        java_home=args.java_home,
    )
    extract_pockets(structure_files, raw_output_dir, output_dir, args.pocket_rank)


if __name__ == '__main__':
    main()
