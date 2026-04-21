import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / 'scripts'))

from mining_utils import get_structure_sequence_and_gvp_residues  # noqa: E402


SUPPORTED_EXTENSIONS = ('.pdb', '.cif')


def resolve_reaction_column(df_reaction, reaction_column=None):
    if reaction_column:
        if reaction_column not in df_reaction.columns:
            raise ValueError(f'Reaction column not found: {reaction_column}')
        return reaction_column

    for candidate in ['CANO_RXN_SMILES', 'reaction']:
        if candidate in df_reaction.columns:
            return candidate

    if len(df_reaction.columns) == 1:
        return df_reaction.columns[0]

    raise ValueError(
        'Unable to determine the reaction column. Please provide --reaction_column '
        'or make sure the file contains CANO_RXN_SMILES.'
    )


def list_structure_files(structure_dir):
    structure_dir = Path(structure_dir)
    if not structure_dir.exists():
        raise FileNotFoundError(f'Structure directory not found: {structure_dir}')

    structure_files = []
    for ext in SUPPORTED_EXTENSIONS:
        structure_files.extend(sorted(structure_dir.glob(f'*{ext}')))

    if not structure_files:
        raise FileNotFoundError(
            f'No structure files found in {structure_dir}. Supported extensions: {SUPPORTED_EXTENSIONS}'
        )

    return structure_files


def build_candidate_enzyme_table(structure_dir):
    rows = []
    for structure_path in tqdm(list_structure_files(structure_dir), desc='Reading structures'):
        uid = structure_path.stem
        sequence, _ = get_structure_sequence_and_gvp_residues(str(structure_path))
        if not sequence:
            raise ValueError(f'Failed to extract a usable sequence from {structure_path}')
        rows.append(
            {
                'UniprotID': uid,
                'sequence': sequence,
                'structure_path': str(structure_path.resolve()),
            }
        )

    return pd.DataFrame(rows).drop_duplicates(subset=['UniprotID']).sort_values('UniprotID').reset_index(drop=True)


def build_pair_table(df_reaction, df_enzyme, reaction_column):
    reactions = (
        df_reaction[reaction_column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if not reactions:
        raise ValueError('No reactions found in the reaction file.')

    pair_rows = []
    for reaction in reactions:
        for _, enzyme_row in df_enzyme.iterrows():
            pair_rows.append(
                {
                    'UniprotID': enzyme_row['UniprotID'],
                    'sequence': enzyme_row['sequence'],
                    'CANO_RXN_SMILES': reaction,
                    'Label': 0,
                }
            )

    return pd.DataFrame(pair_rows)


def main():
    parser = argparse.ArgumentParser(description='Prepare a mining input CSV from reactions and candidate structures.')
    parser.add_argument('--reaction_path', required=True, help='CSV file containing CANO_RXN_SMILES or a single reaction column')
    parser.add_argument('--structure_dir', required=True, help='Directory with candidate enzyme structures named as <UniprotID>.pdb/.cif')
    parser.add_argument('--output_csv', required=True, help='Output pair CSV for EnzymeCAGE inference')
    parser.add_argument('--reaction_column', help='Optional reaction column name')
    parser.add_argument(
        '--enzyme_csv',
        help='Optional path for saving the extracted candidate enzyme table. Default: <output_dir>/candidate_enzymes.csv',
    )
    args = parser.parse_args()

    df_reaction = pd.read_csv(args.reaction_path)
    reaction_column = resolve_reaction_column(df_reaction, args.reaction_column)
    df_enzyme = build_candidate_enzyme_table(args.structure_dir)
    df_pair = build_pair_table(df_reaction, df_enzyme, reaction_column)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_pair.to_csv(output_csv, index=False)

    enzyme_csv = Path(args.enzyme_csv) if args.enzyme_csv else output_csv.parent / 'candidate_enzymes.csv'
    df_enzyme.to_csv(enzyme_csv, index=False)

    print(f'Saved {len(df_pair)} enzyme-reaction pairs to {output_csv}')
    print(f'Saved {len(df_enzyme)} candidate enzymes to {enzyme_csv}')


if __name__ == '__main__':
    main()
