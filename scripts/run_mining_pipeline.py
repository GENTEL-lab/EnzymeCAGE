import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_command(cmd, cwd=ROOT_DIR):
    print('Running:', ' '.join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd))


def write_infer_config(data_path, result_dir, checkpoint_dir, model_name):
    data_path = Path(data_path).resolve()
    data_dir = data_path.parent
    config_path = data_dir / 'infer_p2rank.yaml'

    config = {
        'model': 'EnzymeCAGE',
        'interaction_method': 'geo-enhanced-interaction',
        'rxn_inner_interaction': True,
        'pocket_inner_interaction': True,
        'use_prods_info': False,
        'use_structure': True,
        'use_drfp': True,
        'use_esm': True,
        'esm_model': 'ESM-C_600M',
        'batch_size': 64,
        'model_list': [model_name],
        'data_path': str(data_path),
        'ckpt_dir': str(Path(checkpoint_dir).resolve()),
        'result_dir': str(Path(result_dir).resolve()),
        'rxn_fp': str((data_dir / 'feature' / 'reaction' / 'drfp' / 'rxn2fp.pkl').resolve()),
        'mol_conformation': str((data_dir / 'feature' / 'reaction' / 'molecule_conformation').resolve()),
        'reaction_center': str((data_dir / 'feature' / 'reaction' / 'reacting_center' / 'reacting_center.pkl').resolve()),
        'protein_gvp_feat': str((data_dir / 'feature' / 'protein' / 'gvp_feature' / 'gvp_protein_feature.pt').resolve()),
        'esm_mean_feature': str((data_dir / 'feature' / 'protein' / 'ESM-C_600M' / 'protein_level' / 'seq2feature.pkl').resolve()),
        'esm_node_feature': str((data_dir / 'feature' / 'protein' / 'ESM-C_600M' / 'pocket_node_feature' / 'esm_node_feature.pt').resolve()),
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    return config_path


def rank_predictions(result_dir, data_path, model_name):
    result_dir = Path(result_dir).resolve()
    result_path = result_dir / f'{Path(data_path).stem}_{model_name.replace(".pth", ".csv")}'
    if not result_path.exists():
        raise FileNotFoundError(f'Inference result not found: {result_path}')

    df_result = pd.read_csv(result_path)
    if 'pred' not in df_result.columns:
        raise ValueError(f'Prediction column `pred` not found in {result_path}')

    df_ranked = df_result.sort_values('pred', ascending=False).reset_index(drop=True)
    df_ranked.insert(0, 'rank', range(1, len(df_ranked) + 1))
    ranked_path = result_dir / f'{result_path.stem}_ranked.csv'
    df_ranked.to_csv(ranked_path, index=False)
    return ranked_path


def main():
    parser = argparse.ArgumentParser(description='Run the end-to-end EnzymeCAGE mining pipeline with P2Rank pockets.')
    parser.add_argument('--data_dir', required=True, help='Dataset directory, e.g. dataset/demo')
    parser.add_argument('--p2rank_home', required=True, help='Path to the extracted P2Rank directory')
    parser.add_argument('--input_csv', help='Optional prebuilt EnzymeCAGE pair CSV')
    parser.add_argument('--reaction_path', help='Optional reaction CSV. Default: <data_dir>/reaction.csv')
    parser.add_argument('--structure_dir', help='Optional structure directory. Default: <data_dir>/structures')
    parser.add_argument('--checkpoint_dir', default='checkpoints/full/seed_42', help='Checkpoint directory used by infer.py')
    parser.add_argument('--model_name', default='best_model.pth', help='Checkpoint filename inside --checkpoint_dir')
    parser.add_argument('--threads', type=int, default=4, help='Number of P2Rank threads')
    parser.add_argument('--java_home', help='Optional JAVA_HOME to use for running P2Rank')
    parser.add_argument('--skip_prepare_csv', action='store_true', help='Skip generating the pair CSV')
    parser.add_argument('--skip_p2rank', action='store_true', help='Skip the P2Rank pocket extraction step')
    parser.add_argument('--skip_feature', action='store_true', help='Skip the feature generation step')
    parser.add_argument('--skip_infer', action='store_true', help='Skip the inference step')
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    structure_dir = Path(args.structure_dir).resolve() if args.structure_dir else data_dir / 'structures'
    reaction_path = Path(args.reaction_path).resolve() if args.reaction_path else data_dir / 'reaction.csv'
    input_csv = Path(args.input_csv).resolve() if args.input_csv else data_dir / 'mining.csv'
    pocket_root = data_dir / 'pocket' / 'p2rank'
    pocket_dir = pocket_root / 'pocket'
    pocket_info_path = pocket_root / 'pocket_info.csv'
    result_dir = data_dir / 'predictions'

    if not args.skip_prepare_csv:
        run_command(
            [
                sys.executable,
                str(ROOT_DIR / 'scripts' / 'prepare_mining_input.py'),
                '--reaction_path',
                str(reaction_path),
                '--structure_dir',
                str(structure_dir),
                '--output_csv',
                str(input_csv),
            ]
        )

    if not args.skip_p2rank:
        cmd = [
            sys.executable,
            str(ROOT_DIR / 'scripts' / 'extract_p2rank_pockets.py'),
            '--input_csv',
            str(input_csv),
            '--structure_dir',
            str(structure_dir),
            '--p2rank_home',
            str(Path(args.p2rank_home).resolve()),
            '--output_dir',
            str(pocket_root),
            '--threads',
            str(args.threads),
        ]
        if args.java_home:
            cmd.extend(['--java_home', args.java_home])
        run_command(cmd)

    if not args.skip_feature:
        run_command(
            [
                sys.executable,
                str(ROOT_DIR / 'feature' / 'main.py'),
                '--data_path',
                str(input_csv),
                '--pocket_dir',
                str(pocket_dir),
                '--pocket_info_path',
                str(pocket_info_path),
            ],
            cwd=ROOT_DIR / 'feature',
        )

    if not args.skip_infer:
        config_path = write_infer_config(input_csv, result_dir, args.checkpoint_dir, args.model_name)
        run_command(
            [
                sys.executable,
                str(ROOT_DIR / 'infer.py'),
                '--config',
                str(config_path),
            ]
        )
        ranked_path = rank_predictions(result_dir, input_csv, args.model_name)
        print(f'Ranked prediction table saved to {ranked_path}')


if __name__ == '__main__':
    main()
