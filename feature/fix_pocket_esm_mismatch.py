"""
Fix mismatch between PDB residue numbering and ESM sequence indexing.

Root cause:
- Residue numbers in PDB files may not start from 1 and may be discontinuous.
- ESM features are indexed by contiguous 0-based sequence positions (0, 1, 2, ...).
- `pocket_info.csv` stores original residue numbers from PDB files.
- Directly using those residue numbers can lead to out-of-range indexing.

Solution:
Build a mapping from PDB residue numbers to contiguous sequence indices.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from Bio.PDB import PDBParser

sys.path.append("../")
from enzymecage.base import UID_COL


def build_residue_mapping(pdb_file):
    """
    Build a mapping from PDB residue numbers to sequence indices.

    Returns:
        dict: {pdb_residue_number: sequence_index}
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residue_mapping = {}
    seq_index = 0

    for model in structure:
        for chain in model:
            # Only process chain A (protein chain)
            if chain.id != "A":
                continue
            for residue in chain.get_residues():
                # Only process standard amino-acid residues
                hetero_flag = residue.id[0]
                if hetero_flag == " ":  # standard amino-acid residue
                    pdb_res_num = residue.id[1]  # residue number in PDB
                    residue_mapping[pdb_res_num] = seq_index
                    seq_index += 1

    return residue_mapping


def get_esm_pocket_feature_fixed(
    pocket_info_path, pocket_pdb_dir, esm_node_feat_dir, save_path
):
    """
    Extract ESM pocket features with corrected residue indexing.

    Args:
        pocket_info_path: Path to pocket info CSV
        pocket_pdb_dir: Directory containing pocket PDB files
        esm_node_feat_dir: Directory containing ESM node features
        save_path: Output file path
    """
    print("\n", "=" * 20, "Extracting ESM pocket features (fixed)", "=" * 20, "\n")

    df_pocket_data = pd.read_csv(pocket_info_path)
    uids = [str(each) for each in df_pocket_data[UID_COL].values]
    uid_to_pocket = dict(zip(uids, df_pocket_data["pocket_residues"]))

    uid_to_pocket_node_feature = {}
    skipped_proteins = []

    for uid in tqdm(uids, desc="Processing proteins"):
        # Load ESM feature
        esm_file = os.path.join(esm_node_feat_dir, f"{uid}.npz")
        if not os.path.exists(esm_file):
            skipped_proteins.append((uid, "ESM feature file not found"))
            continue

        esm_node_feature = np.load(esm_file)["node_feature"]
        seq_length = esm_node_feature.shape[0]

        # Get pocket residue ids
        pocket_residue_ids = uid_to_pocket.get(uid)
        if not isinstance(pocket_residue_ids, str):
            skipped_proteins.append((uid, "Pocket residue info is empty"))
            continue

        # Load PDB file and build mapping
        pdb_file = os.path.join(pocket_pdb_dir, f"{uid}.pdb")
        if not os.path.exists(pdb_file):
            skipped_proteins.append((uid, "PDB file not found"))
            continue

        try:
            # Build mapping from residue number to sequence index
            residue_mapping = build_residue_mapping(pdb_file)

            # Convert PDB residue numbers to sequence indices
            pdb_res_numbers = [int(i) for i in pocket_residue_ids.split(",")]
            seq_indices = []

            for pdb_res_num in pdb_res_numbers:
                if pdb_res_num in residue_mapping:
                    seq_idx = residue_mapping[pdb_res_num]
                    if seq_idx < seq_length:  # keep indices in range
                        seq_indices.append(seq_idx)
                    else:
                        print(
                            f"[Warning] {uid}: sequence index {seq_idx} out of range {seq_length}"
                        )
                else:
                    print(f"[Warning] {uid}: no mapping found for PDB residue {pdb_res_num}")

            if len(seq_indices) == 0:
                skipped_proteins.append((uid, "No valid residue indices"))
                continue

            # Extract pocket node feature
            pocket_node_feature = esm_node_feature[seq_indices]
            uid_to_pocket_node_feature[uid] = pocket_node_feature

        except Exception as e:
            skipped_proteins.append((uid, f"Processing error: {str(e)}"))
            continue

    # Save results
    torch.save(uid_to_pocket_node_feature, save_path)
    print(
        f"\nSaved ESM pocket features for {len(uid_to_pocket_node_feature)} proteins to: {save_path}"
    )

    # Print skipped proteins
    if skipped_proteins:
        print(f"\nSkipped {len(skipped_proteins)} proteins:")
        for uid, reason in skipped_proteins[:10]:  # only show first 10
            print(f"  - {uid}: {reason}")
        if len(skipped_proteins) > 10:
            print(f"  ... and {len(skipped_proteins) - 10} more")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix index mismatch when extracting ESM pocket features"
    )
    parser.add_argument(
        "--pocket_info", type=str, required=True, help="Path to pocket info CSV"
    )
    parser.add_argument(
        "--pocket_pdb_dir", type=str, required=True, help="Directory of pocket PDB files"
    )
    parser.add_argument(
        "--esm_node_feat_dir", type=str, required=True, help="Directory of ESM node features"
    )
    parser.add_argument("--save_path", type=str, required=True, help="Output file path")

    args = parser.parse_args()

    get_esm_pocket_feature_fixed(
        args.pocket_info, args.pocket_pdb_dir, args.esm_node_feat_dir, args.save_path
    )


if __name__ == "__main__":
    main()
