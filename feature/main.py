import argparse
import os
import sys
import pickle as pkl
from multiprocessing import Pool
from functools import partial

sys.path.append("../")
sys.path.append("./pkgs/")

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import esm
from Bio import PDB

from enzymecage.base import UID_COL, RXN_COL, SEQ_COL
from utils import check_dir, cano_rxn, tranverse_folder
from gvp_torchdrug_feature import calc_gvp_feature
from extract_pocket import (
    get_pocket_info,
    extract_fix_num_residues,
    get_pocket_info_batch_new,
)
from extract_reacting_center import extract_reacting_center, calc_aam


def calc_drfp(data_path, save_path, append=True):
    print("\n", "=" * 20, "Calculating DRFP", "=" * 20, "\n")
    from drfp import DrfpEncoder

    df_data = pd.read_csv(data_path)
    rxn_list = list(set(df_data[RXN_COL]))

    if os.path.exists(save_path) and append:
        rxn_to_fp = pkl.load(open(save_path, "rb"))
    else:
        rxn_to_fp = {}

    input_list = []
    for rxn in rxn_list:
        input_list.append([rxn])
        rxn_cano = cano_rxn(rxn, remove_stereo=True)
        if rxn_cano != rxn:
            input_list.append([rxn_cano])

    drfp_results = []
    with Pool(20) as p:
        for fp in tqdm(p.imap(DrfpEncoder.encode, input_list), total=len(input_list)):
            drfp_results.append(fp[0])

    for rxn, fp in zip(input_list, drfp_results):
        rxn_to_fp[rxn[0]] = fp
    print(f"Number of reactions: {len(rxn_to_fp)}")

    check_dir(os.path.dirname(save_path))
    pkl.dump(rxn_to_fp, open(save_path, "wb"))
    print(f"Save drfp to {save_path}")


def batch_generator(sequence_list, batch_size):
    for i in range(0, len(sequence_list), batch_size):
        yield sequence_list[i : i + batch_size]


def calc_seq_esm_C_feature(data_path, esm_node_feat_dir, esm_mean_feat_path):
    print("\n", "#" * 20, "Calculating ESM-C feature", "#" * 20, "\n")

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    # Load ESM-C model
    device = "cuda"
    model = ESMC.from_pretrained("esmc_600m").to(device)  # or "cpu"

    df_data = pd.read_csv(data_path)
    uid_to_seq = dict(zip(df_data[UID_COL], df_data[SEQ_COL]))
    os.makedirs(esm_node_feat_dir, exist_ok=True)
    os.makedirs(os.path.dirname(esm_mean_feat_path), exist_ok=True)

    # To ensure the order
    df_data = df_data.drop_duplicates(UID_COL)

    uid_list = []
    for uid in df_data[UID_COL].tolist():
        seq = uid_to_seq.get(uid)
        save_path = os.path.join(esm_node_feat_dir, f"{uid}.npz")

        # skip if already exists
        if os.path.exists(save_path):
            continue
        uid_list.append(uid)
    print(f"\n{len(uid_list)} proteins to calculate features...")

    cnt_fail = 0
    cnt_all = 0
    failed_seqs = []
    failed_uids = []

    if os.path.exists(esm_mean_feat_path):
        seq_to_feature = pkl.load(open(esm_mean_feat_path, "rb"))
    else:
        seq_to_feature = {}

    for uid in tqdm(uid_list):
        seq = uid_to_seq.get(uid)

        save_path = os.path.join(esm_node_feat_dir, f"{uid}.npz")
        if os.path.exists(save_path):
            continue

        protein = ESMProtein(sequence=seq)

        # Extract per-residue representations
        try:
            with torch.no_grad():
                protein_tensor = model.encode(protein)
                logits_output = model.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
        except Exception as e:
            print(f"sequence length: {len(seq)}")
            print(e)
            cnt_fail += 1
            failed_seqs.append(seq)
            failed_uids.append(uid)
            continue

        # dim: (n,1152)
        node_feature = logits_output.embeddings[0].cpu()

        np.savez_compressed(save_path, node_feature=node_feature)

        seq_to_feature[seq] = node_feature.mean(axis=0)

        cnt_all += 1

    with open(esm_mean_feat_path, "wb") as f:
        pkl.dump(seq_to_feature, f)

    print(f"\ncnt_fail: {cnt_fail}")
    df_failed = pd.DataFrame({UID_COL: failed_uids, SEQ_COL: failed_seqs})
    failed_save_path = os.path.join(esm_node_feat_dir, "failed_proteins.csv")
    df_failed.to_csv(failed_save_path, index=False)
    print(f"Save failed proteins to {failed_save_path}")


def calc_seq_esm_feature(data_path, esm_node_feat_dir, esm_mean_feat_path):
    print("\n", "#" * 20, "Calculating ESM feature", "#" * 20, "\n")
    # Load ESM-2 model
    device = "cuda"

    print("Loading ESM-2 model...")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)
    print("Molel loading done!")

    df_data = pd.read_csv(data_path)
    uid_to_seq = dict(zip(df_data[UID_COL], df_data[SEQ_COL]))
    os.makedirs(esm_node_feat_dir, exist_ok=True)
    os.makedirs(os.path.dirname(esm_mean_feat_path), exist_ok=True)

    # To ensure the order
    df_data = df_data.drop_duplicates(UID_COL)

    uid_list = []
    for uid in df_data[UID_COL].tolist():
        seq = uid_to_seq.get(uid)
        save_path = os.path.join(esm_node_feat_dir, f"{uid}.npz")

        # skip if already exists
        if os.path.exists(save_path):
            continue
        uid_list.append(uid)
    print(f"\n{len(uid_list)} proteins to calculate features...")

    cnt_fail = 0
    cnt_all = 0
    failed_seqs = []
    failed_uids = []

    if os.path.exists(esm_mean_feat_path):
        seq_to_feature = pkl.load(open(esm_mean_feat_path, "rb"))
    else:
        seq_to_feature = {}

    for uid in tqdm(uid_list):
        seq = uid_to_seq.get(uid)

        save_path = os.path.join(esm_node_feat_dir, f"{uid}.npz")
        if os.path.exists(save_path):
            continue

        input_data = [(f"seq{cnt_all}", seq)]

        batch_labels, batch_strs, batch_tokens = batch_converter(input_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        batch_tokens = batch_tokens.to(device)

        # Extract per-residue representations
        try:
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
        except Exception as e:
            print(f"sequence length: {len(seq)}")
            print(e)
            cnt_fail += 1
            failed_seqs.append(seq)
            failed_uids.append(uid)
            continue

        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(
                token_representations[i, 1 : tokens_len - 1].cpu().numpy()
            )
        node_feature = sequence_representations[0]

        np.savez_compressed(save_path, node_feature=node_feature)

        seq_to_feature[seq] = node_feature.mean(axis=0)

        cnt_all += 1

    with open(esm_mean_feat_path, "wb") as f:
        pkl.dump(seq_to_feature, f)

    print(f"\ncnt_fail: {cnt_fail}")
    df_failed = pd.DataFrame({UID_COL: failed_uids, SEQ_COL: failed_seqs})
    failed_save_path = os.path.join(esm_node_feat_dir, "failed_proteins.csv")
    df_failed.to_csv(failed_save_path, index=False)
    print(f"Save failed proteins to {failed_save_path}")


def generate_rdkit_conformation_v2(smiles, n_repeat=50):
    try:
        mol = Chem.MolFromSmiles(smiles)
        # mol = Chem.RemoveAllHs(mol)
        # mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv2()
        # rid = AllChem.EmbedMolecule(mol, ps)
        for repeat in range(n_repeat):
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == 0:
                break
        if rid == -1:
            # print("rid", pdb, rid)
            ps.useRandomCoords = True
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == -1:
                mol.Compute2DCoords()
            else:
                AllChem.MMFFOptimizeMolecule(mol, confId=0)
        else:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
    except Exception as e:
        print(e)
        mol = None
    # mol = Chem.RemoveAllHs(mol)
    return mol


def generate_mol_conformation(data_path, save_dir):
    print("\n", "#" * 20, "Generating Mol Conformation", "#" * 20, "\n")
    df_data = pd.read_csv(data_path)
    rxns = set(df_data[RXN_COL])

    all_smiles_list = []
    for rxn in rxns:
        smiles_list = rxn.split(">>")[0].split(".") + rxn.split(">>")[1].split(".")
        all_smiles_list.extend(smiles_list)
    all_smiles_list = [smi.replace("*", "C") for smi in all_smiles_list]
    all_mols = list(set(all_smiles_list))

    mol_index_path = os.path.join(save_dir, "mol2id.csv")
    if not os.path.exists(mol_index_path):
        df_all_mols = pd.DataFrame({"SMILES": all_mols, "ID": range(len(all_mols))})
        df_all_mols.to_csv(mol_index_path, index=False)
    else:
        df_all_mols_old = pd.read_csv(mol_index_path)
        existing_mols = set(df_all_mols_old["SMILES"])
        # Incremental generation, not just re-generate all
        mols_to_generate = [smi for smi in all_mols if smi not in existing_mols]
        start_idx = df_all_mols_old["ID"].max() + 1
        # df_all_mols is mols to generate
        df_all_mols = pd.DataFrame(
            {
                "SMILES": mols_to_generate,
                "ID": range(start_idx, start_idx + len(mols_to_generate)),
            }
        )
        # df_to_save is all mols info
        df_to_save = pd.concat([df_all_mols_old, df_all_mols])
        df_to_save.to_csv(mol_index_path, index=False)

    failed_idx_list = []
    for smiles, idx in tqdm(df_all_mols[["SMILES", "ID"]].values):
        save_path = os.path.join(save_dir, f"{idx}.sdf")
        conformation = generate_rdkit_conformation_v2(smiles)
        if not conformation:
            failed_idx_list.append(idx)
            continue
        Chem.SDWriter(save_path).write(conformation)


def get_pocket_info_batch(
    input_dir, save_path, pocket_save_dir=None, max_residue_num=None
):
    filelist = [path for path in tranverse_folder(input_dir) if path.endswith(".cif")]
    uid_list = [
        os.path.basename(each).replace("_transplant.cif", "") for each in filelist
    ]

    if max_residue_num is None:
        running_func = partial(get_pocket_info, pocket_save_dir=pocket_save_dir)
    else:
        running_func = partial(
            extract_fix_num_residues,
            pocket_save_dir=pocket_save_dir,
            residue_num=max_residue_num,
        )

    pocket_info_list = []
    with Pool(20) as pool:
        for res in tqdm(pool.imap(running_func, filelist), total=len(filelist)):
            pocket_info_list.append(res)
    df_pocket_info = pd.DataFrame(
        {UID_COL: uid_list, "pocket_residues": pocket_info_list}
    )
    df_pocket_info.to_csv(save_path, index=False)
    print(f"\nSave pocket info to {save_path}\n")

    delete_empty(pocket_save_dir)


def build_residue_mapping(pdb_file):
    """
    从PDB文件建立残基编号到序列索引的映射

    Returns:
        dict: {pdb_residue_number: sequence_index}
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residue_mapping = {}
    seq_index = 0

    for model in structure:
        for chain in model:
            # 只处理链A（蛋白质链）
            if chain.id != "A":
                continue
            for residue in chain.get_residues():
                # 只处理标准氨基酸残基
                hetero_flag = residue.id[0]
                if hetero_flag == " ":  # 标准氨基酸
                    pdb_res_num = residue.id[1]  # PDB中的残基编号
                    residue_mapping[pdb_res_num] = seq_index
                    seq_index += 1

    return residue_mapping


def get_esm_pocket_feature(
    pocket_info_path, esm_node_feat_dir, save_path, pocket_pdb_dir=None
):
    """
    提取ESM pocket特征，修复了PDB残基编号与序列索引不匹配的问题

    Args:
        pocket_info_path: pocket信息CSV文件路径
        esm_node_feat_dir: ESM node特征目录
        save_path: 保存路径
        pocket_pdb_dir: pocket PDB文件目录（可选，如果提供则使用残基映射）
    """
    df_pocket_data = pd.read_csv(pocket_info_path)
    uids = [str(each) for each in df_pocket_data[UID_COL].values]
    uid_to_pocket = dict(zip(uids, df_pocket_data["pocket_residues"]))
    esm_file_list = [
        each for each in tranverse_folder(esm_node_feat_dir) if each.endswith(".npz")
    ]

    uid_to_pocket_node_feature = {}
    skipped_count = 0

    for filepath in tqdm(esm_file_list):
        uid = os.path.basename(filepath).replace(".npz", "")
        if uid in uid_to_pocket_node_feature:
            continue
        if not filepath.endswith("npz"):
            continue

        esm_node_feature = np.load(filepath)["node_feature"]
        seq_length = esm_node_feature.shape[0]

        pocket_residue_ids = uid_to_pocket.get(uid)
        if not isinstance(pocket_residue_ids, str):
            continue

        # 如果提供了pocket_pdb_dir，使用残基映射来修复索引问题
        if pocket_pdb_dir:
            pdb_file = os.path.join(pocket_pdb_dir, f"{uid}.pdb")
            if os.path.exists(pdb_file):
                try:
                    # 建立残基编号到序列索引的映射
                    residue_mapping = build_residue_mapping(pdb_file)

                    # 将PDB残基编号转换为序列索引
                    pdb_res_numbers = [int(i) for i in pocket_residue_ids.split(",")]
                    seq_indices = []

                    for pdb_res_num in pdb_res_numbers:
                        if pdb_res_num in residue_mapping:
                            seq_idx = residue_mapping[pdb_res_num]
                            if seq_idx < seq_length:
                                seq_indices.append(seq_idx)
                            else:
                                print(
                                    f"[Warning] UID: {uid} 序列索引 {seq_idx} 超出范围 {seq_length}. 已跳过该残基。"
                                )
                        else:
                            # PDB残基编号未找到映射，跳过
                            pass

                    if len(seq_indices) == 0:
                        print(
                            f"[Warning] UID: {uid} 没有有效的残基索引. 已跳过该蛋白。"
                        )
                        skipped_count += 1
                        continue

                    pocket_node_feature = esm_node_feature[seq_indices]

                except Exception as e:
                    print(f"[Warning] UID: {uid} 处理错误: {str(e)}. 已跳过该蛋白。")
                    skipped_count += 1
                    continue
            else:
                # PDB文件不存在，跳过
                skipped_count += 1
                continue
        else:
            # 使用旧方法（减1）- 可能导致索引越界
            pocket_residue_ids = [int(i) - 1 for i in pocket_residue_ids.split(",")]
            max_idx = max(pocket_residue_ids) if pocket_residue_ids else 0

            if max_idx >= seq_length:
                print(
                    f"[Warning] UID: {uid} 索引越界! 口袋最大索引: {max_idx + 1}, 特征长度: {seq_length}. 已跳过该蛋白。"
                )
                skipped_count += 1
                continue

            try:
                pocket_node_feature = esm_node_feature[pocket_residue_ids]
            except Exception as e:
                print(e)
                print(uid, " Error")
                skipped_count += 1
                continue

        uid_to_pocket_node_feature[uid] = pocket_node_feature

    torch.save(uid_to_pocket_node_feature, save_path)
    print(
        f"Save esm feature of {len(uid_to_pocket_node_feature)} pockets to {save_path}"
    )
    if skipped_count > 0:
        print(
            f"Skipped {skipped_count} proteins due to index mismatch or other errors\n"
        )


def check_pocket_feature(gvp_feature_path, esm_feature_path):
    gvp_feature = torch.load(gvp_feature_path)
    esm_feature = torch.load(esm_feature_path)

    assert len(gvp_feature) == len(esm_feature)
    bad_proteins = []
    for uid, gvp_node_feature in gvp_feature.items():
        esm_node_feature = esm_feature[uid]
        n_gvp_nodes = gvp_node_feature[0].shape[0]
        n_esm_nodes = esm_node_feature.shape[0]
        if n_gvp_nodes != n_esm_nodes:
            bad_proteins.append(uid)

    if len(bad_proteins) > 0:
        print(
            f"Found {len(bad_proteins)} bad proteins, which have different node numbers between gvp and esm features. These proteins should be removed from training data!"
        )
        gvp_feature = {
            uid: val for uid, val in gvp_feature.items() if uid not in bad_proteins
        }
        esm_feature = {
            uid: val for uid, val in esm_feature.items() if uid not in bad_proteins
        }
        torch.save(gvp_feature, gvp_feature_path)
        torch.save(esm_feature, esm_feature_path)


def delete_empty(data_dir):
    file_list = tranverse_folder(data_dir)
    for filepath in tqdm(file_list):
        if os.path.getsize(filepath) < 1000:
            os.remove(filepath)


def calc_reacting_center(data_path, save_dir, append=True):
    print("\n", "#" * 20, "Calculating Reaction Center", "#" * 20, "\n")

    calc_aam(data_path, save_dir, append)

    aam_path = os.path.join(save_dir, "rxn2aam.pkl")
    rxn2aam = pkl.load(open(aam_path, "rb"))

    reacting_center_path = os.path.join(save_dir, "reacting_center.pkl")
    if os.path.exists(reacting_center_path) and append:
        cached_reacting_center_map = pkl.load(open(reacting_center_path, "rb"))
    else:
        cached_reacting_center_map = {}

    df_data = pd.read_csv(data_path)
    rxns_to_run = df_data[RXN_COL].unique()
    rxns_to_run = [rxn for rxn in rxns_to_run if rxn not in cached_reacting_center_map]
    reacting_center_map = {}
    for rxn in tqdm(rxns_to_run):
        reacting_center_map[rxn] = extract_reacting_center(rxn, rxn2aam)

    if append:
        print(
            f"Append {len(reacting_center_map)} reacting center to {reacting_center_path}"
        )

    reacting_center_map.update(cached_reacting_center_map)
    pkl.dump(reacting_center_map, open(reacting_center_path, "wb"))

    if not append:
        print(
            f"Calculate {len(reacting_center_map)} reacting center and save to {reacting_center_path}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--pocket_dir",
        type=str,
        help="If you already have pocket data, you can specify the directory here",
    )
    parser.add_argument("--skip_rxn_feature", action="store_true")
    args = parser.parse_args()

    if args.pocket_dir.endswith("/"):
        # remove '/' in the end of the path
        args.pocket_dir = args.pocket_dir[:-1]

    if not args.pocket_dir:
        pocket_dir = os.path.join(
            os.path.dirname(args.data_path), "pocket/alphafill_8A"
        )
    else:
        pocket_dir = args.pocket_dir

    feature_dir = os.path.join(os.path.dirname(args.data_path), "feature")

    # esm_model = 'esm2_t33_650M_UR50D'
    esm_model = "ESM-C_600M"

    protein_feature_dir = os.path.join(feature_dir, "protein")
    esm_node_feat_dir = os.path.join(protein_feature_dir, f"{esm_model}/node_level")
    esm_mean_feat_path = os.path.join(
        protein_feature_dir, f"{esm_model}/protein_level/seq2feature.pkl"
    )
    esm_pocket_node_feature_path = os.path.join(
        protein_feature_dir, f"{esm_model}/pocket_node_feature/esm_node_feature.pt"
    )
    gvp_feat_path = os.path.join(
        protein_feature_dir, "gvp_feature/gvp_protein_feature.pt"
    )
    pocket_info_save_path = os.path.join(os.path.dirname(pocket_dir), "pocket_info.csv")

    reaction_feat_dir = os.path.join(feature_dir, "reaction")
    drfp_save_path = os.path.join(reaction_feat_dir, "drfp/rxn2fp.pkl")
    mol_conformation_dir = os.path.join(reaction_feat_dir, "molecule_conformation")
    reacting_center_dir = os.path.join(reaction_feat_dir, "reacting_center")

    os.makedirs(pocket_dir, exist_ok=True)
    os.makedirs(esm_node_feat_dir, exist_ok=True)
    os.makedirs(os.path.dirname(esm_mean_feat_path), exist_ok=True)
    os.makedirs(os.path.dirname(esm_pocket_node_feature_path), exist_ok=True)
    os.makedirs(os.path.dirname(gvp_feat_path), exist_ok=True)
    os.makedirs(os.path.dirname(drfp_save_path), exist_ok=True)
    os.makedirs(mol_conformation_dir, exist_ok=True)
    os.makedirs(reacting_center_dir, exist_ok=True)

    if not args.skip_rxn_feature:
        # Calculate reaction fingerprints
        calc_drfp(args.data_path, drfp_save_path)

        # Extract reaction center
        calc_reacting_center(args.data_path, reacting_center_dir)

        # May be the most time consuming step
        # Generate molecular conformation by rdkit
        generate_mol_conformation(args.data_path, mol_conformation_dir)

    # Calculate GVP features of pockets
    calc_gvp_feature(args.data_path, pocket_dir, gvp_feat_path)

    # Calculate ESM features of the full sequence
    if esm_model == "esm2_t33_650M_UR50D":
        calc_seq_esm_feature(args.data_path, esm_node_feat_dir, esm_mean_feat_path)
    elif esm_model == "ESM-C_600M":
        calc_seq_esm_C_feature(args.data_path, esm_node_feat_dir, esm_mean_feat_path)

    # Extract esm feature of pocket nodes
    get_esm_pocket_feature(
        pocket_info_save_path,
        esm_node_feat_dir,
        esm_pocket_node_feature_path,
        pocket_dir,
    )

    check_pocket_feature(gvp_feat_path, esm_pocket_node_feature_path)

    print("\n ###### Feature calculation is finished! ######\n")


if __name__ == "__main__":
    main()
