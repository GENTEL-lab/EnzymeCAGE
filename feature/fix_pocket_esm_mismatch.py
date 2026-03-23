"""
修复PDB残基编号与ESM特征序列索引不匹配的问题

问题根源：
- PDB文件中的残基编号可能不从1开始，也可能不连续
- ESM特征基于序列计算，索引是0-based的连续编号(0, 1, 2, ...)
- pocket_info.csv中存储的是PDB文件的原始残基编号
- 直接使用这些编号会导致索引越界

解决方案：
建立PDB残基编号到序列索引的映射关系
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
    从PDB文件建立残基编号到序列索引的映射

    Returns:
        dict: {pdb_residue_number: sequence_index}
    """
    parser = PDBParser(QUIET=True)
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


def get_esm_pocket_feature_fixed(
    pocket_info_path, pocket_pdb_dir, esm_node_feat_dir, save_path
):
    """
    修复后的ESM pocket特征提取函数

    Args:
        pocket_info_path: pocket信息CSV文件路径
        pocket_pdb_dir: pocket PDB文件所在目录
        esm_node_feat_dir: ESM node特征目录
        save_path: 保存路径
    """
    print("\n", "=" * 20, "提取ESM Pocket特征（修复版）", "=" * 20, "\n")

    df_pocket_data = pd.read_csv(pocket_info_path)
    uids = [str(each) for each in df_pocket_data[UID_COL].values]
    uid_to_pocket = dict(zip(uids, df_pocket_data["pocket_residues"]))

    uid_to_pocket_node_feature = {}
    skipped_proteins = []

    for uid in tqdm(uids, desc="处理蛋白质"):
        # 加载ESM特征
        esm_file = os.path.join(esm_node_feat_dir, f"{uid}.npz")
        if not os.path.exists(esm_file):
            skipped_proteins.append((uid, "ESM特征文件不存在"))
            continue

        esm_node_feature = np.load(esm_file)["node_feature"]
        seq_length = esm_node_feature.shape[0]

        # 获取pocket残基编号
        pocket_residue_ids = uid_to_pocket.get(uid)
        if not isinstance(pocket_residue_ids, str):
            skipped_proteins.append((uid, "Pocket信息为空"))
            continue

        # 加载PDB文件建立映射
        pdb_file = os.path.join(pocket_pdb_dir, f"{uid}.pdb")
        if not os.path.exists(pdb_file):
            skipped_proteins.append((uid, "PDB文件不存在"))
            continue

        try:
            # 建立残基编号到序列索引的映射
            residue_mapping = build_residue_mapping(pdb_file)

            # 将PDB残基编号转换为序列索引
            pdb_res_numbers = [int(i) for i in pocket_residue_ids.split(",")]
            seq_indices = []

            for pdb_res_num in pdb_res_numbers:
                if pdb_res_num in residue_mapping:
                    seq_idx = residue_mapping[pdb_res_num]
                    if seq_idx < seq_length:  # 确保索引在范围内
                        seq_indices.append(seq_idx)
                    else:
                        print(
                            f"[Warning] {uid}: 序列索引 {seq_idx} 超出范围 {seq_length}"
                        )
                else:
                    print(f"[Warning] {uid}: PDB残基编号 {pdb_res_num} 未找到映射")

            if len(seq_indices) == 0:
                skipped_proteins.append((uid, "没有有效的残基索引"))
                continue

            # 提取pocket节点特征
            pocket_node_feature = esm_node_feature[seq_indices]
            uid_to_pocket_node_feature[uid] = pocket_node_feature

        except Exception as e:
            skipped_proteins.append((uid, f"处理错误: {str(e)}"))
            continue

    # 保存结果
    torch.save(uid_to_pocket_node_feature, save_path)
    print(
        f"\n保存了 {len(uid_to_pocket_node_feature)} 个蛋白质的ESM pocket特征到: {save_path}"
    )

    # 输出跳过的蛋白质信息
    if skipped_proteins:
        print(f"\n共跳过 {len(skipped_proteins)} 个蛋白质：")
        for uid, reason in skipped_proteins[:10]:  # 只显示前10个
            print(f"  - {uid}: {reason}")
        if len(skipped_proteins) > 10:
            print(f"  ... 还有 {len(skipped_proteins) - 10} 个")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="修复ESM pocket特征提取中的索引不匹配问题"
    )
    parser.add_argument(
        "--pocket_info", type=str, required=True, help="Pocket信息CSV文件路径"
    )
    parser.add_argument(
        "--pocket_pdb_dir", type=str, required=True, help="Pocket PDB文件目录"
    )
    parser.add_argument(
        "--esm_node_feat_dir", type=str, required=True, help="ESM node特征目录"
    )
    parser.add_argument("--save_path", type=str, required=True, help="输出文件路径")

    args = parser.parse_args()

    get_esm_pocket_feature_fixed(
        args.pocket_info, args.pocket_pdb_dir, args.esm_node_feat_dir, args.save_path
    )


if __name__ == "__main__":
    main()
