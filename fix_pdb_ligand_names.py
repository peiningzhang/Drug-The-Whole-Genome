#!/usr/bin/env python3
"""
修复 PDB 文件名以匹配实际配体名
将文件重命名为 {pdb_id}_{actual_ligand}.pdb 格式
"""

import os
import sys
import re
from pathlib import Path

def get_ligand_from_pdb(pdb_file):
    """从 PDB 文件中提取配体名"""
    ligand_names = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('HET '):
                parts = line.split()
                if len(parts) >= 2:
                    ligand_names.append(parts[1])
            elif line.startswith('HETATM'):
                # 也可以从 HETATM 行提取
                parts = line.split()
                if len(parts) >= 4:
                    resname = parts[3]
                    if resname not in ligand_names:
                        ligand_names.append(resname)
    
    # 去重并返回第一个（通常只有一个配体）
    return list(set(ligand_names))

def rename_pdb_with_ligand(pdb_file, target_dir=None):
    """重命名 PDB 文件以包含实际配体名"""
    pdb_path = Path(pdb_file)
    
    # 获取配体名
    ligands = get_ligand_from_pdb(pdb_file)
    if not ligands:
        print(f"警告: {pdb_file} 中未找到配体")
        return None
    
    ligand = ligands[0]  # 使用第一个配体
    
    # 提取 PDB ID（文件名前缀）
    pdb_id = pdb_path.stem.split('_')[0]
    
    # 生成新文件名
    new_name = f"{pdb_id}_{ligand}.pdb"
    
    if target_dir:
        new_path = Path(target_dir) / new_name
    else:
        new_path = pdb_path.parent / new_name
    
    # 如果目标文件已存在，跳过
    if new_path.exists() and new_path != pdb_path:
        print(f"跳过: {new_path} 已存在")
        return str(new_path)
    
    # 复制文件（不删除原文件）
    import shutil
    shutil.copy2(pdb_path, new_path)
    print(f"创建: {pdb_path.name} -> {new_path.name} (配体: {ligand})")
    
    return str(new_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python fix_pdb_ligand_names.py <pdb_file_or_dir>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if input_path.is_file():
        # 单个文件
        rename_pdb_with_ligand(input_path)
    elif input_path.is_dir():
        # 目录中的所有 PDB 文件
        pdb_files = list(input_path.glob("*.pdb"))
        for pdb_file in pdb_files:
            rename_pdb_with_ligand(pdb_file, input_path)
    else:
        print(f"错误: {input_path} 不存在")
        sys.exit(1)
