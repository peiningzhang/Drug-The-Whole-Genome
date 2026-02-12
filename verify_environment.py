#!/usr/bin/env python
"""验证 DrugCLIP 环境配置"""

import sys

def check_import(module_name, package_name=None):
    """检查模块是否可以导入"""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name} 已安装")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name} 未安装: {e}")
        return False

def check_version(module_name, version_attr="__version__"):
    """检查模块版本"""
    try:
        module = __import__(module_name)
        version = getattr(module, version_attr, "未知版本")
        print(f"  版本: {version}")
        return True
    except:
        return False

def main():
    print("=" * 50)
    print("DrugCLIP 环境验证")
    print("=" * 50)
    
    checks_passed = True
    
    # 检查 Python 版本
    print(f"\nPython 版本: {sys.version}")
    
    # 检查 PyTorch
    print("\n检查 PyTorch...")
    if check_import("torch", "PyTorch"):
        check_version("torch")
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  CUDA 可用: 是")
                print(f"  CUDA 版本: {torch.version.cuda}")
                print(f"  GPU 数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            else:
                print(f"  CUDA 可用: 否")
        except:
            pass
    else:
        checks_passed = False
    
    # 检查其他关键依赖
    print("\n检查其他依赖...")
    dependencies = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("pandas", "Pandas"),
        ("rdkit", "RDKit"),
        ("lmdb", "LMDB"),
        ("transformers", "Transformers"),
        ("biotite", "Biotite"),
        ("Bio", "BioPython"),  # BioPython 导入名是 Bio
    ]
    
    for module, name in dependencies:
        if not check_import(module, name):
            checks_passed = False
    
    # 检查 Uni-Core
    print("\n检查 Uni-Core...")
    check_import("unicore", "Uni-Core")
    
    # 检查数据目录
    print("\n检查数据目录...")
    import os
    data_dir = "./data"
    if os.path.exists(data_dir):
        print(f"✓ 数据目录存在: {data_dir}")
        subdirs = ["model_weights", "encoded_mol_embs", "targets"]
        for subdir in subdirs:
            path = os.path.join(data_dir, subdir)
            if os.path.exists(path):
                print(f"  ✓ {subdir}/ 存在")
            else:
                print(f"  ✗ {subdir}/ 不存在")
    else:
        print(f"✗ 数据目录不存在: {data_dir}")
        print("  请运行: bash download_data.sh")
    
    print("\n" + "=" * 50)
    if checks_passed:
        print("环境验证完成！所有关键依赖已安装。")
    else:
        print("环境验证完成！部分依赖缺失，请检查安装。")
    print("=" * 50)
    
    return 0 if checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
