# DrugCLIP 环境搭建指南

本指南将帮助您在 `/shared/healthinfolab/phz24002/DrugClip` 目录中搭建完整的 DrugCLIP 环境。

## 前置要求

- Conda 已安装（位于 `/shared/healthinfolab/phz24002/anaconda3`）
- 足够的磁盘空间（建议至少 50GB）
- CUDA 12.1 驱动（用于 GPU 支持）

## 快速开始

### 方法 1: 使用自动化脚本（推荐）

```bash
cd /shared/healthinfolab/phz24002/DrugClip

# 1. 安装环境
bash setup_environment.sh

# 2. 下载数据（需要先激活环境）
source /shared/healthinfolab/phz24002/anaconda3/bin/activate drugclip
bash download_data.sh

# 3. 验证环境
python verify_environment.py
```

### 方法 2: 手动安装

#### 步骤 1: 创建 Conda 环境

```bash
source /shared/healthinfolab/phz24002/anaconda3/bin/activate
conda create -n drugclip python=3.8 -y
conda activate drugclip
```

#### 步骤 2: 安装 PyTorch (CUDA 12.1)

```bash
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

#### 步骤 3: 安装基础依赖

```bash
pip install --no-cache-dir \
    iopath \
    lmdb \
    ml_collections \
    numpy \
    scipy \
    tensorboardX \
    tqdm \
    tokenizers
```

#### 步骤 4: 安装 Uni-Core

```bash
cd /tmp
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
python setup.py install
cd /shared/healthinfolab/phz24002/DrugClip
rm -rf /tmp/Uni-Core
```

#### 步骤 5: 安装项目依赖

```bash
cd /shared/healthinfolab/phz24002/DrugClip
pip install --no-cache-dir -r docker/requirements.txt
```

#### 步骤 6: 安装额外依赖

```bash
pip install --no-cache-dir huggingface_hub unimol_tools
```

#### 步骤 7: 下载模型权重和数据

从 HuggingFace 下载以下文件到 `./data` 目录：
- `model_weights.zip`
- `encoded_mol_embs.zip`
- `targets.zip`

下载地址: https://huggingface.co/datasets/bgao95/DrugCLIP_data

```bash
# 使用 huggingface_hub 下载
python -c "
from huggingface_hub import hf_hub_download
import os

repo_id = 'bgao95/DrugCLIP_data'
data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

files = ['model_weights.zip', 'encoded_mol_embs.zip', 'targets.zip']
for file in files:
    print(f'下载 {file}...')
    hf_hub_download(repo_id=repo_id, filename=file, local_dir=data_dir)

# 解压文件
import zipfile
for file in files:
    zip_path = os.path.join(data_dir, file)
    if os.path.exists(zip_path):
        print(f'解压 {file}...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
"
```

#### 步骤 8: 验证环境

```bash
python verify_environment.py
```

## 环境信息

- **Conda 环境名称**: `drugclip`
- **Python 版本**: 3.8
- **PyTorch 版本**: 2.1.2
- **CUDA 版本**: 12.1
- **环境路径**: `/shared/healthinfolab/phz24002/anaconda3/envs/drugclip`

## 激活环境

```bash
source /shared/healthinfolab/phz24002/anaconda3/bin/activate drugclip
```

## 使用 DrugCLIP

### 虚拟筛选

```bash
bash retrieval.sh
```

需要设置：
- `pocket path` 为 `./data/targets/{target}/pocket.lmdb`
- `num_folds` 为 8（对于 5HT2A）

### 基准测试

```bash
bash test.sh
```

在 `test.sh` 中选择 `TASK` 为 `DUDE` 或 `PCBA`

## 常见问题

### 1. RDKit 安装问题

如果 RDKit 安装失败，确保 `numpy<2.0.0`：

```bash
pip install "numpy<2.0.0"
pip install rdkit==2023.9.5
```

### 2. CUDA 不可用

检查 CUDA 驱动版本：
```bash
nvidia-smi
```

确保 CUDA 驱动版本 >= 12.1

### 3. 数据下载失败

如果 HuggingFace 下载失败，可以：
1. 检查网络连接
2. 配置 HuggingFace token（如果需要）
3. 手动从网页下载文件

## 参考

- 项目仓库: https://github.com/THU-ATOM/Drug-The-Whole-Genome
- 数据下载: https://huggingface.co/datasets/bgao95/DrugCLIP_data
- Uni-Mol: https://github.com/dptech-corp/Uni-Mol
