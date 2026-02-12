#!/bin/bash
# DrugCLIP 环境安装脚本
# 使用方法: bash setup_environment.sh

set -e

CONDA_BASE="/shared/healthinfolab/phz24002/anaconda3"
ENV_NAME="drugclip"
PROJECT_DIR="/shared/healthinfolab/phz24002/DrugClip"

echo "=========================================="
echo "DrugCLIP 环境安装脚本"
echo "=========================================="

# 激活 conda base 环境
source ${CONDA_BASE}/bin/activate

# 1. 创建 conda 环境（如果不存在）
if [ ! -d "${CONDA_BASE}/envs/${ENV_NAME}" ]; then
    echo "创建 conda 环境: ${ENV_NAME} (Python 3.8)..."
    conda create -n ${ENV_NAME} python=3.8 -y
else
    echo "环境 ${ENV_NAME} 已存在，跳过创建步骤"
fi

# 激活环境
echo "激活环境: ${ENV_NAME}..."
source activate ${ENV_NAME}

# 2. 安装 PyTorch (CUDA 12.1)
echo "安装 PyTorch 2.1.2 (CUDA 12.1)..."
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. 安装基础依赖
echo "安装基础依赖..."
pip install --no-cache-dir \
    iopath \
    lmdb \
    ml_collections \
    numpy \
    scipy \
    tensorboardX \
    tqdm
# tokenizers 需要 Python 3.9+，暂时跳过，如果需要可以单独安装

# 4. 安装 Uni-Core
echo "安装 Uni-Core..."
cd /tmp
if [ -d "Uni-Core" ]; then
    rm -rf Uni-Core
fi
git clone https://github.com/dptech-corp/Uni-Core.git
cd Uni-Core
python setup.py install
cd ${PROJECT_DIR}
rm -rf /tmp/Uni-Core

# 5. 安装项目依赖
echo "安装项目依赖..."
cd ${PROJECT_DIR}
pip install --no-cache-dir -r docker/requirements.txt

# 6. 安装额外的依赖
echo "安装额外依赖..."
pip install --no-cache-dir \
    huggingface_hub \
    unimol_tools \
    h5py

echo "=========================================="
echo "环境安装完成！"
echo "=========================================="
echo "激活环境: source ${CONDA_BASE}/bin/activate ${ENV_NAME}"
echo "验证安装: python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'"
