#!/bin/bash
# DrugCLIP 数据下载脚本
# 使用方法: bash download_data.sh

set -e

PROJECT_DIR="/shared/healthinfolab/phz24002/DrugClip"
DATA_DIR="${PROJECT_DIR}/data"
HF_DATASET="bgao95/DrugCLIP_data"

echo "=========================================="
echo "DrugCLIP 数据下载脚本"
echo "=========================================="

# 创建 data 目录
mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

# 激活 conda 环境
CONDA_BASE="/shared/healthinfolab/phz24002/anaconda3"
ENV_NAME="drugclip"
source ${CONDA_BASE}/bin/activate ${ENV_NAME}

# 检查是否已安装 huggingface_hub
echo "检查 huggingface_hub..."
python -c "import huggingface_hub" 2>/dev/null || {
    echo "安装 huggingface_hub..."
    pip install huggingface_hub
}

# 下载文件列表
FILES=(
    "model_weights.zip"
    "encoded_mol_embs.zip"
    "targets.zip"
)

echo "开始从 HuggingFace 下载数据..."
echo "数据集: ${HF_DATASET}"
echo "注意: 如果下载失败，请手动从以下地址下载:"
echo "https://huggingface.co/datasets/${HF_DATASET}/tree/main"

for file in "${FILES[@]}"; do
    echo ""
    echo "下载: ${file}..."
    python -c "
from huggingface_hub import hf_hub_download
import os
repo_id = '${HF_DATASET}'
filename = '${file}'
local_dir = '${DATA_DIR}'
try:
    # 尝试使用 repo_type='dataset'
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        repo_type='dataset'
    )
    print(f'成功下载: {downloaded_path}')
except Exception as e1:
    try:
        # 如果失败，尝试不使用 repo_type
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir
        )
        print(f'成功下载: {downloaded_path}')
    except Exception as e2:
        print(f'下载失败: {e2}')
        print(f'请手动下载: https://huggingface.co/datasets/${HF_DATASET}/resolve/main/${file}')
        # 不退出，继续下载其他文件
"
done

# 解压文件
echo "解压文件..."
for file in "${FILES[@]}"; do
    if [ -f "${DATA_DIR}/${file}" ]; then
        echo "解压: ${file}..."
        unzip -q -o "${DATA_DIR}/${file}" -d "${DATA_DIR}" || {
            echo "警告: ${file} 解压失败，请手动检查"
        }
        # 可选：删除 zip 文件以节省空间
        # rm "${DATA_DIR}/${file}"
    else
        echo "警告: ${file} 不存在，跳过解压"
    fi
done

echo "=========================================="
echo "数据下载完成！"
echo "=========================================="
echo "数据目录: ${DATA_DIR}"
ls -lh ${DATA_DIR} | head -20
