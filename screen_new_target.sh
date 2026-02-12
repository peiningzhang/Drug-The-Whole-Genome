#!/bin/bash
# DrugCLIP 新目标筛选脚本
# 使用方法: bash screen_new_target.sh <target_name> <pdb_file_or_dir> [gpu_id]

set -e

if [ $# -lt 2 ]; then
    echo "使用方法: bash screen_new_target.sh <target_name> <pdb_file_or_dir> [gpu_id]"
    echo ""
    echo "参数说明:"
    echo "  target_name    : 目标名称（如 MYPROTEIN）"
    echo "  pdb_file_or_dir: PDB 文件路径或包含多个 PDB 文件的目录"
    echo "  gpu_id         : GPU 设备号（默认: 0）"
    echo ""
    echo "示例:"
    echo "  bash screen_new_target.sh MYPROTEIN ./myprotein.pdb 0"
    echo "  bash screen_new_target.sh MYPROTEIN ./pdb_files/ 0"
    exit 1
fi

TARGET_NAME=$1
PDB_INPUT=$2
GPU_ID=${3:-0}

CONDA_ENV="drugclip"
PROJECT_DIR="/shared/healthinfolab/phz24002/DrugClip"
TARGET_DIR="${PROJECT_DIR}/data/targets/${TARGET_NAME}"

echo "=========================================="
echo "DrugCLIP 新目标筛选"
echo "=========================================="
echo "目标名称: ${TARGET_NAME}"
echo "PDB 输入: ${PDB_INPUT}"
echo "GPU 设备: ${GPU_ID}"
echo "目标目录: ${TARGET_DIR}"
echo "=========================================="

# 激活环境
source /shared/healthinfolab/phz24002/anaconda3/bin/activate ${CONDA_ENV}
cd ${PROJECT_DIR}

# 1. 创建目标目录
echo ""
echo "步骤 1: 创建目标目录..."
mkdir -p ${TARGET_DIR}

# 2. 复制 PDB 文件
echo ""
echo "步骤 2: 复制 PDB 文件..."
if [ -f "${PDB_INPUT}" ]; then
    # 单个文件
    echo "  复制单个文件: ${PDB_INPUT}"
    cp "${PDB_INPUT}" ${TARGET_DIR}/
elif [ -d "${PDB_INPUT}" ]; then
    # 目录
    echo "  复制目录中的所有 PDB 文件..."
    cp ${PDB_INPUT}/*.pdb ${TARGET_DIR}/ 2>/dev/null || true
    if [ $? -ne 0 ] || [ -z "$(ls -A ${TARGET_DIR}/*.pdb 2>/dev/null)" ]; then
        echo "  错误: 目录中没有找到 .pdb 文件"
        exit 1
    fi
else
    echo "  错误: ${PDB_INPUT} 不存在"
    exit 1
fi

echo "  已复制的文件:"
ls -lh ${TARGET_DIR}/*.pdb

# 3. 编码口袋
echo ""
echo "步骤 3: 从 PDB 文件生成 pocket.lmdb..."
echo "  这可能需要几分钟时间..."
bash encode_pocket.sh ${GPU_ID} ${TARGET_DIR}

if [ ! -f "${TARGET_DIR}/pocket.lmdb" ]; then
    echo "  错误: 未能生成 pocket.lmdb"
    exit 1
fi

echo "  ✓ pocket.lmdb 已生成"
if [ -f "${TARGET_DIR}/pocket_reps.pkl" ]; then
    echo "  ✓ pocket_reps.pkl 已生成"
fi

# 4. 运行虚拟筛选
echo ""
echo "步骤 4: 运行虚拟筛选..."
echo "  这可能需要较长时间，取决于分子库大小..."

RESULT_FILE="${TARGET_NAME}_results.txt"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python ./unimol/retrieval.py \
    --user-dir ./unimol \
    --valid-subset test \
    --num-workers 8 \
    --ddp-backend=c10d \
    --batch-size 4 \
    --task drugclip \
    --loss in_batch_softmax \
    --arch drugclip \
    --max-pocket-atoms 511 \
    --fp16 \
    --fp16-init-scale 4 \
    --fp16-scale-window 256 \
    --seed 1 \
    --log-interval 100 \
    --log-format simple \
    --mol-path mols.lmdb \
    --pocket-path ${TARGET_DIR}/pocket.lmdb \
    --fold-version 6_folds \
    --use-cache True \
    --save-path ${RESULT_FILE} \
    "./dict"

if [ -f "${RESULT_FILE}" ]; then
    echo ""
    echo "=========================================="
    echo "筛选完成！"
    echo "=========================================="
    echo "结果文件: ${RESULT_FILE}"
    echo "前 10 个候选分子:"
    head -10 ${RESULT_FILE}
    echo ""
    echo "总候选数: $(wc -l < ${RESULT_FILE})"
else
    echo "  错误: 未能生成结果文件"
    exit 1
fi
