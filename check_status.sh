#!/bin/bash
# 检查 DrugCLIP 环境安装状态

CONDA_BASE="/shared/healthinfolab/phz24002/anaconda3"
ENV_NAME="drugclip"
PROJECT_DIR="/shared/healthinfolab/phz24002/DrugClip"

echo "=========================================="
echo "DrugCLIP 环境安装状态检查"
echo "=========================================="

# 检查 conda 环境
echo -n "1. Conda 环境: "
if [ -d "${CONDA_BASE}/envs/${ENV_NAME}" ]; then
    echo "✓ 存在"
else
    echo "✗ 不存在"
fi

# 检查 Python
echo -n "2. Python: "
if [ -f "${CONDA_BASE}/envs/${ENV_NAME}/bin/python" ]; then
    VERSION=$(${CONDA_BASE}/envs/${ENV_NAME}/bin/python --version 2>&1)
    echo "✓ $VERSION"
else
    echo "✗ 未安装"
fi

# 检查 PyTorch
echo -n "3. PyTorch: "
if ${CONDA_BASE}/envs/${ENV_NAME}/bin/python -c "import torch" 2>/dev/null; then
    VERSION=$(${CONDA_BASE}/envs/${ENV_NAME}/bin/python -c "import torch; print(torch.__version__)" 2>/dev/null)
    CUDA=$(${CONDA_BASE}/envs/${ENV_NAME}/bin/python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CUDA not available')" 2>/dev/null)
    echo "✓ $VERSION ($CUDA)"
else
    echo "✗ 未安装"
fi

# 检查关键依赖
echo -n "4. 关键依赖: "
MISSING=""
for pkg in numpy scipy pandas rdkit lmdb transformers; do
    if ! ${CONDA_BASE}/envs/${ENV_NAME}/bin/python -c "import $pkg" 2>/dev/null; then
        MISSING="$MISSING $pkg"
    fi
done
if [ -z "$MISSING" ]; then
    echo "✓ 全部已安装"
else
    echo "✗ 缺失: $MISSING"
fi

# 检查 Uni-Core
echo -n "5. Uni-Core: "
if ${CONDA_BASE}/envs/${ENV_NAME}/bin/python -c "import unicore" 2>/dev/null; then
    echo "✓ 已安装"
else
    echo "✗ 未安装"
fi

# 检查数据目录
echo -n "6. 数据目录: "
if [ -d "${PROJECT_DIR}/data" ]; then
    echo "✓ 存在"
    if [ -d "${PROJECT_DIR}/data/model_weights" ]; then
        echo "   - model_weights: ✓"
    else
        echo "   - model_weights: ✗"
    fi
    if [ -d "${PROJECT_DIR}/data/targets" ]; then
        echo "   - targets: ✓"
    else
        echo "   - targets: ✗"
    fi
else
    echo "✗ 不存在"
fi

# 检查安装进程
echo -n "7. 安装进程: "
if pgrep -f "setup_environment.sh" > /dev/null; then
    echo "✓ 正在运行"
    echo "   查看日志: tail -f ${PROJECT_DIR}/setup_log.txt"
else
    echo "✗ 未运行"
fi

echo "=========================================="
