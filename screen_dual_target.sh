#!/bin/bash
# DrugCLIP 双目标筛选脚本
# 筛选能同时匹配两个口袋的分子（取交集）
# 使用方法: bash screen_dual_target.sh <target_dir> [gpu_id] [top_n]

set -e

if [ $# -lt 1 ]; then
    echo "使用方法: bash screen_dual_target.sh <target_dir> [gpu_id] [top_n]"
    echo ""
    echo "参数说明:"
    echo "  target_dir: 包含多个 PDB 文件的目录（如 ./data/targets/KOR-5H2A）"
    echo "  gpu_id     : GPU 设备号（默认: 0）"
    echo "  top_n      : 每个口袋取 Top N 分子进行交集（默认: 1000）"
    echo ""
    echo "示例:"
    echo "  bash screen_dual_target.sh ./data/targets/KOR-5H2A 0 1000"
    exit 1
fi

TARGET_DIR=$1
GPU_ID=${2:-0}
TOP_N=${3:-1000}

CONDA_ENV="drugclip"
PROJECT_DIR="/shared/healthinfolab/phz24002/DrugClip"
CONDA_BASE="/shared/healthinfolab/phz24002/anaconda3"

echo "=========================================="
echo "DrugCLIP 双目标筛选（交集）"
echo "=========================================="
echo "目标目录: ${TARGET_DIR}"
echo "GPU 设备: ${GPU_ID}"
echo "Top N: ${TOP_N}"
echo "=========================================="

# 激活环境
source ${CONDA_BASE}/bin/activate ${CONDA_ENV}
cd ${PROJECT_DIR}

# 检查目录是否存在
if [ ! -d "${TARGET_DIR}" ]; then
    echo "错误: 目录 ${TARGET_DIR} 不存在"
    exit 1
fi

# 查找所有 PDB 文件
PDB_FILES=($(find ${TARGET_DIR} -maxdepth 1 -name "*.pdb" -type f | sort))
if [ ${#PDB_FILES[@]} -lt 2 ]; then
    echo "错误: 目录中需要至少 2 个 PDB 文件，找到 ${#PDB_FILES[@]} 个"
    exit 1
fi

echo ""
echo "找到 ${#PDB_FILES[@]} 个 PDB 文件:"
for pdb in "${PDB_FILES[@]}"; do
    echo "  - $(basename ${pdb})"
done

# 为每个 PDB 文件创建单独的口袋目录并处理
POCKET_DIRS=()
POCKET_PATHS=()
RESULTS=()

for pdb_file in "${PDB_FILES[@]}"; do
    pdb_name=$(basename ${pdb_file} .pdb)
    pocket_dir="${TARGET_DIR}/${pdb_name}_pocket"
    
    echo ""
    echo "=========================================="
    echo "处理: ${pdb_name}"
    echo "=========================================="
    
    # 创建口袋目录
    mkdir -p ${pocket_dir}
    
    # 修复 PDB 文件名以匹配实际配体名
    echo "步骤 1: 检查并修复配体名..."
    python3 ${PROJECT_DIR}/fix_pdb_ligand_names.py ${pdb_file} ${pocket_dir} 2>&1 | grep -E "(创建|跳过|警告)" || true
    
    # 确保有 PDB 文件在目录中
    if [ -z "$(ls -A ${pocket_dir}/*.pdb 2>/dev/null)" ]; then
        # 如果没有，复制原文件
        cp ${pdb_file} ${pocket_dir}/
    fi
    
    # 编码口袋
    echo "步骤 2: 从 PDB 文件生成 pocket.lmdb..."
    bash encode_pocket.sh ${GPU_ID} ${pocket_dir}
    
    if [ ! -f "${pocket_dir}/pocket.lmdb" ]; then
        echo "错误: 未能生成 ${pocket_dir}/pocket.lmdb"
        echo "提示: 检查 PDB 文件中的配体名称是否与文件名匹配"
        exit 1
    fi
    
    # 运行筛选
    echo "步骤 3: 运行虚拟筛选..."
    result_file="${TARGET_DIR}/${pdb_name}_results.txt"
    
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
        --pocket-path ${pocket_dir}/pocket.lmdb \
        --fold-version 6_folds \
        --use-cache True \
        --save-path ${result_file} \
        "./dict" 2>&1 | tail -20
    
    if [ ! -f "${result_file}" ]; then
        echo "错误: 未能生成结果文件 ${result_file}"
        exit 1
    fi
    
    POCKET_DIRS+=("${pocket_dir}")
    POCKET_PATHS+=("${pocket_dir}/pocket.lmdb")
    RESULTS+=("${result_file}")
    
    echo "  ✓ 结果保存在: ${result_file}"
    echo "  ✓ 候选数: $(wc -l < ${result_file})"
done

# 计算交集
echo ""
echo "=========================================="
echo "计算交集（每个口袋取 Top ${TOP_N}）"
echo "=========================================="

# 创建 Python 脚本来计算交集
python_script="${TARGET_DIR}/compute_intersection.py"
cat > ${python_script} << 'PYTHON_EOF'
import sys
import csv
from collections import defaultdict

def load_results(result_files, top_n):
    """加载每个结果文件的 Top N 分子"""
    all_molecules = defaultdict(list)
    
    for idx, result_file in enumerate(result_files):
        molecules = set()
        with open(result_file, 'r') as f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if len(row) >= 3:
                    # Format: HIT_ID,SMILES,score
                    smiles = row[1].strip()  # SMILES is in column 1
                    try:
                        score = float(row[2].strip())  # score is in column 2
                        molecules.add((smiles, score))
                        count += 1
                        if count >= top_n:
                            break
                    except ValueError:
                        continue
                elif len(row) >= 2:
                    # Fallback: assume format is SMILES,score (no HIT_ID)
                    smiles = row[0].strip()
                    try:
                        score = float(row[1].strip())
                        molecules.add((smiles, score))
                        count += 1
                        if count >= top_n:
                            break
                    except ValueError:
                        continue
        
        all_molecules[idx] = molecules
        print(f"  口袋 {idx+1}: 加载了 {len(molecules)} 个 Top {top_n} 分子")
    
    return all_molecules

def compute_intersection(all_molecules):
    """计算所有口袋的交集"""
    if len(all_molecules) == 0:
        return set()
    
    # 从第一个口袋开始
    intersection = set(m[0] for m in all_molecules[0])
    
    # 与其他口袋求交集
    for idx in range(1, len(all_molecules)):
        other_molecules = set(m[0] for m in all_molecules[idx])
        intersection = intersection & other_molecules
        print(f"  与口袋 {idx+1} 的交集: {len(intersection)} 个分子")
    
    return intersection

def get_intersection_with_scores(all_molecules, intersection_smiles):
    """获取交集中每个分子的平均分数和每个靶点的分数"""
    molecule_scores = defaultdict(list)
    molecule_scores_by_target = defaultdict(dict)
    
    for idx, molecules in all_molecules.items():
        for smiles, score in molecules:
            if smiles in intersection_smiles:
                molecule_scores[smiles].append(score)
                molecule_scores_by_target[smiles][idx] = score
    
    # 计算平均分数并保存每个靶点的分数
    result = []
    for smiles in intersection_smiles:
        scores = molecule_scores[smiles]
        avg_score = sum(scores) / len(scores)
        # 获取每个靶点的分数（按顺序）
        target_scores = [molecule_scores_by_target[smiles].get(i, None) for i in range(len(all_molecules))]
        result.append((smiles, avg_score, target_scores))
    
    # 按平均分数排序
    result.sort(key=lambda x: x[1], reverse=True)
    return result

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("使用方法: python compute_intersection.py <top_n> <output_file> <target_name1> <result_file1> <target_name2> <result_file2> ...")
        sys.exit(1)
    
    top_n = int(sys.argv[1])
    output_file = sys.argv[2]
    # 参数格式：target_name1, result_file1, target_name2, result_file2, ...
    args = sys.argv[3:]
    target_names = [args[i] for i in range(0, len(args), 2)]
    result_files = [args[i] for i in range(1, len(args), 2)]
    
    if len(target_names) != len(result_files):
        print("错误: 靶点名称和结果文件数量不匹配")
        sys.exit(1)
    
    print(f"加载结果文件（每个取 Top {top_n}）...")
    all_molecules = load_results(result_files, top_n)
    
    print("\n计算交集...")
    intersection_smiles = compute_intersection(all_molecules)
    
    if len(intersection_smiles) == 0:
        print("\n警告: 没有找到共同的分子！")
        print("建议: 增加 top_n 值或检查筛选结果")
        # 仍然创建空文件
        with open(output_file, 'w') as f:
            f.write("SMILES,avg_score\n")
        sys.exit(0)
    
    print(f"\n找到 {len(intersection_smiles)} 个共同分子")
    
    print("\n计算平均分数...")
    intersection_with_scores = get_intersection_with_scores(all_molecules, intersection_smiles)
    
    print(f"\n保存结果到: {output_file}")
    with open(output_file, 'w') as f:
        # 写入表头：SMILES, avg_score, target1_score, target2_score, ...
        header = "SMILES,avg_score"
        for name in target_names:
            header += f",{name}_score"
        f.write(header + "\n")
        
        for smiles, avg_score, target_scores in intersection_with_scores:
            line = f"{smiles},{avg_score:.6f}"
            for score in target_scores:
                if score is not None:
                    line += f",{score:.6f}"
                else:
                    line += ",N/A"
            f.write(line + "\n")
    
    print(f"\n完成！共 {len(intersection_with_scores)} 个共同分子")
    print(f"\n前 10 个共同分子:")
    for i, (smiles, avg_score, target_scores) in enumerate(intersection_with_scores[:10]):
        score_str = f"{avg_score:.6f} ("
        score_str += ", ".join([f"{target_names[j]}:{s:.6f}" if s is not None else f"{target_names[j]}:N/A" 
                                for j, s in enumerate(target_scores)])
        score_str += ")"
        print(f"  {i+1}. {smiles}: {score_str}")
PYTHON_EOF

# 运行交集计算
echo "计算交集（每个口袋取 Top ${TOP_N}）..."
intersection_file="${TARGET_DIR}/intersection_results.txt"

# 构建参数：target_name1, result_file1, target_name2, result_file2, ...
python_args=()
for i in "${!RESULTS[@]}"; do
    pdb_name=$(basename "${PDB_FILES[$i]}" .pdb)
    python_args+=("${pdb_name}")
    python_args+=("${RESULTS[$i]}")
done

python ${python_script} ${TOP_N} ${intersection_file} "${python_args[@]}"

if [ -f "${intersection_file}" ]; then
    intersection_count=$(wc -l < ${intersection_file})
    intersection_count=$((intersection_count - 1))  # 减去标题行
    
    echo ""
    echo "=========================================="
    echo "筛选完成！"
    echo "=========================================="
    echo "交集结果: ${intersection_file}"
    echo "共同分子数: ${intersection_count}"
    
    if [ ${intersection_count} -gt 0 ]; then
        echo ""
        echo "前 10 个共同分子:"
        head -11 ${intersection_file} | tail -10
    else
        echo ""
        echo "注意: 没有找到共同分子，建议增加 top_n 值"
        echo "当前 top_n: ${TOP_N}"
        echo "可以尝试: bash screen_dual_target.sh ${TARGET_DIR} ${GPU_ID} 5000"
    fi
else
    echo "错误: 未能生成交集结果文件"
    exit 1
fi

# 清理临时脚本
rm -f ${python_script}

echo ""
echo "=========================================="
echo "所有结果文件:"
for i in "${!RESULTS[@]}"; do
    pdb_name=$(basename "${PDB_FILES[$i]}" .pdb)
    echo "  ${pdb_name}: ${RESULTS[$i]} ($(wc -l < ${RESULTS[$i]}) 个候选)"
done
echo ""
echo "交集结果:"
echo "  ${intersection_file} (${intersection_count} 个共同分子)"
echo "=========================================="
