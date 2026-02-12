# DrugCLIP 使用大型分子库指南

## 一、ChemDIV 数据库大小

**ChemDIV 当前大小：约 165 万分子（1,648,137）**

虽然 ChemDIV 已经是一个相当大的商业化合物库，但对于大规模虚拟筛选来说，确实有更大的数据库可用。

---

## 二、可用的更大分子库

### 1. ZINC 数据库
- **ZINC-22**: **数十亿级别**（multi-billion scale）
- **特点**：
  - 免费、商业可用
  - 包含可购买化合物
  - 提供 GUI 工具 CartBlanche
  - 支持相似性搜索
- **下载**: https://zinc22.docking.org/
- **格式**: SMILES, SDF

### 2. Enamine REAL 数据库
- **REAL Database**: **约 104 亿分子**（10.4 billion）
- **REAL Space**: **约 781 亿虚拟分子**（78.1 billion）
- **xREAL**: **约 4 万亿分子**（4 trillion）
- **特点**：
  - 可合成化合物（3-4 周交付，~80% 成功率）
  - 提供多种子集（9.6M, 95.6M, 956M, 7.88B 等）
  - 支持按重原子数分区
- **下载**: https://enamine.net/compound-collections/real-compounds
- **格式**: SMILES, SDF

### 3. PubChem
- **大小**: **超过 1 亿化合物**
- **特点**：
  - 免费、开放
  - 包含实验验证数据
  - 提供多种下载格式
- **下载**: https://pubchem.ncbi.nlm.nih.gov/
- **格式**: SMILES, SDF, CSV

### 4. 其他大型数据库
- **ChEMBL**: 约 200 万化合物（含生物活性数据）
- **DrugBank**: 约 1.4 万药物分子
- **GDB-17**: 约 1660 亿虚拟分子（理论可能的小分子）

---

## 三、如何使用更大的分子库

### 方法 1: 使用预编码的分子嵌入（推荐）

如果您有预编码的分子嵌入文件（`.pkl` 或 `.h5` 格式），可以使用分块处理：

#### 步骤 1: 准备分子嵌入文件

将大型分子库编码为嵌入文件（可以分块进行）：

```bash
# 使用 encode_mols.sh 编码分子库
# 可以分批次处理，例如每次处理 30000 个分子
bash encode_mols.sh 0 /path/to/large_mol_library.lmdb /path/to/save_embeddings/ 0 30000

# 继续处理下一批
bash encode_mols.sh 0 /path/to/large_mol_library.lmdb /path/to/save_embeddings/ 30000 60000
# ... 依此类推
```

#### 步骤 2: 使用分块筛选脚本

```bash
python utils/screening_chunk.py \
    --gpu_num 8 \
    --mol_embs /path/to/embeddings/chunk0.pkl /path/to/embeddings/chunk1.pkl ... \
    --zscore_embs /path/to/embeddings/chunk0.pkl \
    --pocket_reps /path/to/pocket_reps.pkl \
    --batch_size 4 \
    --output_dir ./screening_results \
    --rm_intermediate
```

#### 步骤 3: 检索 SMILES 字符串

```bash
python utils/retrieve_chunk.py \
    --input_files screening_results/merge*.pkl \
    --mol_lmdb /path/to/mol_library.lmdb \
    --output_dir final_results \
    --num_threads 8
```

### 方法 2: 实时编码（适合中等大小库）

如果分子库可以放入内存，可以设置 `use_cache=False`：

```bash
# 修改 retrieval.sh
MOL_PATH="/path/to/your/large_mol_library.lmdb"
use_cache=False

# 运行筛选
bash retrieval.sh
```

**注意**: 这会实时编码所有分子，速度较慢，但不需要预先编码。

---

## 四、准备大型分子库的 LMDB 格式

### 步骤 1: 下载分子库

以 ZINC 为例：

```bash
# 下载 ZINC 子集（例如：ZINC15 lead-like，约 400 万分子）
wget https://zinc15.docking.org/substances/subsets/lead-like.tgz
tar -xzf lead-like.tgz
```

### 步骤 2: 转换为 LMDB 格式

您需要编写脚本将 SMILES 转换为 DrugCLIP 所需的 LMDB 格式。参考项目中的数据处理代码：

```python
# 示例：将 SMILES 文件转换为 LMDB
from rdkit import Chem
import lmdb
import pickle
import numpy as np

def smiles_to_lmdb(smiles_file, output_lmdb):
    env = lmdb.open(output_lmdb, map_size=1099511627776)
    
    with env.begin(write=True) as txn:
        idx = 0
        with open(smiles_file, 'r') as f:
            for line in f:
                smiles = line.strip()
                if not smiles:
                    continue
                
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue
                    
                    # 转换为 DrugCLIP 所需的格式
                    # 这里需要根据项目的数据格式要求进行处理
                    data = {
                        'smiles': smiles,
                        'atoms': [...],  # 原子类型
                        'coordinates': [...],  # 3D 坐标
                    }
                    
                    txn.put(str(idx).encode(), pickle.dumps(data))
                    idx += 1
                except:
                    continue
```

### 步骤 3: 验证 LMDB 文件

```bash
python -c "
import lmdb
import pickle
env = lmdb.open('/path/to/your/library.lmdb', readonly=True)
with env.begin() as txn:
    cursor = txn.cursor()
    count = sum(1 for _ in cursor)
    print(f'Total molecules: {count}')
"
```

---

## 五、性能优化建议

### 1. 分块处理策略

对于超大型数据库（>10 亿分子），建议：

```bash
# 1. 先使用小样本集快速测试（如 100 万分子）
# 2. 如果结果满意，再扩展到完整数据库
# 3. 使用分块处理避免内存问题
```

### 2. 多 GPU 并行

```bash
# 使用多个 GPU 加速编码和筛选
CUDA_VISIBLE_DEVICES=0,1,2,3 python utils/screening_chunk.py \
    --gpu_num 4 \
    ...
```

### 3. 使用多样性子集

许多数据库提供预筛选的子集：

- **Enamine REAL**: 提供 9.6M, 95.6M, 956M 等子集
- **ZINC**: 提供 lead-like, drug-like 等子集
- **PubChem**: 可以按分子量、LogP 等筛选

### 4. 分层筛选策略

1. **第一轮**: 使用快速方法筛选到 Top 1%（如使用小样本集）
2. **第二轮**: 对 Top 1% 使用更精确的方法
3. **第三轮**: 对 Top 0.1% 进行详细分析

---

## 六、实际示例：使用 ZINC 数据库

### 完整流程

```bash
# 1. 下载 ZINC 子集（例如：lead-like，约 400 万分子）
cd /shared/healthinfolab/phz24002/DrugClip/data
mkdir -p zinc_library
cd zinc_library

# 下载（需要注册 ZINC 账户）
# wget https://zinc15.docking.org/substances/subsets/lead-like.tgz

# 2. 转换为 LMDB 格式（需要编写转换脚本）
# python convert_zinc_to_lmdb.py zinc_lead-like.smi zinc_lead-like.lmdb

# 3. 编码分子嵌入（分块进行）
mkdir -p zinc_embeddings
bash encode_mols.sh 0 ./zinc_lead-like.lmdb ./zinc_embeddings/ 0 100000
bash encode_mols.sh 0 ./zinc_lead-like.lmdb ./zinc_embeddings/ 100000 200000
# ... 继续处理所有分子

# 4. 使用分块筛选
python utils/screening_chunk.py \
    --gpu_num 8 \
    --mol_embs ./zinc_embeddings/*.pkl \
    --zscore_embs ./zinc_embeddings/chunk0.pkl \
    --pocket_reps ./data/targets/MY_TARGET/pocket_reps.pkl \
    --batch_size 4 \
    --output_dir ./zinc_screening_results \
    --rm_intermediate

# 5. 检索结果
python utils/retrieve_chunk.py \
    --input_files ./zinc_screening_results/merge*.pkl \
    --mol_lmdb ./zinc_lead-like.lmdb \
    --output_dir ./zinc_final_results \
    --num_threads 8
```

---

## 七、数据库大小对比

| 数据库 | 大小 | 特点 | 访问方式 |
|--------|------|------|----------|
| **ChemDIV** | ~165 万 | 商业库，已验证 | 已提供 |
| **ZINC-22** | 数十亿 | 免费，可购买 | 需下载 |
| **Enamine REAL** | 104 亿 | 可合成，多种子集 | 需下载/API |
| **Enamine xREAL** | 4 万亿 | 超大规模，需 API | API 访问 |
| **PubChem** | 1 亿+ | 免费，含实验数据 | 需下载 |
| **ChEMBL** | ~200 万 | 含生物活性数据 | 需下载 |

---

## 八、推荐工作流程

### 对于新目标：

1. **快速测试**（1-2 小时）
   - 使用 ChemDIV（165 万分子）
   - 快速验证方法可行性

2. **扩展筛选**（1-2 天）
   - 使用 ZINC lead-like（400 万分子）
   - 或 Enamine REAL 9.6M 子集

3. **大规模筛选**（1-2 周）
   - 使用完整 ZINC 或 Enamine REAL（10 亿+）
   - 需要分块处理和大量计算资源

---

## 九、注意事项

1. **存储空间**: 大型数据库需要大量存储空间
   - 10 亿分子的嵌入文件可能需要数 TB 空间

2. **计算时间**: 
   - 编码 10 亿分子可能需要数天到数周
   - 筛选时间取决于 GPU 数量和批次大小

3. **内存管理**: 
   - 使用 `screening_chunk.py` 避免内存溢出
   - 及时清理中间文件（`--rm_intermediate`）

4. **数据格式**: 
   - 确保分子库格式与 DrugCLIP 兼容
   - 可能需要预处理和格式转换

---

## 十、获取帮助

如果需要使用特定的大型数据库，建议：

1. 查看数据库官方文档
2. 联系数据库提供商获取技术支持
3. 参考项目 GitHub Issues: https://github.com/THU-ATOM/Drug-The-Whole-Genome
