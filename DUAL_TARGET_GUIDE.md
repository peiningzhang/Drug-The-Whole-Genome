# DrugCLIP 双目标筛选指南

## 功能说明

`screen_dual_target.sh` 脚本用于筛选能**同时匹配多个口袋**的分子（取交集）。

### 工作流程

1. **自动检测 PDB 文件**：扫描目录中的所有 PDB 文件
2. **修复配体名**：自动检测并修复 PDB 文件名以匹配实际配体名
3. **分别筛选**：对每个口袋独立进行虚拟筛选
4. **计算交集**：取每个口袋的 Top N 分子，计算交集
5. **输出结果**：保存共同分子及其平均分数

---

## 使用方法

### 基本用法

```bash
# 在 GPU 节点上运行
ssh gpu28
source /shared/healthinfolab/phz24002/anaconda3/bin/activate drugclip
cd /shared/healthinfolab/phz24002/DrugClip

# 运行双目标筛选
bash screen_dual_target.sh ./data/targets/KOR-5H2A 0 1000
```

### 参数说明

```bash
bash screen_dual_target.sh <target_dir> [gpu_id] [top_n]
```

- `target_dir`: 包含多个 PDB 文件的目录
- `gpu_id`: GPU 设备号（默认: 0）
- `top_n`: 每个口袋取 Top N 分子进行交集（默认: 1000）

---

## 示例：KOR-5H2A 双目标筛选

### 目录结构

```
data/targets/KOR-5H2A/
  ├── 8dzp_KOR.pdb      # KOR 受体
  └── 9as8_5HT2AR.pdb   # 5HT2AR 受体
```

### 运行命令

```bash
bash screen_dual_target.sh ./data/targets/KOR-5H2A 0 1000
```

### 输出文件

脚本会生成以下文件：

1. **单独筛选结果**：
   - `8dzp_KOR_results.txt` - KOR 口袋的筛选结果
   - `9as8_5HT2AR_results.txt` - 5HT2AR 口袋的筛选结果

2. **交集结果**：
   - `intersection_results.txt` - 同时匹配两个口袋的分子

3. **中间文件**：
   - `8dzp_KOR_pocket/` - KOR 口袋的处理目录
   - `9as8_5HT2AR_pocket/` - 5HT2AR 口袋的处理目录

---

## 结果格式

### 单独筛选结果

每行格式：`SMILES,score`

```
CCO,2.345
CCN,2.123
...
```

### 交集结果

每行格式：`SMILES,avg_score`

```
CCO,2.456
CCN,2.234
...
```

其中 `avg_score` 是分子在所有口袋中的平均分数。

---

## 工作原理

### 1. 配体名自动修复

脚本会自动检测 PDB 文件中的实际配体名，并创建正确命名的副本：

- `8dzp_KOR.pdb` → `8dzp_U99.pdb` (实际配体: U99)
- `9as8_5HT2AR.pdb` → `9as8_91Q.pdb` (实际配体: 91Q)

### 2. 口袋提取

对每个 PDB 文件：
- 提取配体结合口袋（距离配体 ≤ 6Å 的残基）
- 生成 `pocket.lmdb` 文件
- 编码口袋特征

### 3. 虚拟筛选

对每个口袋：
- 使用 DrugCLIP 模型计算分子-口袋相似度
- 输出 Top 2% 的候选分子（默认）

### 4. 交集计算

- 从每个结果中取 Top N 分子（默认 1000）
- 计算 SMILES 的交集
- 计算共同分子的平均分数
- 按平均分数排序

---

## 调整参数

### 增加交集数量

如果交集结果为空或太少，可以增加 `top_n`：

```bash
# 每个口袋取 Top 5000
bash screen_dual_target.sh ./data/targets/KOR-5H2A 0 5000

# 每个口袋取 Top 10000
bash screen_dual_target.sh ./data/targets/KOR-5H2A 0 10000
```

### 使用不同的 fold 版本

如果需要修改 fold 版本，可以编辑脚本或直接运行 Python 命令。

---

## 注意事项

1. **配体名匹配**：脚本会自动修复配体名，但如果 PDB 文件中没有 `HET` 记录，可能需要手动检查

2. **交集可能为空**：如果两个口袋差异很大，交集可能为空。建议：
   - 增加 `top_n` 值
   - 检查单独筛选结果，确认是否有重叠

3. **计算时间**：每个口袋的筛选需要一定时间，请耐心等待

4. **存储空间**：确保有足够的存储空间保存中间文件

---

## 故障排除

### 问题 1: 无法提取配体

**症状**：`pocket.lmdb` 生成失败

**解决**：
1. 检查 PDB 文件格式
2. 确认文件中包含配体（HETATM 记录）
3. 手动检查配体名

### 问题 2: 交集为空

**症状**：`intersection_results.txt` 只有标题行

**解决**：
1. 增加 `top_n` 值（如 5000 或 10000）
2. 检查单独筛选结果，查看是否有重叠的 SMILES
3. 考虑使用更宽松的交集策略（如 Top 5% 的交集）

### 问题 3: CUDA 错误

**症状**：`RuntimeError: No CUDA GPUs are available`

**解决**：
1. 确认在 GPU 节点上运行
2. 检查 `CUDA_VISIBLE_DEVICES` 设置
3. 使用 `nvidia-smi` 检查 GPU 状态

---

## 高级用法

### 自定义交集策略

如果需要更复杂的交集策略（如 Top 5% 的交集），可以修改 Python 脚本中的逻辑。

### 批量处理多个目标对

可以编写循环脚本处理多个目标对：

```bash
for target_pair in KOR-5H2A TARGET1-TARGET2 TARGET3-TARGET4; do
    bash screen_dual_target.sh ./data/targets/${target_pair} 0 1000
done
```

---

## 相关脚本

- `screen_new_target.sh` - 单目标筛选
- `retrieval.sh` - 标准虚拟筛选
- `encode_pocket.sh` - 编码单个口袋
- `fix_pdb_ligand_names.py` - 修复 PDB 文件名
