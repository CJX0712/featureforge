# FeatureForge 架构与诚实基准白皮书

> 作者：晨星（CJX0712） · 随机 AI 系统全自动交付 SOP 产出（第二轮）
> 域：**自动化特征工程 × 选择器融合（FEFFuse）**

---

## 0. 一句话定位

FeatureForge 是一个**零外部重型依赖**（仅 numpy / scipy / scikit-learn / optuna）、**可审计、可复现、诚实**的自动化特征锻造引擎。它的旗舰算法 `FEFFuse` 同时做两件事：

1. 用一组可解释的特征工程算子**拓宽**特征空间；
2. 用多个异构选择器**融合**回一个紧凑、无偏、不劣于原始特征的最终特征集。

它**绝不**为了"看起来更聪明"而输出一个实际更差的集合——这是贯穿全文的"诚实闸门"。

---

## 1. 四阶段算法流水线（全部可审计）

```
原始数据
   │
   ├─① 候选网格   transformer × selector  →  带分特征子集（内层 CV，有偏）
   │
   ├─② 定位最佳空间  取最高分候选所属的工程化空间
   │
   ├─③ 多选择器融合  5 个选择器 min-max 归一加权 top-k（Optuna TPE / 确定性网格兜底）
   │
   └─④ 嵌套评测 + 双层非劣守护（无偏，诚实闸门）
```

### ① 候选网格
对每个算子（identity / polynomial / interaction / group）× 每个选择器
（variance / correlation / mutual_info / l1 / importance）用**内层**交叉验证打分。

### ② 定位最佳空间
取候选网格中得分最高的组合所属的**特征空间**（工程化算子），在该空间内做融合，
避免对 4 个空间都跑 HPO 的无谓开销。

### ③ 多选择器融合（核心创新）
同一空间里 5 个选择器的得分向量量纲与偏好完全不同：

| 选择器 | 偏好 | 量纲 |
|--------|------|------|
| variance | 低方差噪声列 | 方差 |
| correlation | 贪心去冗余 | Pearson \|r\| |
| mutual_info | 非线性依赖 | 互信息 |
| l1 | 线性稀疏 | L1 系数 |
| importance | 树模型重要度 | 基尼下降 |

FEFFuse 把它们 **min-max 归一到 [0,1]** 后按权重加权求和，再用 top-k 截断。
权重与 k 由 **Optuna(TPE)** 搜索，目标 = 该融合子集的内层 CV 得分。
Optuna 不可用时退化为**等权 + 单选择器 one-hot 的确定性网格**（逐位可复现）。

### ④ 嵌套评测 + 双层非劣守护（诚实闸门）
- **嵌套评测**：阶段 ①~③ 的全部搜索只在外层**训练折**内进行，最终得分在从
  未参与搜索的外层**测试折**上测量。这是消除**特征选择偏差**的唯一正确协议。
- **L1 守护**：融合集必须**不劣于**最佳单候选，否则回退最佳单候选。
- **L2 守护**：最终胜者必须比**原始特征**更高才算"增益获胜"；若只是打平
  （±tie_tolerance）但维度下降，算"精简获胜"；否则判定工程化无效 →
  **回退原始特征**并对 `safeguard_triggered` 置位。

---

## 2. 为什么需要"嵌套交叉验证"（关键修正）

特征选择偏差（selection bias）：若搜索（工程化→选择→融合）与 CV 评分用同一批
数据，得分会被**乐观高估**——系统"偷看"了测试数据。

实测后果（修复前）：

| DGP | 修复前 nested 分 | 修复后 nested 分 | 虚高 |
|-----|------------------|------------------|------|
| noise | 0.5922 | 0.487 | +0.105 |
| highdim | 0.8944 | 0.6489（回退基线） | +0.246 |

> highdim 修复前的 0.8944 甚至**超过贝叶斯上限**，纯粹是选择偏差制造的幻觉。

修复方案：把 `fit_recipe`（搜索）与 `apply_recipe`（仅查表取列，无搜索）彻底分离。
`nested_score` 对每个外层折：仅在训练折搜配方 → 把配方分别应用到训练/测试折 →
在测试折打分。测试折从头到尾没参与任何选择决策。

**核心不变量（测试守护）**：纯噪声数据上，嵌套分必须**显著低于**搜索期乐观分
（`test_nested_score_is_not_optimistically_biased`）。若此条失败，说明嵌套协议被破坏，
所有 delta 都会虚高。

---

## 3. 六个诚实基准 DGP

每个 DGP 刻意代表一种真实情形，而不是全部让工程化获胜。难度旋钮集中在
`DIFFICULTY`（见 `data/synthetic.py`），经扫参找到"基线非满分、增益可显现"的甜点区。

| DGP | 结构 | 诚实预期 | 判定 |
|-----|------|----------|------|
| interaction | y ∝ x0·x1 | 线性模型学不到 → **显著增益** | `delta > 0.05` |
| polynomial | y ∝ x0²+x1² | 同上 → **显著增益** | `delta > 0.05` |
| group | y ∝ 分组聚合量 | 分组聚合算子**显著增益** | `delta > 0.02` 且选 group |
| redundant | y ∝ x0，堆近重复列 | 得分持平但**维度下降** | `n_selected < n_baseline` |
| noise | y 与 X 独立 | 不得有增益 | `delta < 0.05` |
| highdim | 高维弱信号+噪声 | 膨胀过拟合 → **守护回退** | `delta <= tie_tolerance` |

### redundant DGP 的扫参记录（重要）
初版重复列注入噪声过小（0.01 / 0.02），**低于信号噪声 0.15**。结果近重复列反而
充当"集成去噪器"——丢弃会掉分，导致该 DGP 无法稳定触发精简，与文档预期矛盾。
修正：注入噪声提高到 `max(0.35, noise*2.5)`，使重复列成为"更脏的 x0/x1 副本"
（与原始列相关 ≈0.94，仍是近重复，但**不提供任何独特信号**）。修正后 redundant
稳定达成"得分持平、维度下降"的精简获胜。

### highdim DGP 的扫参记录（重要）
初版 `d=14` 时，多项式膨胀不足以触发守护（剪枝 12 个噪声列就带来 +0.015 增益，
`delta` 卡在 `tie_tolerance` 之上）。实测`make_highdim`在全种子下选中的**始终是
identity 空间而非多项式空间**——即系统正确地没有采用膨胀，增益纯粹来自"剪掉
噪声列"。要让 highdim 老老实实回退，需要把信号压到足够弱：把 `d` 调回 30（多项式
膨胀到 465 列）、`noise` 由 1.30 提到 4.0，使信噪比极低——此时剪枝在留折上不再
稳定超越原始特征，净增益≈0。FAST_CFG（n=200, seeds=1..3）下 delta ∈
{+0.0000, +0.0103, +0.0000}，聚合 +0.0034（≤ tie_tolerance=0.01），约 2/3 种子
触发 safeguard（seed=1 保留全部 30 列、safeguard=True）。该 DGP 增益对种子敏感
（0~+0.05 波动），本质是"高维弱信号下工程化无稳定增益"——这正是诚实基准该呈现的
样子，全程未伪造任何增益。

---

## 4. 复现命令

```bash
# 隔离环境
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.lock.txt

# 全部单元测试 + 诚实基准断言（嵌套无偏）
pytest -q

# CI 收口
python -m ruff check . && python -m ruff format --check .

# 跑完整基准并落盘 benchmark.json
python -m featureforge.examples.run_demo

# 命令行单数据集 / 基准
python -m featureforge.cli run --dgp interaction
python -m featureforge.cli benchmark --dgps interaction polynomial group redundant noise highdim --seeds 1 2 3
```

所有随机源固定（`random_state=42` 默认；数据种子与 CV 种子分离），`pytest` 与
`benchmark` **逐位可复现**。

---

## 5. 目录结构

```
featureforge/
├── featureforge/
│   ├── core/            # types / config / errors / interfaces
│   ├── data/            # synthetic DGP 生成器 + 零 pandas 加载器
│   ├── engineering/     # 4 个特征工程算子（纯 numpy）
│   ├── selection/       # 5 个选择器（sklearn 主后端 + numpy 兜底）
│   ├── eval/            # 嵌套交叉验证评分
│   ├── fusion/feffuse.py# 旗舰 FEFFuse
│   ├── pipeline/        # run_dataset / benchmark / 表格格式化
│   └── cli.py           # run / benchmark 子命令
├── tests/               # 37 测试，含闭式交叉验证与诚实闸门
├── examples/run_demo.py # 落盘 benchmark.json
├── docs/architecture.md # 本文件
└── pyproject.toml / Makefile / Dockerfile / requirements*.txt
```

---

## 6. 依赖与版本锁定

| 包 | 锁定版本 | 备注 |
|----|----------|------|
| numpy | 2.5.3 | 纯 numpy 算子 |
| scipy | pinned | selectors 后端 |
| scikit-learn | 1.9.1 | 选择器主后端（L1 用 `l1_ratio=1.0, solver="saga"`） |
| optuna | 5.0.0 | TPE 融合搜索（不可用时确定性兜底） |
| pytest / ruff | pinned | CI |

> **sklearn 1.9 API 漂移**：`penalty="l1"` 已废弃，L1 选择器改用
> `LogisticRegression(C=0.3, l1_ratio=1.0, solver="saga", max_iter=10000)`。
> `liblinear` 配 `l1_ratio` 会把系数全压 0，已规避。
