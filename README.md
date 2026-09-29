# FeatureForge

> **自动化特征工程 × 选择器融合（FEFFuse）** — 零重型依赖、可审计、可复现、诚实。
> 作者：晨星（GitHub: [CJX0712](https://github.com/CJX0712)）

FeatureForge 把"特征工程"和"特征选择"两件事**融合**成一个自适应、有诚实闸门的
流水线：用一组可解释算子拓宽特征空间，再用多个异构选择器融合回一个紧凑、无偏、
不劣于原始特征的最终特征集。它**绝不**为了"看起来更聪明"而输出一个实际更差的集合。

---

## 特性

- **4 个特征工程算子**（纯 numpy）：Identity / Polynomial(degree-2) / Interaction / GroupAggregation（有状态，fit 学组统计量）。
- **5 个选择器**（sklearn 主后端 + numpy 兜底）：Variance / Correlation / MutualInfo / L1(`l1_ratio=1.0, solver="saga"`) / ModelImportance(RandomForest)。
- **FEFFuse 旗舰**：transformer×selector 候选网格 → 最佳空间 → 多选择器 min-max 归一加权 top-k 融合（Optuna TPE，确定性网格兜底）。
- **嵌套交叉验证**：消除特征选择偏差，得分**无偏**。
- **双层非劣守护**：L1 融合不劣于最佳单候选；L2 不劣于原始特征，否则回退。
- **诚实基准**：6 个 DGP 覆盖"工程化必胜 / 精简 / 无信号 / 有害"，含噪声与高维守护回退。
- **零 pandas 依赖**，单 venv 即可跑；锁定依赖清单见 `requirements.lock.txt`。

---

## 快速开始

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.lock.txt

# 单数据集
python -m featureforge.cli run --dgp interaction

# 跨 6 个 DGP × 多个种子 的完整基准（落盘 benchmark.json）
python -m featureforge.examples.run_demo

# 全部测试（含嵌套无偏诚实闸门）
pytest -q
```

---

## 六个诚实基准 DGP（预期结论）

| DGP | 结构 | 诚实预期 | 判定阈值 |
|-----|------|----------|----------|
| interaction | y ∝ x0·x1 | 工程化**显著增益** | `delta > 0.05` |
| polynomial | y ∝ x0²+x1² | 工程化**显著增益** | `delta > 0.05` |
| group | y ∝ 分组聚合量 | 分组算子**显著增益** | `delta > 0.02` 且选 group |
| redundant | y ∝ x0，堆近重复列 | 得分持平、**维度下降** | `n_selected < n_baseline` |
| noise | y 与 X 独立 | **不得**有增益 | `delta < 0.05` |
| highdim | 高维弱信号+噪声 | 膨胀过拟合 → **守护回退** | `delta <= tie_tolerance` |

> 设计原则：不是所有 DGP 都让工程化获胜。噪声与高维场景必须**老老实实回退**，
> 否则就是选择偏差制造的幻觉（详见 `docs/architecture.md` 第 2 节，含修复前后对比）。

### 最新基准结果（嵌套无偏，与 pytest 诚实闸门断言同源）

由 `python -m featureforge.examples.run_demo` 生成（见 `benchmark.json`，逐位可复现）。
配置：n_samples=200、nested_folds=3、确定性网格融合（use_optuna=False）。

| DGP | base | feffuse | delta | n_base | n_sel | win% | guard% |
|-----|------|---------|-------|--------|-------|------|--------|
| interaction | 0.4533 | 0.7715 | +0.3182 | 6 | 5.0 | 100 | 0 |
| polynomial | 0.4618 | 0.8067 | +0.3449 | 6 | 10.7 | 100 | 0 |
| group | 0.9049 | 0.9734 | +0.0684 | 6 | 13.0 | 100 | 0 |
| redundant | 0.9383 | 0.9584 | +0.0201 | 7 | 2.3 | 100 | 0 |
| noise | 0.5085 | 0.5167 | +0.0082 | 6 | 6.0 | 33 | 67 |
| highdim | 0.5800 | 0.5834 | +0.0034 | 30 | 22.3 | 33 | 67 |
| **AGGREGATE** | 0.6411 | 0.7683 | +0.1272 | — | — | 78 | 22 |

> 诚实结论：interaction/polynomial 工程化带来 **+0.32** 量级的显著增益；group 分组
> 聚合带来 **+0.07** 增益并选中 group 算子；redundant 得分持平但维度从 7 降到 2.3
> （精简获胜）；noise 与 highdim **不提供有意义增益**（delta ≤ tie_tolerance，约
> 1/3~2/3 种子触发守护回退）——系统没有为了"看起来更聪明"而输出更差的特征集。

---

## 命令行

```
python -m featureforge.cli run       --dgp <name> [--seed N] [--no-optuna]
python -m featureforge.cli benchmark --dgps a b c --seeds 1 2 3 [--json FILE]
```

环境变量可覆盖配置（`FEATUREFORGE_RANDOM_STATE` / `FEATUREFORGE_USE_OPTUNA` /
`FEATUREFORGE_CV_FOLDS` / `FEATUREFORGE_NESTED_FOLDS` / `FEATUREFORGE_TIE_TOLERANCE` 等）。

---

## 项目结构

```
featureforge/
├── featureforge/
│   ├── core/            # types / config / errors / interfaces
│   ├── data/            # 6 个 DGP 生成器 + 零 pandas 加载器
│   ├── engineering/     # 4 个特征工程算子（纯 numpy）
│   ├── selection/       # 5 个选择器（sklearn + numpy 兜底）
│   ├── eval/            # 嵌套交叉验证评分
│   ├── fusion/feffuse.py# 旗舰 FEFFuse
│   ├── pipeline/        # run_dataset / benchmark / 表格
│   └── cli.py           # run / benchmark
├── tests/               # 37 测试（闭式交叉验证 + 诚实闸门）
├── examples/run_demo.py # 落盘 benchmark.json
└── docs/architecture.md # 架构与诚实基准白皮书
```

---

## 复现与 CI

```bash
pytest -q                                  # 单元测试 + 诚实基准断言
python -m ruff check . && python -m ruff format --check .   # 代码风格收口
python -m featureforge.examples.run_demo   # 生成 benchmark.json（逐位可复现）
```

所有随机源固定（`random_state=42`；数据种子与 CV 种子分离），**逐位可复现**。

---

## 许可证

MIT © 晨星（CJX0712）
