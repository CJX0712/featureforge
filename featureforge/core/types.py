"""FeatureForge 核心数据类型契约 —— 全系统只依赖这些 dataclass 通信。

设计原则
--------
1. 所有特征工程算子接收/返回 ``Dataset``，保证元数据（特征名、分组列、
   类别列、任务类型）随数据一起流动，不会出现"特征名漂移"。
2. ``SelectionResult.selected`` 恒为**整数索引**，指向其所在特征空间的列；
   跨空间比较时由调用方负责翻译，禁止在算子内部猜测。
3. 任务类型（binary / multiclass / regression）只推断一次，之后显式传递。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 任务类型取值域。regression 与 classification 的评测口径不同，必须区分。
TASKS = ("binary", "multiclass", "regression")


def infer_task(y: np.ndarray) -> str:
    """从标签向量自动推断任务类型。

    - 浮点且非整数取值 → regression（无歧义）
    - 唯一值数量 > 10 → regression（大量离散值按回归处理）
    - 唯一值数量 <= 2 → binary
    - 其余 → multiclass
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if np.issubdtype(y.dtype, np.floating):
        if not np.all(np.isfinite(y)) or not np.allclose(y, np.round(y)):
            return "regression"
    n_unique = int(len(np.unique(y)))
    if n_unique > 10:
        return "regression"
    if n_unique <= 2:
        return "binary"
    return "multiclass"


@dataclass
class Dataset:
    """表格数据集容器。

    Attributes
    ----------
    X : float 数组 (n_samples, n_features)
        特征矩阵。缺失值允许为 NaN，由算子自行处理。
    y : 数组 (n_samples,)
        标签。分类为整数/字符串，回归为浮点。
    feature_names : list[str]
        特征名，长度必须等于 X 的列数。
    categorical : list[int]
        类别型特征列索引（供编码算子使用）。
    group_idx : int | None
        分组列索引（供分组聚合算子使用）。
    task : str
        任务类型；``"auto"`` 表示构造时自动推断。
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    categorical: list[int] = field(default_factory=list)
    group_idx: int | None = None
    task: str = "auto"

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=float)
        self.y = np.asarray(self.y)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2-D, got ndim={self.X.ndim}")
        if self.y.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"X/y sample mismatch: X has {self.X.shape[0]} rows, y has {self.y.shape[0]}"
            )
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) != X columns ({self.X.shape[1]})"
            )
        if self.task == "auto":
            self.task = infer_task(self.y)

    @property
    def n_samples(self) -> int:
        """样本数（属性而非方法，便于 f-string 直接引用）。"""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """特征数。"""
        return int(self.X.shape[1])

    def subset(self, cols: list[int]) -> Dataset:
        """按列索引取子集，同步裁剪元数据（特征名重编号、分组列重映射）。

        分组列若不在 ``cols`` 内则丢弃（聚合已完成，无需保留）。
        """
        cols = [int(c) for c in cols]
        new_group = None
        if self.group_idx is not None and self.group_idx in cols:
            new_group = cols.index(self.group_idx)
        return Dataset(
            X=self.X[:, cols],
            y=self.y,
            feature_names=[self.feature_names[c] for c in cols],
            categorical=[i for i, c in enumerate(cols) if c in self.categorical],
            group_idx=new_group,
            task=self.task,
        )

    def take(self, rows) -> Dataset:
        """按行索引取子集（嵌套交叉验证的折切分用）。元数据整体沿用。"""
        rows = np.asarray(rows, dtype=int)
        return Dataset(
            X=self.X[rows],
            y=self.y[rows],
            feature_names=list(self.feature_names),
            categorical=list(self.categorical),
            group_idx=self.group_idx,
            task=self.task,
        )

    def with_X(self, X_new: np.ndarray, names_new: list[str]) -> Dataset:
        """替换特征矩阵（特征工程算子的标准出口）。分组列沿用原值。"""
        return Dataset(
            X=np.asarray(X_new, dtype=float),
            y=self.y,
            feature_names=list(names_new),
            categorical=list(self.categorical),
            group_idx=self.group_idx,
            task=self.task,
        )


@dataclass
class FeatureOp:
    """一条特征工程算子的产出描述（用于可解释性与审计）。

    Attributes
    ----------
    name : 算子名
    inputs : 该算子消费的原始特征索引
    outputs : 产出的新特征名
    kind : 算子类别标签
    """

    name: str
    inputs: list[int]
    outputs: list[str]
    kind: str


@dataclass
class SelectionResult:
    """特征选择结果。

    Attributes
    ----------
    selected : list[int]
        被选中的列索引，指向传入 Dataset 的特征空间。
    selected_names : list[str]
        被选中的特征名。
    scores : dict[str, np.ndarray]
        方法名 -> 每个特征的得分（长度 = 输入特征数）。用于跨选择器融合。
    method : str
        产生该结果的方法名。
    """

    selected: list[int]
    selected_names: list[str]
    scores: dict[str, np.ndarray]
    method: str


@dataclass
class EvalMetrics:
    """一次「原始特征 vs 工程化选择特征」对比的评测结论。

    Attributes
    ----------
    baseline_score : 原始特征下的交叉验证得分。
    selected_score : 工程化+选择后特征下的交叉验证得分。
    n_features_baseline / n_features_selected : 特征维度。
    delta : selected_score - baseline_score。
    winner : "fusion" 表示采用工程化特征集；"baseline" 表示守护回退原始特征。
    safeguard_triggered : 非劣守护是否触发（工程化未达增益门槛）。
    parsimony : 在得分非劣前提下特征数是否下降（真实"瘦身成功"）。
    """

    baseline_score: float
    selected_score: float
    n_features_baseline: int
    n_features_selected: int
    delta: float
    winner: str
    safeguard_triggered: bool = False
    parsimony: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureForgeResult:
    """FEFFuse 旗舰的一次完整输出。

    Attributes
    ----------
    chosen_names : 最终采用的特征名列表。
    chosen_dataset : 最终采用特征列构成的数据集（已裁剪）。
    transformer : 获胜的特征工程算子名（回退时为 "identity"）。
    selector : 获胜的选择器名（回退时为 "none"）。
    metrics : 与原始特征的对比评测。
    fused_weights : 融合阶段各选择器的最终权重（未用 Optuna 时为等权）。
    candidates : 候选 (transformer, selector) 组合的评测明细，供审计。
    """

    chosen_names: list[str]
    chosen_dataset: Dataset
    transformer: str
    selector: str
    metrics: EvalMetrics
    fused_weights: dict[str, float]
    candidates: list[dict[str, Any]]
