"""特征工程算子：纯 numpy 实现，零 sklearn 依赖。

提供 4 个算子（均实现 :class:`BaseTransformer` 契约）：
    - IdentityTransformer       恒等（保留原始特征，作为"不工程化"的对照空间）
    - PolynomialTransformer     二阶多项式展开（平方 + 唯一交叉，d(d+1)/2 新列）
    - InteractionTransformer     仅交叉项 x_i*x_j (i<j)，d(d-1)/2 新列
    - GroupAggregationTransformer  分组聚合（逐组均值/标准差，**有状态**）

设计铁律
--------
- 算子**只追加**特征（保留原始列），保证"工程化 vs 原始"公平对比。
- ``GroupAggregationTransformer`` 必须**有状态**：``fit`` 在训练集上学分组统计量，
  ``transform`` 只做查表应用。嵌套交叉验证里它会被 fit 在训练折、应用到测试折，
  测试折的未见分组一律回退全局统计量 —— 这是杜绝数据泄漏的关键。
- 任何 NaN/Inf 在出口前被 ``_sanitize`` 夹平，避免下游 CV 崩溃。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..core.types import Dataset


@runtime_checkable
class Transformer(Protocol):
    """特征工程算子契约（标注/导入用，与 ``core.interfaces.BaseTransformer`` 同义）。

    各具体算子（Identity/Polynomial/Interaction/GroupAggregation）在结构上满足本
    协议即可，无需显式继承 —— 这层定义只为 ``fusion.feffuse`` 的 ``transformer_obj``
    标注与包内导入提供统一名字。
    """

    name: str

    def fit(self, ds: Dataset) -> Transformer: ...

    def transform(self, ds: Dataset) -> Dataset: ...


# 数值护栏：特征矩阵绝对值上限，超出视为异常并夹平。
_CLIP = 1e12


def _sanitize(arr: np.ndarray) -> np.ndarray:
    """把所有 NaN/Inf 归零并夹到 ±_CLIP，保证下游 CV 不崩。"""
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, -_CLIP, _CLIP)


def _nanmean(a: np.ndarray) -> float:
    """忽略 NaN 的均值；全 NaN 退化为 0。"""
    m = np.nanmean(a) if a.size else np.nan
    return 0.0 if not np.isfinite(m) else float(m)


def _nanstd(a: np.ndarray) -> float:
    """忽略 NaN 的标准差；退化为 0。"""
    if a.size == 0:
        return 0.0
    s = np.nanstd(a, ddof=0)
    return 0.0 if not np.isfinite(s) else float(s)


class IdentityTransformer:
    """恒等算子：原样返回，作为"不工程化"的对照特征空间。"""

    name = "identity"

    def fit(self, ds: Dataset) -> IdentityTransformer:
        return self

    def transform(self, ds: Dataset) -> Dataset:
        return ds


class PolynomialTransformer:
    """二阶多项式展开：平方 + 唯一交叉项，共 d(d+1)/2 个新特征。"""

    name = "polynomial"

    def fit(self, ds: Dataset) -> PolynomialTransformer:
        return self

    def transform(self, ds: Dataset) -> Dataset:
        X = np.asarray(ds.X, dtype=float)
        n, d = X.shape
        cols, names = [], []
        for i in range(d):
            cols.append(_sanitize(X[:, i] ** 2))
            names.append(f"{ds.feature_names[i]}**2")
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(_sanitize(X[:, i] * X[:, j]))
                names.append(f"{ds.feature_names[i]}*{ds.feature_names[j]}")
        new = np.column_stack(cols)
        return ds.with_X(np.column_stack([X, new]), list(ds.feature_names) + names)


class InteractionTransformer:
    """仅交叉项 x_i*x_j (i<j)，共 d(d-1)/2 个新特征。"""

    name = "interaction"

    def fit(self, ds: Dataset) -> InteractionTransformer:
        return self

    def transform(self, ds: Dataset) -> Dataset:
        X = np.asarray(ds.X, dtype=float)
        n, d = X.shape
        cols, names = [], []
        for i in range(d):
            for j in range(i + 1, d):
                cols.append(_sanitize(X[:, i] * X[:, j]))
                names.append(f"{ds.feature_names[i]}*{ds.feature_names[j]}")
        new = np.column_stack(cols)
        return ds.with_X(np.column_stack([X, new]), list(ds.feature_names) + names)


class GroupAggregationTransformer:
    """分组聚合算子（**有状态**）：为每个非分组特征追加「组内均值 / 组内标准差」。

    为何有状态：分组统计量必须在训练集上学习，测试集只查表。若测试折里出现
    训练折未见过的分组，用全局统计量兜底，绝不跨折泄漏。
    """

    name = "group"

    def __init__(self) -> None:
        self._has_group = False
        self._codes: np.ndarray = np.empty(0)
        self._mean: np.ndarray = np.empty((0, 0))
        self._std: np.ndarray = np.empty((0, 0))
        self._gmean: np.ndarray = np.empty(0)
        self._gstd: np.ndarray = np.empty(0)
        self._feat_idx: list[int] = []

    def fit(self, ds: Dataset) -> GroupAggregationTransformer:
        if ds.group_idx is None:
            self._has_group = False
            return self
        g = int(ds.group_idx)
        X = np.asarray(ds.X, dtype=float)
        feat_idx = [j for j in range(X.shape[1]) if j != g]
        codes = np.unique(X[:, g])
        mean = np.empty((len(codes), len(feat_idx)))
        std = np.empty((len(codes), len(feat_idx)))
        gmean = np.empty(len(feat_idx))
        gstd = np.empty(len(feat_idx))
        for k, code in enumerate(codes):
            mask = X[:, g] == code
            for t, j in enumerate(feat_idx):
                col = X[mask, j]
                mean[k, t] = _nanmean(col)
                std[k, t] = _nanstd(col)
                gmean[t] = _nanmean(X[:, j])
                gstd[t] = _nanstd(X[:, j])
        self._has_group = True
        self._codes = codes
        self._mean = mean
        self._std = std
        self._gmean = gmean
        self._gstd = gstd
        self._feat_idx = feat_idx
        return self

    def transform(self, ds: Dataset) -> Dataset:
        if not self._has_group or ds.group_idx is None:
            return ds
        g = int(ds.group_idx)
        X = np.asarray(ds.X, dtype=float)
        grp = X[:, g]
        pos = np.searchsorted(self._codes, grp)
        # searchsorted 对未见分组值返回 == len(codes) 的插入下标，直接索引会越界；
        # 先夹到合法范围再做等值判断，越界者自然判定为「未命中」→ 走全局兜底。
        safe_pos = np.clip(pos, 0, len(self._codes) - 1)
        ok = (pos < len(self._codes)) & (self._codes[safe_pos] == grp)
        pos = np.where(ok, pos, 0)  # 未见分组统一落到索引 0，再用兜底值覆盖
        cols, names = [], []
        for t, j in enumerate(self._feat_idx):
            mean_col = np.where(ok, self._mean[pos, t], self._gmean[t])
            std_col = np.where(ok, self._std[pos, t], self._gstd[t])
            cols.append(_sanitize(mean_col))
            cols.append(_sanitize(std_col))
            names.append(f"mean({ds.feature_names[j]}|g)")
            names.append(f"std({ds.feature_names[j]}|g)")
        new = np.column_stack(cols)
        return ds.with_X(np.column_stack([X, new]), list(ds.feature_names) + names)


TRANSFORMERS = {
    "identity": IdentityTransformer,
    "polynomial": PolynomialTransformer,
    "interaction": InteractionTransformer,
    "group": GroupAggregationTransformer,
}


def build_transformers(names=None) -> list:
    """构造全部（或指定）特征工程算子实例。

    Parameters
    ----------
    names : list[str] | None
        限定返回的算子名；None 表示全部。
    """
    all_t = {
        "identity": IdentityTransformer(),
        "polynomial": PolynomialTransformer(),
        "interaction": InteractionTransformer(),
        "group": GroupAggregationTransformer(),
    }
    if names is None:
        return [all_t[n] for n in ("identity", "polynomial", "interaction", "group")]
    return [all_t[n] for n in names if n in all_t]


def build_engineered_datasets(ds: Dataset, names=None) -> dict:
    """对每个算子 fit+transform，返回 {算子名: 工程化 Dataset}。"""
    out = {}
    for t in build_transformers(names):
        t.fit(ds)
        out[t.name] = t.transform(ds)
    return out


def describe_ops() -> dict:
    """返回各算子的元信息（可解释性 / 审计用）。"""
    return {
        "identity": {"kind": "passthrough", "new_features": "0"},
        "polynomial": {"kind": "degree-2 expansion", "new_features": "d(d+1)/2"},
        "interaction": {"kind": "cross terms", "new_features": "d(d-1)/2"},
        "group": {"kind": "per-group mean/std", "new_features": "2*(d-1)"},
    }
