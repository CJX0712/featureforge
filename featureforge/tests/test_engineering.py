"""特征工程算子单测 —— 校验各算子的**不变量**而非泛化能力。

重点守护两条铁律：
1. 算子只追加特征（原始列数不变、顺序不变）；
2. GroupAggregationTransformer 必须**有状态**，未见分组回退全局，杜绝泄漏。
"""

from __future__ import annotations

import numpy as np

from featureforge.core.types import Dataset
from featureforge.data.synthetic import make_group
from featureforge.engineering.transformers import (
    GroupAggregationTransformer,
    IdentityTransformer,
    InteractionTransformer,
    PolynomialTransformer,
    _sanitize,
    build_transformers,
)


def _ds(n: int = 120, d: int = 5, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = (X[:, 0] > 0).astype(int)
    return Dataset(X=X, y=y, feature_names=[f"x{i}" for i in range(d)], task="binary")


def test_identity_preserves_everything() -> None:
    """恒等算子：形状、数值、特征名原样返回。"""
    ds = _ds()
    out = IdentityTransformer().fit(ds).transform(ds)
    assert out.X.shape == ds.X.shape
    assert np.allclose(out.X, ds.X)
    assert out.feature_names == ds.feature_names


def test_polynomial_adds_d_times_dplus1_over2_columns() -> None:
    """多项式展开新增列数 = d(d+1)/2（平方 + 唯一交叉）。"""
    ds = _ds(d=6)
    out = PolynomialTransformer().fit(ds).transform(ds)
    expected = 6 + 6 * 7 // 2
    assert out.X.shape[1] == expected
    # 原始列必须原封不动保留在前面
    assert np.allclose(out.X[:, :6], ds.X)


def test_interaction_adds_d_times_dminus1_over2_columns() -> None:
    """交互算子新增列数 = d(d-1)/2（仅交叉项）。"""
    ds = _ds(d=6)
    out = InteractionTransformer().fit(ds).transform(ds)
    expected = 6 + 6 * 5 // 2
    assert out.X.shape[1] == expected
    assert np.allclose(out.X[:, :6], ds.X)


def test_engineered_features_are_finite() -> None:
    """工程化特征不得含 NaN/Inf（_sanitize 已兜底）。"""
    ds = _ds(d=8)
    for t in (PolynomialTransformer(), InteractionTransformer()):
        out = t.fit(ds).transform(ds)
        assert np.all(np.isfinite(out.X)), t.name


def test_group_aggregation_dimension_and_stateful_fallback() -> None:
    """分组聚合：维度正确，且未见分组回退全局均值（无泄漏）。"""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 2))
    grp = np.repeat([0, 1, 2], 20)
    X = np.column_stack([grp.astype(float), X])
    ds = Dataset(X=X, y=(X[:, 1] > 0).astype(int), feature_names=["g", "a", "b"], group_idx=0)
    t = GroupAggregationTransformer().fit(ds)
    # 注入一个训练集未见过的分组码 9
    X2 = X.copy()
    X2[:, 0] = 9.0
    ds2 = ds.with_X(X2, ds.feature_names)
    out = t.transform(ds2)
    # 2 个非分组特征 × (mean,std) = 4 个新列 → 共 3 + 4 = 7
    assert out.X.shape[1] == 7
    # 未见分组的均值列必须等于该特征的全局均值
    gmean_a = float(np.mean(X[:, 1]))
    gmean_b = float(np.mean(X[:, 2]))
    assert np.allclose(out.X[:, 3], gmean_a)  # mean(a|g)
    assert np.allclose(out.X[:, 5], gmean_b)  # mean(b|g)


def test_group_aggregation_recovers_group_signal_closed_form() -> None:
    """闭式校验：group DGP 下 mean(x1|g) 与标签强相关（增益机制成立）。

    注意：y 经中位数阈值化成**二值**，而点二列相关的理论上限由二值化决定，
    即使关系完全确定也仅≈0.798。故断言门槛取 0.7（远高于"无信号"的 0），
    既证明分组聚合还原了信号，又不臆造物理上不可达的 0.8。
    """
    ds = make_group(rs=2)
    t = GroupAggregationTransformer().fit(ds)
    out = t.transform(ds)
    # 新特征名形如 mean(x1|g) / std(x1|g)
    names = out.feature_names
    mean_x1 = [i for i, n in enumerate(names) if n == "mean(x1|g)"][0]
    r = np.corrcoef(out.X[:, mean_x1], ds.y.astype(float))[0, 1]
    assert abs(r) > 0.7, f"group aggregation should recover signal, got r={r:.3f}"


def test_sanitize_removes_nan_and_inf() -> None:
    """_sanitize 把 NaN/Inf 归零并夹平。"""
    arr = np.array([np.nan, np.inf, -np.inf, 5.0, 1e20])
    out = _sanitize(arr)
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0
    assert out[3] == 5.0
    assert out[4] <= 1e12


def test_build_transformers_returns_all_four() -> None:
    """build_transformers 默认返回 4 个算子，名字齐全。"""
    ts = build_transformers()
    assert [t.name for t in ts] == ["identity", "polynomial", "interaction", "group"]


def test_transformers_append_not_mutate_input() -> None:
    """算子不得原地修改入参 X（返回新对象）。"""
    ds = _ds()
    before = ds.X.copy()
    _ = PolynomialTransformer().fit(ds).transform(ds)
    assert np.allclose(ds.X, before)
