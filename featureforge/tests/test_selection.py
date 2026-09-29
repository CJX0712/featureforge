"""特征选择单测 —— 5 路选择器 + sklearn/numpy 双后端 + 护栏。

重点：
- ``_safe_corr`` 护栏（<3 有效点返回 0）；
- CorrelationSelector 贪心去冗余（近重复列只留其一）；
- 任一后端不可用时静默回落 numpy 兜底，链路不崩。
"""

from __future__ import annotations

import numpy as np
import pytest

from featureforge.core.types import Dataset
from featureforge.selection import selectors as sel_mod
from featureforge.selection.selectors import (
    CorrelationSelector,
    L1Selector,
    ModelImportanceSelector,
    MutualInfoSelector,
    VarianceSelector,
    _safe_corr,
    available_sklearn,
    build_selectors,
    select_all,
)


def _signal_ds(n: int = 200, seed: int = 0) -> Dataset:
    """y 由 x0 线性决定 + 噪声，x1/x2 为纯噪声。"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    y = (X[:, 0] + 0.2 * rng.standard_normal(n) > 0).astype(int)
    return Dataset(X=X, y=y, feature_names=[f"x{i}" for i in range(4)], task="binary")


def test_available_sklearn_is_bool() -> None:
    """原函数仍可调用，且返回布尔。"""
    assert isinstance(available_sklearn(), bool)


def test_variance_selector_drops_constant_column() -> None:
    """常量列方差为 0 → 被丢弃。"""
    X = np.column_stack([np.ones(50), np.arange(50.0)])
    ds = Dataset(X=X, y=(np.arange(50) > 25).astype(int), feature_names=["c", "v"], task="binary")
    res = VarianceSelector().select(ds)
    assert 0 not in res.selected  # 常量列 c 被丢
    assert 1 in res.selected


def test_correlation_drops_redundant_near_duplicate() -> None:
    """近重复列（与母列相关 >0.85）只保留其一，不双选。"""
    rng = np.random.default_rng(3)
    x0 = rng.standard_normal(150)
    dup = x0 + 0.4 * rng.standard_normal(150)  # 与 x0 相关 ~0.93
    X = np.column_stack([x0, dup, rng.standard_normal((150, 2))])
    y = (x0 > 0).astype(int)
    ds = Dataset(X=X, y=y, feature_names=["x0", "dup0", "n1", "n2"], task="binary")
    res = CorrelationSelector().select(ds)
    # x0 与 dup0 不应同时入选
    assert not (0 in res.selected and 1 in res.selected)


def test_all_selectors_pick_at_least_one_feature() -> None:
    """每个选择器在含信号的数据集上至少选 1 列。"""
    ds = _signal_ds()
    for s in (
        MutualInfoSelector(random_state=0),
        L1Selector(random_state=0),
        ModelImportanceSelector(random_state=0),
        VarianceSelector(),
        CorrelationSelector(),
    ):
        res = s.select(ds)
        assert len(res.selected) >= 1, s.name
        assert len(res.scores[s.name]) == ds.n_features, s.name


def test_safe_corr_guards_small_samples() -> None:
    """_safe_corr 护栏：2 点返回 0，3 点完美相关返回 1。"""
    assert _safe_corr(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == 0.0
    r = _safe_corr(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert r == pytest.approx(1.0)


def test_numpy_fallback_when_sklearn_unavailable(monkeypatch) -> None:
    """sklearn 不可用 → 选择器回落 numpy 兜底，仍产出合法结果。"""
    monkeypatch.setattr(sel_mod, "_SKLEAR_AVAILABLE", False)
    ds = _signal_ds()
    for s in (L1Selector(random_state=0), MutualInfoSelector(random_state=0)):
        res = s.select(ds)
        assert len(res.selected) >= 1
        assert len(res.scores[s.name]) == ds.n_features


def test_select_all_skips_broken_selector() -> None:
    """select_all 中某路选择器崩溃时，其余照常返回。"""

    class Broken:
        name = "broken"

        def select(self, ds):  # noqa: ANN001, ARG001
            raise RuntimeError("boom")

    sel_list = {"broken": Broken(), "variance": VarianceSelector()}
    out = select_all(_signal_ds(), sel_list)
    assert "variance" in out
    assert "broken" not in out


def test_build_selectors_returns_all_five() -> None:
    """build_selectors 默认构造 5 路选择器。"""
    sels = build_selectors()
    assert set(sels) == {"variance", "correlation", "mutual_info", "l1", "importance"}
