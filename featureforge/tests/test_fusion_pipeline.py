"""融合 / 评测 / 流水线 单测 —— 重点守护"诚实闸门"不被悄悄绕过。

所有涉及"增益/守护"的断言一律基于**嵌套无偏分**（FAST_CFG.nested_folds=3），
绝不使用搜索期的乐观分。
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FAST_CFG

from featureforge.core.config import Config
from featureforge.core.errors import FEDataError
from featureforge.core.types import Dataset, infer_task
from featureforge.data.loader import load_array
from featureforge.data.synthetic import DGPS, make_dataset
from featureforge.eval.cv_score import cv_score
from featureforge.fusion.feffuse import feffuse
from featureforge.pipeline.featureforge_pipeline import benchmark, run_dataset


def test_infer_task_binary_multiclass_regression() -> None:
    """任务类型推断必须把三类任务分开（评测口径依赖它）。"""
    assert infer_task(np.array([0, 1, 1, 0])) == "binary"
    assert infer_task(np.array([0, 1, 2, 2, 1])) == "multiclass"
    assert infer_task(np.linspace(0.0, 1.0, 50)) == "regression"
    assert infer_task(np.array([0.0, 1.0, 2.0])) == "multiclass"


def test_dataset_metadata_stays_consistent_after_subset() -> None:
    """子集化后特征名长度必须等于列数；分组列索引要随之重映射。"""
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.standard_normal((20, 3)), np.repeat([0.0, 1.0], 10)])
    ds = load_array(X, np.zeros(20, dtype=int), ["a", "b", "c", "g"], group_idx=3)
    sub = ds.subset([0, 2, 3])
    assert len(sub.feature_names) == sub.X.shape[1] == 3
    assert sub.group_idx == 2, "分组列索引必须重映射到新位置"
    sub2 = ds.subset([0, 1])
    assert sub2.group_idx is None, "分组列被裁掉后必须为 None"


def test_cv_score_is_deterministic() -> None:
    """同一 cfg 两次调用必须逐位一致（基准可复现的前提）。"""
    ds = make_dataset("interaction", random_state=1)
    cfg = Config()
    assert cv_score(ds.X, ds.y, ds.task, cfg) == cv_score(ds.X, ds.y, ds.task, cfg)


def test_cv_score_rejects_empty_feature_set() -> None:
    """空特征集必须报 F207，而不是静默返回垃圾分数。"""
    ds = make_dataset("interaction", random_state=1)
    with pytest.raises(FEDataError) as ei:
        cv_score(np.empty((10, 0)), ds.y[:10], ds.task, Config())
    assert ei.value.code == "F207"


def test_cv_score_regression_uses_r2() -> None:
    """回归口径为 R²：完全线性可解时接近 1.0。"""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 3))
    y = X[:, 0] * 2.0 + 1.0
    assert cv_score(X, y, "regression", Config()) > 0.95


def test_feffuse_gains_on_interaction_dgp() -> None:
    """interaction：y 依赖 x0*x1，工程化必须带来**显著**增益。"""
    res = run_dataset("interaction", FAST_CFG, seed=1)
    m = res.metrics
    assert m.delta > 0.05, f"expected engineering gain, got delta={m.delta:.4f}"
    assert m.winner == "fusion"
    assert m.safeguard_triggered is False


def test_feffuse_gains_on_polynomial_dgp() -> None:
    """polynomial：y 依赖平方项，多项式展开必须带来增益。"""
    res = run_dataset("polynomial", FAST_CFG, seed=1)
    assert res.metrics.delta > 0.05, f"delta={res.metrics.delta:.4f}"
    assert res.metrics.winner == "fusion"


def test_feffuse_gains_on_group_dgp() -> None:
    """group：y 依赖分组聚合量，分组聚合算子必须被选中并带来增益。"""
    res = run_dataset("group", FAST_CFG, seed=1)
    assert res.metrics.delta > 0.02, f"delta={res.metrics.delta:.4f}"
    assert res.transformer == "group", f"expected group transformer, got {res.transformer}"


def test_feffuse_reduces_dimensions_on_redundant_dgp() -> None:
    """redundant：得分持平但维度必须下降（精简获胜）。"""
    m = run_dataset("redundant", FAST_CFG, seed=1).metrics
    assert m.n_features_selected < m.n_features_baseline, (
        f"expected fewer features: {m.n_features_selected} vs {m.n_features_baseline}"
    )


def test_feffuse_no_gain_on_pure_noise_dgp() -> None:
    """noise：y 与特征独立，嵌套评测下**不得**出现增益（选择偏差已消除）。"""
    m = run_dataset("noise", FAST_CFG, seed=1).metrics
    assert m.delta < 0.05, f"signal-free data must not show gain, delta={m.delta:+.4f}"


def test_feffuse_safeguard_on_highdim_dgp() -> None:
    """highdim：特征膨胀稀释信号 → 不增益即回退原始特征。"""
    m = run_dataset("highdim", FAST_CFG, seed=1).metrics
    assert m.delta <= FAST_CFG.tie_tolerance, f"expected no gain on highdim, delta={m.delta:+.4f}"


def test_feffuse_returns_consistent_result_object() -> None:
    """返回结果自洽：特征名数 == 数据集列数 == metrics 记录数。"""
    ds = make_dataset("interaction", random_state=2, n=FAST_CFG.n_samples)
    res = feffuse(ds, FAST_CFG)
    assert len(res.chosen_names) == res.chosen_dataset.n_features
    assert len(res.chosen_names) == res.metrics.n_features_selected
    assert np.all(np.isfinite(res.chosen_dataset.X))


def test_nested_score_is_not_optimistically_biased() -> None:
    """核心不变量：纯噪声数据上，嵌套分必须显著低于搜索期乐观分。

    若这条断言失败，说明嵌套协议被破坏（搜索泄漏到了评估折），
    整个基准的所有 delta 都会虚高。
    """
    ds = make_dataset("noise", random_state=3, n=FAST_CFG.n_samples)
    res = feffuse(ds, FAST_CFG)
    apparent = res.metrics.detail["apparent_score"]
    nested = res.metrics.detail["nested_score"]
    assert nested < apparent, (
        f"nested({nested:.4f}) must be below apparent({apparent:.4f}) on noise data"
    )


def test_feffuse_never_returns_much_worse_than_baseline() -> None:
    """非劣守护的核心不变量：最终得分不得显著低于原始基线。"""
    for kind in DGPS:
        m = run_dataset(kind, FAST_CFG, seed=2).metrics
        assert m.selected_score >= m.baseline_score - 0.05, (
            f"{kind}: selected {m.selected_score:.4f} << baseline {m.baseline_score:.4f}"
        )


def test_config_env_override(monkeypatch) -> None:
    """ENV 覆盖必须按字段类型正确解析（CI 关闭 Optuna 走兜底全靠它）。"""
    monkeypatch.setenv("FEATUREFORGE_RANDOM_STATE", "7")
    monkeypatch.setenv("FEATUREFORGE_USE_OPTUNA", "false")
    monkeypatch.setenv("FEATUREFORGE_CV_FOLDS", "3")
    monkeypatch.setenv("FEATUREFORGE_TIE_TOLERANCE", "0.02")
    monkeypatch.setenv("FEATUREFORGE_NESTED_FOLDS", "4")
    cfg = Config.from_env()
    assert cfg.random_state == 7
    assert cfg.use_optuna is False
    assert cfg.cv_folds == 3
    assert cfg.nested_folds == 4
    assert cfg.tie_tolerance == pytest.approx(0.02)


def test_benchmark_structure_and_aggregate() -> None:
    """基准输出必须含 per_dgp 与 aggregate 两段，且字段完整。"""
    report = benchmark(FAST_CFG, dgps=["interaction", "noise"], seeds=[1])
    assert set(report["per_dgp"]) == {"interaction", "noise"}
    assert "aggregate" in report
    for v in report["per_dgp"].values():
        assert {"baseline", "selected", "delta", "rows"} <= set(v)
    assert report["aggregate"]["n_dgp"] == 2


def test_dgp_registry_covers_all_documented_cases() -> None:
    """6 个 DGP 必须全部注册（文档与代码的契约）。"""
    assert set(DGPS) == {"interaction", "polynomial", "group", "redundant", "noise", "highdim"}


def test_all_dgps_are_balanced_binary() -> None:
    """所有 DGP 均为均衡二分类 → accuracy 口径才具备跨 DGP 可比性。"""
    for kind in DGPS:
        ds = make_dataset(kind, random_state=1, n=200)
        assert ds.task == "binary", kind
        counts = np.bincount(np.asarray(ds.y, dtype=int))
        assert counts[0] == counts[1], f"{kind} is not balanced: {counts}"


def test_nested_folds_one_falls_back_to_apparent() -> None:
    """``nested_folds<=1`` 是显式调试开关，此时返回搜索期分数（有偏，仅调试）。"""
    from featureforge.fusion.feffuse import fit_recipe, nested_score

    ds = make_dataset("interaction", random_state=1, n=200)
    debug_cfg = Config(nested_folds=1, cv_folds=1, use_optuna=False)
    assert nested_score(ds, debug_cfg) == fit_recipe(ds, debug_cfg).apparent_score


def test_raw_dataset_pipeline_entrypoint() -> None:
    """外部数据集入口 load_array + feffuse 必须可用（集成路径）。"""
    from featureforge.pipeline.featureforge_pipeline import run_raw

    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # 交互结构
    ds = load_array(X, y, ["a", "b", "c", "d"])
    res = run_raw(ds, FAST_CFG)
    assert isinstance(res.chosen_dataset, Dataset)
    assert res.metrics.delta > 0.0
