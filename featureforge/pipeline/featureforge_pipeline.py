"""FeatureForge 流水线：单数据集运行 + 跨 DGP 基准（含聚合与制表）。

数据种子与 CV 种子**分离**：``seed`` 只控制合成数据的生成，
``cfg.random_state`` 只控制折划分与 Optuna 采样。
这样"换一份数据"与"换一次验证"是正交的两件事，基准才可读。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.config import Config
from ..core.types import Dataset, FeatureForgeResult
from ..data.synthetic import DGPS
from ..fusion.feffuse import feffuse


def run_dataset(
    kind: str,
    cfg: Config | None = None,
    seed: int = 1,
) -> FeatureForgeResult:
    """在指定 DGP 上跑一次完整 FEFFuse。

    Parameters
    ----------
    kind : DGP 名（见 ``data.synthetic.DGPS``）
    cfg : 配置；CV/Optuna 种子取自其中
    seed : **数据**种子
    """
    cfg = cfg or Config()
    ds = make_dataset_for(kind, seed, cfg)
    return feffuse(ds, cfg)


def make_dataset_for(kind: str, seed: int, cfg: Config | None = None) -> Dataset:
    """按 DGP 名与数据种子构造数据集（样本数取自 cfg.n_samples）。"""
    from ..data.synthetic import make_dataset

    cfg = cfg or Config()
    if kind not in DGPS:
        raise KeyError(f"unknown DGP '{kind}'; available: {sorted(DGPS)}")
    return make_dataset(kind, random_state=seed, n=cfg.n_samples)


def run_raw(ds: Dataset, cfg: Config | None = None) -> FeatureForgeResult:
    """在**任意**外部数据集上跑 FEFFuse（供集成使用）。"""
    return feffuse(ds, cfg or Config())


def benchmark(
    cfg: Config | None = None,
    dgps: Sequence[str] | None = None,
    seeds: Sequence[int] = (1, 2, 3),
) -> dict:
    """跨 DGP × 多种子基准。

    Returns
    -------
    dict
        ``{"config":..., "per_dgp": {...}, "aggregate": {...}}``
        每个 DGP 记录逐 seed 的原始/工程化得分、均值、增益、获胜率、守护触发率。
    """
    cfg = cfg or Config()
    if dgps is None:
        dgps = list(DGPS)
    per_dgp: dict[str, dict] = {}
    for kind in dgps:
        base_list: list[float] = []
        sel_list: list[float] = []
        delta_list: list[float] = []
        n_sel_list: list[int] = []
        winners: list[str] = []
        safeguards: list[bool] = []
        transformers: list[str] = []
        rows: list[dict] = []
        for seed in seeds:
            res = run_dataset(kind, cfg, seed=seed)
            m = res.metrics
            base_list.append(m.baseline_score)
            sel_list.append(m.selected_score)
            delta_list.append(m.delta)
            n_sel_list.append(m.n_features_selected)
            winners.append(m.winner)
            safeguards.append(m.safeguard_triggered)
            transformers.append(res.transformer)
            rows.append(
                {
                    "seed": int(seed),
                    "baseline": float(m.baseline_score),
                    "selected": float(m.selected_score),
                    "delta": float(m.delta),
                    "n_features_selected": int(m.n_features_selected),
                    "winner": m.winner,
                    "safeguard": bool(m.safeguard_triggered),
                    "transformer": res.transformer,
                    "selector": res.selector,
                }
            )
        n_base = make_dataset_for(kind, seeds[0], cfg).n_features
        per_dgp[kind] = {
            "baseline": float(np.mean(base_list)),
            "selected": float(np.mean(sel_list)),
            "delta": float(np.mean(delta_list)),
            "baseline_std": float(np.std(base_list)),
            "selected_std": float(np.std(sel_list)),
            "baseline_per_seed": [float(v) for v in base_list],
            "selected_per_seed": [float(v) for v in sel_list],
            "n_features_baseline": int(n_base),
            "n_features_selected": float(np.mean(n_sel_list)),
            "fusion_win_rate": float(np.mean([w == "fusion" for w in winners])),
            "safeguard_rate": float(np.mean(safeguards)),
            "transformers": transformers,
            "seeds": [int(s) for s in seeds],
            "rows": rows,
        }

    all_base = [v["baseline"] for v in per_dgp.values()]
    all_sel = [v["selected"] for v in per_dgp.values()]
    all_delta = [v["delta"] for v in per_dgp.values()]
    return {
        "config": {
            "random_state": cfg.random_state,
            "cv_folds": cfg.cv_folds,
            "non_inferior_margin": cfg.non_inferior_margin,
            "tie_tolerance": cfg.tie_tolerance,
            "use_optuna": cfg.use_optuna,
            "optuna_trials": cfg.optuna_trials,
            "n_samples": cfg.n_samples,
            "seeds": [int(s) for s in seeds],
        },
        "per_dgp": per_dgp,
        "aggregate": {
            "baseline": float(np.mean(all_base)),
            "selected": float(np.mean(all_sel)),
            "delta": float(np.mean(all_delta)),
            "n_dgp": len(per_dgp),
            "fusion_win_rate": float(np.mean([v["fusion_win_rate"] for v in per_dgp.values()])),
            "safeguard_rate": float(np.mean([v["safeguard_rate"] for v in per_dgp.values()])),
        },
    }


def format_table(report: dict) -> str:
    """把基准结果渲染成固定列宽表格（中文列宽用 ANSI 无关的固定宽度，避免错位）。"""

    def row(cells: Sequence[str], widths: Sequence[int]) -> str:
        return "  ".join(f"{c:<{w}}" for c, w in zip(cells, widths, strict=False))

    header = ("DGP", "base", "feffuse", "delta", "n_base", "n_sel", "win%", "guard%")
    widths = (12, 7, 9, 8, 7, 7, 6, 7)
    lines = [row(header, widths), "-" * (sum(widths) + 2 * (len(widths) - 1))]
    for kind, v in report["per_dgp"].items():
        lines.append(
            row(
                (
                    kind,
                    f"{v['baseline']:.4f}",
                    f"{v['selected']:.4f}",
                    f"{v['delta']:+.4f}",
                    str(v["n_features_baseline"]),
                    f"{v['n_features_selected']:.1f}",
                    f"{v['fusion_win_rate'] * 100:.0f}",
                    f"{v['safeguard_rate'] * 100:.0f}",
                ),
                widths,
            )
        )
    agg = report["aggregate"]
    lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
    lines.append(
        row(
            (
                "AGGREGATE",
                f"{agg['baseline']:.4f}",
                f"{agg['selected']:.4f}",
                f"{agg['delta']:+.4f}",
                "-",
                "-",
                f"{agg['fusion_win_rate'] * 100:.0f}",
                f"{agg['safeguard_rate'] * 100:.0f}",
            ),
            widths,
        )
    )
    return "\n".join(lines)
