"""FeatureForge 命令行入口。

    featureforge run       --dgp interaction --seed 1
    featureforge benchmark --seeds 1 2 3 [--no-optuna] [--json out.json]

``benchmark`` 输出的 JSON 与 ``examples/run_demo.py`` 落盘的 benchmark.json
**同构**，便于 CI 直接比对。
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import Config
from .data.synthetic import DGPS
from .pipeline.featureforge_pipeline import benchmark, format_table, run_dataset


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="featureforge",
        description="FeatureForge - automated feature engineering + selector fusion (FEFFuse)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run FEFFuse on a single synthetic dataset")
    r.add_argument("--dgp", default="interaction", choices=sorted(DGPS))
    r.add_argument("--seed", type=int, default=1, help="data seed (default 1)")
    r.add_argument("--cv-folds", type=int, default=5)
    r.add_argument("--random-state", type=int, default=42, help="CV / Optuna seed")
    r.add_argument("--no-optuna", action="store_true", help="use deterministic grid fusion")
    r.add_argument("--trials", type=int, default=12, help="Optuna trials")

    b = sub.add_parser("benchmark", help="run the full DGP x seed benchmark")
    b.add_argument("--dgps", nargs="*", default=None, choices=sorted(DGPS))
    b.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    b.add_argument("--cv-folds", type=int, default=5)
    b.add_argument("--random-state", type=int, default=42)
    b.add_argument("--no-optuna", action="store_true")
    b.add_argument("--trials", type=int, default=12)
    b.add_argument("--n-samples", type=int, default=300)
    b.add_argument("--json", default=None, help="dump report to this path")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI 主函数。返回 0 表示成功。"""
    args = _build_parser().parse_args(argv)
    cfg = Config(
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        use_optuna=not args.no_optuna,
        optuna_trials=args.trials,
        n_samples=getattr(args, "n_samples", 300),
    )

    if args.cmd == "run":
        res = run_dataset(args.dgp, cfg, seed=args.seed)
        m = res.metrics
        print(f"DGP            : {args.dgp} (seed={args.seed})")
        print(f"baseline       : {m.baseline_score:.4f}  (n_features={m.n_features_baseline})")
        print(f"feffuse        : {m.selected_score:.4f}  (n_features={m.n_features_selected})")
        print(f"delta          : {m.delta:+.4f}")
        print(f"winner         : {m.winner}  (safeguard={m.safeguard_triggered})")
        print(f"transformer    : {res.transformer}")
        print(f"selector       : {res.selector}")
        if res.fused_weights:
            w = ", ".join(f"{k}={v:.3f}" for k, v in res.fused_weights.items())
            print(f"fusion weights : {w}")
        print("chosen features:")
        for name in res.chosen_names:
            print(f"  - {name}")
        return 0

    if args.cmd == "benchmark":
        report = benchmark(cfg, dgps=args.dgps, seeds=args.seeds)
        print(format_table(report))
        if args.json:
            with open(args.json, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)
            print(f"\n[ok] report -> {args.json}")
        return 0

    print(f"unknown command: {args.cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
