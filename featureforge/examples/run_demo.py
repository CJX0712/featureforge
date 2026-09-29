"""端到端演示：跑完整基准 → 打印表格 → 落盘 benchmark.json。

    python -m featureforge.examples.run_demo

落盘路径固定为**项目根**下的 ``benchmark.json``（与 README 中引用的数字同源），
保证文档里的每一个数字都能被这条命令重新跑出来。
"""

from __future__ import annotations

import json
import os
import sys
import time

# 项目根 = <root>/featureforge（本文件位于 <root>/featureforge/featureforge/examples/）
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from featureforge.core.config import Config  # noqa: E402
from featureforge.pipeline.featureforge_pipeline import (  # noqa: E402
    benchmark,
    format_table,
)


def main() -> int:
    """执行基准并落盘。返回进程退出码。"""
    # 与测试套件（FAST_CFG）完全一致的配置：n_samples=200、内层 cv_folds=1、
    # 外层 nested_folds=3、确定性网格融合（use_optuna=False）。这样落盘的
    # benchmark.json 与 pytest 的诚实闸门断言逐一对齐，避免"基准与测试两套口径"。
    cfg = Config(random_state=42, cv_folds=1, nested_folds=3, use_optuna=False, n_samples=200)
    t0 = time.time()
    report = benchmark(cfg, seeds=(1, 2, 3))
    elapsed = time.time() - t0

    print("=" * 78)
    print("FeatureForge benchmark - FEFFuse vs raw features")
    print("=" * 78)
    print(format_table(report))
    print()
    agg = report["aggregate"]
    print(f"aggregate delta : {agg['delta']:+.4f}")
    print(f"fusion win rate : {agg['fusion_win_rate'] * 100:.1f}%")
    print(f"safeguard rate  : {agg['safeguard_rate'] * 100:.1f}%")
    print(f"elapsed         : {elapsed:.2f}s")

    out = os.path.join(_ROOT, "benchmark.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n[ok] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
