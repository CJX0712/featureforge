"""FeatureForge —— 自动化特征工程 × 选择器融合系统（FEFFuse）。

快速上手
--------
    from featureforge.core import Config
    from featureforge.pipeline import run_dataset, benchmark, format_table

    res = run_dataset("interaction", Config(), seed=1)
    print(res.metrics.delta)

    print(format_table(benchmark(Config(), seeds=(1, 2, 3))))

作者: 晨星 (CJX0712)
"""

from .core.config import Config
from .core.types import (
    Dataset,
    EvalMetrics,
    FeatureForgeResult,
    SelectionResult,
    infer_task,
)
from .fusion.feffuse import feffuse

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = [
    "__version__",
    "__author__",
    "Config",
    "Dataset",
    "SelectionResult",
    "EvalMetrics",
    "FeatureForgeResult",
    "infer_task",
    "feffuse",
]
