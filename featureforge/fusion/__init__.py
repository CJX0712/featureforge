"""FeatureForge 融合层：FEFFuse 旗舰（工程化 × 选择器融合 + 非劣守护）。"""

from .feffuse import (
    Recipe,
    apply_recipe,
    available_optuna,
    feffuse,
    fit_recipe,
    nested_score,
    run,
)

__all__ = [
    "feffuse",
    "run",
    "available_optuna",
    "Recipe",
    "fit_recipe",
    "apply_recipe",
    "nested_score",
]
