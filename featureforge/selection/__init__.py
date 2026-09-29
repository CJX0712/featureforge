"""FeatureForge 特征选择层：5 路选择器 + sklearn/numpy 双后端。"""

from .selectors import (
    CorrelationSelector,
    L1Selector,
    ModelImportanceSelector,
    MutualInfoSelector,
    Selector,
    VarianceSelector,
    available_sklearn,
    build_selectors,
    select_all,
)

__all__ = [
    "Selector",
    "VarianceSelector",
    "CorrelationSelector",
    "MutualInfoSelector",
    "L1Selector",
    "ModelImportanceSelector",
    "available_sklearn",
    "build_selectors",
    "select_all",
]
