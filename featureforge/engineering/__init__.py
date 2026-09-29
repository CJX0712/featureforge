"""FeatureForge 特征工程层：纯 numpy 算子（多项式 / 交互 / 分组聚合 / 恒等）。"""

from .transformers import (
    TRANSFORMERS,
    GroupAggregationTransformer,
    IdentityTransformer,
    InteractionTransformer,
    PolynomialTransformer,
    Transformer,
    build_engineered_datasets,
    build_transformers,
    describe_ops,
)

__all__ = [
    "Transformer",
    "IdentityTransformer",
    "PolynomialTransformer",
    "InteractionTransformer",
    "GroupAggregationTransformer",
    "TRANSFORMERS",
    "build_transformers",
    "build_engineered_datasets",
    "describe_ops",
]
