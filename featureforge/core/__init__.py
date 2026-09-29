"""FeatureForge 核心契约层。"""

from . import errors, interfaces, types
from .config import ENV_PREFIX, Config
from .errors import (
    FeatureForgeError,
    FEConfigError,
    FEDataError,
    FEEngineeringError,
    FEFusionError,
    FESelectionError,
)
from .interfaces import BaseSelector, BaseTransformer
from .types import (
    Dataset,
    EvalMetrics,
    FeatureForgeResult,
    FeatureOp,
    SelectionResult,
    infer_task,
)

__all__ = [
    "errors",
    "interfaces",
    "types",
    "Config",
    "ENV_PREFIX",
    "FeatureForgeError",
    "FEConfigError",
    "FEDataError",
    "FEEngineeringError",
    "FESelectionError",
    "FEFusionError",
    "BaseTransformer",
    "BaseSelector",
    "Dataset",
    "FeatureOp",
    "SelectionResult",
    "EvalMetrics",
    "FeatureForgeResult",
    "infer_task",
]
