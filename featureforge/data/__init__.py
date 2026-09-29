"""FeatureForge 数据层：合成 DGP 生成 + 外部文件载入。"""

from .loader import load_array, load_csv
from .synthetic import DGPS, DIFFICULTY, make_dataset

__all__ = ["DGPS", "DIFFICULTY", "make_dataset", "load_csv", "load_array"]
