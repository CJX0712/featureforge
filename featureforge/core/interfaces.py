"""FeatureForge 抽象接口（Protocol，可运行时检查）。

调用方向恒为单向无环：
    cli -> pipeline -> {data, engineering, selection, fusion, eval} -> core
本模块只定义契约，不含实现；所有算子通过 ``isinstance`` 校验即可插拔。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import Dataset, SelectionResult


@runtime_checkable
class BaseTransformer(Protocol):
    """特征工程算子契约。

    约定
    ----
    - ``fit`` 必须从训练集学习统计量（如分组均值），返回 self。
    - ``transform`` 产出**新** Dataset，不得原地修改入参。
    - 算子只**追加**特征（保留原始列），便于"工程化与否"公平对比。
    """

    name: str

    def fit(self, ds: Dataset) -> BaseTransformer:
        """学习算子参数。"""
        ...

    def transform(self, ds: Dataset) -> Dataset:
        """产出追加了新特征的 Dataset。"""
        ...


@runtime_checkable
class BaseSelector(Protocol):
    """特征选择算子契约。

    约定
    ----
    - ``select`` 返回 ``SelectionResult``，``selected`` 为传入 Dataset 的列索引。
    - ``scores`` 必须给出**每个**特征一个得分（长度 = n_features），
      即使某些特征被淘汰也要给出 0 分，否则融合阶段无法对齐加权。
    - 得分语义统一为"越大越好"，跨选择器可比。
    """

    name: str

    def select(self, ds: Dataset) -> SelectionResult:
        """在给定特征空间内打分并选出子集。"""
        ...


def score_vector(res: SelectionResult, method: str, n_features: int) -> np.ndarray:
    """从 SelectionResult 取出指定方法的得分向量，缺失时返回全零向量。

    融合阶段需要把不同选择器的得分对齐到同一长度，缺项必须补零而非报错，
    否则某一路后端不可用时整条融合链路会崩。
    """
    vec = res.scores.get(method)
    if vec is None:
        return np.zeros(n_features, dtype=float)
    vec = np.asarray(vec, dtype=float)
    if vec.shape[0] != n_features:
        return np.zeros(n_features, dtype=float)
    return vec
