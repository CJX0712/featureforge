"""外部数据载入：CSV / NPY / NPZ —— 全部走 numpy，不引入 pandas 依赖。

保持零额外依赖是刻意的：本系统承诺"零下载可跑 demo"，pandas 属于可选增强，
核心链路必须能在只有 numpy 的环境里完整运行。
"""

from __future__ import annotations

import os

import numpy as np

from ..core.errors import FEDataError
from ..core.types import Dataset, infer_task


def load_csv(
    path: str,
    target: str | int = -1,
    *,
    has_header: bool = True,
    group: str | None = None,
    task: str = "auto",
) -> Dataset:
    """从 CSV 载入表格数据。

    Parameters
    ----------
    path : CSV 文件路径。
    target : 目标列（列名或列索引，默认最后一列）。
    has_header : 是否含表头。False 时特征名按 ``f0, f1, ...`` 生成。
    group : 分组列名（可选）。必须存在于 header 中。
    task : 任务类型，``"auto"`` 自动推断。

    Raises
    ------
    FEDataError
        文件不存在 / 空文件 / 目标列越界。
    """
    if not os.path.isfile(path):
        raise FEDataError(f"csv file not found: {path}", code="F202")
    try:
        raw = np.genfromtxt(path, delimiter=",", names=has_header, dtype=None, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - 任何解析异常都转为领域错误
        raise FEDataError(f"failed to parse csv {path}: {exc}", code="F203") from exc
    if raw.size == 0:
        raise FEDataError(f"csv file is empty: {path}", code="F204")

    if has_header:
        col_names = list(raw.dtype.names)
    else:
        col_names = [f"f{i}" for i in range(raw.shape[1])]

    if isinstance(target, str):
        if target not in col_names:
            raise FEDataError(f"target column '{target}' not in {col_names}", code="F205")
        t_idx = col_names.index(target)
    else:
        t_idx = target if target >= 0 else len(col_names) + target
        if not (0 <= t_idx < len(col_names)):
            raise FEDataError(f"target index {target} out of range", code="F205")

    feature_names = [c for i, c in enumerate(col_names) if i != t_idx]
    X = np.column_stack(
        [np.asarray(raw[c], dtype=float) for i, c in enumerate(col_names) if i != t_idx]
    )
    y_raw = np.asarray(raw[col_names[t_idx]])

    # 字符串标签 → 整数编码（模型只吃数字），但保留原始标签的类别数供任务推断
    if y_raw.dtype.kind in ("U", "S", "O"):
        uniq, y = np.unique(y_raw, return_inverse=True)
        del uniq
    else:
        y = y_raw.astype(float)

    group_idx = None
    if group is not None:
        if group not in feature_names:
            raise FEDataError(
                f"group column '{group}' not among features {feature_names}", code="F206"
            )
        group_idx = feature_names.index(group)

    if task == "auto":
        task = infer_task(y)
    return Dataset(X=X, y=y, feature_names=feature_names, group_idx=group_idx, task=task)


def load_array(
    X, y, feature_names: list[str] | None = None, group_idx: int | None = None
) -> Dataset:
    """从内存数组直接构造 Dataset（供外部集成使用）。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return Dataset(X=X, y=y, feature_names=feature_names, group_idx=group_idx)
