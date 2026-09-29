"""交叉验证评分 —— 融合与基准的**唯一**打分口径。

为什么单独成模块
----------------
整个系统的胜负判定（工程化 vs 原始、融合 vs 单候选）全部归结到这一个函数。
把它独立出来，才能保证"同一口径下比较"——历史踩坑里多次出现"两套口径混用
导致结论失真"，这里从架构上消除该可能。

口径约定
--------
- 分类：**accuracy**（所有 DGP 均按中位数阈值化，正负类 1:1，故 accuracy 可比）
- 回归：**R²**（对常量标签退化为 0.0）
- 两者都**越大越好**
- 折内标准化用**训练折**统计量，绝不用全量（防泄漏）
"""

from __future__ import annotations

import numpy as np

from ..core.config import Config
from ..core.errors import FEDataError
from ..core.types import Dataset

_LAMBDA = 1.0  # numpy 兜底岭回归的正则系数


def _train_scaler(Ztr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """从训练折学习标准化参数（mu, sd）。常量列 sd 退化为 1。"""
    mu = Ztr.mean(axis=0)
    sd = Ztr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return mu, sd


def _apply_scaler(Z: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """应用标准化参数。"""
    return (Z - mu) / sd


def _ridge_fit(Z: np.ndarray, t: np.ndarray, lam: float = _LAMBDA) -> np.ndarray:
    """岭回归闭式解：w = (ZᵀZ + λI)⁻¹ Zᵀt（含截距项由 Z 的首列 1 承担）。

    用 ``np.linalg.solve`` 而非显式求逆：数值更稳、更快。
    奇异时（如特征数 > 样本数且 λ 过小）用 lstsq 兜底。
    """
    d = Z.shape[1]
    A = Z.T @ Z + lam * np.eye(d)
    b = Z.T @ t
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _metrics(pred: np.ndarray, y: np.ndarray, task: str) -> float:
    """按任务口径计算得分（越大越好）。"""
    y = np.asarray(y)
    pred = np.asarray(pred)
    if task == "regression":
        y = y.astype(float)
        sst = float(((y - y.mean()) ** 2).sum())
        if sst < 1e-12:
            return 0.0
        sse = float(((y - pred) ** 2).sum())
        return 1.0 - sse / sst
    return float(np.mean(pred == y))


def _numpy_predict(Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray, task: str) -> np.ndarray:
    """纯 numpy 预测（离线兜底路径，零 sklearn 依赖）。

    - 回归：岭回归闭式解
    - 二分类：标签映射 ±1 + 岭回归 + 符号判别
    - 多分类：One-vs-Rest（每类一个岭回归，取分最大者）
    """
    if task == "regression":
        w = _ridge_fit(Ztr, ytr.astype(float))
        return Zte @ w
    classes = np.unique(ytr)
    if len(classes) <= 2:
        pos = classes[-1]
        t = np.where(ytr == pos, 1.0, -1.0)
        w = _ridge_fit(Ztr, t)
        scores = Zte @ w
        return np.where(scores >= 0.0, pos, classes[0])
    # 多分类 OvR
    score_mat = np.zeros((Zte.shape[0], len(classes)), dtype=float)
    for ci, c in enumerate(classes):
        t = np.where(ytr == c, 1.0, -1.0)
        score_mat[:, ci] = Zte @ _ridge_fit(Ztr, t)
    return classes[np.argmax(score_mat, axis=1)]


def _sklearn_predict(
    Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray, task: str, cfg: Config
) -> np.ndarray | None:
    """sklearn 预测路径。任何异常返回 None，由调用方降级到 numpy。"""
    try:
        if task == "regression":
            from sklearn.linear_model import Ridge as _sk_ridge

            model = _sk_ridge(alpha=1.0)
            model.fit(Ztr, ytr.astype(float))
            return np.asarray(model.predict(Zte))
        from sklearn.linear_model import LogisticRegression as _sk_lr

        model = _sk_lr(max_iter=2000, solver="lbfgs")
        model.fit(Ztr, ytr)
        return np.asarray(model.predict(Zte))
    except Exception:  # noqa: BLE001 - sklearn 不可用/不收敛 → 降级
        return None


def fold_score(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    task: str,
    cfg: Config,
) -> float:
    """训练一次、在测试折上打一次分（嵌套评测的最小单元）。

    与 ``cv_score`` 共用**完全相同**的预处理与模型（折内标准化 + 补截距列 +
    sklearn 优先/numpy 兜底），保证内层搜索分与外层泛化分口径一致。
    """
    Xtr = np.where(np.isfinite(np.asarray(Xtr, dtype=float)), np.asarray(Xtr, dtype=float), 0.0)
    Xte = np.where(np.isfinite(np.asarray(Xte, dtype=float)), np.asarray(Xte, dtype=float), 0.0)
    ytr = np.asarray(ytr)
    yte = np.asarray(yte)
    mu, sd = _train_scaler(Xtr)
    Ztr = np.column_stack([np.ones(Xtr.shape[0]), _apply_scaler(Xtr, mu, sd)])
    Zte = np.column_stack([np.ones(Xte.shape[0]), _apply_scaler(Xte, mu, sd)])
    pred = None
    try:
        import sklearn  # noqa: F401

        pred = _sklearn_predict(Ztr, ytr, Zte, task, cfg)
    except Exception:  # noqa: BLE001
        pred = None
    if pred is None:
        pred = _numpy_predict(Ztr, ytr, Zte, task)
    return _metrics(pred, yte, task)


def split_folds(n: int, cfg: Config) -> list[np.ndarray]:
    """确定性折划分：先按种子洗牌，再顺序均分。

    分类任务做**分层**划分（按类别分组后轮转分配），保证每折都有各类样本，
    避免小数据上某折只剩一类导致得分失真。
    """
    rng = np.random.default_rng(cfg.random_state)
    k = int(cfg.cv_folds)
    idx = rng.permutation(n)
    if k < 2:
        # 单折 holdout 模式：75% 训练 / 25% 验证（算力最省，仅用于内层排序）
        cut = max(1, int(0.75 * n))
        return [np.sort(idx[cut:])]
    return [np.sort(f) for f in np.array_split(idx, k)]


def split_folds_nested(n: int, cfg: Config) -> list[np.ndarray]:
    """外层嵌套折划分：与 ``split_folds``（cv_folds 折）**共用**同一随机排列，
    当折数相同时会得到逐位一致的折，从而让"基线"与"嵌套得分"构成配对比较。

    折数取 ``cfg.nested_folds``（>=2）。
    """
    rng = np.random.default_rng(cfg.random_state)
    k = max(2, int(cfg.nested_folds))
    idx = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(idx, k)]


def cv_score(X: np.ndarray, y: np.ndarray, task: str, cfg: Config) -> float:
    """对给定特征子集做 k 折交叉验证，返回平均得分（越大越好）。

    Parameters
    ----------
    X : 特征矩阵 (n, d) —— 已是**选定后**的子集
    y : 标签 (n,)
    task : "binary" | "multiclass" | "regression"
    cfg : 配置（提供 random_state / cv_folds）

    Returns
    -------
    float
        各折得分均值。

    Raises
    ------
    FEDataError
        特征数为 0 或样本数不足以分折。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] == 0:
        raise FEDataError("cv_score requires at least one feature column", code="F207")
    n = X.shape[0]
    if n < 2:
        raise FEDataError("cv_score requires at least 2 samples", code="F208")

    X = np.where(np.isfinite(X), X, 0.0)
    folds = split_folds(n, cfg)
    scores: list[float] = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        scores.append(fold_score(X[train_idx], y[train_idx], X[test_idx], y[test_idx], task, cfg))

    if not scores:
        raise FEDataError("cv_score produced no valid folds", code="F209")
    return float(np.mean(scores))


def cv_score_dataset(ds: Dataset, cfg: Config) -> float:
    """对 Dataset 的**全部**特征打分（基线口径）。"""
    return cv_score(ds.X, ds.y, ds.task, cfg)
