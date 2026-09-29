"""5 路特征选择器 + sklearn / numpy 双后端（sklearn 不可用时静默回落 numpy 兜底）。

设计要点
--------
- 每个选择器都返回 ``SelectionResult``：``selected``（列索引）+ ``scores``
  （**每个**特征一个得分，长度 = 输入特征数，被淘汰的特征给 0 分）。
  融合阶段需要把不同选择器的得分对齐到同一长度再加权，缺项必须补零。
- 得分语义统一为"越大越好"，跨选择器可比。
- ``_safe_corr`` 护栏：<3 个有效点或常量/含 NaN 一律返回 0，避免 Pearson
  在退化输入上抛错或给出无意义的相关性。
- sklearn 不可用（或显式被单测关掉）时，互信息 / L1 / 树重要性三路统一回落到
  ``|corr|`` 兜底，保证链路不崩、仍产出合法结果。
"""

from __future__ import annotations

import numpy as np

from ..core.types import Dataset, SelectionResult

# sklearn 可用性（模块级缓存，单测可 monkeypatch 成 False 验证兜底路径）。
_SKLEAR_AVAILABLE: bool | None = None


def available_sklearn() -> bool:
    """探测 sklearn 是否可导入（缓存）。不可用时选择链路自动回落 numpy 兜底。"""
    global _SKLEAR_AVAILABLE
    if _SKLEAR_AVAILABLE is None:
        try:
            import sklearn  # noqa: F401

            _SKLEAR_AVAILABLE = True
        except Exception:  # noqa: BLE001 - 任何导入失败都视为不可用
            _SKLEAR_AVAILABLE = False
    return bool(_SKLEAR_AVAILABLE)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson 相关系数，带护栏：

    - 有效点数 < 3 → 0（样本太少，相关系数不可靠）；
    - 任一数组非有限 / 常量为 0 → 0（退化输入无意义）；
    - 其余返回有符号 r（调用方按需取 abs）。
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 3 or b.size < 3:
        return 0.0
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return 0.0
    sa = float(a.std())
    sb = float(b.std())
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else 0.0


def _scores_by_target_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """numpy 兜底的逐特征得分：每个特征与目标 |corr|。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    d = X.shape[1]
    return np.array([abs(_safe_corr(X[:, j], y)) for j in range(d)], dtype=float)


class Selector:
    """选择器基类（运行时可 isinstance 校验，便于插拔）。"""

    name: str

    def select(self, ds: Dataset) -> SelectionResult:
        raise NotImplementedError


class VarianceSelector(Selector):
    """方差阈值：丢弃近常量列（方差 ≈ 0，不携带信息）。"""

    name = "variance"

    def select(self, ds: Dataset) -> SelectionResult:
        X = np.asarray(ds.X, dtype=float)
        var = X.var(axis=0)
        selected = [int(j) for j in range(X.shape[1]) if var[j] > 1e-9]
        if not selected:  # 退化保护：全常量则至少保留第 0 列
            selected = [0]
        return SelectionResult(
            selected=selected,
            selected_names=[ds.feature_names[j] for j in selected],
            scores={"variance": np.asarray(var, dtype=float)},
            method=self.name,
        )


class CorrelationSelector(Selector):
    """贪心去冗余：按 |corr(y)| 降序，跳过与已选列相关 > 0.85 的近重复列。"""

    name = "correlation"

    def select(self, ds: Dataset) -> SelectionResult:
        X = np.asarray(ds.X, dtype=float)
        y = np.asarray(ds.y, dtype=float).ravel()
        d = X.shape[1]
        c = np.array([abs(_safe_corr(X[:, j], y)) for j in range(d)], dtype=float)
        order = np.argsort(-c, kind="stable")
        selected: list[int] = []
        for j in order:
            if all(abs(_safe_corr(X[:, j], X[:, k])) <= 0.85 for k in selected):
                selected.append(int(j))
        if not selected:  # 全部 |corr| ≈ 0：退化为保留最相关的一列
            selected = [int(order[0])]
        return SelectionResult(
            selected=selected,
            selected_names=[ds.feature_names[j] for j in selected],
            scores={"correlation": c},
            method=self.name,
        )


class MutualInfoSelector(Selector):
    """互信息：衡量每个特征与目标的（非线性）依赖强度。"""

    name = "mutual_info"

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = int(random_state)

    def select(self, ds: Dataset) -> SelectionResult:
        X = np.asarray(ds.X, dtype=float)
        y = np.asarray(ds.y)
        if _SKLEAR_AVAILABLE:
            try:
                from sklearn.feature_selection import mutual_info_classif

                mi = np.asarray(
                    mutual_info_classif(X, y, random_state=self.random_state),
                    dtype=float,
                )
            except Exception:  # noqa: BLE001 - 回落 numpy 兜底
                mi = _scores_by_target_corr(X, y)
        else:
            mi = _scores_by_target_corr(X, y)
        selected = [int(j) for j in range(X.shape[1]) if mi[j] > 1e-9]
        if not selected:
            selected = [int(np.argmax(mi))]
        return SelectionResult(
            selected=selected,
            selected_names=[ds.feature_names[j] for j in selected],
            scores={"mutual_info": mi},
            method=self.name,
        )


class L1Selector(Selector):
    """L1 逻辑回归：系数绝对值作为特征得分，零系数列被淘汰（稀疏性）。

    注意 sklearn 1.9 中 ``penalty="l1"`` 已废弃，改用 ``l1_ratio=1.0`` +
    ``solver="saga"``；``liblinear`` 配 ``l1_ratio`` 会把系数全压成 0。
    """

    name = "l1"

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = int(random_state)

    def select(self, ds: Dataset) -> SelectionResult:
        X = np.asarray(ds.X, dtype=float)
        y = np.asarray(ds.y)
        if _SKLEAR_AVAILABLE:
            try:
                from sklearn.linear_model import LogisticRegression

                model = LogisticRegression(
                    C=0.3,
                    l1_ratio=1.0,
                    solver="saga",
                    max_iter=10000,
                    random_state=self.random_state,
                )
                model.fit(X, y)
                coef = np.abs(np.asarray(model.coef_[0], dtype=float))
            except Exception:  # noqa: BLE001 - 回落 numpy 兜底
                coef = _scores_by_target_corr(X, y)
        else:
            coef = _scores_by_target_corr(X, y)
        selected = [int(j) for j in range(X.shape[1]) if coef[j] > 1e-9]
        if not selected:
            selected = [int(np.argmax(coef))]
        return SelectionResult(
            selected=selected,
            selected_names=[ds.feature_names[j] for j in selected],
            scores={"l1": coef},
            method=self.name,
        )


class ModelImportanceSelector(Selector):
    """随机森林特征重要性：捕获非线性、非单调的信号。"""

    name = "importance"

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = int(random_state)

    def select(self, ds: Dataset) -> SelectionResult:
        X = np.asarray(ds.X, dtype=float)
        y = np.asarray(ds.y)
        if _SKLEAR_AVAILABLE:
            try:
                from sklearn.ensemble import RandomForestClassifier

                rf = RandomForestClassifier(
                    n_estimators=100,
                    n_jobs=1,
                    random_state=self.random_state,
                )
                rf.fit(X, y)
                imp = np.asarray(rf.feature_importances_, dtype=float)
            except Exception:  # noqa: BLE001 - 回落 numpy 兜底
                imp = _scores_by_target_corr(X, y)
        else:
            imp = _scores_by_target_corr(X, y)
        selected = [int(j) for j in range(X.shape[1]) if imp[j] > 1e-9]
        if not selected:
            selected = [int(np.argmax(imp))]
        return SelectionResult(
            selected=selected,
            selected_names=[ds.feature_names[j] for j in selected],
            scores={"importance": imp},
            method=self.name,
        )


def build_selectors(
    selectors: list[str] | None = None, random_state: int = 0
) -> dict[str, Selector]:
    """构造选择器实例字典。

    Parameters
    ----------
    selectors : 限定返回的选择器名；None 表示全部 5 路。
    random_state : 传给需要随机性的选择器（互信息 / L1 / 树）。
    """
    all_defs: dict[str, Selector] = {
        "variance": VarianceSelector(),
        "correlation": CorrelationSelector(),
        "mutual_info": MutualInfoSelector(random_state=random_state),
        "l1": L1Selector(random_state=random_state),
        "importance": ModelImportanceSelector(random_state=random_state),
    }
    if selectors is None:
        return dict(all_defs)
    return {k: v for k, v in all_defs.items() if k in selectors}


def select_all(ds: Dataset, sel_list: dict[str, Selector]) -> dict[str, SelectionResult]:
    """对给定数据集跑完所有选择器；单路崩溃时静默跳过，不影响其余。"""
    out: dict[str, SelectionResult] = {}
    for name, sel in sel_list.items():
        try:
            out[name] = sel.select(ds)
        except Exception:  # noqa: BLE001 - 某一选择器失败不应拖垮整条链路
            continue
    return out
