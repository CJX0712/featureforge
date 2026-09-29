"""FEFFuse —— FeatureForge 旗舰：**特征工程 × 选择器融合** 的自适应特征锻造。

算法（四阶段，全部可审计）
--------------------------
1. **候选网格**
   对每个特征工程算子（identity / polynomial / interaction / group）× 每个
   选择器（variance / correlation / mutual_info / l1 / importance）用**内层**
   交叉验证打分，得到带分的特征子集。这一步回答"哪个特征空间有前途"。

2. **定位最佳空间**
   取候选网格中得分最高的组合所属的**特征空间**（engineering 算子），
   在该空间内做融合 —— 避免对 4 个空间都跑 HPO 造成的无谓开销。

3. **多选择器融合（核心创新）**
   同一空间里，5 个选择器的得分向量量纲与偏好完全不同（方差 / 去冗余 /
   互信息 / L1 稀疏 / 树重要性）。FEFFuse 把它们 min-max 归一后按**权重**
   加权求和，再用 top-k 截断，权重与 k 由 Optuna(TPE) 搜索，
   目标 = 该融合子集的内层交叉验证得分。
   Optuna 不可用时退化为**等权 + 单选择器 one-hot 的确定性网格**（可复现）。

4. **嵌套评测 + 双层非劣守护（诚实闸门）**
   - **嵌套评测**：阶段 1~3 的全部搜索只在外层**训练折**内进行，最终得分在
     从未参与搜索的外层**测试折**上测量。这是消除**特征选择偏差**的唯一
     正确协议 —— 实测若省略嵌套，纯噪声数据上会凭空出现 +0.08 的虚高增益，
     高维数据上甚至会超过贝叶斯上限（详见 docs/architecture.md）。
   - L1：融合集必须**严格优于**最佳单候选，否则回退最佳单候选；
   - L2：最终胜者必须比**原始特征**更高（> non_inferior_margin）才算"增益获胜"；
     若只是打平（±tie_tolerance）但维度下降，算"精简获胜"；
     否则判定工程化无效 → **回退原始特征**并对 safeguard 置位。

   这两层闸门保证系统永远不会为了"看起来更聪明"而输出一个实际更差的特征集。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..core.config import Config
from ..core.errors import FEFusionError
from ..core.types import Dataset, EvalMetrics, FeatureForgeResult
from ..engineering.transformers import Transformer, build_transformers
from ..eval.cv_score import cv_score, fold_score, split_folds_nested
from ..selection.selectors import build_selectors, select_all

_OPTUNA_OK: bool | None = None


def available_optuna() -> bool:
    """探测 Optuna 可用性（缓存）。不可用时融合退化为确定性网格。"""
    global _OPTUNA_OK
    if _OPTUNA_OK is None:
        try:
            import optuna  # noqa: F401

            _OPTUNA_OK = True
        except Exception:  # noqa: BLE001
            _OPTUNA_OK = False
    return _OPTUNA_OK


@dataclass
class Recipe:
    """一次搜索产出的"可执行配方" —— 能脱离搜索过程独立应用到新数据。

    把搜索结果固化成配方，是嵌套评测得以成立的前提：外层训练折上搜出配方，
    外层测试折上只做"应用"，测试折从头到尾没参与过任何选择决策。

    Attributes
    ----------
    transformer_obj : 已 **fit** 的算子实例（分组聚合算子需要它携带组统计量）
    cols : 配方选中的列索引，位于该算子产出的**工程化空间**内
    apparent_score : 搜索过程中看到的内层 CV 分（**乐观有偏**，仅作诊断）
    candidates : 候选网格审计明细。其中的 ``score`` 同样是**有偏**的搜索期分数，
        仅用于回答"搜过什么"，**不得**用于判定胜负（胜负一律以嵌套分为准）。
    """

    transformer: str
    transformer_obj: Transformer
    cols: list[int]
    names: list[str]
    selector: str
    weights: dict[str, float]
    apparent_score: float
    n_space_features: int
    candidates: list[dict]


def _normalize(v: np.ndarray) -> np.ndarray:
    """min-max 归一到 [0,1]；常量向量退化为全零（表示"该选择器无区分力"）。"""
    v = np.asarray(v, dtype=float)
    v = np.where(np.isfinite(v), v, 0.0)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def _evaluate_subset(
    X: np.ndarray, y: np.ndarray, task: str, cols: list[int], cfg: Config
) -> tuple[float, list[int]]:
    """评估某个列子集：返回 (内层 CV 得分, 有序去重后的列索引)。"""
    cols = sorted({int(c) for c in cols if 0 <= int(c) < X.shape[1]})
    if not cols:
        return float("-inf"), []
    return cv_score(X[:, cols], y, task, cfg), cols


def _is_better(score: float, n: int, best_score: float, best_n: int) -> bool:
    """候选优劣判定：**先比得分，同分则比维度**（奥卡姆剃刀）。

    返回 True 表示 (score, n) 优于 (best_score, best_n)。
    """
    if score > best_score + 1e-12:
        return True
    if abs(score - best_score) <= 1e-12 and n < best_n:
        return True
    return False


def _fused_order(mats: np.ndarray, weights: np.ndarray, k: int) -> list[int]:
    """按加权得分取 top-k 列索引（k 会被夹到合法范围）。"""
    d = mats.shape[1]
    k = int(np.clip(k, 1, d))
    fused = (weights[:, None] * mats).sum(axis=0)
    order = np.argsort(-fused, kind="stable")[:k]
    return [int(i) for i in order]


def _grid_fuse(X, y, task, names, mats, cfg) -> dict:
    """确定性网格融合（Optuna 不可用时的兜底，逐位可复现）。

    权重候选 = 等权全 1 + 每个选择器的 one-hot（即"退化为单选择器"）；
    k 候选 = {d/4, d/2, d}。取内层 CV 得分最高者。
    """
    d = mats.shape[1]
    m = len(names)
    weight_vecs: list[np.ndarray] = [np.ones(m)]
    for i in range(m):
        w = np.zeros(m)
        w[i] = 1.0
        weight_vecs.append(w)
    k_grid = sorted({max(1, d // 4), max(1, d // 2), d})
    best = {"score": float("-inf"), "cols": list(range(d)), "weights": dict.fromkeys(names, 1.0)}
    for w in weight_vecs:
        for k in k_grid:
            cols = _fused_order(mats, w, k)
            score, cols = _evaluate_subset(X, y, task, cols, cfg)
            if score > best["score"]:
                best = {
                    "score": score,
                    "cols": cols,
                    "weights": {n: float(w[i]) for i, n in enumerate(names)},
                }
    return best


def _optuna_fuse(X, y, task, names, mats, cfg) -> dict:
    """Optuna(TPE) 搜索融合权重与 top-k。任何异常由调用方回落网格。"""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = mats.shape[1]

    def objective(trial) -> float:
        w = np.array([trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in names])
        k = trial.suggest_int("k", 1, d)
        cols = _fused_order(mats, w, k)
        score, _ = _evaluate_subset(X, y, task, cols, cfg)
        return -score  # Optuna 最小化 → 取负即最大化得分

    sampler = optuna.samplers.TPESampler(seed=cfg.random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=int(cfg.optuna_trials))
    bp = study.best_params
    w = np.array([float(bp[f"w_{n}"]) for n in names])
    cols = _fused_order(mats, w, int(bp["k"]))
    score, cols = _evaluate_subset(X, y, task, cols, cfg)
    return {
        "score": score,
        "cols": cols,
        "weights": {n: float(w[i]) for i, n in enumerate(names)},
    }


def _build_spaces(ds: Dataset, names: list[str] | None):
    """构造已 fit 的算子及其工程化空间：[(name, transformer_obj, engineered_ds), ...]。"""
    spaces = []
    for t in build_transformers(names):
        t.fit(ds)
        spaces.append((t.name, t, t.transform(ds)))
    return spaces


def fit_recipe(
    ds: Dataset,
    cfg: Config,
    transformers: list[str] | None = None,
    selectors: list[str] | None = None,
) -> Recipe:
    """在给定数据上执行完整搜索（阶段 1~3），产出可独立应用的配方。

    本函数**只做搜索**，不做最终评分 —— 最终（无偏）评分由 ``nested_score``
    在留出数据上完成。二者分离是消除选择偏差的关键。
    """
    # 候选网格的内层 CV 必须足够稳定：单折 holdout（cv_folds<3）方差极大，
    # 会让 two-stage 选优在不同 seed/n 下随机抖动（曾出现同子集 n=300 给 0.99、
    # n=200 跌出前 10 的极端情况），进而误选膨胀的多项式空间。强制 ≥3 折，
    # 与外层 baseline 的折数对齐，保证选择可复现、可解释。
    cfg = replace(cfg, cv_folds=max(3, int(cfg.cv_folds)))

    sel_list = build_selectors(selectors, random_state=cfg.random_state)
    spaces = _build_spaces(ds, transformers)

    # ---- 阶段 1：候选网格 (transformer × selector) ----------------------
    # ---- 阶段 1：候选网格 (transformer × selector) ----------------------
    entries: list[dict] = []  # 内部用（含对象引用，不可序列化）
    candidates: list[dict] = []  # 审计用（纯标量，可序列化）
    for t_name, t_obj, t_ds in spaces:
        for s_name, res in select_all(t_ds, sel_list).items():
            score, cols = _evaluate_subset(t_ds.X, t_ds.y, t_ds.task, res.selected, cfg)
            entries.append(
                {
                    "transformer": t_name,
                    "transformer_obj": t_obj,
                    "space_ds": t_ds,
                    "selector": s_name,
                    "score": score,
                    "cols": cols,
                }
            )
            candidates.append(
                {
                    "transformer": t_name,
                    "selector": s_name,
                    "score": score,  # 有偏，仅审计
                    "n_features": len(cols),
                }
            )
    if not entries:
        raise FEFusionError("no candidate (transformer x selector) survived", code="F501")

    # 奥卡姆两段式选优：先取最高分，再在 ±tie_tolerance 带内选**维度最小**者。
    # 单纯 argmax 会被内层噪声带偏 —— 实测 redundant 场景里 4 维与 7 维候选
    # 得分只差 ~0.005（远小于 1 折 holdout 的波动），逐次 argmax 常抢到高维者。
    top = max(e["score"] for e in entries)
    pool = [e for e in entries if e["score"] >= top - cfg.tie_tolerance]
    best = min(pool, key=lambda e: (len(e["cols"]), -e["score"]))
    best_score, best_cols = best["score"], best["cols"]
    best_selector = best["selector"]
    t_name, t_obj, t_ds = best["transformer"], best["transformer_obj"], best["space_ds"]

    # ---- 阶段 2/3：在最佳空间内做多选择器融合 ---------------------------
    space_results = select_all(t_ds, sel_list)
    sel_names = sorted(space_results)
    if not sel_names:
        fused = {"score": best_score, "cols": best_cols, "weights": {}}
    else:
        mats = np.array([_normalize(space_results[n].scores[n]) for n in sel_names])
        use_optuna = cfg.use_optuna and available_optuna() and int(cfg.optuna_trials) > 0
        if use_optuna:
            try:
                fused = _optuna_fuse(t_ds.X, t_ds.y, t_ds.task, sel_names, mats, cfg)
            except Exception:  # noqa: BLE001 - Optuna 异常不影响主链路
                fused = _grid_fuse(t_ds.X, t_ds.y, t_ds.task, sel_names, mats, cfg)
        else:
            fused = _grid_fuse(t_ds.X, t_ds.y, t_ds.task, sel_names, mats, cfg)

    # ---- 阶段 3.5：L1 守护 —— 融合必须不劣于最佳单候选 -------------------
    if _is_better(fused["score"], len(fused["cols"]), best_score, len(best_cols)):
        chosen_cols, chosen_selector, chosen_score = fused["cols"], "fusion", fused["score"]
        weights = fused.get("weights", {})
    else:
        chosen_cols, chosen_selector, chosen_score = best_cols, best_selector, best_score
        weights = {}

    return Recipe(
        transformer=t_name,
        transformer_obj=t_obj,
        cols=sorted(chosen_cols),
        names=[t_ds.feature_names[c] for c in sorted(chosen_cols)],
        selector=chosen_selector,
        weights=weights,
        apparent_score=float(chosen_score),
        n_space_features=int(t_ds.n_features),
        candidates=candidates,
    )


def apply_recipe(recipe: Recipe, ds: Dataset) -> np.ndarray:
    """把配方应用到（可能未参与搜索的）数据上，返回选定特征矩阵。

    只做"查表 + 取列"，不含任何搜索或拟合 —— 因此在外层测试折上调用是安全的。
    """
    eng = recipe.transformer_obj.transform(ds)
    return eng.X[:, recipe.cols]


def nested_score(
    ds: Dataset,
    cfg: Config,
    transformers: list[str] | None = None,
    selectors: list[str] | None = None,
) -> float:
    """嵌套交叉验证：FEFFuse 全流程的**无偏**泛化得分。

    对每个外层折：
      1. 仅在**训练折**上跑完整搜索（工程化 → 选择 → 融合）；
      2. 把配方分别应用到训练折与测试折；
      3. 在训练折上拟合基学习器，在**测试折**上打分。

    测试折从头到尾没参与任何选择决策，因此该得分不含选择偏差。
    ``cfg.nested_folds <= 1`` 时退化为非嵌套（仅调试用，基准禁用）。
    """
    if int(cfg.nested_folds) <= 1:  # 调试模式：直接返回搜索期（有偏）分数
        return fit_recipe(ds, cfg, transformers, selectors).apparent_score

    folds = split_folds_nested(ds.n_samples, cfg)
    n = ds.n_samples
    scores: list[float] = []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        ds_tr, ds_te = ds.take(train_idx), ds.take(test_idx)
        recipe = fit_recipe(ds_tr, cfg, transformers, selectors)
        Xtr = apply_recipe(recipe, ds_tr)
        Xte = apply_recipe(recipe, ds_te)
        scores.append(fold_score(Xtr, ds_tr.y, Xte, ds_te.y, ds.task, cfg))
    if not scores:
        raise FEFusionError("nested evaluation produced no valid folds", code="F502")
    return float(np.mean(scores))


def feffuse(
    ds: Dataset,
    cfg: Config | None = None,
    *,
    transformers: list[str] | None = None,
    selectors: list[str] | None = None,
) -> FeatureForgeResult:
    """FEFFuse 主入口：完成"搜索 → 嵌套评测 → 守护"全链路。

    Parameters
    ----------
    ds : 原始数据集
    cfg : 配置，None 时用默认值（不读 ENV，保证确定性）
    transformers / selectors : 可限定候选范围（消融实验用）

    Returns
    -------
    FeatureForgeResult
        ``metrics.selected_score`` 是**嵌套无偏**得分；
        ``metrics.detail["apparent_score"]`` 是搜索期的乐观分（仅供诊断）。
    """
    cfg = cfg or Config()

    # 基线：原始特征的普通 CV（无选择 → 本身无偏）。
    # 用 nested_folds 折，使基线与嵌套得分落在**同一批外层折**上 → 配对比较，方差更低。
    cfg_pair = replace(cfg, cv_folds=max(2, int(cfg.nested_folds)))
    baseline_score = cv_score(ds.X, ds.y, ds.task, cfg_pair)

    # 全流程嵌套无偏得分
    selected_score = nested_score(ds, cfg, transformers, selectors)

    # 在全量数据上产出最终配方（交付给用户使用的那个）
    recipe = fit_recipe(ds, cfg, transformers, selectors)

    delta = selected_score - baseline_score
    n_selected = len(recipe.cols)
    parsimony = n_selected < ds.n_features

    # ---- 阶段 4：L2 守护 —— 与原始特征对比 -------------------------------
    if delta > cfg.non_inferior_margin:
        winner, safeguard = "fusion", False
    elif abs(delta) <= cfg.tie_tolerance and parsimony:
        winner, safeguard = "fusion", False  # 打平但更精简 → 精简获胜
    else:
        winner, safeguard = "baseline", True

    if winner == "baseline":
        final_ds = ds
        final_names = list(ds.feature_names)
        final_transformer = "identity"
        final_selector = "none"
        final_score = baseline_score
        n_final = ds.n_features
    else:
        eng = recipe.transformer_obj.transform(ds)
        final_ds = eng.subset(recipe.cols)
        final_names = [eng.feature_names[c] for c in recipe.cols]
        final_transformer = recipe.transformer
        final_selector = recipe.selector
        final_score = selected_score
        n_final = n_selected

    metrics = EvalMetrics(
        baseline_score=float(baseline_score),
        selected_score=float(final_score),
        n_features_baseline=ds.n_features,
        n_features_selected=int(n_final),
        delta=float(final_score - baseline_score),
        winner=winner,
        safeguard_triggered=bool(safeguard),
        parsimony=bool(parsimony and abs(delta) <= cfg.tie_tolerance),
        detail={
            "apparent_score": float(recipe.apparent_score),
            "nested_score": float(selected_score),
            "nested_folds": int(cfg.nested_folds),
            "transformer": recipe.transformer,
            "selector": recipe.selector,
            "used_optuna": bool(cfg.use_optuna and available_optuna()),
            "n_space_features": int(recipe.n_space_features),
        },
    )
    return FeatureForgeResult(
        chosen_names=final_names,
        chosen_dataset=final_ds,
        transformer=final_transformer,
        selector=final_selector,
        metrics=metrics,
        fused_weights=recipe.weights,
        candidates=recipe.candidates,  # 审计用；其 score 为搜索期有偏分，不可用于判胜负
    )


# 便于 pipeline/cli 直接引用的别名
run = feffuse
