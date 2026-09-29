"""合成数据生成器（6 个 DGP，覆盖"工程化必胜 / 精简 / 无信号 / 有害"四种情形）。

设计意图 —— 让基准**有区分度且诚实**
------------------------------------
每个 DGP 都刻意代表一种真实情形，而不是全部让工程化获胜：

===========  ============================================================
DGP          预期结论（诚实预期，非事后编造）
===========  ============================================================
interaction  y 依赖 x0*x1 → 原始线性模型学不到 → 工程化**显著增益**
polynomial   y 依赖 x0²+x1² → 同上 → 工程化**显著增益**
group        y 依赖分组聚合量 → 分组聚合算子**显著增益**
redundant    特征高度冗余 → 得分持平但维度下降 → **精简获胜**
noise        y 与特征完全独立 → 得分恒 ≈0.5，工程化**不应有增益**
highdim      高维弱信号 + 大量噪声 → 多项式膨胀导致过拟合 → **守护回退**
===========  ============================================================

难度梯度由各函数的 ``noise`` 参数控制，取值均经扫参找到"基线非满分、增益可显现"
的甜点区（详见 docs/architecture.md 的扫参记录）。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.errors import FEDataError
from ..core.types import Dataset

# 每个 DGP 的基线难度旋钮。集中在此便于扫参与复现，禁止散落到函数体。
DIFFICULTY = {
    "interaction": {"n": 300, "d": 6, "noise": 0.35},
    "polynomial": {"n": 300, "d": 6, "noise": 0.45},
    "group": {"n": 500, "d": 5, "n_groups": 8, "noise": 0.25},
    "redundant": {"n": 300, "d": 4, "noise": 0.15},
    "noise": {"n": 300, "d": 6, "noise": 1.0},
    "highdim": {"n": 300, "d": 30, "noise": 4.0},
}


def _threshold(logit: np.ndarray) -> np.ndarray:
    """按中位数阈值化，保证正负类均衡（避免类别不平衡污染 accuracy 口径）。

    用中位数而非 0，可让所有 DGP 的正负比例恒为 1:1，
    使不同 DGP 之间的 accuracy 具备可比性。
    """
    return (logit > np.median(logit)).astype(int)


def make_interaction(rs: int, n: int = 300, d: int = 6, noise: float = 0.35) -> Dataset:
    """y 由交互项 x0*x1 决定 —— 纯线性模型在原始空间学不到该结构。"""
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, d))
    logit = 1.2 * (X[:, 0] * X[:, 1]) + noise * rng.standard_normal(n)
    y = _threshold(logit)
    return Dataset(
        X=X,
        y=y,
        feature_names=[f"x{i}" for i in range(d)],
        task="binary",
    )


def make_polynomial(rs: int, n: int = 300, d: int = 6, noise: float = 0.45) -> Dataset:
    """y 由平方项 x0²+x1² 决定 —— 方向对称，线性模型在原始空间无从下手。"""
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, d))
    logit = 0.60 * (X[:, 0] ** 2 + X[:, 1] ** 2) + noise * rng.standard_normal(n)
    y = _threshold(logit)
    return Dataset(
        X=X,
        y=y,
        feature_names=[f"x{i}" for i in range(d)],
        task="binary",
    )


def make_group(
    rs: int, n: int = 300, d: int = 5, n_groups: int = 8, noise: float = 0.25
) -> Dataset:
    """y 由**分组层面的聚合量**决定，必须由分组聚合算子才能显式提取。

    构造方式：给每个分组一个隐藏偏移 ``group_offset``，并把它注入 x1 的取值
    （x1 = 0.3·噪声 + group_offset）。于是：

    - 原始特征 x1 里确实含分组信号，但被 0.3 的组内噪声**稀释**；
    - 分组均值特征 mean(x1 | group) 能把稀释后的信号还原出来（组内约 60 个样本，
      均值标准误约 0.04，远小于组间偏移的标准差 1.0）。

    因此该 DGP 能让"分组聚合"产生真实、可解释的增益，而非数据泄漏式虚高。
    """
    rng = np.random.default_rng(rs)
    grp = rng.integers(0, n_groups, size=n)
    base = rng.standard_normal((n, d))
    group_offset = rng.standard_normal(n_groups)
    # x1 携带分组偏移（索引 1），其余列保持纯噪声
    base[:, 1] = 0.3 * base[:, 1] + group_offset[grp]
    # y 主要由 group_offset 决定；x0 与噪声作为轻微稀释项。
    # 注意：y 经中位数阈值化成二值后，mean(x1|g) 与 y 的**点二列相关**
    # 理论上限≈0.798（由二值化本身决定，与构造无关）。把稀释项压到很小
    # （0.08 量级）并加大样本（n=500）让 r 稳定逼近上限 ~0.79，同时保留少量噪声。
    logit = 1.0 * group_offset[grp] + 0.08 * base[:, 0] + 0.08 * noise * rng.standard_normal(n)
    y = _threshold(logit)
    X = np.column_stack([base, grp.astype(float)])
    names = [f"x{i}" for i in range(d)] + ["group"]
    return Dataset(
        X=X,
        y=y,
        feature_names=names,
        group_idx=d,  # 分组列是最后一列
        task="binary",
    )


def make_redundant(rs: int, n: int = 300, d: int = 4, noise: float = 0.15) -> Dataset:
    """y 只依赖 x0，但特征空间堆了近重复列 —— 考察**精简能力**而非增益。

    预期：得分持平（x0 本来就在），但选择后维度下降 → 判"精简获胜"。
    """
    rng = np.random.default_rng(rs)
    base = rng.standard_normal((n, d))
    logit = 1.5 * base[:, 0] + noise * rng.standard_normal(n)
    y = _threshold(logit)
    # 近重复列的注入噪声**高于**信号噪声(0.15)：它们是"更脏的 x0/x1 副本"，
    # 与原始列的相关系数仍 ≈0.94（仍是近重复），但**不提供任何独特信号**。
    # 因此理智的特征锻造应把它们丢弃 —— 得分持平但维度下降（"精简获胜"）。
    # 若注入噪声过小（< 信号噪声），近重复列反而充当"集成去噪器"，丢弃会掉分，
    # 导致该 DGP 无法稳定触发精简（与文档预期矛盾）。
    dup_noise = max(0.35, noise * 2.5)
    X = np.column_stack(
        [
            base,
            base[:, 0] + dup_noise * rng.standard_normal(n),  # x0 的近重复（更脏）
            base[:, 0] + (dup_noise + 0.2) * rng.standard_normal(n),
            base[:, 1] + dup_noise * rng.standard_normal(n),  # x1 的近重复（更脏）
        ]
    )
    names = [f"x{i}" for i in range(d)] + ["dup0a", "dup0b", "dup1a"]
    return Dataset(X=X, y=y, feature_names=names, task="binary")


def make_noise(rs: int, n: int = 300, d: int = 6, noise: float = 1.0) -> Dataset:
    """y 与 X **完全独立**（均匀随机标签）—— 工程化不应凭空造出增益。

    诚实预期：两条路径得分都 ≈0.5，delta ≈ 0。任何声称的"增益"都是噪声。
    """
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, d))
    y = rng.integers(0, 2, size=n)
    return Dataset(X=X, y=y, feature_names=[f"x{i}" for i in range(d)], task="binary")


def make_highdim(rs: int, n: int = 300, d: int = 30, noise: float = 4.0) -> Dataset:
    """高维弱信号 + 大量纯噪声列 —— 考察**守护回退**。

    30 列原始特征里只有 x0/x1 有信号（信号强度 1.0 / 0.5，噪声 std=4.0，
    信噪比极低）。多项式（degree=2）会额外造出 465 个交互/平方特征，
    其中绝大多数是纯噪声组合；在嵌套 CV 下，融合既不会选中膨胀空间、
    也**无法稳定地靠剪枝超越原始特征**（噪声太强时剪掉的列在留折上
    不再带来增益）→ 净增益≈0，非劣守护触发并回退到**原始特征集**。

    实测（FAST_CFG, n=200, seeds=1..3）：delta ∈ {+0.0000, +0.0103, +0.0000}，
    聚合 +0.0034（≤ tie_tolerance=0.01），约 2/3 种子触发 safeguard；
    seed=1 时 transformer=identity、n_selected=30、safeguard=True。
    注：该 DGP 的增益对种子敏感（逐种子在 0~+0.05 波动），本质是"高维弱信号下
    工程化无稳定增益"，而非每次都干净回退——这正是诚实基准该呈现的样子。
    """
    rng = np.random.default_rng(rs)
    X = rng.standard_normal((n, d))
    logit = 1.0 * X[:, 0] + 0.5 * X[:, 1] + noise * rng.standard_normal(n)
    y = _threshold(logit)
    return Dataset(X=X, y=y, feature_names=[f"x{i}" for i in range(d)], task="binary")


# DGP 注册表。新增 DGP 只需在此登记，pipeline 自动纳入基准。
DGPS: dict[str, Callable[..., Dataset]] = {
    "interaction": make_interaction,
    "polynomial": make_polynomial,
    "group": make_group,
    "redundant": make_redundant,
    "noise": make_noise,
    "highdim": make_highdim,
}


def make_dataset(kind: str, random_state: int = 42, **overrides) -> Dataset:
    """按名字构造 DGP。难度参数可用 ``**overrides`` 覆盖（扫参用）。

    Raises
    ------
    FEDataError
        未知 DGP 名。
    """
    if kind not in DGPS:
        raise FEDataError(f"unknown DGP '{kind}'; available: {sorted(DGPS)}", code="F201")
    params = dict(DIFFICULTY[kind])
    params.update(overrides)
    return DGPS[kind](rs=random_state, **params)
