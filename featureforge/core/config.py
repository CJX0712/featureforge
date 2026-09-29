"""FeatureForge 配置：默认值 + ``FEATUREFORGE_`` 前缀环境变量覆盖。

所有可调旋钮集中在此，禁止在各模块散落魔法数字。ENV 覆盖让 CI / 复现脚本
无需改代码即可切换（如关闭 Optuna 走纯 numpy 兜底路径）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

ENV_PREFIX = "FEATUREFORGE_"


@dataclass
class Config:
    """FeatureForge 全局配置。

    Attributes
    ----------
    random_state : 全局随机种子，保证基准逐位可复现。
    cv_folds : 交叉验证折数。
    cv_folds : **内层**验证折数（用于候选搜索/融合调权的打分）。内层只负责"给候选
        排序"，最终对外报告的分数一律取外层 ``nested_folds`` 折的无偏分，因此内层
        折数可以按算力下调而不损害结论的诚实性；``cv_folds=1`` 时退化为单次
        75/25 holdout，最省算力（排序噪声最大）。
    nested_folds : **外层**嵌套折数。整个"工程化→选择→融合"搜索只在外层训练折
        内进行，得分在未见过的外层测试折上测量 —— 这是消除**特征选择偏差**
        的唯一正确协议（详见 docs/architecture.md）。设为 <=1 时退化为
        非嵌套（仅用于调试，基准禁用）。
    non_inferior_margin : 非劣门槛。工程化特征必须比原始特征高出该值才获胜，
        否则触发守护回退。设为 0.0 表示"严格不劣即胜"。
    tie_tolerance : 持平容差。得分差落在 ±该区间内视为"打平"，此时若特征数
        下降则按**精简原则**判工程化获胜（得分不变但维度更低）。
    use_optuna : 是否启用 Optuna 调融合权重；False 时走等权网格兜底。
    optuna_trials : Optuna 试验次数。
    n_samples : 合成数据默认样本数。
    n_jobs : 并行度。受限 Windows 环境必须保持 1（joblib 并发会崩）。
    """

    random_state: int = 42
    cv_folds: int = 3
    nested_folds: int = 3
    non_inferior_margin: float = 0.0
    tie_tolerance: float = 0.01
    use_optuna: bool = True
    optuna_trials: int = 10
    n_samples: int = 300
    n_jobs: int = 1
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, prefix: str = ENV_PREFIX) -> Config:
        """读取环境变量覆盖默认值。类型按字段当前类型推断，不支持的类型跳过。"""
        cfg = cls()
        for env_key, attr in (
            ("RANDOM_STATE", "random_state"),
            ("CV_FOLDS", "cv_folds"),
            ("NESTED_FOLDS", "nested_folds"),
            ("NON_INFERIOR_MARGIN", "non_inferior_margin"),
            ("TIE_TOLERANCE", "tie_tolerance"),
            ("USE_OPTUNA", "use_optuna"),
            ("OPTUNA_TRIALS", "optuna_trials"),
            ("N_SAMPLES", "n_samples"),
            ("N_JOBS", "n_jobs"),
        ):
            raw = os.environ.get(prefix + env_key)
            if raw is None or raw == "":
                continue
            current = getattr(cfg, attr)
            if isinstance(current, bool):
                setattr(cfg, attr, raw.strip().lower() in ("1", "true", "yes", "on"))
            elif isinstance(current, int):
                setattr(cfg, attr, int(raw))
            elif isinstance(current, float):
                setattr(cfg, attr, float(raw))
        return cfg
