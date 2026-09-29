"""FeatureForge 错误码体系（F100 ~ F500）。

统一错误码让流水线在"某一路候选失败"时能精准降级而不是整体崩掉：
单条候选失败只记录日志并跳过，只有 E/F2xx（数据层）才允许向上抛出。
"""

from __future__ import annotations


class FeatureForgeError(Exception):
    """FeatureForge 所有异常的基类。前缀 F 便于与其他 forge 系统区分。"""

    code = "F000"

    def __init__(self, message: str, *, code: str | None = None, cause: Exception | None = None):
        self.message = message
        self.code = code or self.code
        self.cause = cause
        super().__init__(f"[{self.code}] {message}")


class FEConfigError(FeatureForgeError):
    """F1xx —— 配置非法（未知 DGP、ENV 解析失败、参数冲突）。"""

    code = "F100"


class FEDataError(FeatureForgeError):
    """F2xx —— 数据层错误（形状不匹配、全 NaN 列、载入失败）。"""

    code = "F200"


class FEEngineeringError(FeatureForgeError):
    """F3xx —— 特征工程算子失败（分组列缺失、维度爆炸）。"""

    code = "F300"


class FESelectionError(FeatureForgeError):
    """F4xx —— 特征选择失败（全零得分、后端不可用且无兜底）。"""

    code = "F400"


class FEFusionError(FeatureForgeError):
    """F5xx —— 融合/搜索阶段失败（Optuna 异常、候选全空）。"""

    code = "F500"
