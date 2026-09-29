"""测试共享配置。

``FAST_CFG`` 的定位：**快，但不牺牲诚实性**。

- ``nested_folds=3`` 保留 —— 测试断言的是**外层嵌套无偏分**，
  若为了快把它关掉，断言就退化成有偏的搜索期分数，测试将失去意义。
- ``cv_folds=1``（内层 75/25 holdout）与 ``use_optuna=False``（确定性网格融合）
  只影响"候选排序"，不影响最终报告的口径，因此是安全的提速点。
"""

from __future__ import annotations

from featureforge.core.config import Config

FAST_CFG = Config(
    random_state=42,
    cv_folds=1,  # 内层：单折 holdout，仅用于给候选排序
    nested_folds=3,  # 外层：嵌套无偏评测，断言依据
    use_optuna=False,  # 确定性网格融合，逐位可复现
    n_samples=200,
)
