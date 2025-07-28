"""VaR计算器模块"""

from typing import Optional
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

class VaRCalculator:
    """Value at Risk (VaR) 计算器"""

    def __init__(self, confidence_level: float = 0.05, method: str = "historical"):
        """初始化VaR计算器

        Args:
            confidence_level: 置信水平，默认0.05表示95%置信区间下的VaR
            method: VaR计算方法，支持"historical"或"parametric"
        """
        self.confidence_level = confidence_level
        self.method = method

    def calculate_var(self, returns: pd.Series) -> float:
        """计算VaR

        Args:
            returns: 收益率序列，按时间顺序排列
        """
        if returns is None or returns.empty:
            logger.warning("收益率序列为空，无法计算VaR")
            return 0.0

        if self.method == "historical":
            var = returns.quantile(self.confidence_level)
        elif self.method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            # 使用正态分布假设计算VaR，使用numpy近似
            z = np.percentile(np.random.standard_normal(100000), self.confidence_level * 100)
            var = mu + sigma * z
        else:
            raise ValueError(f"未知的VaR计算方法: {self.method}")

        logger.debug(f"计算得到VaR: {var}")
        return float(var)

    def calculate_cvar(self, returns: pd.Series) -> float:
        """计算条件VaR (CVaR)"""
        if returns is None or returns.empty:
            logger.warning("收益率序列为空，无法计算CVaR")
            return 0.0

        var = self.calculate_var(returns)
        cvar = returns[returns <= var].mean()
        logger.debug(f"计算得到CVaR: {cvar}")
        return float(cvar)

