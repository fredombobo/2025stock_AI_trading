"""回撤控制模块"""

import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

class DrawdownControl:
    """基于最大回撤的风控"""

    def __init__(self, max_drawdown: float = 0.2):
        """初始化回撤控制器

        Args:
            max_drawdown: 允许的最大回撤比例，如0.2表示20%
        """
        self.max_drawdown = max_drawdown

    def is_within_drawdown(self, equity_curve: pd.Series) -> bool:
        """检查最新回撤是否超过阈值

        Args:
            equity_curve: 组合净值曲线
        Returns:
            True表示在允许范围内，False表示超过限制
        """
        if equity_curve is None or equity_curve.empty:
            return True

        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        current_dd = drawdown.iloc[-1]

        if current_dd < -self.max_drawdown:
            logger.warning(
                f"组合回撤 {abs(current_dd):.2%} 超过限制 {self.max_drawdown:.2%}")
            return False
        return True

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """计算历史最大回撤"""
        if equity_curve is None or equity_curve.empty:
            return 0.0
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max
        return float(drawdown.min())

