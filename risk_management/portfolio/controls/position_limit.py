"""仓位限制控制模块"""

from typing import Dict, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

class PositionLimitControl:
    """仓位限制控制类"""

    def __init__(self, max_position_weight: float = 0.1, max_sector_weight: float = 0.3):
        """初始化仓位限制控制器

        Args:
            max_position_weight: 单个标的的最大仓位占比
            max_sector_weight: 单个板块的最大仓位占比
        """
        self.max_position_weight = max_position_weight
        self.max_sector_weight = max_sector_weight

    def check_position_limit(self, current_positions: Dict[str, float], ts_code: str,
                              new_value: float, total_portfolio_value: float) -> bool:
        """检查单个标的仓位限制"""
        existing = current_positions.get(ts_code, 0.0)
        weight = (existing + new_value) / total_portfolio_value
        within = weight <= self.max_position_weight
        if not within:
            logger.warning(
                f"{ts_code} 仓位比例 {weight:.2%} 超过上限 {self.max_position_weight:.2%}")
        return within

    def check_sector_limit(self, sector_exposure: Dict[str, float], sector: Optional[str],
                           new_value: float, total_portfolio_value: float) -> bool:
        """检查板块仓位限制"""
        if not sector:
            return True
        existing = sector_exposure.get(sector, 0.0)
        weight = (existing + new_value) / total_portfolio_value
        within = weight <= self.max_sector_weight
        if not within:
            logger.warning(
                f"板块 {sector} 仓位比例 {weight:.2%} 超过上限 {self.max_sector_weight:.2%}")
        return within

