# risk_management/portfolio/portfolio_manager.py
"""组合管理器"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json

from data.storage.database_manager import data_repository
from portfolio_manager.risk_models.var_calculator import VaRCalculator
from portfolio_manager.controls.position_limit import PositionLimitControl
from portfolio_manager.controls.drawdown_control import DrawdownControl
from utils.logger import get_logger
from utils.exceptions import RiskManagementError
from utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown

logger = get_logger(__name__)

@dataclass
class Position:
    """持仓信息"""
    ts_code: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    entry_date: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'ts_code': self.ts_code,
            'quantity': self.quantity,
            'avg_cost': self.avg_cost,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'weight': self.weight,
            'entry_date': self.entry_date.isoformat()
        }

@dataclass
class PortfolioSnapshot:
    """组合快照"""
    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    total_pnl: float
    daily_return: float
    positions: List[Position]
    risk_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'cash': self.cash,
            'positions_value': self.positions_value,
            'total_pnl': self.total_pnl,
            'daily_return': self.daily_return,
            'positions': [pos.to_dict() for pos in self.positions],
            'risk_metrics': self.risk_metrics
        }

class PortfolioManager:
    """组合管理器"""
    
    def __init__(self, initial_capital: float = 1000000, 
                 max_position_weight: float = 0.1,
                 max_sector_weight: float = 0.3,
                 cash_reserve_ratio: float = 