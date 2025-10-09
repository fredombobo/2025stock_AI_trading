"""Strategy package shortcuts."""

from .base import (
    SignalType,
    PositionType,
    TradingSignal,
    StrategyConfig,
    StrategyBase,
    SignalCondition,
    SignalGenerator,
    Trade,
    BacktestEngine,
    MACrossoverStrategy,
    MACDStrategy,
)
from .china_limit_up_strategy import ChinaLimitUpConfig, ChinaLimitUpMomentumStrategy

__all__ = [
    "SignalType",
    "PositionType",
    "TradingSignal",
    "StrategyConfig",
    "StrategyBase",
    "SignalCondition",
    "SignalGenerator",
    "Trade",
    "BacktestEngine",
    "MACrossoverStrategy",
    "MACDStrategy",
    "ChinaLimitUpConfig",
    "ChinaLimitUpMomentumStrategy",
]
