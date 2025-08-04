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
)

from .master_tracking_strategy import MasterTrackingStrategy
from .sector_rotation_strategy import SectorRotationStrategy
from .multi_factor_strategy import MultiFactorStrategy
from .capital_flow_strategy import CapitalFlowStrategy

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
    "MasterTrackingStrategy",
    "SectorRotationStrategy",
    "MultiFactorStrategy",
    "CapitalFlowStrategy",
]
