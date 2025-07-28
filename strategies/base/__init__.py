"""Expose core strategy classes."""

from .strategy_base import (
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
]
