"""Minimal moving average crossover strategy for demonstration."""
from dataclasses import dataclass
from enum import Enum
from typing import List

import pandas as pd


class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradingSignal:
    ts_code: str
    trade_date: pd.Timestamp
    signal_type: SignalType
    price: float

    def to_dict(self) -> dict:
        return {
            "ts_code": self.ts_code,
            "trade_date": self.trade_date.isoformat(),
            "signal_type": self.signal_type.name,
            "price": self.price,
        }


class SimpleMovingAverageStrategy:
    """Generate trading signals when short MA crosses long MA."""

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if data.empty or len(data) < self.long_window:
            return []

        df = data.copy()
        df["short_ma"] = df["close_price"].rolling(self.short_window).mean()
        df["long_ma"] = df["close_price"].rolling(self.long_window).mean()

        signals: List[TradingSignal] = []
        prev_short = df["short_ma"].shift(1)
        prev_long = df["long_ma"].shift(1)

        crossover_buy = (df["short_ma"] > df["long_ma"]) & (prev_short <= prev_long)
        crossover_sell = (df["short_ma"] < df["long_ma"]) & (prev_short >= prev_long)

        for idx, row in df[crossover_buy | crossover_sell].iterrows():
            sig_type = SignalType.BUY if crossover_buy.loc[idx] else SignalType.SELL
            signals.append(
                TradingSignal(
                    ts_code=row["ts_code"],
                    trade_date=row["trade_date"],
                    signal_type=sig_type,
                    price=row["close_price"],
                )
            )
        return signals
