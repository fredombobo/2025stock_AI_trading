"""Simple moving-average crossover strategy."""
from dataclasses import dataclass
from typing import List
import pandas as pd

from strategies.base.strategy_base import StrategyBase, StrategyConfig, TradingSignal, SignalType
from utils.validators import validate_dataframe_columns


@dataclass
class MovingAverageConfig(StrategyConfig):
    short_window: int = 5
    long_window: int = 20


class MovingAverageCrossoverStrategy(StrategyBase):
    """Generates buy/sell signals based on moving average crossovers."""

    def __init__(self, config: MovingAverageConfig | None = None) -> None:
        if config is None:
            config = MovingAverageConfig(name="MA Crossover")
        super().__init__(config)
        self.short_window = config.short_window
        self.long_window = config.long_window

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals from close price data.

        Args:
            data: DataFrame with at least ts_code, trade_date and close_price columns.
        Returns:
            List of TradingSignal objects.
        """
        required_cols = ["ts_code", "trade_date", "close_price"]
        validate_dataframe_columns(data, required_cols)
        df = data.sort_values("trade_date").copy()
        df["short_ma"] = df["close_price"].rolling(self.short_window).mean()
        df["long_ma"] = df["close_price"].rolling(self.long_window).mean()

        signals: List[TradingSignal] = []
        for i in range(1, len(df)):
            row_prev = df.iloc[i - 1]
            row = df.iloc[i]
            if pd.isna(row_prev.short_ma) or pd.isna(row_prev.long_ma):
                continue
            if row.short_ma > row.long_ma and row_prev.short_ma <= row_prev.long_ma:
                signals.append(
                    TradingSignal(
                        ts_code=row.ts_code,
                        trade_date=row.trade_date,
                        signal_type=SignalType.BUY,
                        price=row.close_price,
                        reason="short_ma crossed above long_ma",
                    )
                )
            elif row.short_ma < row.long_ma and row_prev.short_ma >= row_prev.long_ma:
                signals.append(
                    TradingSignal(
                        ts_code=row.ts_code,
                        trade_date=row.trade_date,
                        signal_type=SignalType.SELL,
                        price=row.close_price,
                        reason="short_ma crossed below long_ma",
                    )
                )
        return signals
