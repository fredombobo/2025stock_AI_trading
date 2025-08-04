"""板块轮动策略"""
from __future__ import annotations

from typing import Dict, List, Set
import pandas as pd
import numpy as np

from strategies.base.strategy_base import (
    StrategyBase,
    StrategyConfig,
    TradingSignal,
    SignalType,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class SectorRotationStrategy(StrategyBase):
    """简单的板块轮动策略"""

    def __init__(self, config: StrategyConfig, sector_map: Dict[str, List[str]],
                 lookback: int = 20, top_n: int = 3):
        super().__init__(config)
        self.sector_map = sector_map
        self.lookback = lookback
        self.top_n = top_n
        self.stock_to_sector = {
            code: sector for sector, codes in sector_map.items() for code in codes
        }
        self.top_sectors: Set[str] = set()

    def _compute_sector_strength(self, start_date: str, end_date: str):
        sector_returns: Dict[str, float] = {}
        for sector, codes in self.sector_map.items():
            rets = []
            for code in codes:
                df = self.repository.get_daily_price(code, start_date, end_date)
                if df.empty or len(df) <= self.lookback:
                    continue
                close = df.sort_values('trade_date')['close_price']
                ret = close.iloc[-1] / close.iloc[-self.lookback-1] - 1
                rets.append(ret)
            sector_returns[sector] = float(np.mean(rets)) if rets else -np.inf
        self.top_sectors = set(
            sorted(sector_returns, key=sector_returns.get, reverse=True)[: self.top_n]
        )

    def run_strategy(self, ts_codes: List[str], start_date: str, end_date: str):
        self._compute_sector_strength(start_date, end_date)
        return super().run_strategy(ts_codes, start_date, end_date)

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if data.empty:
            return []
        ts_code = data['ts_code'].iloc[0]
        sector = self.stock_to_sector.get(ts_code)
        if sector not in self.top_sectors:
            return []
        if len(data) <= self.lookback:
            return []
        close = data.sort_values('trade_date')['close_price']
        ret = close.iloc[-1] / close.iloc[-self.lookback-1] - 1
        if ret <= 0:
            return []
        signal = TradingSignal(
            ts_code=ts_code,
            trade_date=data['trade_date'].iloc[-1],
            signal_type=SignalType.BUY,
            price=data['close_price'].iloc[-1],
            confidence=0.6,
            reason=f"sector:{sector}"
        )
        return [signal]
