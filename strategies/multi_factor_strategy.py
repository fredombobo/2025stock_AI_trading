"""多因子选股策略"""
from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd

from strategies.base.strategy_base import (
    StrategyBase,
    StrategyConfig,
    TradingSignal,
    SignalType,
)
from data.providers.tushare_provider import TushareProvider
from utils.logger import get_logger

logger = get_logger(__name__)


class MultiFactorStrategy(StrategyBase):
    """动量、估值与换手率的简单多因子模型"""

    def __init__(self, config: StrategyConfig, lookback: int = 20):
        super().__init__(config)
        self.lookback = lookback
        try:
            self.provider = TushareProvider()
        except Exception as e:  # pragma: no cover
            logger.warning(f"无法初始化TushareProvider: {e}")
            self.provider = None

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if data.empty:
            return []
        ts_code = data['ts_code'].iloc[0]
        if len(data) <= self.lookback:
            return []
        close = data.sort_values('trade_date')['close_price']
        momentum = close.iloc[-1] / close.iloc[-self.lookback-1] - 1

        pe = np.nan
        turnover = np.nan
        if self.provider is not None:
            try:
                basic = self.provider.get_daily_basic(
                    ts_code=ts_code,
                    trade_date=data['trade_date'].iloc[-1].strftime('%Y%m%d'),
                )
                if not basic.empty:
                    pe = basic.get('pe_ttm', np.nan).iloc[0]
                    turnover = basic.get('turnover_rate', np.nan).iloc[0]
            except Exception as e:
                logger.debug(f"获取daily_basic失败: {e}")

        scores = []
        if not np.isnan(momentum):
            scores.append(momentum)
        if not np.isnan(pe) and pe > 0:
            scores.append(-pe / 100)  # 估值越低越好
        if not np.isnan(turnover):
            scores.append(turnover / 100)
        if not scores:
            return []
        score = float(np.mean(scores))
        if score <= 0:
            return []
        signal = TradingSignal(
            ts_code=ts_code,
            trade_date=data['trade_date'].iloc[-1],
            signal_type=SignalType.BUY,
            price=data['close_price'].iloc[-1],
            confidence=min(1.0, 0.5 + score),
            reason='multi_factor'
        )
        return [signal]
