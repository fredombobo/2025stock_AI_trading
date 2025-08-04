"""资金博弈分析策略"""
from __future__ import annotations

from datetime import timedelta
from typing import List
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


class CapitalFlowStrategy(StrategyBase):
    """基于资金流向的简单短线策略"""

    def __init__(self, config: StrategyConfig, lookback: int = 3):
        super().__init__(config)
        self.lookback = lookback
        try:
            self.provider = TushareProvider()
        except Exception as e:  # pragma: no cover
            logger.warning(f"无法初始化TushareProvider: {e}")
            self.provider = None

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if data.empty or self.provider is None:
            return []
        ts_code = data['ts_code'].iloc[0]
        end = data['trade_date'].iloc[-1]
        start = (end - timedelta(days=self.lookback * 2)).strftime('%Y%m%d')
        end_str = end.strftime('%Y%m%d')
        try:
            mf = self.provider.get_moneyflow(ts_code, start_date=start, end_date=end_str)
        except Exception as e:
            logger.debug(f"获取资金流数据失败: {e}")
            return []
        if mf.empty or 'net_mf_amount' not in mf.columns:
            return []
        mf = mf.sort_values('trade_date')
        recent = mf['net_mf_amount'].tail(self.lookback)
        if recent.sum() > 0 and recent.iloc[-1] > 0:
            signal = TradingSignal(
                ts_code=ts_code,
                trade_date=end,
                signal_type=SignalType.BUY,
                price=data['close_price'].iloc[-1],
                confidence=0.6,
                reason='moneyflow_positive'
            )
            return [signal]
        return []
