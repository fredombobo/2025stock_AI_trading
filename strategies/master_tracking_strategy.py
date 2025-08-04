"""主力资金跟踪策略"""
from __future__ import annotations

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


class MasterTrackingStrategy(StrategyBase):
    """通过龙虎榜和大宗交易跟踪主力资金的策略"""

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        try:
            self.provider = TushareProvider()
        except Exception as e:  # pragma: no cover - 无token环境下不报错
            logger.warning(f"无法初始化TushareProvider: {e}")
            self.provider = None

    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        if data.empty or self.provider is None:
            return []

        ts_code = data['ts_code'].iloc[0]
        trade_date = data['trade_date'].iloc[-1]
        date_str = trade_date.strftime('%Y%m%d')

        reasons = []
        confidence = 0.5

        try:
            lhb = self.provider.get_top_list(trade_date=date_str)
            if not lhb.empty:
                record = lhb[lhb['ts_code'] == ts_code]
                if not record.empty and record['net_amount'].iloc[0] > 1e7:
                    reasons.append('top_list_net_buy')
                    confidence += 0.2
        except Exception as e:
            logger.debug(f"获取龙虎榜数据失败: {e}")

        try:
            block = self.provider.get_block_trade(ts_code=ts_code,
                                                  start_date=date_str,
                                                  end_date=date_str)
            if not block.empty:
                premium = (block['price'] - block['pre_close']) / block['pre_close']
                if premium.mean() > 0:
                    reasons.append('block_trade_premium')
                    confidence += 0.2
        except Exception as e:
            logger.debug(f"获取大宗交易数据失败: {e}")

        if not reasons:
            return []

        price = data['close_price'].iloc[-1]
        signal = TradingSignal(
            ts_code=ts_code,
            trade_date=trade_date,
            signal_type=SignalType.BUY,
            price=price,
            confidence=min(1.0, confidence),
            reason=','.join(reasons),
        )
        return [signal]
