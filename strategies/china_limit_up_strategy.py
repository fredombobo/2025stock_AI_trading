"""商业级中国A股打板策略实现。

该模块实现了一个面向中国A股市场的“打板”动量策略。策略聚焦于识别
连续涨停（俗称“连板”）股票，并结合量能、换手、筹码结构等维度对
信号进行过滤，同时给出完善的风控参数配置。代码遵循项目内的策略
基类接口，方便接入现有的回测与交易执行框架。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.base import StrategyBase, StrategyConfig, TradingSignal, SignalType
from utils.logger import get_logger
from utils.validators import validate_dataframe_columns


logger = get_logger(__name__)


@dataclass
class ChinaLimitUpConfig(StrategyConfig):
    """中国A股打板策略专用配置。

    Attributes:
        limit_up_pct_main: 主板股票涨停阈值（默认9.7%，兼顾部分差价）。
        limit_up_pct_growth: 创业板/科创板涨停阈值（默认19.5%）。
        limit_up_pct_st: ST股票涨停阈值（默认4.8%）。
        volume_surge_ratio: 成交量放大倍数阈值，相对N日均量。
        turnover_lower_bound: 换手率下限，确保有足够换手。
        turnover_upper_bound: 换手率上限，防止过度博弈。
        breakout_lookback: 历史高点回看周期。
        consecutive_limit_up_days: 要求的连续涨停天数。
        intraday_pullback_pct: 当日最大回撤阈值，用于判断封板强度。
        max_open_gap: 开盘跳空幅度上限，避免情绪见顶。
        hold_days: 计划持有天数，用于仓位管理（辅助信息）。
    """

    limit_up_pct_main: float = 0.097
    limit_up_pct_growth: float = 0.195
    limit_up_pct_st: float = 0.048
    volume_surge_ratio: float = 1.8
    turnover_lower_bound: float = 2.0
    turnover_upper_bound: float = 25.0
    breakout_lookback: int = 20
    consecutive_limit_up_days: int = 2
    intraday_pullback_pct: float = 0.03
    max_open_gap: float = 0.05
    hold_days: int = 2


class ChinaLimitUpMomentumStrategy(StrategyBase):
    """针对中国A股市场的商业级打板策略实现。

    策略核心思想：

    1. 针对不同板块设置差异化的涨停阈值，识别合规的涨停板。
    2. 通过连续涨停、量能放大、换手区间、突破结构等多维度过滤信号。
    3. 自动估算信号置信度，并给出配套止损、止盈、仓位建议。
    4. 兼容项目内统一的 ``StrategyBase`` 接口，可直接用于回测或实盘。
    """

    #: 默认需要的数据列
    REQUIRED_COLUMNS = {
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
    }

    #: 可接受的价格字段别名
    PRICE_ALIASES: Dict[str, str] = {
        "open_price": "open",
        "high_price": "high",
        "low_price": "low",
        "close_price": "close",
        "prev_close": "pre_close",
        "preclose": "pre_close",
    }

    def __init__(self, config: Optional[ChinaLimitUpConfig] = None):
        config = config or ChinaLimitUpConfig(name="China Limit Up Momentum", max_stocks=5)
        super().__init__(config)

        # 记录用于评估的策略属性
        self.limit_up_thresholds = {
            "main": config.limit_up_pct_main,
            "growth": config.limit_up_pct_growth,
            "st": config.limit_up_pct_st,
        }
        self.consecutive_limit_up_days = config.consecutive_limit_up_days
        self.required_columns = self.REQUIRED_COLUMNS

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """生成打板买入信号。

        Args:
            data: 经过 ``StrategyBase.prepare_data`` 或外部处理后的行情数据。

        Returns:
            交易信号列表。
        """

        if data.empty:
            logger.warning("输入数据为空，无法生成信号")
            return []

        standardized = self._standardize_columns(data.copy())

        if not validate_dataframe_columns(standardized, list(self.REQUIRED_COLUMNS)):
            missing = self.REQUIRED_COLUMNS - set(standardized.columns)
            raise ValueError(f"数据缺少必需列: {missing}")

        standardized["trade_date"] = pd.to_datetime(standardized["trade_date"])
        standardized.sort_values(["ts_code", "trade_date"], inplace=True)

        signals: List[TradingSignal] = []
        for ts_code, stock_df in standardized.groupby("ts_code"):
            enriched = self._enrich_stock_frame(ts_code, stock_df)
            if enriched.empty:
                continue
            stock_signals = self._generate_stock_signals(ts_code, enriched)
            signals.extend(stock_signals)

        if not signals:
            logger.info("未触发打板信号")

        return signals

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """统一不同数据源的字段命名。"""

        rename_map: Dict[str, str] = {}
        for alias, target in self.PRICE_ALIASES.items():
            if alias in data.columns and target not in data.columns:
                rename_map[alias] = target
        if rename_map:
            data = data.rename(columns=rename_map)

        # 将 vol/amount 等转换为数值
        numeric_cols = [col for col in ["open", "high", "low", "close", "pre_close", "vol", "amount"] if col in data.columns]
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors="coerce")
        data = data.dropna(subset=[col for col in ["open", "high", "low", "close", "pre_close", "vol"] if col in data.columns])

        return data

    def _enrich_stock_frame(self, ts_code: str, df: pd.DataFrame) -> pd.DataFrame:
        """计算策略所需的衍生指标。"""

        df = df.copy()
        df["pct_change"] = np.where(
            df["pre_close"] > 0,
            (df["close"] - df["pre_close"]) / df["pre_close"],
            np.nan,
        )

        if "pct_chg" in df.columns and df["pct_chg"].notna().any():
            df.loc[df["pct_chg"].notna(), "pct_change"] = df.loc[df["pct_chg"].notna(), "pct_chg"] / 100.0

        df["limit_up"] = df.apply(lambda row: self._is_limit_up(ts_code, row), axis=1)

        # 计算量能指标
        df["volume_ma5"] = df["vol"].rolling(window=5, min_periods=1).mean()
        df["volume_ma20"] = df["vol"].rolling(window=20, min_periods=1).mean()
        df["volume_ratio"] = np.where(df["volume_ma5"] > 0, df["vol"] / df["volume_ma5"], 0)
        df["volume_surge"] = np.where(df["volume_ma20"] > 0, df["vol"] / df["volume_ma20"], 0)

        # 换手率（若缺失则估算）
        if "turnover_rate" not in df.columns:
            df["turnover_rate"] = np.nan
        rolling_turnover = df["turnover_rate"].rolling(window=5, min_periods=1).mean()
        df["turnover_rate"] = df["turnover_rate"].fillna(rolling_turnover)

        # 最高价突破
        df["rolling_high"] = df["high"].rolling(window=self.config.breakout_lookback, min_periods=1).max()
        df["is_breakout"] = df["high"] >= df["rolling_high"]

        # 当日回落幅度
        df["intraday_pullback"] = np.where(
            df["high"] > 0,
            (df["high"] - df["close"]) / df["high"],
            0,
        )

        df["gap_open"] = np.where(
            df["pre_close"] > 0,
            (df["open"] - df["pre_close"]) / df["pre_close"],
            0,
        )

        # 连板统计
        df["limit_up_streak"] = df["limit_up"].astype(int)
        df["limit_up_streak"] = df["limit_up_streak"].groupby((df["limit_up_streak"] == 0).cumsum()).cumsum()

        return df

    def _generate_stock_signals(self, ts_code: str, df: pd.DataFrame) -> List[TradingSignal]:
        """在单只股票维度生成信号。"""

        signals: List[TradingSignal] = []

        if len(df) < self.config.lookback_period:
            recent_df = df.copy()
        else:
            recent_df = df.iloc[-self.config.lookback_period :].copy()

        latest = recent_df.iloc[-1]

        if not self._passes_core_filters(recent_df, latest):
            return signals

        confidence = self._estimate_confidence(recent_df)

        signal = TradingSignal(
            ts_code=ts_code,
            trade_date=latest["trade_date"],
            signal_type=SignalType.BUY,
            price=float(latest["close"]),
            confidence=confidence,
            reason=self._build_signal_reason(recent_df, latest),
            position_size=min(0.08 + confidence * 0.12, self.config.max_position_size),
        )

        signal.stop_loss = signal.price * (1 - max(self.config.stop_loss_pct, 0.07))
        signal.take_profit = signal.price * (1 + max(self.config.take_profit_pct, 0.15))

        signals.append(signal)

        logger.debug(
            "生成打板信号: %s 日期=%s 置信度=%.2f", ts_code, latest["trade_date"].date(), confidence
        )

        return signals

    def _passes_core_filters(self, recent_df: pd.DataFrame, latest: pd.Series) -> bool:
        """核心过滤条件。"""

        if not latest["limit_up"]:
            return False

        if latest["limit_up_streak"] < self.config.consecutive_limit_up_days:
            return False

        if latest["intraday_pullback"] > self.config.intraday_pullback_pct:
            return False

        if latest["gap_open"] > self.config.max_open_gap:
            return False

        volume_condition = (
            latest["volume_surge"] >= self.config.volume_surge_ratio
            and latest["volume_ratio"] >= 1.5
        )
        if not volume_condition:
            return False

        turnover = latest.get("turnover_rate", np.nan)
        if not np.isnan(turnover):
            if not (self.config.turnover_lower_bound <= turnover <= self.config.turnover_upper_bound):
                return False

        breakout_confirmed = bool(latest.get("is_breakout", False))
        if not breakout_confirmed:
            return False

        # 防止连续巨量后衰竭：近3日量能加权判断
        last_three_vol = recent_df["vol"].tail(3)
        if last_three_vol.isna().any() or len(last_three_vol) < 3:
            return False
        if last_three_vol.iloc[-1] < last_three_vol.mean():
            return False

        return True

    def _estimate_confidence(self, recent_df: pd.DataFrame) -> float:
        """根据多维指标估算信号置信度。"""

        latest = recent_df.iloc[-1]

        confidence_components: List[float] = []

        # 连板力度
        streak_score = min(1.0, latest["limit_up_streak"] / (self.config.consecutive_limit_up_days + 1))
        confidence_components.append(0.25 + 0.35 * streak_score)

        # 量能及换手
        volume_factor = min(2.0, latest["volume_surge"] / self.config.volume_surge_ratio)
        confidence_components.append(0.2 * volume_factor)

        turnover = latest.get("turnover_rate", np.nan)
        if not np.isnan(turnover):
            turnover_mid = (self.config.turnover_lower_bound + self.config.turnover_upper_bound) / 2
            turnover_score = 1 - abs(turnover - turnover_mid) / turnover_mid
            confidence_components.append(0.15 * max(0.0, min(turnover_score, 1.0)))
        else:
            confidence_components.append(0.05)

        # 回撤越小越好
        pullback_score = max(0.0, 1 - latest["intraday_pullback"] / self.config.intraday_pullback_pct)
        confidence_components.append(0.15 * min(1.0, pullback_score))

        # 近5日平均涨幅平滑动量
        pct_window = recent_df["pct_change"].tail(5).replace([np.inf, -np.inf], np.nan).dropna()
        if not pct_window.empty:
            momentum = pct_window.mean()
            momentum_score = max(0.0, min(momentum / 0.05, 1.0))
            confidence_components.append(0.15 * momentum_score)
        else:
            confidence_components.append(0.05)

        confidence = float(np.clip(sum(confidence_components), 0.6, 0.99))

        return confidence

    def _build_signal_reason(self, recent_df: pd.DataFrame, latest: pd.Series) -> str:
        """构建可读的信号说明。"""

        reason_parts = [
            f"{int(latest['limit_up_streak'])}连板封死",
            f"量能{latest['volume_surge']:.1f}倍",
        ]

        turnover = latest.get("turnover_rate", np.nan)
        if not np.isnan(turnover):
            reason_parts.append(f"换手{turnover:.1f}%")

        momentum = recent_df["pct_change"].tail(3).mean()
        reason_parts.append(f"近3日均涨幅{momentum * 100:.1f}%")

        if latest.get("is_breakout", False):
            reason_parts.append("突破前高")

        return "；".join(reason_parts)

    def _is_limit_up(self, ts_code: str, row: pd.Series) -> bool:
        """依据股票所属板块推断涨停状态。"""

        if "limit_status" in row and isinstance(row["limit_status"], str):
            return row["limit_status"].upper() in {"UP", "LIMIT_UP", "涨停"}

        pct_change = row.get("pct_change")
        if pd.isna(pct_change):
            return False

        pct_change = float(pct_change)

        threshold = self._determine_threshold(ts_code, row)

        return pct_change >= threshold * 0.9  # 留出容错空间，考虑收盘价低于理论涨停价的情况

    def _determine_threshold(self, ts_code: str, row: pd.Series) -> float:
        """根据股票类型给出涨停阈值。"""

        ts_code = ts_code or row.get("ts_code", "")
        ts_code = str(ts_code)

        is_star = ts_code.startswith("688") or ts_code.startswith("689")
        is_chinext = ts_code.startswith("300") or ts_code.startswith("301")
        is_beijing = ts_code.startswith("83") or ts_code.startswith("87")

        if row.get("is_st", False) or "ST" in str(row.get("name", "")):
            return self.config.limit_up_pct_st

        if is_star or is_chinext or is_beijing:
            return self.config.limit_up_pct_growth

        return self.config.limit_up_pct_main

