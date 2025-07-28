# strategies/base/strategy_base.py
"""策略基类"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from data.storage.database_manager import data_repository
from data.processors.technical_indicators import technical_indicators
from utils.logger import get_logger
from utils.exceptions import StrategyError
from utils.validators import validate_dataframe_columns

logger = get_logger(__name__)

class SignalType(Enum):
    """信号类型枚举"""
    BUY = 1
    SELL = -1
    HOLD = 0

class PositionType(Enum):
    """持仓类型枚举"""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

@dataclass
class TradingSignal:
    """交易信号数据类"""
    ts_code: str
    trade_date: datetime
    signal_type: SignalType
    price: float
    confidence: float = 0.5  # 信号置信度 0-1
    reason: str = ""         # 信号原因
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: float = 1.0  # 建议仓位大小 0-1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'ts_code': self.ts_code,
            'trade_date': self.trade_date.isoformat(),
            'signal_type': self.signal_type.value,
            'price': self.price,
            'confidence': self.confidence,
            'reason': self.reason,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'position_size': self.position_size
        }

@dataclass
class StrategyConfig:
    """策略配置数据类"""
    name: str
    version: str = "1.0.0"
    lookback_period: int = 60  # 回看天数
    min_volume: int = 10000    # 最小成交量过滤
    max_stocks: int = 10       # 最大持股数量
    rebalance_frequency: str = "daily"  # 调仓频率
    commission_rate: float = 0.0003     # 手续费率
    slippage_rate: float = 0.001        # 滑点率
    
    # 风控参数
    max_position_size: float = 0.1      # 单股最大仓位
    stop_loss_pct: float = 0.1          # 止损百分比
    take_profit_pct: float = 0.2        # 止盈百分比
    max_drawdown_limit: float = 0.15    # 最大回撤限制

class StrategyBase(ABC):
    """策略基类"""
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.repository = data_repository
        self.tech_indicators = technical_indicators
        
        # 策略状态
        self.positions = {}  # 当前持仓
        self.signals_history = []  # 信号历史
        self.performance_metrics = {}  # 绩效指标
        
        logger.info(f"初始化策略: {self.config.name} v{self.config.version}")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """生成交易信号 - 子类必须实现"""
        pass
    
    def prepare_data(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """准备策略所需数据"""
        try:
            # 获取价格数据
            price_data = self.repository.get_daily_price(ts_code, start_date, end_date)
            
            if price_data.empty:
                logger.warning(f"股票 {ts_code} 无价格数据")
                return pd.DataFrame()
            
            # 获取技术指标
            indicators = self.repository.get_technical_indicators(ts_code, start_date, end_date)
            
            # 合并数据
            if not indicators.empty:
                data = price_data.merge(indicators, on=['ts_code', 'trade_date'], how='left')
            else:
                data = price_data.copy()
                logger.info(f"股票 {ts_code} 无技术指标数据，将计算")
                # 计算基础技术指标
                self.tech_indicators.calculate_all_indicators(ts_code, start_date, end_date)
                indicators = self.repository.get_technical_indicators(ts_code, start_date, end_date)
                if not indicators.empty:
                    data = price_data.merge(indicators, on=['ts_code', 'trade_date'], how='left')
            
            # 数据预处理
            data = self.preprocess_data(data)
            
            return data.sort_values('trade_date').reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"准备数据失败: {e}")
            raise StrategyError(f"数据准备失败: {e}")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        if data.empty:
            return data
        
        # 过滤成交量
        if 'vol' in data.columns:
            data = data[data['vol'] >= self.config.min_volume]
        
        # 移除异常值
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['ts_code']:
                # 使用3σ原则移除异常值
                mean_val = data[col].mean()
                std_val = data[col].std()
                data = data[abs(data[col] - mean_val) <= 3 * std_val]
        
        return data
    
    def calculate_position_size(self, signal: TradingSignal, available_capital: float) -> float:
        """计算建议仓位大小"""
        # 基础仓位
        base_position = min(signal.position_size, self.config.max_position_size)
        
        # 根据信号置信度调整
        confidence_adjusted = base_position * signal.confidence
        
        # 根据可用资金调整
        max_affordable = available_capital / (signal.price * 100)  # 假设最小100股
        
        return min(confidence_adjusted, max_affordable)
    
    def apply_risk_management(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """应用风险管理规则"""
        filtered_signals = []
        
        for signal in signals:
            # 设置止损止盈
            if signal.stop_loss is None and self.config.stop_loss_pct > 0:
                if signal.signal_type == SignalType.BUY:
                    signal.stop_loss = signal.price * (1 - self.config.stop_loss_pct)
                elif signal.signal_type == SignalType.SELL:
                    signal.stop_loss = signal.price * (1 + self.config.stop_loss_pct)
            
            if signal.take_profit is None and self.config.take_profit_pct > 0:
                if signal.signal_type == SignalType.BUY:
                    signal.take_profit = signal.price * (1 + self.config.take_profit_pct)
                elif signal.signal_type == SignalType.SELL:
                    signal.take_profit = signal.price * (1 - self.config.take_profit_pct)
            
            # 仓位大小限制
            signal.position_size = min(signal.position_size, self.config.max_position_size)
            
            filtered_signals.append(signal)
        
        # 限制总信号数量
        if len(filtered_signals) > self.config.max_stocks:
            # 按置信度排序，取前N个
            filtered_signals = sorted(filtered_signals, key=lambda x: x.confidence, reverse=True)
            filtered_signals = filtered_signals[:self.config.max_stocks]
        
        return filtered_signals
    
    def run_strategy(self, ts_codes: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """运行策略"""
        try:
            all_signals = []
            strategy_data = {}
            
            logger.info(f"运行策略 {self.config.name}，股票数量: {len(ts_codes)}")
            
            for i, ts_code in enumerate(ts_codes):
                logger.debug(f"处理股票 {ts_code} ({i+1}/{len(ts_codes)})")
                
                try:
                    # 准备数据
                    data = self.prepare_data(ts_code, start_date, end_date)
                    
                    if data.empty:
                        logger.warning(f"股票 {ts_code} 数据为空，跳过")
                        continue
                    
                    # 生成信号
                    signals = self.generate_signals(data)
                    
                    if signals:
                        # 应用风险管理
                        filtered_signals = self.apply_risk_management(signals)
                        all_signals.extend(filtered_signals)
                        
                        logger.debug(f"股票 {ts_code} 生成 {len(signals)} 个信号，过滤后 {len(filtered_signals)} 个")
                    
                    # 保存数据用于分析
                    strategy_data[ts_code] = data
                    
                except Exception as e:
                    logger.error(f"处理股票 {ts_code} 失败: {e}")
                    continue
            
            # 记录信号历史
            self.signals_history.extend(all_signals)
            
            logger.info(f"策略运行完成，共生成 {len(all_signals)} 个信号")
            
            return {
                'signals': all_signals,
                'strategy_data': strategy_data,
                'summary': {
                    'total_signals': len(all_signals),
                    'buy_signals': len([s for s in all_signals if s.signal_type == SignalType.BUY]),
                    'sell_signals': len([s for s in all_signals if s.signal_type == SignalType.SELL]),
                    'processed_stocks': len(strategy_data),
                    'strategy_name': self.config.name,
                    'run_date': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"策略运行失败: {e}")
            raise StrategyError(f"策略运行失败: {e}")
    
    def backtest(self, ts_codes: List[str], start_date: str, end_date: str,
                initial_capital: float = 1000000) -> Dict[str, Any]:
        """回测策略"""
        try:
            logger.info(f"开始回测策略 {self.config.name}")
            
            # 运行策略获取信号
            strategy_result = self.run_strategy(ts_codes, start_date, end_date)
            signals = strategy_result['signals']
            
            if not signals:
                logger.warning("无交易信号，回测结束")
                return {'error': '无交易信号'}
            
            # 创建回测引擎
            backtest_engine = BacktestEngine(
                initial_capital=initial_capital,
                commission_rate=self.config.commission_rate,
                slippage_rate=self.config.slippage_rate
            )
            
            # 执行回测
            backtest_results = backtest_engine.run_backtest(signals)
            
            # 合并策略结果
            backtest_results.update({
                'strategy_config': self.config.__dict__,
                'strategy_summary': strategy_result['summary']
            })
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"回测失败: {e}")
            raise StrategyError(f"回测失败: {e}")
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        return {
            'name': self.config.name,
            'version': self.config.version,
            'config': self.config.__dict__,
            'signals_count': len(self.signals_history),
            'positions_count': len(self.positions),
            'performance_metrics': self.performance_metrics
        }


# strategies/base/signal_generator.py
"""信号生成器"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from enum import Enum

from strategies.base.strategy_base import TradingSignal, SignalType
from utils.logger import get_logger

logger = get_logger(__name__)

class SignalCondition(Enum):
    """信号条件枚举"""
    CROSSOVER = "crossover"          # 上穿
    CROSSUNDER = "crossunder"        # 下穿
    ABOVE = "above"                  # 高于
    BELOW = "below"                  # 低于
    RISING = "rising"                # 上升
    FALLING = "falling"              # 下降
    DIVERGENCE = "divergence"        # 背离

class SignalGenerator:
    """信号生成器工具类"""
    
    @staticmethod
    def crossover_signals(series1: pd.Series, series2: pd.Series, 
                         min_confidence: float = 0.5) -> List[Tuple[int, SignalType, float]]:
        """生成交叉信号
        
        Args:
            series1: 快线序列
            series2: 慢线序列  
            min_confidence: 最小置信度
        
        Returns:
            List of (index, signal_type, confidence)
        """
        signals = []
        
        if len(series1) != len(series2) or len(series1) < 2:
            return signals
        
        # 计算交叉点
        cross_up = (series1 > series2) & (series1.shift(1) <= series2.shift(1))
        cross_down = (series1 < series2) & (series1.shift(1) >= series2.shift(1))
        
        for i in range(1, len(series1)):
            if cross_up.iloc[i]:
                # 上穿 - 买入信号
                confidence = SignalGenerator._calculate_crossover_confidence(
                    series1.iloc[i-5:i+1], series2.iloc[i-5:i+1], 'up'
                )
                if confidence >= min_confidence:
                    signals.append((i, SignalType.BUY, confidence))
            
            elif cross_down.iloc[i]:
                # 下穿 - 卖出信号
                confidence = SignalGenerator._calculate_crossover_confidence(
                    series1.iloc[i-5:i+1], series2.iloc[i-5:i+1], 'down'
                )
                if confidence >= min_confidence:
                    signals.append((i, SignalType.SELL, confidence))
        
        return signals
    
    @staticmethod
    def threshold_signals(series: pd.Series, upper_threshold: float, 
                         lower_threshold: float, min_confidence: float = 0.5) -> List[Tuple[int, SignalType, float]]:
        """生成阈值信号"""
        signals = []
        
        if len(series) < 2:
            return signals
        
        for i in range(1, len(series)):
            current_val = series.iloc[i]
            prev_val = series.iloc[i-1]
            
            # 突破上阈值 - 卖出信号（超买）
            if current_val > upper_threshold and prev_val <= upper_threshold:
                confidence = min(1.0, (current_val - upper_threshold) / (100 - upper_threshold))
                if confidence >= min_confidence:
                    signals.append((i, SignalType.SELL, confidence))
            
            # 突破下阈值 - 买入信号（超卖）
            elif current_val < lower_threshold and prev_val >= lower_threshold:
                confidence = min(1.0, (lower_threshold - current_val) / lower_threshold)
                if confidence >= min_confidence:
                    signals.append((i, SignalType.BUY, confidence))
        
        return signals
    
    @staticmethod
    def momentum_signals(price_series: pd.Series, lookback_period: int = 10,
                        threshold: float = 0.02, min_confidence: float = 0.5) -> List[Tuple[int, SignalType, float]]:
        """生成动量信号"""
        signals = []
        
        if len(price_series) < lookback_period + 1:
            return signals
        
        # 计算动量
        momentum = price_series.pct_change(lookback_period)
        
        for i in range(lookback_period, len(price_series)):
            mom_val = momentum.iloc[i]
            
            if abs(mom_val) < threshold:
                continue
            
            if mom_val > threshold:
                # 正动量 - 买入信号
                confidence = min(1.0, mom_val / (threshold * 3))
                if confidence >= min_confidence:
                    signals.append((i, SignalType.BUY, confidence))
            
            elif mom_val < -threshold:
                # 负动量 - 卖出信号
                confidence = min(1.0, abs(mom_val) / (threshold * 3))
                if confidence >= min_confidence:
                    signals.append((i, SignalType.SELL, confidence))
        
        return signals
    
    @staticmethod
    def volume_confirmation(price_signals: List[Tuple[int, SignalType, float]], 
                           volume_series: pd.Series, 
                           volume_threshold: float = 1.5) -> List[Tuple[int, SignalType, float]]:
        """成交量确认信号"""
        confirmed_signals = []
        
        if volume_series.empty:
            return price_signals
        
        # 计算成交量移动平均
        vol_ma = volume_series.rolling(window=20).mean()
        
        for idx, signal_type, confidence in price_signals:
            if idx >= len(volume_series) or idx >= len(vol_ma):
                continue
            
            current_vol = volume_series.iloc[idx]
            avg_vol = vol_ma.iloc[idx]
            
            if pd.isna(avg_vol) or avg_vol == 0:
                # 无法确认，保持原置信度
                confirmed_signals.append((idx, signal_type, confidence * 0.8))
                continue
            
            volume_ratio = current_vol / avg_vol
            
            if volume_ratio >= volume_threshold:
                # 成交量放大，提高置信度
                boosted_confidence = min(1.0, confidence * (1 + volume_ratio * 0.1))
                confirmed_signals.append((idx, signal_type, boosted_confidence))
            elif volume_ratio >= 0.8:
                # 成交量正常，保持置信度
                confirmed_signals.append((idx, signal_type, confidence))
            else:
                # 成交量萎缩，降低置信度
                reduced_confidence = confidence * volume_ratio
                if reduced_confidence >= 0.3:  # 最低置信度阈值
                    confirmed_signals.append((idx, signal_type, reduced_confidence))
        
        return confirmed_signals
    
    @staticmethod
    def trend_filter(signals: List[Tuple[int, SignalType, float]], 
                    trend_series: pd.Series, 
                    trend_period: int = 50) -> List[Tuple[int, SignalType, float]]:
        """趋势过滤器"""
        filtered_signals = []
        
        if trend_series.empty or len(trend_series) < trend_period:
            return signals
        
        # 计算长期趋势
        trend_ma = trend_series.rolling(window=trend_period).mean()
        
        for idx, signal_type, confidence in signals:
            if idx >= len(trend_series) or idx >= len(trend_ma):
                continue
            
            current_price = trend_series.iloc[idx]
            trend_level = trend_ma.iloc[idx]
            
            if pd.isna(trend_level):
                continue
            
            # 判断趋势方向
            is_uptrend = current_price > trend_level
            is_downtrend = current_price < trend_level
            
            # 只在趋势方向一致时保留信号
            if signal_type == SignalType.BUY and is_uptrend:
                filtered_signals.append((idx, signal_type, confidence))
            elif signal_type == SignalType.SELL and is_downtrend:
                filtered_signals.append((idx, signal_type, confidence))
            # 在趋势相反时，可以考虑降低置信度而不是完全过滤
            elif signal_type == SignalType.BUY and is_downtrend:
                # 逆趋势买入，大幅降低置信度
                contrarian_confidence = confidence * 0.3
                if contrarian_confidence >= 0.4:
                    filtered_signals.append((idx, signal_type, contrarian_confidence))
            elif signal_type == SignalType.SELL and is_uptrend:
                # 逆趋势卖出，大幅降低置信度
                contrarian_confidence = confidence * 0.3
                if contrarian_confidence >= 0.4:
                    filtered_signals.append((idx, signal_type, contrarian_confidence))
        
        return filtered_signals
    
    @staticmethod
    def _calculate_crossover_confidence(series1: pd.Series, series2: pd.Series, 
                                      direction: str) -> float:
        """计算交叉信号的置信度"""
        if len(series1) < 2 or len(series2) < 2:
            return 0.5
        
        # 计算序列的斜率
        x = np.arange(len(series1))
        slope1 = np.polyfit(x, series1.values, 1)[0]
        slope2 = np.polyfit(x, series2.values, 1)[0]
        
        # 计算分离度
        separation = abs(series1.iloc[-1] - series2.iloc[-1]) / series2.iloc[-1]
        
        if direction == 'up':
            # 上穿：快线上升，慢线平缓或下降
            slope_score = min(1.0, max(0.0, slope1 / (abs(slope1) + abs(slope2) + 1e-8)))
            trend_score = 1.0 if slope1 > slope2 else 0.5
        else:  # down
            # 下穿：快线下降，慢线平缓或上升
            slope_score = min(1.0, max(0.0, abs(slope1) / (abs(slope1) + abs(slope2) + 1e-8)))
            trend_score = 1.0 if slope1 < slope2 else 0.5
        
        # 分离度得分
        sep_score = min(1.0, separation * 10)
        
        # 综合置信度
        confidence = (slope_score * 0.4 + trend_score * 0.3 + sep_score * 0.3)
        
        return max(0.1, min(1.0, confidence))


# strategies/base/backtest_engine.py
"""回测引擎"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from strategies.base.strategy_base import TradingSignal, SignalType
from utils.logger import get_logger
from utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown

logger = get_logger(__name__)

@dataclass
class Trade:
    """交易记录"""
    ts_code: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    side: str = "long"  # long/short
    pnl: float = 0.0
    commission: float = 0.0
    
    @property
    def is_closed(self) -> bool:
        return self.exit_date is not None
    
    @property
    def return_pct(self) -> float:
        if not self.is_closed:
            return 0.0
        if self.side == "long":
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, initial_capital: float = 1000000,
                 commission_rate: float = 0.0003,
                 slippage_rate: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
        # 回测状态
        self.current_capital = initial_capital
        self.positions = {}  # ts_code -> quantity
        self.trades = []     # 交易记录
        self.daily_values = []  # 每日资产价值
        self.daily_returns = []  # 每日收益率
        
        from data.storage.database_manager import data_repository
        self.repository = data_repository
    
    def run_backtest(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """运行回测"""
        try:
            logger.info(f"开始回测，信号数量: {len(signals)}")
            
            if not signals:
                return {'error': '无交易信号'}
            
            # 按日期排序信号
            signals = sorted(signals, key=lambda x: x.trade_date)
            
            # 获取回测日期范围
            start_date = signals[0].trade_date
            end_date = signals[-1].trade_date
            
            logger.info(f"回测期间: {start_date} 到 {end_date}")
            
            # 按日期分组信号
            signals_by_date = {}
            for signal in signals:
                date_str = signal.trade_date.strftime('%Y-%m-%d')
                if date_str not in signals_by_date:
                    signals_by_date[date_str] = []
                signals_by_date[date_str].append(signal)
            
            # 获取所有涉及的股票代码
            all_ts_codes = list(set([signal.ts_code for signal in signals]))
            
            # 获取价格数据
            price_data = self._get_price_data(
                all_ts_codes, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if price_data.empty:
                return {'error': '无价格数据'}
            
            # 执行逐日回测
            trading_dates = sorted(signals_by_date.keys())
            
            for i, date in enumerate(trading_dates):
                date_signals = signals_by_date[date]
                
                # 处理当日信号
                self._process_daily_signals(date, date_signals, price_data)
                
                # 计算当日资产价值
                daily_value = self._calculate_portfolio_value(date, price_data)
                self.daily_values.append({
                    'date': date,
                    'total_value': daily_value,
                    'cash': self.current_capital,
                    'positions_value': daily_value - self.current_capital
                })
                
                # 计算日收益率
                if i > 0:
                    prev_value = self.daily_values[i-1]['total_value']
                    daily_return = (daily_value - prev_value) / prev_value
                    self.daily_returns.append(daily_return)
            
            # 计算绩效指标
            performance_metrics = self._calculate_performance_metrics()
            
            # 生成回测报告
            backtest_report = {
                'summary': {
                    'initial_capital': self.initial_capital,
                    'final_value': self.daily_values[-1]['total_value'] if self.daily_values else self.initial_capital,
                    'total_return': (self.daily_values[-1]['total_value'] - self.initial_capital) / self.initial_capital if self.daily_values else 0,
                    'total_trades': len([t for t in self.trades if t.is_closed]),
                    'winning_trades': len([t for t in self.trades if t.is_closed and t.pnl > 0]),
                    'losing_trades': len([t for t in self.trades if t.is_closed and t.pnl < 0]),
                },
                'performance_metrics': performance_metrics,
                'trades': [self._trade_to_dict(trade) for trade in self.trades if trade.is_closed],
                'daily_values': self.daily_values,
                'positions': self.positions
            }
            
            logger.info(f"回测完成，总收益率: {backtest_report['summary']['total_return']:.2%}")
            
            return backtest_report
            
        except Exception as e:
            logger.error(f"回测执行失败: {e}")
            return {'error': str(e)}
    
    def _get_price_data(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取价格数据"""
        all_data = []
        
        for ts_code in ts_codes:
            try:
                data = self.repository.get_daily_price(ts_code, start_date, end_date)
                if not data.empty:
                    all_data.append(data)
            except Exception as e:
                logger.warning(f"获取股票 {ts_code} 价格数据失败: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _process_daily_signals(self, date: str, signals: List[TradingSignal], price_data: pd.DataFrame):
        """处理当日交易信号"""
        date_price_data = price_data[price_data['trade_date'] == date]
        
        for signal in signals:
            try:
                # 获取当日价格
                stock_price_data = date_price_data[date_price_data['ts_code'] == signal.ts_code]
                
                if stock_price_data.empty:
                    logger.warning(f"股票 {signal.ts_code} 在 {date} 无价格数据")
                    continue
                
                current_price = stock_price_data.iloc[0]['close_price']
                
                # 考虑滑点
                if signal.signal_type == SignalType.BUY:
                    execution_price = current_price * (1 + self.slippage_rate)
                else:
                    execution_price = current_price * (1 - self.slippage_rate)
                
                # 执行交易
                self._execute_trade(signal, execution_price, date)
                
            except Exception as e:
                logger.error(f"处理信号失败: {signal.ts_code} {date} {e}")
    
    def _execute_trade(self, signal: TradingSignal, price: float, date: str):
        """执行交易"""
        ts_code = signal.ts_code
        
        if signal.signal_type == SignalType.BUY:
            # 买入逻辑
            available_capital = self.current_capital * signal.position_size
            quantity = int(available_capital / (price * 100)) * 100  # 整手买入
            
            if quantity > 0:
                total_cost = quantity * price
                commission = total_cost * self.commission_rate
                
                if total_cost + commission <= self.current_capital:
                    # 执行买入
                    self.current_capital -= (total_cost + commission)
                    
                    if ts_code in self.positions:
                        self.positions[ts_code] += quantity
                    else:
                        self.positions[ts_code] = quantity
                    
                    # 记录交易
                    trade = Trade(
                        ts_code=ts_code,
                        entry_date=datetime.strptime(date, '%Y-%m-%d'),
                        entry_price=price,
                        quantity=quantity,
                        side="long",
                        commission=commission
                    )
                    self.trades.append(trade)
                    
                    logger.debug(f"买入 {ts_code}: {quantity}股 @ {price:.2f}")
        
        elif signal.signal_type == SignalType.SELL:
            # 卖出逻辑
            if ts_code in self.positions and self.positions[ts_code] > 0:
                # 计算卖出数量
                current_position = self.positions[ts_code]
                sell_quantity = int(current_position * signal.position_size)
                sell_quantity = min(sell_quantity, current_position)
                
                if sell_quantity > 0:
                    total_proceeds = sell_quantity * price
                    commission = total_proceeds * self.commission_rate
                    
                    # 执行卖出
                    self.current_capital += (total_proceeds - commission)
                    self.positions[ts_code] -= sell_quantity
                    
                    if self.positions[ts_code] == 0:
                        del self.positions[ts_code]
                    
                    # 更新对应的买入交易记录
                    self._close_trades(ts_code, sell_quantity, price, date, commission)
                    
                    logger.debug(f"卖出 {ts_code}: {sell_quantity}股 @ {price:.2f}")
    
    def _close_trades(self, ts_code: str, quantity: int, exit_price: float, 
                     exit_date: str, commission: float):
        """平仓交易"""
        remaining_quantity = quantity
        
        # FIFO (先进先出) 平仓
        for trade in self.trades:
            if (trade.ts_code == ts_code and 
                trade.side == "long" and 
                not trade.is_closed and 
                remaining_quantity > 0):
                
                close_quantity = min(remaining_quantity, trade.quantity)
                
                # 计算盈亏
                pnl = (exit_price - trade.entry_price) * close_quantity - commission
                
                if close_quantity == trade.quantity:
                    # 完全平仓
                    trade.exit_date = datetime.strptime(exit_date, '%Y-%m-%d')
                    trade.exit_price = exit_price
                    trade.pnl = pnl
                    trade.commission += commission
                else:
                    # 部分平仓，创建新的已平仓交易记录
                    new_trade = Trade(
                        ts_code=ts_code,
                        entry_date=trade.entry_date,
                        entry_price=trade.entry_price,
                        exit_date=datetime.strptime(exit_date, '%Y-%m-%d'),
                        exit_price=exit_price,
                        quantity=close_quantity,
                        side="long",
                        pnl=pnl,
                        commission=commission
                    )
                    self.trades.append(new_trade)
                    
                    # 更新原交易记录
                    trade.quantity -= close_quantity
                
                remaining_quantity -= close_quantity
    
    def _calculate_portfolio_value(self, date: str, price_data: pd.DataFrame) -> float:
        """计算组合价值"""
        total_value = self.current_capital
        
        date_prices = price_data[price_data['trade_date'] == date]
        
        for ts_code, quantity in self.positions.items():
            stock_data = date_prices[date_prices['ts_code'] == ts_code]
            if not stock_data.empty:
                current_price = stock_data.iloc[0]['close_price']
                position_value = quantity * current_price
                total_value += position_value
        
        return total_value
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """计算绩效指标"""
        if not self.daily_values or not self.daily_returns:
            return {}
        
        # 提取数值序列
        values = [dv['total_value'] for dv in self.daily_values]
        returns = pd.Series(self.daily_returns)
        
        # 计算基础指标
        total_return = (values[-1] - values[0]) / values[0]
        
        # 年化收益率（假设252个交易日）
        trading_days = len(values)
        annualized_return = (1 + total_return) ** (252 / trading_days) - 1
        
        # 夏普比率
        sharpe_ratio = calculate_sharpe_ratio(returns)
        
        # 最大回撤
        cumulative_returns = pd.Series(values) / values[0]
        max_drawdown_info = calculate_max_drawdown(cumulative_returns)
        
        # 胜率
        closed_trades = [t for t in self.trades if t.is_closed]
        winning_trades = len([t for t in closed_trades if t.pnl > 0])
        total_trades = len(closed_trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # 平均盈亏
        if closed_trades:
            avg_profit = np.mean([t.pnl for t in closed_trades if t.pnl > 0]) if winning_trades > 0 else 0
            avg_loss = np.mean([t.pnl for t in closed_trades if t.pnl < 0]) if (total_trades - winning_trades) > 0 else 0
            profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
        else:
            avg_profit = avg_loss = profit_loss_ratio = 0
        
        # 波动率
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown_info['max_drawdown'],
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_loss_ratio': profit_loss_ratio,
            'trading_days': trading_days
        }
    
    def _trade_to_dict(self, trade: Trade) -> Dict[str, Any]:
        """交易记录转字典"""
        return {
            'ts_code': trade.ts_code,
            'entry_date': trade.entry_date.strftime('%Y-%m-%d'),
            'entry_price': trade.entry_price,
            'exit_date': trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else None,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'side': trade.side,
            'pnl': trade.pnl,
            'return_pct': trade.return_pct,
            'commission': trade.commission
        }


# strategies/momentum/ma_crossover.py
"""均线交叉策略"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

from strategies.base.strategy_base import StrategyBase, StrategyConfig, TradingSignal, SignalType
from strategies.base.signal_generator import SignalGenerator
from utils.logger import get_logger

logger = get_logger(__name__)

class MACrossoverStrategy(StrategyBase):
    """均线交叉策略"""
    
    def __init__(self, short_window: int = 5, long_window: int = 20, 
                 volume_confirmation: bool = True, trend_filter: bool = True):
        
        # 创建策略配置
        config = StrategyConfig(
            name="MA_Crossover",
            version="1.0.0",
            lookback_period=max(long_window * 2, 60),
            max_position_size=0.1,
            stop_loss_pct=0.08,
            take_profit_pct=0.15
        )
        
        super().__init__(config)
        
        # 策略参数
        self.short_window = short_window
        self.long_window = long_window
        self.volume_confirmation = volume_confirmation
        self.trend_filter = trend_filter
        
        logger.info(f"初始化均线交叉策略: MA({short_window}, {long_window})")
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """生成均线交叉信号"""
        signals = []
        
        try:
            if data.empty or len(data) < self.long_window:
                logger.warning("数据不足，无法生成信号")
                return signals
            
            # 确保有必要的列
            required_columns = ['ts_code', 'trade_date', 'close_price']
            if not all(col in data.columns for col in required_columns):
                logger.error(f"数据缺少必要列: {required_columns}")
                return signals
            
            # 计算移动平均线
            data = data.copy()
            data[f'ma_{self.short_window}'] = data['close_price'].rolling(window=self.short_window).mean()
            data[f'ma_{self.long_window}'] = data['close_price'].rolling(window=self.long_window).mean()
            
            # 生成基础交叉信号
            short_ma = data[f'ma_{self.short_window}']
            long_ma = data[f'ma_{self.long_window}']
            
            crossover_signals = SignalGenerator.crossover_signals(short_ma, long_ma, min_confidence=0.4)
            
            # 成交量确认
            if self.volume_confirmation and 'vol' in data.columns:
                crossover_signals = SignalGenerator.volume_confirmation(
                    crossover_signals, data['vol'], volume_threshold=1.3
                )
            
            # 趋势过滤
            if self.trend_filter:
                trend_period = min(50, len(data) // 2)
                if trend_period >= 10:
                    crossover_signals = SignalGenerator.trend_filter(
                        crossover_signals, data['close_price'], trend_period=trend_period
                    )
            
            # 转换为TradingSignal对象
            for idx, signal_type, confidence in crossover_signals:
                if idx >= len(data):
                    continue
                
                row = data.iloc[idx]
                
                # 计算止损止盈位
                current_price = row['close_price']
                
                if signal_type == SignalType.BUY:
                    stop_loss = current_price * (1 - self.config.stop_loss_pct)
                    take_profit = current_price * (1 + self.config.take_profit_pct)
                    reason = f"MA({self.short_window})上穿MA({self.long_window})"
                else:
                    stop_loss = current_price * (1 + self.config.stop_loss_pct)
                    take_profit = current_price * (1 - self.config.take_profit_pct)
                    reason = f"MA({self.short_window})下穿MA({self.long_window})"
                
                # 根据距离均线的位置调整仓位大小
                ma_distance = abs(current_price - long_ma.iloc[idx]) / long_ma.iloc[idx]
                position_size = min(self.config.max_position_size, 
                                  self.config.max_position_size * (1 - ma_distance * 2))
                
                signal = TradingSignal(
                    ts_code=row['ts_code'],
                    trade_date=row['trade_date'],
                    signal_type=signal_type,
                    price=current_price,
                    confidence=confidence,
                    reason=reason,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=max(0.02, position_size)  # 最小2%仓位
                )
                
                signals.append(signal)
            
            logger.debug(f"生成 {len(signals)} 个均线交叉信号")
            return signals
            
        except Exception as e:
            logger.error(f"生成均线交叉信号失败: {e}")
            return []
    
    def get_strategy_description(self) -> str:
        """获取策略描述"""
        return f"""
        均线交叉策略 MA({self.short_window}, {self.long_window})
        
        策略逻辑:
        - 当短期均线({self.short_window}日)上穿长期均线({self.long_window}日)时买入
        - 当短期均线下穿长期均线时卖出
        - 成交量确认: {'开启' if self.volume_confirmation else '关闭'}
        - 趋势过滤: {'开启' if self.trend_filter else '关闭'}
        
        风控参数:
        - 止损: {self.config.stop_loss_pct:.1%}
        - 止盈: {self.config.take_profit_pct:.1%}
        - 最大单股仓位: {self.config.max_position_size:.1%}
        """


# strategies/momentum/macd_strategy.py
"""MACD策略"""
import pandas as pd
import numpy as np
from typing import List

from strategies.base.strategy_base import StrategyBase, StrategyConfig, TradingSignal, SignalType
from strategies.base.signal_generator import SignalGenerator
from utils.logger import get_logger

logger = get_logger(__name__)

class MACDStrategy(StrategyBase):
    """MACD策略"""
    
    def __init__(self, divergence_detection: bool = True, histogram_confirmation: bool = True):
        
        config = StrategyConfig(
            name="MACD_Strategy",
            version="1.0.0",
            lookback_period=60,
            max_position_size=0.12,
            stop_loss_pct=0.06,
            take_profit_pct=0.18
        )
        
        super().__init__(config)
        
        self.divergence_detection = divergence_detection
        self.histogram_confirmation = histogram_confirmation
        
        logger.info("初始化MACD策略")
    
    def generate_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """生成MACD信号"""
        signals = []
        
        try:
            if data.empty or len(data) < 34:  # MACD需要至少34个数据点
                return signals
            
            # 检查是否有MACD指标
            macd_cols = ['macd_dif', 'macd_dea', 'macd_histogram']
            if not all(col in data.columns for col in macd_cols):
                logger.warning("数据中缺少MACD指标")
                return signals
            
            data = data.copy()
            
            # 1. MACD线交叉信号
            dif_dea_signals = SignalGenerator.crossover_signals(
                data['macd_dif'], data['macd_dea'], min_confidence=0.5
            )
            
            # 2. 零轴交叉信号
            zero_line = pd.Series([0] * len(data))
            zero_cross_signals = SignalGenerator.crossover_signals(
                data['macd_dif'], zero_line, min_confidence=0.4
            )
            
            # 3. 柱状图确认
            if self.histogram_confirmation:
                # 柱状图转折信号
                histogram = data['macd_histogram']
                histogram_signals = self._detect_histogram_signals(histogram)
            else:
                histogram_signals = []
            
            # 4. 背离检测
            if self.divergence_detection:
                divergence_signals = self._detect_divergence(data)
            else:
                divergence_signals = []
            
            # 合并所有信号
            all_signals = dif_dea_signals + zero_cross_signals + histogram_signals + divergence_signals
            
            # 转换为TradingSignal对象
            for idx, signal_type, confidence in all_signals:
                if idx >= len(data):
                    continue
                
                row = data.iloc[idx]
                current_price = row['close_price']
                
                # 确定信号原因
                reason = self._determine_signal_reason(row, signal_type)
                
                # 根据MACD强度调整仓位
                macd_strength = abs(row['macd_dif'] - row['macd_dea'])
                position_size = min(self.config.max_position_size,
                                  self.config.max_position_size * (0.5 + macd_strength * 10))
                
                signal = TradingSignal(
                    ts_code=row['ts_code'],
                    trade_date=row['trade_date'],
                    signal_type=signal_type,
                    price=current_price,
                    confidence=confidence,
                    reason=reason,
                    position_size=max(0.03, position_size)
                )
                
                signals.append(signal)
            
            logger.debug(f"生成 {len(signals)} 个MACD信号")
            return signals
            
        except Exception as e:
            logger.error(f"生成MACD信号失败: {e}")
            return []
    
    def _detect_histogram_signals(self, histogram: pd.Series) -> List[tuple]:
        """检测柱状图信号"""
        signals = []
        
        for i in range(2, len(histogram)):
            if pd.isna(histogram.iloc[i]) or pd.isna(histogram.iloc[i-1]):
                continue
            
            current = histogram.iloc[i]
            prev = histogram.iloc[i-1]
            prev2 = histogram.iloc[i-2]
            
            # 柱状图由负转正且递增
            if current > 0 and prev <= 0 and current > prev:
                confidence = min(1.0, abs(current - prev) * 50)
                signals.append((i, SignalType.BUY, confidence))
            
            # 柱状图由正转负且递减
            elif current < 0 and prev >= 0 and current < prev:
                confidence = min(1.0, abs(current - prev) * 50)
                signals.append((i, SignalType.SELL, confidence))
        
        return signals
    
    def _detect_divergence(self, data: pd.DataFrame) -> List[tuple]:
        """检测背离信号"""
        signals = []
        
        if len(data) < 20:
            return signals
        
        price = data['close_price']
        macd_dif = data['macd_dif']
        
        # 寻找价格和MACD的高点和低点
        price_peaks = self._find_peaks(price.values)
        price_troughs = self._find_troughs(price.values)
        macd_peaks = self._find_peaks(macd_dif.values)
        macd_troughs = self._find_troughs(macd_dif.values)
        
        # 顶背离检测
        for i, peak_idx in enumerate(price_peaks[1:], 1):
            prev_peak_idx = price_peaks[i-1]
            
            # 找到对应时期的MACD峰值
            macd_peak_in_range = [p for p in macd_peaks 
                                if prev_peak_idx <= p <= peak_idx]
            
            if len(macd_peak_in_range) >= 2:
                # 价格创新高，MACD未创新高 -> 顶背离
                if (price.iloc[peak_idx] > price.iloc[prev_peak_idx] and
                    macd_dif.iloc[macd_peak_in_range[-1]] < macd_dif.iloc[macd_peak_in_range[-2]]):
                    
                    confidence = 0.7
                    signals.append((peak_idx, SignalType.SELL, confidence))
        
        # 底背离检测
        for i, trough_idx in enumerate(price_troughs[1:], 1):
            prev_trough_idx = price_troughs[i-1]
            
            # 找到对应时期的MACD谷值
            macd_trough_in_range = [t for t in macd_troughs 
                                  if prev_trough_idx <= t <= trough_idx]
            
            if len(macd_trough_in_range) >= 2:
                # 价格创新低，MACD未创新低 -> 底背离
                if (price.iloc[trough_idx] < price.iloc[prev_trough_idx] and
                    macd_dif.iloc[macd_trough_in_range[-1]] > macd_dif.iloc[macd_trough_in_range[-2]]):
                    
                    confidence = 0.7
                    signals.append((trough_idx, SignalType.BUY, confidence))
        
        return signals
    
    def _find_peaks(self, series: np.ndarray, min_distance: int = 5) -> List[int]:
        """寻找峰值"""
        peaks = []
        
        for i in range(min_distance, len(series) - min_distance):
            is_peak = True
            current_val = series[i]
            
            # 检查左右邻域
            for j in range(1, min_distance + 1):
                if current_val <= series[i-j] or current_val <= series[i+j]:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
        
        return peaks
    
    def _find_troughs(self, series: np.ndarray, min_distance: int = 5) -> List[int]:
        """寻找谷值"""
        troughs = []
        
        for i in range(min_distance, len(series) - min_distance):
            is_trough = True
            current_val = series[i]
            
            # 检查左右邻域
            for j in range(1, min_distance + 1):
                if current_val >= series[i-j] or current_val >= series[i+j]:
                    is_trough = False
                    break
            
            if is_trough:
                troughs.append(i)
        
        return troughs
    
    def _determine_signal_reason(self, row: pd.Series, signal_type: SignalType) -> str:
        """确定信号原因"""
        macd_dif = row['macd_dif']
        macd_dea = row['macd_dea']
        macd_hist = row['macd_histogram']
        
        reasons = []
        
        if signal_type == SignalType.BUY:
            if macd_dif > macd_dea:
                reasons.append("MACD金叉")
            if macd_dif > 0:
                reasons.append("MACD线上穿零轴")
            if macd_hist > 0:
                reasons.append("MACD柱线转正")
        else:
            if macd_dif < macd_dea:
                reasons.append("MACD死叉")
            if macd_dif < 0:
                reasons.append("MACD线下穿零轴")
            if macd_hist < 0:
                reasons.append("MACD柱线转负")
        
        return ", ".join(reasons) if reasons else "MACD信号"