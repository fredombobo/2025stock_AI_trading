# data/processors/technical_indicators.py
"""技术指标计算模块"""
import pandas as pd
import numpy as np
import talib
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import warnings

from data.storage.database_manager import data_repository
from utils.logger import get_logger
from utils.decorators import log_execution_time
from utils.exceptions import CalculationError

logger = get_logger(__name__)
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """技术指标计算器"""
    
    def __init__(self):
        self.repository = data_repository
    
    @log_execution_time
    def calculate_all_indicators(self, ts_code: str, start_date: Optional[str] = None,
                               end_date: Optional[str] = None, save_to_db: bool = True) -> pd.DataFrame:
        """计算所有技术指标
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            save_to_db: 是否保存到数据库
        """
        try:
            logger.info(f"开始计算股票 {ts_code} 的技术指标")
            
            # 获取价格数据
            price_data = self.repository.get_daily_price(ts_code, start_date, end_date)
            
            if price_data.empty:
                logger.warning(f"股票 {ts_code} 无价格数据")
                return pd.DataFrame()
            
            # 确保数据按日期排序
            price_data = price_data.sort_values('trade_date').reset_index(drop=True)
            
            # 计算各类指标
            indicators_data = pd.DataFrame({
                'ts_code': price_data['ts_code'],
                'trade_date': price_data['trade_date']
            })
            
            # 移动平均线
            ma_indicators = self._calculate_moving_averages(price_data)
            indicators_data = pd.concat([indicators_data, ma_indicators], axis=1)
            
            # MACD指标
            macd_indicators = self._calculate_macd(price_data)
            indicators_data = pd.concat([indicators_data, macd_indicators], axis=1)
            
            # RSI指标
            rsi_indicators = self._calculate_rsi(price_data)
            indicators_data = pd.concat([indicators_data, rsi_indicators], axis=1)
            
            # 布林带
            boll_indicators = self._calculate_bollinger_bands(price_data)
            indicators_data = pd.concat([indicators_data, boll_indicators], axis=1)
            
            # KDJ指标
            kdj_indicators = self._calculate_kdj(price_data)
            indicators_data = pd.concat([indicators_data, kdj_indicators], axis=1)
            
            # 威廉指标
            wr_indicators = self._calculate_williams_r(price_data)
            indicators_data = pd.concat([indicators_data, wr_indicators], axis=1)
            
            # 成交量指标
            vol_indicators = self._calculate_volume_indicators(price_data)
            indicators_data = pd.concat([indicators_data, vol_indicators], axis=1)
            
            # 动量指标
            momentum_indicators = self._calculate_momentum_indicators(price_data)
            indicators_data = pd.concat([indicators_data, momentum_indicators], axis=1)
            
            # 波动率指标
            volatility_indicators = self._calculate_volatility_indicators(price_data)
            indicators_data = pd.concat([indicators_data, volatility_indicators], axis=1)
            
            # 移除包含NaN的行
            indicators_data = indicators_data.dropna()
            
            if save_to_db and not indicators_data.empty:
                self._save_indicators_to_db(indicators_data)
            
            logger.info(f"成功计算股票 {ts_code} 的技术指标，共 {len(indicators_data)} 条记录")
            return indicators_data
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            raise CalculationError(f"计算技术指标失败: {e}")
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线"""
        close_prices = df['close_price'].values
        
        indicators = pd.DataFrame()
        
        # 简单移动平均线
        periods = [5, 10, 20, 30, 60, 120, 250]
        for period in periods:
            if len(close_prices) >= period:
                indicators[f'ma{period}'] = talib.SMA(close_prices, timeperiod=period)
        
        # 指数移动平均线
        ema_periods = [5, 10, 20, 30]
        for period in ema_periods:
            if len(close_prices) >= period:
                indicators[f'ema{period}'] = talib.EMA(close_prices, timeperiod=period)
        
        return indicators
    
    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        close_prices = df['close_price'].values
        
        if len(close_prices) < 34:  # MACD需要至少34个数据点
            return pd.DataFrame()
        
        # 计算MACD
        macd, macdsignal, macdhist = talib.MACD(close_prices, 
                                              fastperiod=12, 
                                              slowperiod=26, 
                                              signalperiod=9)
        
        return pd.DataFrame({
            'macd_dif': macd,
            'macd_dea': macdsignal, 
            'macd_histogram': macdhist
        })
    
    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算RSI指标"""
        close_prices = df['close_price'].values
        
        indicators = pd.DataFrame()
        
        # 不同周期的RSI
        periods = [6, 12, 24]
        for period in periods:
            if len(close_prices) >= period + 1:
                indicators[f'rsi{period}'] = talib.RSI(close_prices, timeperiod=period)
        
        return indicators
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带"""
        close_prices = df['close_price'].values
        
        if len(close_prices) < 20:
            return pd.DataFrame()
        
        # 布林带参数：20日均线，2倍标准差
        upper, middle, lower = talib.BBANDS(close_prices, 
                                          timeperiod=20, 
                                          nbdevup=2, 
                                          nbdevdn=2, 
                                          matype=0)
        
        return pd.DataFrame({
            'boll_upper': upper,
            'boll_mid': middle,
            'boll_lower': lower
        })
    
    def _calculate_kdj(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算KDJ指标"""
        high_prices = df['high_price'].values
        low_prices = df['low_price'].values
        close_prices = df['close_price'].values
        
        if len(close_prices) < 14:
            return pd.DataFrame()
        
        # 计算KDJ
        slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices,
                                 fastk_period=9,
                                 slowk_period=3,
                                 slowk_matype=0,
                                 slowd_period=3,
                                 slowd_matype=0)
        
        # J值 = 3*K - 2*D
        j_values = 3 * slowk - 2 * slowd
        
        return pd.DataFrame({
            'kdj_k': slowk,
            'kdj_d': slowd,
            'kdj_j': j_values
        })
    
    def _calculate_williams_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算威廉指标"""
        high_prices = df['high_price'].values
        low_prices = df['low_price'].values
        close_prices = df['close_price'].values
        
        indicators = pd.DataFrame()
        
        # 不同周期的威廉指标
        periods = [6, 10]
        for period in periods:
            if len(close_prices) >= period:
                wr = talib.WILLR(high_prices, low_prices, close_prices, timeperiod=period)
                indicators[f'wr{period}'] = wr
        
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量指标"""
        volumes = df['vol'].values
        close_prices = df['close_price'].values
        
        indicators = pd.DataFrame()
        
        # 成交量移动平均
        if len(volumes) >= 5:
            indicators['vol_ma5'] = talib.SMA(volumes.astype(float), timeperiod=5)
        if len(volumes) >= 10:
            indicators['vol_ma10'] = talib.SMA(volumes.astype(float), timeperiod=10)
        
        # 量比计算（当前成交量 / 5日平均成交量）
        if len(volumes) >= 5:
            vol_ma5 = talib.SMA(volumes.astype(float), timeperiod=5)
            indicators['vol_ratio'] = volumes / vol_ma5
        
        # OBV指标
        if len(volumes) >= 2:
            obv = talib.OBV(close_prices, volumes.astype(float))
            indicators['obv'] = obv
        
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算动量指标"""
        close_prices = df['close_price'].values
        high_prices = df['high_price'].values
        low_prices = df['low_price'].values
        
        indicators = pd.DataFrame()
        
        # 动量指标
        if len(close_prices) >= 10:
            momentum = talib.MOM(close_prices, timeperiod=10)
            indicators['momentum'] = momentum
        
        # 变化率
        if len(close_prices) >= 10:
            roc = talib.ROC(close_prices, timeperiod=10)
            indicators['roc'] = roc
        
        # CCI指标
        if len(close_prices) >= 14:
            cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['cci'] = cci
        
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率指标"""
        close_prices = df['close_price'].values
        high_prices = df['high_price'].values
        low_prices = df['low_price'].values
        
        indicators = pd.DataFrame()
        
        # 真实波动幅度均值
        if len(close_prices) >= 14:
            atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            indicators['atr'] = atr
        
        # 标准差
        if len(close_prices) >= 20:
            std = talib.STDDEV(close_prices, timeperiod=20, nbdev=1)
            indicators['std'] = std
        
        # 历史波动率（20日）
        if len(close_prices) >= 21:
            returns = np.log(close_prices[1:] / close_prices[:-1])
            rolling_std = pd.Series(returns).rolling(window=20).std()
            historical_vol = rolling_std * np.sqrt(252)  # 年化波动率
            indicators['historical_volatility'] = np.concatenate([[np.nan], historical_vol.values])
        
        return indicators
    
    def _save_indicators_to_db(self, indicators_data: pd.DataFrame):
        """保存技术指标到数据库"""
        try:
            # 删除已存在的数据
            ts_code = indicators_data['ts_code'].iloc[0]
            start_date = indicators_data['trade_date'].min().strftime('%Y-%m-%d')
            end_date = indicators_data['trade_date'].max().strftime('%Y-%m-%d')
            
            with self.repository.db_manager.get_session() as session:
                delete_sql = """
                DELETE FROM technical_indicators 
                WHERE ts_code = :ts_code 
                AND trade_date >= :start_date 
                AND trade_date <= :end_date
                """
                session.execute(delete_sql, {
                    'ts_code': ts_code,
                    'start_date': start_date,
                    'end_date': end_date
                })
            
            # 插入新数据
            indicators_data.to_sql('technical_indicators', 
                                 self.repository.db_manager._engine,
                                 if_exists='append', 
                                 index=False, 
                                 chunksize=1000)
            
            # 清除相关缓存
            self.repository.cache_manager.clear_pattern(f"tech_indicators:{ts_code}:*")
            
            logger.info(f"成功保存技术指标数据到数据库，记录数: {len(indicators_data)}")
            
        except Exception as e:
            logger.error(f"保存技术指标到数据库失败: {e}")
            raise
    
    def get_latest_indicators(self, ts_code: str, count: int = 1) -> pd.DataFrame:
        """获取最新的技术指标数据"""
        try:
            # 先尝试从数据库获取
            indicators = self.repository.get_technical_indicators(ts_code)
            
            if not indicators.empty and len(indicators) >= count:
                return indicators.tail(count)
            
            # 如果数据库中没有或数据不足，重新计算
            logger.info(f"重新计算股票 {ts_code} 的技术指标")
            self.calculate_all_indicators(ts_code, save_to_db=True)
            
            # 再次获取
            indicators = self.repository.get_technical_indicators(ts_code)
            return indicators.tail(count) if not indicators.empty else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"获取最新技术指标失败: {e}")
            return pd.DataFrame()
    
    def batch_calculate_indicators(self, ts_codes: List[str], 
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Dict[str, bool]:
        """批量计算技术指标"""
        results = {}
        
        for i, ts_code in enumerate(ts_codes):
            logger.info(f"计算股票 {ts_code} 技术指标 ({i+1}/{len(ts_codes)})")
            
            try:
                self.calculate_all_indicators(ts_code, start_date, end_date, save_to_db=True)
                results[ts_code] = True
                
            except Exception as e:
                logger.error(f"计算股票 {ts_code} 技术指标失败: {e}")
                results[ts_code] = False
        
        success_count = sum(results.values())
        logger.info(f"批量计算完成: 成功 {success_count}/{len(ts_codes)}")
        
        return results
    
    def get_signal_analysis(self, ts_code: str) -> Dict[str, Any]:
        """获取技术指标信号分析"""
        try:
            # 获取最新的技术指标
            indicators = self.get_latest_indicators(ts_code, count=5)
            
            if indicators.empty:
                return {'error': '无技术指标数据'}
            
            latest = indicators.iloc[-1]
            signals = {}
            
            # MACD信号
            if not pd.isna(latest.get('macd_dif')) and not pd.isna(latest.get('macd_dea')):
                if latest['macd_dif'] > latest['macd_dea']:
                    signals['macd'] = 'bullish'
                else:
                    signals['macd'] = 'bearish'
            
            # RSI信号
            if not pd.isna(latest.get('rsi12')):
                rsi = latest['rsi12']
                if rsi > 70:
                    signals['rsi'] = 'overbought'
                elif rsi < 30:
                    signals['rsi'] = 'oversold'
                else:
                    signals['rsi'] = 'neutral'
            
            # 布林带信号
            if all(not pd.isna(latest.get(col)) for col in ['boll_upper', 'boll_lower']):
                close_price = self.repository.get_latest_prices([ts_code])
                if not close_price.empty:
                    current_price = close_price.iloc[0]['close_price']
                    if current_price > latest['boll_upper']:
                        signals['bollinger'] = 'overbought'
                    elif current_price < latest['boll_lower']:
                        signals['bollinger'] = 'oversold'
                    else:
                        signals['bollinger'] = 'neutral'
            
            # KDJ信号
            if not pd.isna(latest.get('kdj_k')) and not pd.isna(latest.get('kdj_d')):
                if latest['kdj_k'] > latest['kdj_d'] and latest['kdj_k'] < 80:
                    signals['kdj'] = 'bullish'
                elif latest['kdj_k'] < latest['kdj_d'] and latest['kdj_k'] > 20:
                    signals['kdj'] = 'bearish'
                else:
                    signals['kdj'] = 'neutral'
            
            return {
                'ts_code': ts_code,
                'signals': signals,
                'latest_data': latest.to_dict(),
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"技术指标信号分析失败: {e}")
            return {'error': str(e)}
    
    def get_trend_analysis(self, ts_code: str, period: int = 20) -> Dict[str, Any]:
        """获取趋势分析"""
        try:
            # 获取历史数据和技术指标
            price_data = self.repository.get_daily_price(ts_code)
            indicators = self.repository.get_technical_indicators(ts_code)
            
            if price_data.empty or indicators.empty:
                return {'error': '数据不足'}
            
            # 取最近period天的数据
            recent_prices = price_data.tail(period)
            recent_indicators = indicators.tail(period)
            
            # 价格趋势分析
            price_trend = self._analyze_price_trend(recent_prices)
            
            # 均线趋势分析
            ma_trend = self._analyze_ma_trend(recent_indicators)
            
            # 成交量趋势分析
            volume_trend = self._analyze_volume_trend(recent_prices)
            
            return {
                'ts_code': ts_code,
                'period': period,
                'price_trend': price_trend,
                'ma_trend': ma_trend,
                'volume_trend': volume_trend,
                'analysis_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"趋势分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_price_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析价格趋势"""
        if len(df) < 2:
            return {'trend': 'unknown'}
        
        prices = df['close_price'].values
        
        # 计算线性回归斜率
        x = np.arange(len(prices))
        slope = np.polyfit(x, prices, 1)[0]
        
        # 计算价格变化率
        price_change = (prices[-1] - prices[0]) / prices[0] * 100
        
        # 判断趋势
        if slope > 0 and price_change > 2:
            trend = 'uptrend'
        elif slope < 0 and price_change < -2:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        return {
            'trend': trend,
            'slope': slope,
            'price_change_pct': price_change,
            'volatility': np.std(prices) / np.mean(prices) * 100
        }
    
    def _analyze_ma_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析均线趋势"""
        if df.empty:
            return {'trend': 'unknown'}
        
        latest = df.iloc[-1]
        
        # 检查多头排列
        mas = []
        for period in [5, 10, 20, 30]:
            ma_col = f'ma{period}'
            if ma_col in df.columns and not pd.isna(latest.get(ma_col)):
                mas.append(latest[ma_col])
        
        if len(mas) >= 3:
            # 检查是否多头排列（短期均线在上）
            is_bullish = all(mas[i] > mas[i+1] for i in range(len(mas)-1))
            is_bearish = all(mas[i] < mas[i+1] for i in range(len(mas)-1))
            
            if is_bullish:
                trend = 'bullish_alignment'
            elif is_bearish:
                trend = 'bearish_alignment'
            else:
                trend = 'mixed'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'ma_values': dict(zip([f'ma{p}' for p in [5, 10, 20, 30]], mas))
        }
    
    def _analyze_volume_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析成交量趋势"""
        if len(df) < 5:
            return {'trend': 'unknown'}
        
        volumes = df['vol'].values
        recent_avg = np.mean(volumes[-5:])
        historical_avg = np.mean(volumes[:-5]) if len(volumes) > 5 else recent_avg
        
        volume_ratio = recent_avg / historical_avg if historical_avg > 0 else 1
        
        if volume_ratio > 1.5:
            trend = 'increasing'
        elif volume_ratio < 0.7:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'volume_ratio': volume_ratio,
            'recent_avg': recent_avg,
            'historical_avg': historical_avg
        }

# 全局技术指标计算器实例
technical_indicators = TechnicalIndicators()# data/processors/technical_indicators.py