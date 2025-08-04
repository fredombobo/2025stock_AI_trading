# data/providers/tushare_provider.py
"""Tushare数据提供商"""
import time
import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import threading
from queue import Queue
import traceback

from config.system_config import system_config
from utils.logger import get_logger
from utils.exceptions import DataProviderError, RateLimitError
from utils.decorators import retry, rate_limit

logger = get_logger(__name__)

class TushareProvider:
    """Tushare数据提供商"""
    
    def __init__(self):
        """初始化Tushare客户端"""
        if not system_config.tushare.token:
            raise ValueError("Tushare token未配置")
        
        # 设置token
        ts.set_token(system_config.tushare.token)
        self.pro = ts.pro_api()
        
        # 请求限制器
        self._request_times = Queue()
        self._lock = threading.Lock()
        
        logger.info("Tushare数据提供商初始化成功")
    
    def _check_rate_limit(self):
        """检查请求频率限制"""
        with self._lock:
            current_time = time.time()
            
            # 清理60秒前的请求记录
            while not self._request_times.empty():
                request_time = self._request_times.queue[0]
                if current_time - request_time > 60:
                    self._request_times.get()
                else:
                    break
            
            # 检查是否超过频率限制
            if self._request_times.qsize() >= system_config.tushare.max_requests_per_minute:
                wait_time = 60 - (current_time - self._request_times.queue[0])
                if wait_time > 0:
                    logger.warning(f"达到请求频率限制，等待 {wait_time:.2f} 秒")
                    time.sleep(wait_time + 1)
            
            # 记录当前请求时间
            self._request_times.put(current_time)
    
    @retry(max_attempts=3, delay=1, exceptions=(Exception,))
    @rate_limit(calls_per_minute=200)
    def _make_request(self, api_name: str, **kwargs) -> pd.DataFrame:
        """发起API请求的通用方法"""
        try:
            self._check_rate_limit()
            
            # 获取API方法
            api_method = getattr(self.pro, api_name)
            
            # 发起请求
            logger.debug(f"调用Tushare API: {api_name}, 参数: {kwargs}")
            result = api_method(**kwargs)
            
            if result is None or result.empty:
                logger.warning(f"API {api_name} 返回空数据")
                return pd.DataFrame()
            
            logger.debug(f"API {api_name} 返回 {len(result)} 条记录")
            return result
            
        except Exception as e:
            logger.error(f"Tushare API调用失败: {api_name}, 错误: {e}")
            logger.error(f"详细错误: {traceback.format_exc()}")
            raise DataProviderError(f"Tushare API调用失败: {e}")
    
    def get_stock_basic(self, exchange: Optional[str] = None, 
                       list_status: str = 'L') -> pd.DataFrame:
        """获取股票基本信息
        
        Args:
            exchange: 交易所代码 SSE上交所 SZSE深交所
            list_status: 上市状态 L上市 D退市 P暂停上市
        """
        try:
            params = {'list_status': list_status}
            if exchange:
                params['exchange'] = exchange
            
            df = self._make_request('stock_basic', **params)
            
            if not df.empty:
                # 数据类型转换
                df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
                df['delist_date'] = pd.to_datetime(df['delist_date'], errors='coerce')
                
                # 添加市场标识
                df['market'] = df['ts_code'].apply(
                    lambda x: 'SH' if x.endswith('.SH') else 'SZ' if x.endswith('.SZ') else 'BJ'
                )
            
            return df
            
        except Exception as e:
            logger.error(f"获取股票基本信息失败: {e}")
            raise DataProviderError(f"获取股票基本信息失败: {e}")
    
    def get_daily_price(self, ts_code: Optional[str] = None, 
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       trade_date: Optional[str] = None) -> pd.DataFrame:
        """获取日线行情数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD
            trade_date: 交易日期 YYYYMMDD
        """
        try:
            params = {}
            
            if ts_code:
                params['ts_code'] = ts_code
            if start_date:
                params['start_date'] = start_date.replace('-', '')
            if end_date:
                params['end_date'] = end_date.replace('-', '')
            if trade_date:
                params['trade_date'] = trade_date.replace('-', '')
            
            df = self._make_request('daily', **params)
            
            if not df.empty:
                # 数据清洗和转换
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.rename(columns={
                    'open': 'open_price',
                    'high': 'high_price', 
                    'low': 'low_price',
                    'close': 'close_price'
                })
                
                # 排序
                df = df.sort_values(['ts_code', 'trade_date'])
                
                # 数据验证
                df = self._validate_price_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"获取日线数据失败: {e}")
            raise DataProviderError(f"获取日线数据失败: {e}")
    
    def get_minute_price(self, ts_code: str, trade_date: str, 
                        freq: str = '1min') -> pd.DataFrame:
        """获取分钟级行情数据
        
        Args:
            ts_code: 股票代码
            trade_date: 交易日期 YYYYMMDD
            freq: 频率 1min, 5min, 15min, 30min, 60min
        """
        try:
            params = {
                'ts_code': ts_code,
                'trade_date': trade_date.replace('-', ''),
                'freq': freq
            }
            
            df = self._make_request('stk_mins', **params)
            
            if not df.empty:
                # 数据转换
                df['trade_time'] = pd.to_datetime(df['trade_time'])
                df = df.rename(columns={
                    'open': 'open_price',
                    'high': 'high_price',
                    'low': 'low_price', 
                    'close': 'close_price'
                })
                
                df = df.sort_values('trade_time')
            
            return df
            
        except Exception as e:
            logger.error(f"获取分钟数据失败: {e}")
            raise DataProviderError(f"获取分钟数据失败: {e}")
    
    def get_financial_data(self, ts_code: str, start_date: Optional[str] = None,
                          end_date: Optional[str] = None, 
                          report_type: Optional[str] = None) -> pd.DataFrame:
        """获取财务数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            report_type: 报告类型 1合并报表 2单季合并 3调整单季合并表 4调整合并报表 5调整前合并报表
        """
        try:
            params = {'ts_code': ts_code}
            
            if start_date:
                params['start_date'] = start_date.replace('-', '')
            if end_date:
                params['end_date'] = end_date.replace('-', '')
            if report_type:
                params['report_type'] = report_type
            
            # 获取利润表数据
            income_df = self._make_request('income', **params)
            
            # 获取资产负债表数据
            balance_df = self._make_request('balancesheet', **params)
            
            # 获取现金流量表数据
            cashflow_df = self._make_request('cashflow', **params)
            
            # 合并财务数据
            df = self._merge_financial_data(income_df, balance_df, cashflow_df)
            
            if not df.empty:
                # 日期转换
                df['ann_date'] = pd.to_datetime(df['ann_date'], errors='coerce')
                df['f_ann_date'] = pd.to_datetime(df['f_ann_date'], errors='coerce')
                df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"获取财务数据失败: {e}")
            raise DataProviderError(f"获取财务数据失败: {e}")
    
    def get_trade_calendar(self, start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          exchange: str = 'SSE') -> pd.DataFrame:
        """获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期  
            exchange: 交易所 SSE上交所 SZSE深交所
        """
        try:
            params = {'exchange': exchange}
            
            if start_date:
                params['start_date'] = start_date.replace('-', '')
            if end_date:
                params['end_date'] = end_date.replace('-', '')
            
            df = self._make_request('trade_cal', **params)
            
            if not df.empty:
                df['cal_date'] = pd.to_datetime(df['cal_date'])
                df['pretrade_date'] = pd.to_datetime(df['pretrade_date'], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            raise DataProviderError(f"获取交易日历失败: {e}")
    
    def get_index_daily(self, ts_code: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """获取指数日线数据
        
        Args:
            ts_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
        """
        try:
            params = {'ts_code': ts_code}
            
            if start_date:
                params['start_date'] = start_date.replace('-', '')
            if end_date:
                params['end_date'] = end_date.replace('-', '')
            
            df = self._make_request('index_daily', **params)
            
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df.rename(columns={
                    'open': 'open_price',
                    'high': 'high_price',
                    'low': 'low_price',
                    'close': 'close_price'
                })
            
            return df
            
        except Exception as e:
            logger.error(f"获取指数数据失败: {e}")
            raise DataProviderError(f"获取指数数据失败: {e}")
    
    def get_concept_detail(self, concept_code: str) -> pd.DataFrame:
        """获取概念股详情"""
        try:
            df = self._make_request('concept_detail', id=concept_code)
            return df
        except Exception as e:
            logger.error(f"获取概念股详情失败: {e}")
            raise DataProviderError(f"获取概念股详情失败: {e}")
    
    def get_daily_basic(self, ts_code: Optional[str] = None,
                       trade_date: Optional[str] = None) -> pd.DataFrame:
        """获取每日基本面数据"""
        try:
            params = {}
            if ts_code:
                params['ts_code'] = ts_code
            if trade_date:
                params['trade_date'] = trade_date.replace('-', '')
            
            df = self._make_request('daily_basic', **params)
            
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            return df
            
        except Exception as e:
            logger.error(f"获取每日基本面数据失败: {e}")
            raise DataProviderError(f"获取每日基本面数据失败: {e}")

    def get_top_list(self, start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     trade_date: Optional[str] = None) -> pd.DataFrame:
        """获取龙虎榜数据"""
        try:
            params: Dict[str, Any] = {}
            if trade_date:
                params['trade_date'] = trade_date.replace('-', '')
            if start_date:
                params['start_date'] = start_date.replace('-', '')
            if end_date:
                params['end_date'] = end_date.replace('-', '')

            df = self._make_request('top_list', **params)
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df
        except Exception as e:
            logger.error(f"获取龙虎榜数据失败: {e}")
            raise DataProviderError(f"获取龙虎榜数据失败: {e}")

    def get_block_trade(self, ts_code: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """获取大宗交易数据"""
        try:
            params: Dict[str, Any] = {}
            if ts_code:
                params['ts_code'] = ts_code
            if start_date:
                params['start_date'] = start_date.replace('-', '')
            if end_date:
                params['end_date'] = end_date.replace('-', '')

            df = self._make_request('block_trade', **params)
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df
        except Exception as e:
            logger.error(f"获取大宗交易数据失败: {e}")
            raise DataProviderError(f"获取大宗交易数据失败: {e}")

    def get_moneyflow(self, ts_code: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """获取资金流向数据"""
        try:
            params: Dict[str, Any] = {'ts_code': ts_code}
            if start_date:
                params['start_date'] = start_date.replace('-', '')
            if end_date:
                params['end_date'] = end_date.replace('-', '')

            df = self._make_request('moneyflow', **params)
            if not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            return df
        except Exception as e:
            logger.error(f"获取资金流向数据失败: {e}")
            raise DataProviderError(f"获取资金流向数据失败: {e}")
    
    def _validate_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证价格数据的有效性"""
        if df.empty:
            return df
        
        # 移除价格为空或为0的数据
        price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
        for col in price_columns:
            if col in df.columns:
                df = df[~(df[col].isna() | (df[col] <= 0))]
        
        # 检查价格逻辑关系
        if len(df) > 0:
            # high >= max(open, close, low)
            # low <= min(open, close, high)
            mask = (
                (df['high_price'] >= df[['open_price', 'close_price', 'low_price']].max(axis=1)) &
                (df['low_price'] <= df[['open_price', 'close_price', 'high_price']].min(axis=1))
            )
            invalid_count = (~mask).sum()
            if invalid_count > 0:
                logger.warning(f"发现 {invalid_count} 条价格数据不符合逻辑，已移除")
                df = df[mask]
        
        return df
    
    def _merge_financial_data(self, income_df: pd.DataFrame, 
                            balance_df: pd.DataFrame, 
                            cashflow_df: pd.DataFrame) -> pd.DataFrame:
        """合并财务数据"""
        if income_df.empty:
            return pd.DataFrame()
        
        # 以利润表为主表进行合并
        result = income_df.copy()
        
        # 合并资产负债表
        if not balance_df.empty:
            merge_cols = ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type']
            balance_cols = [col for col in balance_df.columns if col not in merge_cols]
            result = result.merge(
                balance_df[merge_cols + balance_cols], 
                on=merge_cols, 
                how='left',
                suffixes=('', '_balance')
            )
        
        # 合并现金流量表
        if not cashflow_df.empty:
            merge_cols = ['ts_code', 'ann_date', 'f_ann_date', 'end_date', 'report_type']
            cashflow_cols = [col for col in cashflow_df.columns if col not in merge_cols]
            result = result.merge(
                cashflow_df[merge_cols + cashflow_cols],
                on=merge_cols,
                how='left', 
                suffixes=('', '_cashflow')
            )
        
        return result
    
    def get_stock_list_by_date(self, trade_date: str) -> List[str]:
        """获取指定日期的股票列表"""
        try:
            # 获取当日有交易的股票
            df = self.get_daily_price(trade_date=trade_date)
            if not df.empty:
                return df['ts_code'].unique().tolist()
            return []
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def batch_get_daily_price(self, ts_codes: List[str], 
                            start_date: str, end_date: str,
                            batch_size: int = 100) -> pd.DataFrame:
        """批量获取日线数据"""
        all_data = []
        
        try:
            # 分批处理股票代码
            for i in range(0, len(ts_codes), batch_size):
                batch_codes = ts_codes[i:i + batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}, 股票数量: {len(batch_codes)}")
                
                for ts_code in batch_codes:
                    try:
                        df = self.get_daily_price(
                            ts_code=ts_code,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if not df.empty:
                            all_data.append(df)
                        
                        # 避免请求过于频繁
                        time.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"获取股票 {ts_code} 数据失败: {e}")
                        continue
                
                # 批次间休息
                if i + batch_size < len(ts_codes):
                    time.sleep(1)
            
            # 合并所有数据
            if all_data:
                result = pd.concat(all_data, ignore_index=True)
                logger.info(f"批量获取完成，共 {len(result)} 条记录")
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"批量获取日线数据失败: {e}")
            raise DataProviderError(f"批量获取日线数据失败: {e}")

# 全局Tushare提供商实例（可能因未配置token而为空）
try:
    tushare_provider = TushareProvider()
except Exception as e:
    tushare_provider = None
    logger.warning(f"无法初始化TushareProvider: {e}")
