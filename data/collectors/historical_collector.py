# data/collectors/historical_collector.py
"""历史数据收集器"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from data.providers.tushare_provider import tushare_provider
from data.storage.database_manager import db_manager, data_repository
from data.storage.data_models import StockBasic, DailyPrice, FinancialData, DataUpdateLog
from config.system_config import system_config
from utils.logger import get_logger
from utils.decorators import log_execution_time
from utils.exceptions import DataCollectionError

logger = get_logger(__name__)

class HistoricalDataCollector:
    """历史数据收集器"""
    
    def __init__(self):
        self.provider = tushare_provider
        self.db_manager = db_manager
        self.repository = data_repository
        self.max_workers = system_config.max_workers
        self.batch_size = system_config.batch_size
    
    @log_execution_time
    def collect_stock_basic_info(self, force_update: bool = False) -> bool:
        """收集股票基本信息
        
        Args:
            force_update: 强制更新，忽略现有数据
        """
        try:
            logger.info("开始收集股票基本信息")
            
            # 检查是否需要更新
            if not force_update:
                existing_count = self.db_manager.get_table_row_count('stock_basic')
                if existing_count > 0:
                    logger.info(f"股票基本信息已存在 {existing_count} 条记录，跳过收集")
                    return True
            
            # 获取数据
            stock_data = self.provider.get_stock_basic()
            if stock_data.empty:
                logger.warning("未获取到股票基本信息")
                return False
            
            # 数据清理和转换
            stock_data = self._clean_stock_basic_data(stock_data)
            
            # 保存到数据库
            with self.db_manager.get_session() as session:
                if force_update:
                    # 清空现有数据
                    session.query(StockBasic).delete()
                
                # 批量插入
                stock_data.to_sql('stock_basic', session.bind, if_exists='append', 
                                index=False, chunksize=1000)
            
            # 记录更新日志
            self._log_data_update('stock_basic', None, None, None, 
                                'success', len(stock_data))
            
            logger.info(f"成功收集股票基本信息 {len(stock_data)} 条")
            return True
            
        except Exception as e:
            logger.error(f"收集股票基本信息失败: {e}")
            self._log_data_update('stock_basic', None, None, None, 
                                'failed', 0, str(e))
            return False
    
    @log_execution_time
    def collect_daily_price_range(self, start_date: str, end_date: str,
                                 ts_codes: Optional[List[str]] = None,
                                 parallel: bool = True) -> bool:
        """收集指定日期范围的日线数据
        
        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            ts_codes: 股票代码列表，None表示所有股票
            parallel: 是否并行处理
        """
        try:
            logger.info(f"开始收集日线数据: {start_date} 到 {end_date}")
            
            # 获取股票代码列表
            if ts_codes is None:
                stock_basic = self.repository.get_stock_basic_info()
                if stock_basic.empty:
                    logger.error("未找到股票基本信息，请先收集股票基本信息")
                    return False
                ts_codes = stock_basic['ts_code'].tolist()
            
            logger.info(f"共需处理 {len(ts_codes)} 只股票")
            
            if parallel and len(ts_codes) > 10:
                # 并行处理
                return self._collect_daily_price_parallel(ts_codes, start_date, end_date)
            else:
                # 串行处理
                return self._collect_daily_price_sequential(ts_codes, start_date, end_date)
                
        except Exception as e:
            logger.error(f"收集日线数据失败: {e}")
            return False
    
    def _collect_daily_price_parallel(self, ts_codes: List[str], 
                                    start_date: str, end_date: str) -> bool:
        """并行收集日线数据"""
        success_count = 0
        failed_count = 0
        
        # 分批处理
        batch_size = 50  # 每批处理50只股票
        
        for i in range(0, len(ts_codes), batch_size):
            batch_codes = ts_codes[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(ts_codes)-1)//batch_size + 1}")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交任务
                future_to_code = {
                    executor.submit(self._collect_single_stock_daily, code, start_date, end_date): code
                    for code in batch_codes
                }
                
                # 处理结果
                for future in as_completed(future_to_code):
                    ts_code = future_to_code[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                        else:
                            failed_count += 1
                    except Exception as e:
                        logger.error(f"处理股票 {ts_code} 失败: {e}")
                        failed_count += 1
            
            # 批次间休息
            if i + batch_size < len(ts_codes):
                time.sleep(2)
        
        logger.info(f"并行收集完成: 成功 {success_count}, 失败 {failed_count}")
        return failed_count == 0
    
    def _collect_daily_price_sequential(self, ts_codes: List[str],
                                      start_date: str, end_date: str) -> bool:
        """串行收集日线数据"""
        success_count = 0
        failed_count = 0
        
        for i, ts_code in enumerate(ts_codes):
            logger.info(f"处理股票 {ts_code} ({i+1}/{len(ts_codes)})")
            
            try:
                success = self._collect_single_stock_daily(ts_code, start_date, end_date)
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    
                # 避免请求过于频繁
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"处理股票 {ts_code} 失败: {e}")
                failed_count += 1
        
        logger.info(f"串行收集完成: 成功 {success_count}, 失败 {failed_count}")
        return failed_count == 0
    
    def _collect_single_stock_daily(self, ts_code: str, start_date: str, end_date: str) -> bool:
        """收集单只股票的日线数据"""
        try:
            start_time = time.time()
            
            # 检查数据是否已存在
            existing_data = self.repository.get_daily_price(ts_code, start_date, end_date)
            if not existing_data.empty:
                # 计算需要补充的日期范围
                existing_dates = set(existing_data['trade_date'].dt.strftime('%Y-%m-%d'))
                all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
                all_dates_str = set(all_dates.strftime('%Y-%m-%d'))
                missing_dates = all_dates_str - existing_dates
                
                if not missing_dates:
                    logger.debug(f"股票 {ts_code} 数据已完整，跳过")
                    return True
            
            # 获取数据
            daily_data = self.provider.get_daily_price(ts_code, start_date, end_date)
            
            if daily_data.empty:
                logger.debug(f"股票 {ts_code} 无数据")
                return True
            
            # 数据清理
            daily_data = self._clean_daily_price_data(daily_data)
            
            # 保存数据
            success = self.repository.batch_upsert_daily_price(daily_data)
            
            duration = time.time() - start_time
            
            # 记录日志
            self._log_data_update('daily_price', ts_code, start_date, end_date,
                                'success' if success else 'failed', 
                                len(daily_data), None, duration)
            
            if success:
                logger.debug(f"成功收集股票 {ts_code} 日线数据 {len(daily_data)} 条")
            
            return success
            
        except Exception as e:
            logger.error(f"收集股票 {ts_code} 日线数据失败: {e}")
            self._log_data_update('daily_price', ts_code, start_date, end_date,
                                'failed', 0, str(e))
            return False
    
    @log_execution_time
    def collect_financial_data(self, ts_codes: Optional[List[str]] = None,
                             years: int = 5) -> bool:
        """收集财务数据
        
        Args:
            ts_codes: 股票代码列表
            years: 收集最近几年的数据
        """
        try:
            logger.info(f"开始收集财务数据，年限: {years} 年")
            
            # 获取股票代码列表
            if ts_codes is None:
                stock_basic = self.repository.get_stock_basic_info()
                if stock_basic.empty:
                    logger.error("未找到股票基本信息")
                    return False
                ts_codes = stock_basic['ts_code'].tolist()
            
            # 计算日期范围
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y%m%d')
            
            success_count = 0
            failed_count = 0
            
            for i, ts_code in enumerate(ts_codes):
                logger.info(f"收集股票 {ts_code} 财务数据 ({i+1}/{len(ts_codes)})")
                
                try:
                    start_time = time.time()
                    
                    # 获取财务数据
                    financial_data = self.provider.get_financial_data(
                        ts_code, start_date, end_date
                    )
                    
                    if not financial_data.empty:
                        # 保存数据
                        financial_data.to_sql('financial_data', self.db_manager._engine,
                                            if_exists='append', index=False, chunksize=100)
                        
                        success_count += 1
                        duration = time.time() - start_time
                        
                        self._log_data_update('financial_data', ts_code, start_date, end_date,
                                            'success', len(financial_data), None, duration)
                        
                        logger.debug(f"成功收集股票 {ts_code} 财务数据 {len(financial_data)} 条")
                    else:
                        logger.debug(f"股票 {ts_code} 无财务数据")
                    
                    # 避免请求过于频繁
                    time.sleep(0.3)
                    
                except Exception as e:
                    logger.error(f"收集股票 {ts_code} 财务数据失败: {e}")
                    failed_count += 1
                    self._log_data_update('financial_data', ts_code, start_date, end_date,
                                        'failed', 0, str(e))
            
            logger.info(f"财务数据收集完成: 成功 {success_count}, 失败 {failed_count}")
            return failed_count == 0
            
        except Exception as e:
            logger.error(f"收集财务数据失败: {e}")
            return False
    
    def collect_incremental_daily_data(self, days_back: int = 5) -> bool:
        """增量收集最近几天的日线数据
        
        Args:
            days_back: 回溯天数
        """
        try:
            logger.info(f"开始增量收集最近 {days_back} 天的日线数据")
            
            # 计算日期范围
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # 获取活跃股票列表（有最近交易记录的股票）
            latest_date = self.db_manager.get_latest_trade_date('daily_price')
            if latest_date:
                # 获取最近有交易的股票
                sql = """
                SELECT DISTINCT ts_code 
                FROM daily_price 
                WHERE trade_date >= :date 
                ORDER BY ts_code
                """
                result = self.db_manager.execute_sql(sql, {'date': latest_date})
                ts_codes = [row[0] for row in result]
            else:
                # 如果没有历史数据，获取所有股票
                stock_basic = self.repository.get_stock_basic_info()
                ts_codes = stock_basic['ts_code'].tolist()
            
            logger.info(f"共需更新 {len(ts_codes)} 只股票")
            
            # 收集数据
            return self.collect_daily_price_range(start_date, end_date, ts_codes, parallel=True)
            
        except Exception as e:
            logger.error(f"增量收集日线数据失败: {e}")
            return False
    
    def _clean_stock_basic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理股票基本信息数据"""
        if df.empty:
            return df
        
        # 移除异常数据
        df = df.dropna(subset=['ts_code', 'symbol', 'name'])
        
        # 标准化字段
        df['area'] = df['area'].fillna('未知')
        df['industry'] = df['industry'].fillna('未知')
        
        # 确保日期格式正确
        df['list_date'] = pd.to_datetime(df['list_date'], errors='coerce')
        df['delist_date'] = pd.to_datetime(df['delist_date'], errors='coerce')
        
        return df
    
    def _clean_daily_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理日线价格数据"""
        if df.empty:
            return df
        
        # 移除空值和异常值
        df = df.dropna(subset=['ts_code', 'trade_date'])
        df = df[df['close_price'] > 0]
        
        # 确保数据类型正确
        numeric_cols = ['open_price', 'high_price', 'low_price', 'close_price', 
                       'pre_close', 'change', 'pct_chg', 'amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
        
        # 移除价格逻辑错误的记录
        mask = (
            (df['high_price'] >= df[['open_price', 'close_price', 'low_price']].max(axis=1)) &
            (df['low_price'] <= df[['open_price', 'close_price', 'high_price']].min(axis=1))
        )
        df = df[mask]
        
        return df
    
    def _log_data_update(self, data_type: str, ts_code: Optional[str],
                        start_date: Optional[str], end_date: Optional[str],
                        status: str, records_count: int, 
                        error_message: Optional[str] = None,
                        duration_seconds: Optional[float] = None):
        """记录数据更新日志"""
        try:
            log_data = {
                'data_type': data_type,
                'ts_code': ts_code,
                'start_date': pd.to_datetime(start_date) if start_date else None,
                'end_date': pd.to_datetime(end_date) if end_date else None,
                'status': status,
                'records_count': records_count,
                'error_message': error_message,
                'duration_seconds': duration_seconds
            }
            
            log_df = pd.DataFrame([log_data])
            log_df.to_sql('data_update_log', self.db_manager._engine,
                         if_exists='append', index=False)
                         
        except Exception as e:
            logger.error(f"记录数据更新日志失败: {e}")

# 全局历史数据收集器实例
historical_collector = HistoricalDataCollector()


# data/collectors/realtime_collector.py
"""实时数据收集器"""
import pandas as pd
import asyncio
from datetime import datetime, time as dt_time
from typing import Optional, List, Dict, Any, Callable
import threading
import time
from queue import Queue
import schedule

from data.providers.tushare_provider import tushare_provider
from data.storage.database_manager import db_manager, data_repository
from data.collectors.historical_collector import historical_collector
from utils.logger import get_logger
from utils.exceptions import DataCollectionError

logger = get_logger(__name__)

class RealtimeDataCollector:
    """实时数据收集器"""
    
    def __init__(self):
        self.provider = tushare_provider
        self.db_manager = db_manager
        self.repository = data_repository
        self.historical_collector = historical_collector
        
        # 运行状态
        self.is_running = False
        self.collection_thread = None
        
        # 数据队列
        self.data_queue = Queue()
        
        # 订阅的股票列表
        self.subscribed_stocks = set()
        
        # 回调函数
        self.callbacks = {}
    
    def start_collection(self):
        """开始实时数据收集"""
        if self.is_running:
            logger.warning("实时数据收集器已在运行")
            return
        
        self.is_running = True
        
        # 启动数据收集线程
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        # 启动定时任务
        self._setup_scheduled_tasks()
        
        logger.info("实时数据收集器启动成功")
    
    def stop_collection(self):
        """停止实时数据收集"""
        self.is_running = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        logger.info("实时数据收集器已停止")
    
    def subscribe_stock(self, ts_code: str, callback: Optional[Callable] = None):
        """订阅股票实时数据
        
        Args:
            ts_code: 股票代码
            callback: 数据回调函数
        """
        self.subscribed_stocks.add(ts_code)
        
        if callback:
            self.callbacks[ts_code] = callback
        
        logger.info(f"订阅股票: {ts_code}")
    
    def unsubscribe_stock(self, ts_code: str):
        """取消订阅股票"""
        self.subscribed_stocks.discard(ts_code)
        self.callbacks.pop(ts_code, None)
        
        logger.info(f"取消订阅股票: {ts_code}")
    
    def _collection_loop(self):
        """数据收集主循环"""
        while self.is_running:
            try:
                current_time = datetime.now().time()
                
                # 检查是否在交易时间
                if self._is_trading_time(current_time):
                    self._collect_realtime_data()
                    time.sleep(1)  # 1秒更新一次
                else:
                    time.sleep(60)  # 非交易时间，1分钟检查一次
                    
            except Exception as e:
                logger.error(f"实时数据收集异常: {e}")
                time.sleep(5)
    
    def _is_trading_time(self, current_time: dt_time) -> bool:
        """检查是否在交易时间"""
        # A股交易时间: 9:30-11:30, 13:00-15:00
        morning_start = dt_time(9, 30)
        morning_end = dt_time(11, 30)
        afternoon_start = dt_time(13, 0)
        afternoon_end = dt_time(15, 0)
        
        is_morning = morning_start <= current_time <= morning_end
        is_afternoon = afternoon_start <= current_time <= afternoon_end
        
        return is_morning or is_afternoon
    
    def _collect_realtime_data(self):
        """收集实时数据"""
        if not self.subscribed_stocks:
            return
        
        try:
            # 获取实时价格数据
            current_date = datetime.now().strftime('%Y%m%d')
            
            for ts_code in self.subscribed_stocks:
                try:
                    # 获取当日分钟数据
                    minute_data = self.provider.get_minute_price(ts_code, current_date, '1min')
                    
                    if not minute_data.empty:
                        # 获取最新数据
                        latest_data = minute_data.iloc[-1]
                        
                        # 触发回调
                        if ts_code in self.callbacks:
                            self.callbacks[ts_code](latest_data)
                        
                        # 缓存最新价格
                        self._cache_latest_price(ts_code, latest_data)
                    
                except Exception as e:
                    logger.error(f"获取股票 {ts_code} 实时数据失败: {e}")
                    
            logger.debug(f"实时数据收集完成，股票数量: {len(self.subscribed_stocks)}")
            
        except Exception as e:
            logger.error(f"收集实时数据失败: {e}")
    
    def _cache_latest_price(self, ts_code: str, data: pd.Series):
        """缓存最新价格到Redis"""
        try:
            cache_data = {
                'ts_code': ts_code,
                'price': float(data['close_price']),
                'volume': int(data['vol']),
                'timestamp': data['trade_time'].isoformat(),
                'cached_at': datetime.now().isoformat()
            }
            
            cache_key = f"realtime_price:{ts_code}"
            self.repository.cache_manager.set_cache(cache_key, cache_data, 300)  # 5分钟过期
            
        except Exception as e:
            logger.error(f"缓存实时价格失败: {e}")
    
    def get_latest_price(self, ts_code: str) -> Optional[Dict]:
        """获取最新价格"""
        try:
            cache_key = f"realtime_price:{ts_code}"
            return self.repository.cache_manager.get_cache(cache_key)
        except Exception as e:
            logger.error(f"获取最新价格失败: {e}")
            return None
    
    def _setup_scheduled_tasks(self):
        """设置定时任务"""
        # 每日收盘后更新日线数据
        schedule.every().day.at("15:30").do(self._daily_data_update)
        
        # 每周末更新股票基本信息
        schedule.every().saturday.at("20:00").do(self._weekly_basic_info_update)
        
        # 启动定时任务线程
        schedule_thread = threading.Thread(target=self._run_schedule, daemon=True)
        schedule_thread.start()
    
    def _run_schedule(self):
        """运行定时任务"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)
    
    def _daily_data_update(self):
        """每日数据更新"""
        try:
            logger.info("开始每日数据更新")
            
            # 更新当日数据
            current_date = datetime.now().strftime('%Y-%m-%d')
            
            # 获取当日有交易的股票列表
            trade_stocks = self.provider.get_stock_list_by_date(current_date.replace('-', ''))
            
            if trade_stocks:
                success = self.historical_collector.collect_daily_price_range(
                    current_date, current_date, trade_stocks, parallel=True
                )
                
                if success:
                    logger.info(f"每日数据更新成功，股票数量: {len(trade_stocks)}")
                else:
                    logger.error("每日数据更新失败")
            else:
                logger.info("今日无交易数据")
                
        except Exception as e:
            logger.error(f"每日数据更新异常: {e}")
    
    def _weekly_basic_info_update(self):
        """每周基本信息更新"""
        try:
            logger.info("开始每周基本信息更新")
            success = self.historical_collector.collect_stock_basic_info(force_update=True)
            
            if success:
                logger.info("每周基本信息更新成功")
            else:
                logger.error("每周基本信息更新失败")
                
        except Exception as e:
            logger.error(f"每周基本信息更新异常: {e}")

# 全局实时数据收集器实例
realtime_collector = RealtimeDataCollector()