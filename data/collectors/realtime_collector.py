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
