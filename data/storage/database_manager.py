# data/storage/database_manager.py
"""数据库管理器"""
import logging
import redis
from contextlib import contextmanager
from typing import Optional, Generator, Any, Dict, List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
from datetime import datetime, timedelta
import json

from config.database_config import db_config
from data.storage.data_models import Base
from utils.logger import get_logger
from utils.exceptions import DatabaseError, ConnectionError

logger = get_logger(__name__)

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._redis_client = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """初始化数据库连接"""
        try:
            # 初始化PostgreSQL连接
            postgres_config = db_config.get_postgres_config()
            self._engine = create_engine(
                db_config.get_postgres_url(),
                poolclass=QueuePool,
                pool_size=postgres_config.pool_size,
                max_overflow=postgres_config.max_overflow,
                pool_timeout=postgres_config.pool_timeout,
                pool_recycle=postgres_config.pool_recycle,
                echo=postgres_config.echo,
                connect_args={
                    "options": "-c timezone=Asia/Shanghai"
                }
            )
            
            # 创建Session工厂
            self._session_factory = sessionmaker(bind=self._engine)
            
            # 初始化Redis连接
            redis_config = db_config.get_redis_config()
            self._redis_client = redis.ConnectionPool(
                host=redis_config.host,
                port=redis_config.port,
                db=redis_config.database,
                password=redis_config.password if redis_config.password else None,
                max_connections=redis_config.pool_size,
                decode_responses=True
            )

            logger.info("数据库连接初始化成功")

        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise ConnectionError(f"数据库连接失败: {e}")

class CacheManager:
    """缓存管理器"""
    
    def __init__(self, database_manager: DatabaseManager):
        self.db_manager = database_manager
        self.redis_client = database_manager.get_redis_client()
        self.default_expire = 3600  # 1小时默认过期时间
    
    def set_cache(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """设置缓存"""
        try:
            expire_time = expire or self.default_expire
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, default=str)
            elif isinstance(value, pd.DataFrame):
                value = value.to_json(orient='records', force_ascii=False)
            
            return self.redis_client.setex(key, expire_time, value)
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False
    
    def get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            
            # 尝试解析JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None
    
    def delete_cache(self, key: str) -> bool:
        """删除缓存"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"清除缓存模式失败: {e}")
            return 0
    
    def cache_dataframe(self, key: str, df: pd.DataFrame, expire: Optional[int] = None) -> bool:
        """缓存DataFrame"""
        try:
            data = {
                'data': df.to_dict('records'),
                'columns': df.columns.tolist(),
                'index': df.index.tolist(),
                'cached_at': datetime.now().isoformat()
            }
            return self.set_cache(key, data, expire)
        except Exception as e:
            logger.error(f"缓存DataFrame失败: {e}")
            return False
    
    def get_cached_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存的DataFrame"""
        try:
            cached_data = self.get_cache(key)
            if not cached_data or 'data' not in cached_data:
                return None
            
            df = pd.DataFrame(cached_data['data'])
            if 'columns' in cached_data:
                df.columns = cached_data['columns']
            
            return df
        except Exception as e:
            logger.error(f"获取缓存DataFrame失败: {e}")
            return None
    
    def is_cache_valid(self, key: str, max_age_hours: int = 24) -> bool:
        """检查缓存是否有效"""
        try:
            cached_data = self.get_cache(key)
            if not cached_data or 'cached_at' not in cached_data:
                return False
            
            cached_time = datetime.fromisoformat(cached_data['cached_at'])
            return datetime.now() - cached_time < timedelta(hours=max_age_hours)
        except Exception as e:
            logger.error(f"检查缓存有效性失败: {e}")
            return False

class DataRepository:
    """数据仓库基类"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.cache_manager = CacheManager(db_manager)
    
    def get_stock_basic_info(self, ts_code: Optional[str] = None) -> pd.DataFrame:
        """获取股票基本信息"""
        cache_key = f"stock_basic:{ts_code or 'all'}"
        
        # 尝试从缓存获取
        cached_df = self.cache_manager.get_cached_dataframe(cache_key)
        if cached_df is not None and self.cache_manager.is_cache_valid(cache_key, 24):
            logger.debug(f"从缓存获取股票基本信息: {ts_code or 'all'}")
            return cached_df
        
        # 从数据库查询
        sql = "SELECT * FROM stock_basic"
        params = {}
        
        if ts_code:
            sql += " WHERE ts_code = :ts_code"
            params['ts_code'] = ts_code
        
        sql += " ORDER BY ts_code"
        
        df = self.db_manager.query_to_dataframe(sql, params)
        
        # 缓存结果
        self.cache_manager.cache_dataframe(cache_key, df, 86400)  # 24小时
        
        return df
    
    def get_daily_price(self, ts_code: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """获取日线行情数据"""
        cache_key = f"daily_price:{ts_code}:{start_date or 'none'}:{end_date or 'none'}"
        
        # 尝试从缓存获取
        cached_df = self.cache_manager.get_cached_dataframe(cache_key)
        if cached_df is not None and self.cache_manager.is_cache_valid(cache_key, 1):
            logger.debug(f"从缓存获取日线数据: {ts_code}")
            return cached_df
        
        # 构建SQL查询
        sql = "SELECT * FROM daily_price WHERE ts_code = :ts_code"
        params = {'ts_code': ts_code}
        
        if start_date:
            sql += " AND trade_date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            sql += " AND trade_date <= :end_date"
            params['end_date'] = end_date
        
        sql += " ORDER BY trade_date"
        
        df = self.db_manager.query_to_dataframe(sql, params)
        
        # 缓存结果（1小时）
        self.cache_manager.cache_dataframe(cache_key, df, 3600)
        
        return df
    
    def get_latest_prices(self, ts_codes: List[str], limit: int = 1) -> pd.DataFrame:
        """获取最新价格数据"""
        if not ts_codes:
            return pd.DataFrame()
        
        placeholders = ','.join([':code' + str(i) for i in range(len(ts_codes))])
        params = {f'code{i}': code for i, code in enumerate(ts_codes)}
        
        sql = f"""
        SELECT DISTINCT ON (ts_code) ts_code, trade_date, close_price, pct_chg, vol, amount
        FROM daily_price 
        WHERE ts_code IN ({placeholders})
        ORDER BY ts_code, trade_date DESC
        """
        
        if limit > 1:
            # 如果需要多天数据，使用窗口函数
            sql = f"""
            SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
                FROM daily_price 
                WHERE ts_code IN ({placeholders})
            ) t WHERE rn <= :limit
            ORDER BY ts_code, trade_date DESC
            """
            params['limit'] = limit
        
        return self.db_manager.query_to_dataframe(sql, params)
    
    def get_technical_indicators(self, ts_code: str, start_date: Optional[str] = None,
                               end_date: Optional[str] = None) -> pd.DataFrame:
        """获取技术指标数据"""
        cache_key = f"tech_indicators:{ts_code}:{start_date or 'none'}:{end_date or 'none'}"
        
        # 尝试从缓存获取
        cached_df = self.cache_manager.get_cached_dataframe(cache_key)
        if cached_df is not None and self.cache_manager.is_cache_valid(cache_key, 1):
            return cached_df
        
        sql = "SELECT * FROM technical_indicators WHERE ts_code = :ts_code"
        params = {'ts_code': ts_code}
        
        if start_date:
            sql += " AND trade_date >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            sql += " AND trade_date <= :end_date"
            params['end_date'] = end_date
        
        sql += " ORDER BY trade_date"
        
        df = self.db_manager.query_to_dataframe(sql, params)
        
        # 缓存结果
        self.cache_manager.cache_dataframe(cache_key, df, 3600)
        
        return df
    
    def get_financial_data(self, ts_code: str, report_type: Optional[str] = None) -> pd.DataFrame:
        """获取财务数据"""
        cache_key = f"financial:{ts_code}:{report_type or 'all'}"
        
        # 尝试从缓存获取
        cached_df = self.cache_manager.get_cached_dataframe(cache_key)
        if cached_df is not None and self.cache_manager.is_cache_valid(cache_key, 24):
            return cached_df
        
        sql = "SELECT * FROM financial_data WHERE ts_code = :ts_code"
        params = {'ts_code': ts_code}
        
        if report_type:
            sql += " AND report_type = :report_type"
            params['report_type'] = report_type
        
        sql += " ORDER BY end_date DESC"
        
        df = self.db_manager.query_to_dataframe(sql, params)
        
        # 缓存结果（24小时）
        self.cache_manager.cache_dataframe(cache_key, df, 86400)
        
        return df
    
    def batch_upsert_daily_price(self, df: pd.DataFrame) -> bool:
        """批量更新插入日线数据"""
        try:
            if df.empty:
                return True
            
            # 使用PostgreSQL的ON CONFLICT功能进行upsert
            with self.db_manager.get_session() as session:
                # 先删除可能存在的重复数据
                unique_combinations = df[['ts_code', 'trade_date']].drop_duplicates()
                
                for _, row in unique_combinations.iterrows():
                    session.execute(
                        text("DELETE FROM daily_price WHERE ts_code = :ts_code AND trade_date = :trade_date"),
                        {'ts_code': row['ts_code'], 'trade_date': row['trade_date']}
                    )
                
                # 批量插入新数据
                df.to_sql('daily_price', session.bind, if_exists='append', index=False, chunksize=1000)
                
            # 清除相关缓存
            for ts_code in df['ts_code'].unique():
                self.cache_manager.clear_pattern(f"daily_price:{ts_code}:*")
            
            logger.info(f"成功批量更新 {len(df)} 条日线数据")
            return True
            
        except Exception as e:
            logger.error(f"批量更新日线数据失败: {e}")
            return False
    
    def get_data_update_status(self, data_type: str, ts_code: Optional[str] = None) -> pd.DataFrame:
        """获取数据更新状态"""
        sql = "SELECT * FROM data_update_log WHERE data_type = :data_type"
        params = {'data_type': data_type}
        
        if ts_code:
            sql += " AND ts_code = :ts_code"
            params['ts_code'] = ts_code
        
        sql += " ORDER BY created_at DESC LIMIT 100"
        
        return self.db_manager.query_to_dataframe(sql, params)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """获取数据库会话（上下文管理器）"""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"数据库操作失败: {e}")
            raise DatabaseError(f"数据库操作失败: {e}")
        except Exception as e:
            session.rollback()
            logger.error(f"未知错误: {e}")
            raise
        finally:
            session.close()
    
    def get_redis_client(self) -> redis.Redis:
        """获取Redis客户端"""
        return redis.Redis(connection_pool=self._redis_client)
    
    def create_tables(self):
        """创建所有表"""
        try:
            Base.metadata.create_all(self._engine)
            logger.info("数据表创建成功")
        except Exception as e:
            logger.error(f"数据表创建失败: {e}")
            raise DatabaseError(f"数据表创建失败: {e}")
    
    def drop_tables(self):
        """删除所有表"""
        try:
            Base.metadata.drop_all(self._engine)
            logger.info("数据表删除成功")
        except Exception as e:
            logger.error(f"数据表删除失败: {e}")
            raise DatabaseError(f"数据表删除失败: {e}")
    
    def execute_sql(self, sql: str, params: Optional[Dict] = None) -> Any:
        """执行SQL语句"""
        try:
            with self.get_session() as session:
                result = session.execute(text(sql), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"SQL执行失败: {sql}, error: {e}")
            raise DatabaseError(f"SQL执行失败: {e}")
    
    def bulk_insert_dataframe(self, df: pd.DataFrame, table_name: str, 
                            if_exists: str = 'append', chunksize: int = 10000):
        """批量插入DataFrame数据"""
        try:
            df.to_sql(
                table_name, 
                self._engine, 
                if_exists=if_exists,
                index=False,
                chunksize=chunksize,
                method='multi'
            )
            logger.info(f"成功插入 {len(df)} 条记录到表 {table_name}")
        except Exception as e:
            logger.error(f"批量插入失败: {e}")
            raise DatabaseError(f"批量插入失败: {e}")
    
    def query_to_dataframe(self, sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """查询结果转为DataFrame"""
        try:
            return pd.read_sql(sql, self._engine, params=params)
        except Exception as e:
            logger.error(f"查询转DataFrame失败: {e}")
            raise DatabaseError(f"查询失败: {e}")
    
    def get_table_row_count(self, table_name: str) -> int:
        """获取表的行数"""
        sql = f"SELECT COUNT(*) as count FROM {table_name}"
        result = self.execute_sql(sql)
        return result[0][0] if result else 0
    
    def get_latest_trade_date(self, table_name: str, ts_code: Optional[str] = None) -> Optional[datetime]:
        """获取最新交易日期"""
        try:
            sql = f"SELECT MAX(trade_date) as max_date FROM {table_name}"
            params = {}
            
            if ts_code:
                sql += " WHERE ts_code = :ts_code"
                params['ts_code'] = ts_code
            
            result = self.execute_sql(sql, params)
            return result[0][0] if result and result[0][0] else None
        except Exception as e:
            logger.error(f"获取最新交易日期失败: {e}")
            return None
    
    def check_data_exists(self, table_name: str, ts_code: str, trade_date: datetime) -> bool:
        """检查数据是否存在"""
        try:
            sql = f"SELECT 1 FROM {table_name} WHERE ts_code = :ts_code AND trade_date = :trade_date LIMIT 1"
            result = self.execute_sql(sql, {'ts_code': ts_code, 'trade_date': trade_date})
            return len(result) > 0
        except Exception as e:
            logger.error(f"检查数据存在性失败: {e}")
            return False


# 全局数据库管理器实例
db_manager = DatabaseManager()
data_repository = DataRepository(db_manager)

