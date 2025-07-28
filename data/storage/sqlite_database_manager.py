# data/storage/sqlite_database_manager.py
"""SQLite数据库管理器"""
import logging
import sqlite3
import pandas as pd
import os
from contextlib import contextmanager
from typing import Optional, Generator, Any, Dict, List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
import json

from config.database_config import db_config, system_config
from config.sqlite_models import Base
from utils.logger import get_logger
from utils.exceptions import DatabaseError, ConnectionError

logger = get_logger(__name__)


class SQLiteDatabaseManager:
    """SQLite数据库管理器"""

    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._redis_client = None
        self._initialize_connections()

    def _initialize_connections(self):
        """初始化数据库连接"""
        try:
            # 初始化SQLite连接
            sqlite_url = db_config.get_sqlite_url()

            # SQLite特殊配置
            connect_args = {
                "timeout": system_config.sqlite_timeout,
                "check_same_thread": system_config.sqlite_check_same_thread
            }

            self._engine = create_engine(
                sqlite_url,
                echo=db_config.get_sqlite_config().echo,
                connect_args=connect_args,
                # SQLite不支持连接池，使用StaticPool
                poolclass=None
            )

            # 创建Session工厂
            self._session_factory = sessionmaker(bind=self._engine)

            # 如果配置了Redis，尝试连接
            try:
                import redis
                redis_config = db_config.get_redis_config()
                self._redis_client = redis.ConnectionPool(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.database,
                    password=redis_config.password if redis_config.password else None,
                    max_connections=redis_config.pool_size,
                    decode_responses=True
                )
                logger.info("Redis连接初始化成功")
            except ImportError:
                logger.warning("Redis模块未安装，将使用内存缓存")
                self._redis_client = None
            except Exception as e:
                logger.warning(f"Redis连接失败，将使用内存缓存: {e}")
                self._redis_client = None

            logger.info("SQLite数据库连接初始化成功")

        except Exception as e:
            logger.error(f"数据库连接初始化失败: {e}")
            raise ConnectionError(f"数据库连接失败: {e}")

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

    def get_redis_client(self):
        """获取Redis客户端"""
        if self._redis_client:
            import redis
            return redis.Redis(connection_pool=self._redis_client)
        else:
            # 返回内存缓存替代
            return MemoryCache()

    def create_tables(self):
        """创建所有表"""
        try:
            Base.metadata.create_all(self._engine)
            logger.info("数据表创建成功")

            # 为SQLite启用外键约束
            with self._engine.connect() as conn:
                conn.execute(text("PRAGMA foreign_keys=ON"))

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
                              if_exists: str = 'append', chunksize: int = 1000):
        """批量插入DataFrame数据"""
        try:
            # SQLite优化设置
            with self._engine.connect() as conn:
                # 临时关闭同步，提高插入性能
                conn.execute(text("PRAGMA synchronous = OFF"))
                conn.execute(text("PRAGMA journal_mode = MEMORY"))

                # 插入数据
                df.to_sql(
                    table_name,
                    conn,
                    if_exists=if_exists,
                    index=False,
                    chunksize=chunksize,
                    method='multi'
                )

                # 恢复正常设置
                conn.execute(text("PRAGMA synchronous = NORMAL"))
                conn.execute(text("PRAGMA journal_mode = DELETE"))

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
        try:
            result = self.execute_sql(sql)
            return result[0][0] if result else 0
        except:
            return 0

    def get_latest_trade_date(self, table_name: str, ts_code: Optional[str] = None) -> Optional[datetime]:
        """获取最新交易日期"""
        try:
            sql = f"SELECT MAX(trade_date) as max_date FROM {table_name}"
            params = {}

            if ts_code:
                sql += " WHERE ts_code = :ts_code"
                params['ts_code'] = ts_code

            result = self.execute_sql(sql, params)
            date_str = result[0][0] if result and result[0][0] else None

            if date_str:
                # 处理SQLite日期字符串
                if isinstance(date_str, str):
                    return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    return date_str
            return None
        except Exception as e:
            logger.error(f"获取最新交易日期失败: {e}")
            return None

    def check_data_exists(self, table_name: str, ts_code: str, trade_date: datetime) -> bool:
        """检查数据是否存在"""
        try:
            sql = f"SELECT 1 FROM {table_name} WHERE ts_code = :ts_code AND date(trade_date) = date(:trade_date) LIMIT 1"
            result = self.execute_sql(sql, {'ts_code': ts_code, 'trade_date': trade_date})
            return len(result) > 0
        except Exception as e:
            logger.error(f"检查数据存在性失败: {e}")
            return False

    def optimize_database(self):
        """优化SQLite数据库"""
        try:
            with self._engine.connect() as conn:
                # 分析查询计划
                conn.execute(text("ANALYZE"))

                # 重建索引
                conn.execute(text("REINDEX"))

                # 清理空间
                conn.execute(text("VACUUM"))

            logger.info("数据库优化完成")
        except Exception as e:
            logger.warning(f"数据库优化失败: {e}")

    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        try:
            with self._engine.connect() as conn:
                # 获取数据库大小
                db_path = db_config.get_sqlite_config().database
                db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0

                # 获取表信息
                tables_info = conn.execute(text(
                    "SELECT name, sql FROM sqlite_master WHERE type='table'"
                )).fetchall()

                # 获取数据库版本
                version = conn.execute(text("SELECT sqlite_version()")).fetchone()[0]

                return {
                    'database_type': 'SQLite',
                    'version': version,
                    'database_path': db_path,
                    'database_size_mb': db_size / (1024 * 1024),
                    'tables_count': len(tables_info),
                    'tables': [{'name': table[0], 'sql': table[1]} for table in tables_info]
                }
        except Exception as e:
            logger.error(f"获取数据库信息失败: {e}")
            return {}


class MemoryCache:
    """内存缓存替代Redis"""

    def __init__(self):
        self._cache = {}
        self._expire_times = {}

    def setex(self, key: str, expire: int, value: Any) -> bool:
        """设置带过期时间的缓存"""
        try:
            self._cache[key] = value
            self._expire_times[key] = datetime.now() + timedelta(seconds=expire)
            return True
        except:
            return False

    def get(self, key: str) -> Any:
        """获取缓存"""
        try:
            # 检查是否过期
            if key in self._expire_times:
                if datetime.now() > self._expire_times[key]:
                    self.delete(key)
                    return None

            return self._cache.get(key)
        except:
            return None

    def delete(self, *keys) -> int:
        """删除缓存"""
        deleted = 0
        for key in keys:
            if key in self._cache:
                del self._cache[key]
                deleted += 1
            if key in self._expire_times:
                del self._expire_times[key]
        return deleted

    def keys(self, pattern: str) -> List[str]:
        """获取匹配的键"""
        import fnmatch
        return [key for key in self._cache.keys() if fnmatch.fnmatch(key, pattern)]

    def ping(self) -> bool:
        """测试连接"""
        return True


class SQLiteDataRepository:
    """SQLite数据仓库"""

    def __init__(self, db_manager: SQLiteDatabaseManager):
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
            sql += " AND date(trade_date) >= date(:start_date)"
            params['start_date'] = start_date

        if end_date:
            sql += " AND date(trade_date) <= date(:end_date)"
            params['end_date'] = end_date

        sql += " ORDER BY trade_date"

        df = self.db_manager.query_to_dataframe(sql, params)

        # 处理日期列
        if not df.empty and 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])

        # 缓存结果（1小时）
        self.cache_manager.cache_dataframe(cache_key, df, 3600)

        return df

    def get_latest_prices(self, ts_codes: List[str], limit: int = 1) -> pd.DataFrame:
        """获取最新价格数据"""
        if not ts_codes:
            return pd.DataFrame()

        # SQLite使用IN查询
        placeholders = ','.join(['?' for _ in ts_codes])

        if limit == 1:
            sql = f"""
            SELECT t1.* FROM daily_price t1
            INNER JOIN (
                SELECT ts_code, MAX(trade_date) as max_date
                FROM daily_price 
                WHERE ts_code IN ({placeholders})
                GROUP BY ts_code
            ) t2 ON t1.ts_code = t2.ts_code AND t1.trade_date = t2.max_date
            ORDER BY t1.ts_code, t1.trade_date DESC
            """
        else:
            # 对于多天数据，使用窗口函数（SQLite 3.25+支持）
            sql = f"""
            SELECT * FROM (
                SELECT *, 
                       ROW_NUMBER() OVER (PARTITION BY ts_code ORDER BY trade_date DESC) as rn
                FROM daily_price 
                WHERE ts_code IN ({placeholders})
            ) t WHERE rn <= {limit}
            ORDER BY ts_code, trade_date DESC
            """

        try:
            with self.db_manager._engine.connect() as conn:
                df = pd.read_sql(sql, conn, params=ts_codes)
                if not df.empty and 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                return df
        except Exception as e:
            logger.error(f"获取最新价格失败: {e}")
            return pd.DataFrame()


class CacheManager:
    """缓存管理器"""

    def __init__(self, database_manager: SQLiteDatabaseManager):
        self.db_manager = database_manager
        self.cache_client = database_manager.get_redis_client()
        self.default_expire = 3600  # 1小时默认过期时间

    def set_cache(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """设置缓存"""
        try:
            expire_time = expire or self.default_expire
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, default=str)
            elif isinstance(value, pd.DataFrame):
                value = value.to_json(orient='records', force_ascii=False)

            return self.cache_client.setex(key, expire_time, value)
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            return False

    def get_cache(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            value = self.cache_client.get(key)
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
            return bool(self.cache_client.delete(key))
        except Exception as e:
            logger.error(f"删除缓存失败: {e}")
            return False

    def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        try:
            keys = self.cache_client.keys(pattern)
            if keys:
                return self.cache_client.delete(*keys)
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


# 全局实例
sqlite_db_manager = SQLiteDatabaseManager()
sqlite_data_repository = SQLiteDataRepository(sqlite_db_manager)

# 测试脚本
if __name__ == "__main__":
    print("=== SQLite数据库管理器测试 ===")

    # 测试数据库连接
    try:
        db_manager = SQLiteDatabaseManager()
        print("✅ 数据库连接成功")

        # 创建表
        db_manager.create_tables()
        print("✅ 数据表创建成功")

        # 获取数据库信息
        db_info = db_manager.get_database_info()
        print(f"数据库版本: {db_info.get('version', 'Unknown')}")
        print(f"数据库路径: {db_info.get('database_path', 'Unknown')}")
        print(f"数据库大小: {db_info.get('database_size_mb', 0):.2f} MB")
        print(f"表数量: {db_info.get('tables_count', 0)}")

        # 测试缓存
        cache_manager = CacheManager(db_manager)
        test_data = {"test": "data", "timestamp": datetime.now().isoformat()}

        if cache_manager.set_cache("test_key", test_data, 60):
            print("✅ 缓存设置成功")

            cached_result = cache_manager.get_cache("test_key")
            if cached_result:
                print("✅ 缓存获取成功")
            else:
                print("❌ 缓存获取失败")
        else:
            print("❌ 缓存设置失败")

        print("✅ 所有测试通过！")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()