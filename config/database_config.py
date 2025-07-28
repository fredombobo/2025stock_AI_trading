# config/database_config.py
"""数据库配置模块 - SQLite版本"""
import os
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """数据库配置类"""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False


class DBConfig:
    """数据库配置管理器"""

    def __init__(self):
        self.database_type = os.getenv('DATABASE_TYPE', 'sqlite')  # 默认使用SQLite
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, DatabaseConfig]:
        """加载数据库配置"""
        configs = {}

        # SQLite配置
        configs['sqlite'] = DatabaseConfig(
            host='',
            port=0,
            database=os.getenv('SQLITE_DB_PATH', 'data/quant_trading.db'),
            username='',
            password='',
            pool_size=1,  # SQLite不支持连接池
            max_overflow=0,
            echo=os.getenv('DB_ECHO', 'False').lower() == 'true'
        )

        # PostgreSQL配置（保留原有功能）
        configs['postgresql'] = DatabaseConfig(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            database=os.getenv('POSTGRES_DB', 'quant_trading'),
            username=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'password'),
            pool_size=20,
            max_overflow=30,
            echo=os.getenv('DB_ECHO', 'False').lower() == 'true'
        )

        # Redis配置
        configs['redis'] = DatabaseConfig(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            database=int(os.getenv('REDIS_DB', '0')),
            username=os.getenv('REDIS_USER', ''),
            password=os.getenv('REDIS_PASSWORD', ''),
            pool_size=50,
            max_overflow=100
        )

        return configs

    def get_database_config(self) -> DatabaseConfig:
        """获取当前数据库配置"""
        return self.configs[self.database_type]

    def get_postgres_config(self) -> DatabaseConfig:
        """获取PostgreSQL配置"""
        return self.configs['postgresql']

    def get_sqlite_config(self) -> DatabaseConfig:
        """获取SQLite配置"""
        return self.configs['sqlite']

    def get_redis_config(self) -> DatabaseConfig:
        """获取Redis配置"""
        return self.configs['redis']

    def get_database_url(self) -> str:
        """获取当前数据库连接URL"""
        if self.database_type == 'sqlite':
            return self.get_sqlite_url()
        elif self.database_type == 'postgresql':
            return self.get_postgres_url()
        else:
            raise ValueError(f"不支持的数据库类型: {self.database_type}")

    def get_postgres_url(self) -> str:
        """获取PostgreSQL连接URL"""
        config = self.configs['postgresql']
        return f"postgresql+psycopg2://{config.username}:{config.password}@{config.host}:{config.port}/{config.database}"

    def get_sqlite_url(self) -> str:
        """获取SQLite连接URL"""
        config = self.configs['sqlite']
        # 确保数据目录存在
        db_dir = os.path.dirname(config.database)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        return f"sqlite:///{config.database}"

    def get_redis_url(self) -> str:
        """获取Redis连接URL"""
        config = self.configs['redis']
        auth_part = f":{config.password}@" if config.password else ""
        return f"redis://{auth_part}{config.host}:{config.port}/{config.database}"

    def is_sqlite(self) -> bool:
        """检查是否使用SQLite"""
        return self.database_type == 'sqlite'

    def is_postgresql(self) -> bool:
        """检查是否使用PostgreSQL"""
        return self.database_type == 'postgresql'


# config/system_config.py
"""系统配置模块"""
import os
from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class TushareConfig:
    """Tushare配置"""
    token: str
    timeout: int = 60
    retry_count: int = 3
    retry_delay: int = 1
    max_requests_per_minute: int = 200


@dataclass
class SystemConfig:
    """系统配置"""
    # 环境配置
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'False').lower() == 'true')
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))

    # 数据库类型配置
    database_type: str = field(default_factory=lambda: os.getenv('DATABASE_TYPE', 'sqlite'))

    # Tushare配置
    tushare: TushareConfig = field(default_factory=lambda: TushareConfig(
        token=os.getenv('TUSHARE_TOKEN', ''),
        timeout=int(os.getenv('TUSHARE_TIMEOUT', '60')),
        retry_count=int(os.getenv('TUSHARE_RETRY_COUNT', '3')),
        retry_delay=int(os.getenv('TUSHARE_RETRY_DELAY', '1')),
        max_requests_per_minute=int(os.getenv('TUSHARE_MAX_REQUESTS', '200'))
    ))

    # 数据存储配置
    data_path: str = field(default_factory=lambda: os.getenv('DATA_PATH', './data'))
    cache_expire_hours: int = field(default_factory=lambda: int(os.getenv('CACHE_EXPIRE_HOURS', '24')))

    # 系统性能配置
    max_workers: int = field(default_factory=lambda: int(os.getenv('MAX_WORKERS', '4')))
    batch_size: int = field(default_factory=lambda: int(os.getenv('BATCH_SIZE', '1000')))

    # SQLite特定配置
    sqlite_timeout: int = field(default_factory=lambda: int(os.getenv('SQLITE_TIMEOUT', '30')))
    sqlite_check_same_thread: bool = field(
        default_factory=lambda: os.getenv('SQLITE_CHECK_SAME_THREAD', 'False').lower() == 'true')

    def __post_init__(self):
        """配置验证"""
        if not self.tushare.token and self.environment == 'production':
            raise ValueError("生产环境必须配置TUSHARE_TOKEN")

        # 创建数据目录
        os.makedirs(self.data_path, exist_ok=True)

        # 如果使用SQLite，确保数据库目录存在
        if self.database_type == 'sqlite':
            db_path = os.getenv('SQLITE_DB_PATH', 'data/quant_trading.db')
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)


# 全局配置实例
db_config = DBConfig()
system_config = SystemConfig()


# 兼容性函数 - 保持与原代码的兼容性
def get_database_url():
    """获取数据库URL - 兼容性函数"""
    return db_config.get_database_url()


def is_sqlite():
    """是否使用SQLite"""
    return db_config.is_sqlite()


def is_postgresql():
    """是否使用PostgreSQL"""
    return db_config.is_postgresql()


# config/sqlite_models.py
"""SQLite数据模型定义"""
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean,
    Text, BigInteger, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()


class StockBasic(Base):
    """股票基础信息表 - SQLite版本"""
    __tablename__ = 'stock_basic'

    id = Column(Integer, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), unique=True, nullable=False)
    symbol = Column(String(10), nullable=False)
    name = Column(String(50), nullable=False)
    area = Column(String(10))
    industry = Column(String(50))
    market = Column(String(10))
    exchange = Column(String(10))
    curr_type = Column(String(10))
    list_status = Column(String(10))
    list_date = Column(DateTime)
    delist_date = Column(DateTime)
    is_hs = Column(String(10))

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # SQLite索引
    __table_args__ = (
        Index('idx_ts_code', 'ts_code'),
        Index('idx_symbol', 'symbol'),
        Index('idx_industry', 'industry'),
        Index('idx_market', 'market'),
    )


class DailyPrice(Base):
    """日线行情数据表 - SQLite版本"""
    __tablename__ = 'daily_price'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), nullable=False)
    trade_date = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    pre_close = Column(Float)
    change = Column(Float)
    pct_chg = Column(Float)
    vol = Column(BigInteger)
    amount = Column(Float)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # SQLite约束和索引
    __table_args__ = (
        UniqueConstraint('ts_code', 'trade_date', name='uk_code_date'),
        Index('idx_trade_date', 'trade_date'),
        Index('idx_ts_code_date', 'ts_code', 'trade_date'),
        Index('idx_vol', 'vol'),
        Index('idx_amount', 'amount'),
    )


class TechnicalIndicators(Base):
    """技术指标数据表 - SQLite版本"""
    __tablename__ = 'technical_indicators'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), nullable=False)
    trade_date = Column(DateTime, nullable=False)

    # 移动平均线
    ma5 = Column(Float)
    ma10 = Column(Float)
    ma20 = Column(Float)
    ma30 = Column(Float)
    ma60 = Column(Float)
    ma120 = Column(Float)
    ma250 = Column(Float)

    # 指数移动平均线
    ema5 = Column(Float)
    ema10 = Column(Float)
    ema20 = Column(Float)
    ema30 = Column(Float)

    # MACD指标
    macd_dif = Column(Float)
    macd_dea = Column(Float)
    macd_histogram = Column(Float)

    # RSI指标
    rsi6 = Column(Float)
    rsi12 = Column(Float)
    rsi24 = Column(Float)

    # 布林带
    boll_upper = Column(Float)
    boll_mid = Column(Float)
    boll_lower = Column(Float)

    # KDJ指标
    kdj_k = Column(Float)
    kdj_d = Column(Float)
    kdj_j = Column(Float)

    # 威廉指标
    wr10 = Column(Float)
    wr6 = Column(Float)

    # 成交量指标
    vol_ma5 = Column(Float)
    vol_ma10 = Column(Float)
    vol_ratio = Column(Float)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('ts_code', 'trade_date', name='uk_tech_code_date'),
        Index('idx_tech_trade_date', 'trade_date'),
        Index('idx_tech_ts_code_date', 'ts_code', 'trade_date'),
    )


class DataUpdateLog(Base):
    """数据更新日志表 - SQLite版本"""
    __tablename__ = 'data_update_log'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    data_type = Column(String(50), nullable=False)
    ts_code = Column(String(20))
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    status = Column(String(20), nullable=False)
    records_count = Column(Integer)
    error_message = Column(Text)
    duration_seconds = Column(Float)

    created_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index('idx_data_type', 'data_type'),
        Index('idx_status', 'status'),
        Index('idx_created_at', 'created_at'),
        Index('idx_ts_code_type', 'ts_code', 'data_type'),
    )


if __name__ == "__main__":
    # 测试配置
    print("=== 数据库配置测试 ===")

    print(f"数据库类型: {db_config.database_type}")
    print(f"数据库URL: {db_config.get_database_url()}")

    if db_config.is_sqlite():
        print("✅ 使用SQLite数据库")
        sqlite_config = db_config.get_sqlite_config()
        print(f"数据库文件: {sqlite_config.database}")

    print(f"Tushare Token配置: {'已配置' if system_config.tushare.token else '未配置'}")
    print(f"数据路径: {system_config.data_path}")

    print("配置测试完成！")