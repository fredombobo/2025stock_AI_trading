"""系统配置模块"""
import os
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
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # 数据库类型配置
    database_type: str = field(default_factory=lambda: os.getenv("DATABASE_TYPE", "sqlite"))

    # Tushare配置
    tushare: TushareConfig = field(default_factory=lambda: TushareConfig(
        token=os.getenv("TUSHARE_TOKEN", ""),
        timeout=int(os.getenv("TUSHARE_TIMEOUT", "60")),
        retry_count=int(os.getenv("TUSHARE_RETRY_COUNT", "3")),
        retry_delay=int(os.getenv("TUSHARE_RETRY_DELAY", "1")),
        max_requests_per_minute=int(os.getenv("TUSHARE_MAX_REQUESTS", "200"))
    ))

    # 数据存储配置
    data_path: str = field(default_factory=lambda: os.getenv("DATA_PATH", "./data"))
    cache_expire_hours: int = field(default_factory=lambda: int(os.getenv("CACHE_EXPIRE_HOURS", "24")))

    # 系统性能配置
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "4")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "1000")))

    # SQLite特定配置
    sqlite_timeout: int = field(default_factory=lambda: int(os.getenv("SQLITE_TIMEOUT", "30")))
    sqlite_check_same_thread: bool = field(default_factory=lambda: os.getenv("SQLITE_CHECK_SAME_THREAD", "False").lower() == "true")

    def __post_init__(self):
        """配置验证"""
        if not self.tushare.token and self.environment == "production":
            raise ValueError("生产环境必须配置TUSHARE_TOKEN")

        # 创建数据目录
        os.makedirs(self.data_path, exist_ok=True)

        # 如果使用SQLite，确保数据库目录存在
        if self.database_type == "sqlite":
            db_path = os.getenv("SQLITE_DB_PATH", "data/quant_trading.db")
            db_dir = os.path.dirname(db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)


# 全局配置实例
system_config = SystemConfig()
