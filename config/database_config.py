"""数据库连接配置模块"""
import os
from typing import Dict
from dataclasses import dataclass


@dataclass
class DatabaseConfig:
    """数据库配置"""
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
        self.database_type = os.getenv("DATABASE_TYPE", "sqlite")
        self.configs = self._load_configs()

    def _load_configs(self) -> Dict[str, DatabaseConfig]:
        configs: Dict[str, DatabaseConfig] = {}

        # SQLite配置
        configs["sqlite"] = DatabaseConfig(
            host="",
            port=0,
            database=os.getenv("SQLITE_DB_PATH", "data/quant_trading.db"),
            username="",
            password="",
            pool_size=1,
            max_overflow=0,
            echo=os.getenv("DB_ECHO", "False").lower() == "true",
        )

        # PostgreSQL配置
        configs["postgresql"] = DatabaseConfig(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "quant_trading"),
            username=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", "password"),
            pool_size=20,
            max_overflow=30,
            echo=os.getenv("DB_ECHO", "False").lower() == "true",
        )

        # Redis配置
        configs["redis"] = DatabaseConfig(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            database=int(os.getenv("REDIS_DB", "0")),
            username=os.getenv("REDIS_USER", ""),
            password=os.getenv("REDIS_PASSWORD", ""),
            pool_size=50,
            max_overflow=100,
        )

        return configs

    def get_database_config(self) -> DatabaseConfig:
        return self.configs[self.database_type]

    def get_postgres_config(self) -> DatabaseConfig:
        return self.configs["postgresql"]

    def get_sqlite_config(self) -> DatabaseConfig:
        return self.configs["sqlite"]

    def get_redis_config(self) -> DatabaseConfig:
        return self.configs["redis"]

    def get_database_url(self) -> str:
        if self.database_type == "sqlite":
            return self.get_sqlite_url()
        if self.database_type == "postgresql":
            return self.get_postgres_url()
        raise ValueError(f"不支持的数据库类型: {self.database_type}")

    def get_postgres_url(self) -> str:
        config = self.configs["postgresql"]
        return (
            f"postgresql+psycopg2://{config.username}:{config.password}@"
            f"{config.host}:{config.port}/{config.database}"
        )

    def get_sqlite_url(self) -> str:
        config = self.configs["sqlite"]
        db_dir = os.path.dirname(config.database)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        return f"sqlite:///{config.database}"

    def get_redis_url(self) -> str:
        config = self.configs["redis"]
        auth_part = f":{config.password}@" if config.password else ""
        return f"redis://{auth_part}{config.host}:{config.port}/{config.database}"

    def is_sqlite(self) -> bool:
        return self.database_type == "sqlite"

    def is_postgresql(self) -> bool:
        return self.database_type == "postgresql"


# 全局配置实例
db_config = DBConfig()


# 兼容性函数
def get_database_url() -> str:
    return db_config.get_database_url()


def is_sqlite() -> bool:
    return db_config.is_sqlite()


def is_postgresql() -> bool:
    return db_config.is_postgresql()
