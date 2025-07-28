"""SQLite数据模型定义"""
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    BigInteger,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()


class StockBasic(Base):
    """股票基础信息表 - SQLite版本"""

    __tablename__ = "stock_basic"

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

    __table_args__ = (
        Index("idx_ts_code", "ts_code"),
        Index("idx_symbol", "symbol"),
        Index("idx_industry", "industry"),
        Index("idx_market", "market"),
    )


class DailyPrice(Base):
    """日线行情数据表 - SQLite版本"""

    __tablename__ = "daily_price"

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

    __table_args__ = (
        UniqueConstraint("ts_code", "trade_date", name="uk_code_date"),
        Index("idx_trade_date", "trade_date"),
        Index("idx_ts_code_date", "ts_code", "trade_date"),
        Index("idx_vol", "vol"),
        Index("idx_amount", "amount"),
    )


class TechnicalIndicators(Base):
    """技术指标数据表 - SQLite版本"""

    __tablename__ = "technical_indicators"

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
        UniqueConstraint("ts_code", "trade_date", name="uk_tech_code_date"),
        Index("idx_tech_trade_date", "trade_date"),
        Index("idx_tech_ts_code_date", "ts_code", "trade_date"),
    )


class DataUpdateLog(Base):
    """数据更新日志表 - SQLite版本"""

    __tablename__ = "data_update_log"

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
        Index("idx_data_type", "data_type"),
        Index("idx_status", "status"),
        Index("idx_created_at", "created_at"),
        Index("idx_ts_code_type", "ts_code", "data_type"),
    )

