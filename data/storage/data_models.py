# data/storage/data_models.py
"""数据模型定义"""
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Boolean, 
    Text, BigInteger, Index, UniqueConstraint, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional

Base = declarative_base()

class StockBasic(Base):
    """股票基础信息表"""
    __tablename__ = 'stock_basic'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), unique=True, nullable=False, comment='股票代码')
    symbol = Column(String(10), nullable=False, comment='股票简称')
    name = Column(String(50), nullable=False, comment='股票名称')
    area = Column(String(10), comment='地域')
    industry = Column(String(50), comment='所属行业')
    market = Column(String(10), comment='市场类型')
    exchange = Column(String(10), comment='交易所代码')
    curr_type = Column(String(10), comment='交易货币')
    list_status = Column(String(10), comment='上市状态')
    list_date = Column(DateTime, comment='上市日期')
    delist_date = Column(DateTime, comment='退市日期')
    is_hs = Column(String(10), comment='是否沪深港通标的')
    
    created_at = Column(DateTime, default=func.now(), comment='创建时间')
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment='更新时间')
    
    # 索引
    __table_args__ = (
        Index('idx_ts_code', 'ts_code'),
        Index('idx_symbol', 'symbol'),
        Index('idx_industry', 'industry'),
        Index('idx_market', 'market'),
    )

class DailyPrice(Base):
    """日线行情数据表"""
    __tablename__ = 'daily_price'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), nullable=False, comment='股票代码')
    trade_date = Column(DateTime, nullable=False, comment='交易日期')
    open_price = Column(Float, comment='开盘价')
    high_price = Column(Float, comment='最高价')
    low_price = Column(Float, comment='最低价')
    close_price = Column(Float, comment='收盘价')
    pre_close = Column(Float, comment='昨收价')
    change = Column(Float, comment='涨跌额')
    pct_chg = Column(Float, comment='涨跌幅')
    vol = Column(BigInteger, comment='成交量(手)')
    amount = Column(Float, comment='成交额(千元)')
    
    created_at = Column(DateTime, default=func.now(), comment='创建时间')
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment='更新时间')
    
    # 唯一约束和索引
    __table_args__ = (
        UniqueConstraint('ts_code', 'trade_date', name='uk_code_date'),
        Index('idx_trade_date', 'trade_date'),
        Index('idx_ts_code_date', 'ts_code', 'trade_date'),
        Index('idx_vol', 'vol'),
        Index('idx_amount', 'amount'),
    )

class MinutePrice(Base):
    """分钟级行情数据表"""
    __tablename__ = 'minute_price'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), nullable=False, comment='股票代码')
    trade_time = Column(DateTime, nullable=False, comment='交易时间')
    open_price = Column(Float, comment='开盘价')
    high_price = Column(Float, comment='最高价')
    low_price = Column(Float, comment='最低价')
    close_price = Column(Float, comment='收盘价')
    vol = Column(BigInteger, comment='成交量')
    amount = Column(Float, comment='成交额')
    
    created_at = Column(DateTime, default=func.now(), comment='创建时间')
    
    # 分区和索引
    __table_args__ = (
        UniqueConstraint('ts_code', 'trade_time', name='uk_code_time'),
        Index('idx_trade_time', 'trade_time'),
        Index('idx_ts_code_time', 'ts_code', 'trade_time'),
    )

class FinancialData(Base):
    """财务数据表"""
    __tablename__ = 'financial_data'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), nullable=False, comment='股票代码')
    ann_date = Column(DateTime, comment='公告日期')
    f_ann_date = Column(DateTime, comment='实际公告日期')
    end_date = Column(DateTime, nullable=False, comment='报告期')
    report_type = Column(String(20), comment='报告类型')
    comp_type = Column(String(20), comment='公司类型')
    
    # 主要财务指标
    total_revenue = Column(Float, comment='营业总收入')
    revenue = Column(Float, comment='营业收入')
    int_income = Column(Float, comment='利息收入')
    prem_earned = Column(Float, comment='已赚保费')
    comm_income = Column(Float, comment='手续费及佣金收入')
    n_commis_income = Column(Float, comment='手续费及佣金净收入')
    n_oth_income = Column(Float, comment='其他经营净收益')
    n_oth_b_income = Column(Float, comment='加:其他业务净收益')
    prem_income = Column(Float, comment='保险业务收入')
    out_prem = Column(Float, comment='减:分出保费')
    une_prem_reser = Column(Float, comment='提取未到期责任准备金')
    reins_income = Column(Float, comment='其中:分保费收入')
    n_sec_tb_income = Column(Float, comment='代理买卖证券业务净收入')
    n_sec_uw_income = Column(Float, comment='证券承销业务净收入')
    n_asset_mg_income = Column(Float, comment='受托客户资产管理业务净收入')
    oth_b_income = Column(Float, comment='其他业务收入')
    fv_value_chg_gain = Column(Float, comment='加:公允价值变动净收益')
    invest_income = Column(Float, comment='加:投资净收益')
    ass_invest_income = Column(Float, comment='其中:对联营企业和合营企业的投资收益')
    forex_gain = Column(Float, comment='加:汇兑净收益')
    total_cogs = Column(Float, comment='营业总成本')
    oper_cost = Column(Float, comment='减:营业成本')
    int_exp = Column(Float, comment='减:利息支出')
    comm_exp = Column(Float, comment='减:手续费及佣金支出')
    biz_tax_surchg = Column(Float, comment='减:营业税金及附加')
    sell_exp = Column(Float, comment='减:销售费用')
    admin_exp = Column(Float, comment='减:管理费用')
    fin_exp = Column(Float, comment='减:财务费用')
    assets_impair_loss = Column(Float, comment='减:资产减值损失')
    prem_refund = Column(Float, comment='退保金')
    compens_payout = Column(Float, comment='赔付总支出')
    reser_insur_liab = Column(Float, comment='提取保险责任准备金')
    div_payt = Column(Float, comment='保户红利支出')
    reins_exp = Column(Float, comment='分保费用')
    oper_exp = Column(Float, comment='营业支出')
    compens_payout_refu = Column(Float, comment='减:摊回赔付支出')
    insur_reser_refu = Column(Float, comment='减:摊回保险责任准备金')
    reins_cost_refund = Column(Float, comment='减:摊回分保费用')
    other_bus_cost = Column(Float, comment='其他业务成本')
    operate_profit = Column(Float, comment='营业利润')
    non_oper_income = Column(Float, comment='加:营业外收入')
    non_oper_exp = Column(Float, comment='减:营业外支出')
    nca_disploss = Column(Float, comment='其中:减:非流动资产处置净损失')
    total_profit = Column(Float, comment='利润总额')
    income_tax = Column(Float, comment='减:所得税费用')
    n_income = Column(Float, comment='净利润(含少数股东损益)')
    n_income_attr_p = Column(Float, comment='净利润(不含少数股东损益)')
    minority_gain = Column(Float, comment='少数股东损益')
    oth_compr_income = Column(Float, comment='其他综合收益')
    t_compr_income = Column(Float, comment='综合收益总额')
    compr_inc_attr_p = Column(Float, comment='归属于母公司(或股东)的综合收益总额')
    compr_inc_attr_m_s = Column(Float, comment='归属于少数股东的综合收益总额')
    ebit = Column(Float, comment='息税前利润')
    ebitda = Column(Float, comment='息税折旧摊销前利润')
    insurance_exp = Column(Float, comment='保险业务支出')
    undist_profit = Column(Float, comment='年初未分配利润')
    distable_profit = Column(Float, comment='可分配利润')
    
    created_at = Column(DateTime, default=func.now(), comment='创建时间')
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment='更新时间')
    
    __table_args__ = (
        UniqueConstraint('ts_code', 'end_date', 'report_type', name='uk_fin_code_date_type'),
        Index('idx_end_date', 'end_date'),
        Index('idx_ann_date', 'ann_date'),
        Index('idx_ts_code_end_date', 'ts_code', 'end_date'),
    )

class TechnicalIndicators(Base):
    """技术指标数据表"""
    __tablename__ = 'technical_indicators'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts_code = Column(String(20), nullable=False, comment='股票代码')
    trade_date = Column(DateTime, nullable=False, comment='交易日期')
    
    # 移动平均线
    ma5 = Column(Float, comment='5日移动平均')
    ma10 = Column(Float, comment='10日移动平均')
    ma20 = Column(Float, comment='20日移动平均')
    ma30 = Column(Float, comment='30日移动平均')
    ma60 = Column(Float, comment='60日移动平均')
    ma120 = Column(Float, comment='120日移动平均')
    ma250 = Column(Float, comment='250日移动平均')
    
    # 指数移动平均线
    ema5 = Column(Float, comment='5日指数移动平均')
    ema10 = Column(Float, comment='10日指数移动平均')
    ema20 = Column(Float, comment='20日指数移动平均')
    ema30 = Column(Float, comment='30日指数移动平均')
    
    # MACD指标
    macd_dif = Column(Float, comment='MACD DIF')
    macd_dea = Column(Float, comment='MACD DEA')
    macd_histogram = Column(Float, comment='MACD柱状图')
    
    # RSI指标
    rsi6 = Column(Float, comment='6日RSI')
    rsi12 = Column(Float, comment='12日RSI')
    rsi24 = Column(Float, comment='24日RSI')
    
    # 布林带
    boll_upper = Column(Float, comment='布林带上轨')
    boll_mid = Column(Float, comment='布林带中轨')
    boll_lower = Column(Float, comment='布林带下轨')
    
    # KDJ指标
    kdj_k = Column(Float, comment='KDJ K值')
    kdj_d = Column(Float, comment='KDJ D值')
    kdj_j = Column(Float, comment='KDJ J值')
    
    # 威廉指标
    wr10 = Column(Float, comment='10日威廉指标')
    wr6 = Column(Float, comment='6日威廉指标')
    
    # 成交量指标
    vol_ma5 = Column(Float, comment='5日成交量均线')
    vol_ma10 = Column(Float, comment='10日成交量均线')
    vol_ratio = Column(Float, comment='量比')
    
    created_at = Column(DateTime, default=func.now(), comment='创建时间')
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment='更新时间')
    
    __table_args__ = (
        UniqueConstraint('ts_code', 'trade_date', name='uk_tech_code_date'),
        Index('idx_tech_trade_date', 'trade_date'),
        Index('idx_tech_ts_code_date', 'ts_code', 'trade_date'),
    )

class MarketNews(Base):
    """市场新闻表"""
    __tablename__ = 'market_news'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False, comment='新闻标题')
    content = Column(Text, comment='新闻内容')
    source = Column(String(100), comment='新闻来源')
    author = Column(String(100), comment='作者')
    pub_time = Column(DateTime, nullable=False, comment='发布时间')
    url = Column(String(1000), comment='新闻链接')
    
    # 情感分析结果
    sentiment_score = Column(Float, comment='情感得分(-1到1)')
    sentiment_label = Column(String(20), comment='情感标签')
    
    # 相关股票
    related_stocks = Column(String(1000), comment='相关股票代码，逗号分隔')
    
    # 关键词
    keywords = Column(String(1000), comment='关键词，逗号分隔')
    
    created_at = Column(DateTime, default=func.now(), comment='创建时间')
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment='更新时间')
    
    __table_args__ = (
        Index('idx_pub_time', 'pub_time'),
        Index('idx_sentiment_score', 'sentiment_score'),
        Index('idx_source', 'source'),
    )

class DataUpdateLog(Base):
    """数据更新日志表"""
    __tablename__ = 'data_update_log'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    data_type = Column(String(50), nullable=False, comment='数据类型')
    ts_code = Column(String(20), comment='股票代码')
    start_date = Column(DateTime, comment='开始日期')
    end_date = Column(DateTime, comment='结束日期')
    status = Column(String(20), nullable=False, comment='状态')
    records_count = Column(Integer, comment='记录数量')
    error_message = Column(Text, comment='错误信息')
    duration_seconds = Column(Float, comment='耗时(秒)')
    
    created_at = Column(DateTime, default=func.now(), comment='创建时间')
    
    __table_args__ = (
        Index('idx_data_type', 'data_type'),
        Index('idx_status', 'status'),
        Index('idx_created_at', 'created_at'),
        Index('idx_ts_code_type', 'ts_code', 'data_type'),
    )
