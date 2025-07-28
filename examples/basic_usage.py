# examples/basic_usage.py
"""基础使用示例"""
import sys
import os
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.storage.database_manager import data_repository
from data.collectors.historical_collector import historical_collector
from data.collectors.realtime_collector import realtime_collector
from data.processors.technical_indicators import technical_indicators
from utils.logger import get_logger

logger = get_logger(__name__)

class QuickStartExample:
    """快速开始示例"""
    
    def __init__(self):
        self.repository = data_repository
        self.historical_collector = historical_collector
        self.realtime_collector = realtime_collector
        self.technical_indicators = technical_indicators
    
    def example_1_basic_data_query(self):
        """示例1: 基础数据查询"""
        logger.info("=== 示例1: 基础数据查询 ===")
        
        try:
            # 获取股票基本信息
            logger.info("获取股票基本信息...")
            stock_basic = self.repository.get_stock_basic_info()
            logger.info(f"共有 {len(stock_basic)} 只股票")
            
            if not stock_basic.empty:
                # 显示前5只股票信息
                logger.info("前5只股票信息:")
                for _, stock in stock_basic.head().iterrows():
                    logger.info(f"  {stock['ts_code']} - {stock['name']} ({stock['industry']})")
            
            # 获取特定股票的日线数据
            if not stock_basic.empty:
                ts_code = stock_basic.iloc[0]['ts_code']
                logger.info(f"\n获取股票 {ts_code} 的最近30天数据...")
                
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                daily_data = self.repository.get_daily_price(ts_code, start_date, end_date)
                
                if not daily_data.empty:
                    logger.info(f"获取到 {len(daily_data)} 条日线数据")
                    latest = daily_data.iloc[-1]
                    logger.info(f"最新价格: {latest['close_price']:.2f}, 涨跌幅: {latest['pct_chg']:.2f}%")
                else:
                    logger.warning("未获取到日线数据")
            
        except Exception as e:
            logger.error(f"示例1执行失败: {e}")
    
    def example_2_technical_indicators(self):
        """示例2: 技术指标计算"""
        logger.info("\n=== 示例2: 技术指标计算 ===")
        
        try:
            # 获取一个有数据的股票
            sql = "SELECT ts_code FROM daily_price GROUP BY ts_code HAVING COUNT(*) > 50 LIMIT 1"
            result = self.repository.db_manager.execute_sql(sql)
            
            if not result:
                logger.warning("未找到有足够数据的股票")
                return
            
            ts_code = result[0][0]
            logger.info(f"为股票 {ts_code} 计算技术指标...")
            
            # 计算技术指标
            indicators = self.technical_indicators.calculate_all_indicators(ts_code, save_to_db=True)
            
            if not indicators.empty:
                logger.info(f"计算完成，共 {len(indicators)} 条记录")
                
                # 显示最新的技术指标
                latest = indicators.iloc[-1]
                logger.info("最新技术指标:")
                logger.info(f"  MA5: {latest.get('ma5', 'N/A')}")
                logger.info(f"  MA20: {latest.get('ma20', 'N/A')}")
                logger.info(f"  RSI12: {latest.get('rsi12', 'N/A')}")
                logger.info(f"  MACD DIF: {latest.get('macd_dif', 'N/A')}")
                
                # 获取信号分析
                signals = self.technical_indicators.get_signal_analysis(ts_code)
                if 'signals' in signals:
                    logger.info("技术指标信号:")
                    for indicator, signal in signals['signals'].items():
                        logger.info(f"  {indicator}: {signal}")
            else:
                logger.warning("技术指标计算失败")
                
        except Exception as e:
            logger.error(f"示例2执行失败: {e}")
    
    def example_3_data_collection(self):
        """示例3: 数据收集"""
        logger.info("\n=== 示例3: 数据收集 ===")
        
        try:
            # 获取股票列表（限制数量）
            stock_basic = self.repository.get_stock_basic_info()
            if stock_basic.empty:
                logger.info("股票基本信息为空，开始收集...")
                success = self.historical_collector.collect_stock_basic_info()
                if success:
                    logger.info("股票基本信息收集成功")
                    stock_basic = self.repository.get_stock_basic_info()
            
            if not stock_basic.empty:
                # 选择前5只股票收集最近7天的数据
                sample_stocks = stock_basic.head(5)['ts_code'].tolist()
                
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                
                logger.info(f"收集 {len(sample_stocks)} 只股票最近7天的数据...")
                
                success = self.historical_collector.collect_daily_price_range(
                    start_date, end_date, sample_stocks, parallel=False
                )
                
                if success:
                    logger.info("数据收集成功")
                else:
                    logger.warning("数据收集部分失败")
        
        except Exception as e:
            logger.error(f"示例3执行失败: {e}")
    
    def example_4_realtime_data(self):
        """示例4: 实时数据处理"""
        logger.info("\n=== 示例4: 实时数据处理 ===")
        
        try:
            # 定义价格更新回调函数
            def price_callback(data):
                logger.info(f"价格更新: {data['ts_code']} - {data['price']}")
            
            # 获取一只股票进行演示
            stock_basic = self.repository.get_stock_basic_info()
            if stock_basic.empty:
                logger.warning("无股票数据")
                return
            
            ts_code = stock_basic.iloc[0]['ts_code']
            
            # 订阅实时数据（演示用，不实际启动）
            logger.info(f"演示订阅股票 {ts_code} 的实时数据")
            self.realtime_collector.subscribe_stock(ts_code, callback=price_callback)
            
            # 模拟获取最新价格
            cached_price = self.realtime_collector.get_latest_price(ts_code)
            if cached_price:
                logger.info(f"缓存的最新价格: {cached_price}")
            else:
                logger.info("暂无缓存价格数据")
            
            # 取消订阅
            self.realtime_collector.unsubscribe_stock(ts_code)
            logger.info("取消订阅成功")
        
        except Exception as e:
            logger.error(f"示例4执行失败: {e}")
    
    def example_5_performance_analysis(self):
        """示例5: 性能分析"""
        logger.info("\n=== 示例5: 性能分析 ===")
        
        try:
            # 获取系统统计信息
            from utils.helpers import get_memory_usage
            
            memory_info = get_memory_usage()
            logger.info(f"内存使用情况: {memory_info}")
            
            # 获取数据库统计信息
            tables = ['stock_basic', 'daily_price', 'technical_indicators']
            for table in tables:
                try:
                    count = self.repository.db_manager.get_table_row_count(table)
                    logger.info(f"表 {table}: {count} 条记录")
                except:
                    logger.info(f"表 {table}: 无法获取记录数")
            
            # 获取最新更新时间
            latest_date = self.repository.db_manager.get_latest_trade_date('daily_price')
            if latest_date:
                logger.info(f"最新交易数据日期: {latest_date}")
            
        except Exception as e:
            logger.error(f"示例5执行失败: {e}")
    
    def run_all_examples(self):
        """运行所有示例"""
        logger.info("🚀 开始运行量化交易系统示例")
        
        examples = [
            self.example_1_basic_data_query,
            self.example_2_technical_indicators,
            self.example_3_data_collection,
            self.example_4_realtime_data,
            self.example_5_performance_analysis
        ]
        
        for i, example in enumerate(examples, 1):
            try:
                example()
                logger.info(f"✅ 示例 {i} 执行完成\n")
            except Exception as e:
                logger.error(f"❌ 示例 {i} 执行失败: {e}\n")
        
        logger.info("🎉 所有示例运行完成！")

def main():
    """主函数"""
    try:
        example = QuickStartExample()
        example.run_all_examples()
    except Exception as e:
        logger.error(f"示例运行失败: {e}")

if __name__ == '__main__':
    main()


# examples/strategy_example.py
"""策略开发示例"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.storage.database_manager import data_repository
from data.processors.technical_indicators import technical_indicators
from utils.logger import get_logger
from utils.helpers import calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown

logger = get_logger(__name__)

class SimpleMAStrategy:
    """简单均线策略示例"""
    
    def __init__(self, short_window=5, long_window=20):
        self.short_window = short_window
        self.long_window = long_window
        self.repository = data_repository
        self.tech_indicators = technical_indicators
    
    def generate_signals(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成交易信号"""
        try:
            # 获取价格数据
            price_data = self.repository.get_daily_price(ts_code, start_date, end_date)
            
            if price_data.empty:
                logger.warning(f"股票 {ts_code} 无价格数据")
                return pd.DataFrame()
            
            # 确保数据按日期排序
            price_data = price_data.sort_values('trade_date').reset_index(drop=True)
            
            # 计算移动平均线
            price_data['ma_short'] = price_data['close_price'].rolling(window=self.short_window).mean()
            price_data['ma_long'] = price_data['close_price'].rolling(window=self.long_window).mean()
            
            # 生成交易信号
            price_data['signal'] = 0
            price_data['position'] = 0
            
            # 当短期均线上穿长期均线时买入(signal=1)
            # 当短期均线下穿长期均线时卖出(signal=-1)
            price_data.loc[
                (price_data['ma_short'] > price_data['ma_long']) & 
                (price_data['ma_short'].shift(1) <= price_data['ma_long'].shift(1)), 
                'signal'
            ] = 1
            
            price_data.loc[
                (price_data['ma_short'] < price_data['ma_long']) & 
                (price_data['ma_short'].shift(1) >= price_data['ma_long'].shift(1)), 
                'signal'
            ] = -1
            
            # 计算持仓状态
            price_data['position'] = price_data['signal'].fillna(0).cumsum()
            price_data['position'] = price_data['position'].clip(-1, 1)  # 限制持仓在-1到1之间
            
            return price_data
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return pd.DataFrame()
    
    def backtest_strategy(self, ts_code: str, start_date: str, end_date: str) -> dict:
        """回测策略"""
        try:
            # 生成交易信号
            signals_data = self.generate_signals(ts_code, start_date, end_date)
            
            if signals_data.empty:
                return {'error': '无交易信号数据'}
            
            # 计算收益率
            signals_data['returns'] = signals_data['close_price'].pct_change()
            signals_data['strategy_returns'] = signals_data['returns'] * signals_data['position'].shift(1)
            
            # 计算累积收益
            signals_data['cumulative_returns'] = (1 + signals_data['returns']).cumprod()
            signals_data['cumulative_strategy_returns'] = (1 + signals_data['strategy_returns']).cumprod()
            
            # 移除NaN值
            clean_data = signals_data.dropna()
            
            if clean_data.empty:
                return {'error': '无有效回测数据'}
            
            # 计算性能指标
            total_return = clean_data['cumulative_strategy_returns'].iloc[-1] - 1
            benchmark_return = clean_data['cumulative_returns'].iloc[-1] - 1
            
            # 计算夏普比率
            sharpe_ratio = calculate_sharpe_ratio(clean_data['strategy_returns'].dropna())
            
            # 计算最大回撤
            max_drawdown_info = calculate_max_drawdown(clean_data['cumulative_strategy_returns'])
            
            # 计算胜率
            winning_trades = (clean_data['strategy_returns'] > 0).sum()
            total_trades = (clean_data['strategy_returns'] != 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 计算交易次数
            trade_signals = (clean_data['signal'] != 0).sum()
            
            results = {
                'ts_code': ts_code,
                'start_date': start_date,
                'end_date': end_date,
                'total_return': total_return,
                'benchmark_return': benchmark_return,
                'excess_return': total_return - benchmark_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown_info['max_drawdown'],
                'win_rate': win_rate,
                'total_trades': int(total_trades),
                'trade_signals': int(trade_signals),
                'data_points': len(clean_data),
                'strategy_params': {
                    'short_window': self.short_window,
                    'long_window': self.long_window
                }
            }
            
            return results
            
        except Exception as e:
            logger.error(f"策略回测失败: {e}")
            return {'error': str(e)}
    
    def batch_backtest(self, ts_codes: list, start_date: str, end_date: str) -> pd.DataFrame:
        """批量回测多只股票"""
        results = []
        
        for i, ts_code in enumerate(ts_codes):
            logger.info(f"回测股票 {ts_code} ({i+1}/{len(ts_codes)})")
            
            result = self.backtest_strategy(ts_code, start_date, end_date)
            
            if 'error' not in result:
                results.append(result)
                logger.info(f"  总收益: {result['total_return']:.2%}, 夏普比率: {result['sharpe_ratio']:.3f}")
            else:
                logger.warning(f"  跳过: {result['error']}")
        
        if results:
            results_df = pd.DataFrame(results)
            return results_df
        else:
            return pd.DataFrame()

def run_strategy_example():
    """运行策略示例"""
    logger.info("🚀 开始运行策略示例")
    
    try:
        # 创建策略实例
        strategy = SimpleMAStrategy(short_window=5, long_window=20)
        
        # 获取测试股票
        stock_basic = data_repository.get_stock_basic_info()
        if stock_basic.empty:
            logger.error("无股票基本信息，请先运行系统初始化")
            return
        
        # 选择前10只股票进行测试
        test_stocks = stock_basic.head(10)['ts_code'].tolist()
        
        # 设置回测时间范围
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        logger.info(f"回测时间范围: {start_date} 到 {end_date}")
        logger.info(f"测试股票数量: {len(test_stocks)}")
        
        # 单只股票详细回测
        logger.info(f"\n=== 单只股票详细回测 ===")
        sample_stock = test_stocks[0]
        detailed_result = strategy.backtest_strategy(sample_stock, start_date, end_date)
        
        if 'error' not in detailed_result:
            logger.info(f"股票: {detailed_result['ts_code']}")
            logger.info(f"策略总收益: {detailed_result['total_return']:.2%}")
            logger.info(f"基准收益: {detailed_result['benchmark_return']:.2%}")
            logger.info(f"超额收益: {detailed_result['excess_return']:.2%}")
            logger.info(f"夏普比率: {detailed_result['sharpe_ratio']:.3f}")
            logger.info(f"最大回撤: {detailed_result['max_drawdown']:.2%}")
            logger.info(f"胜率: {detailed_result['win_rate']:.2%}")
            logger.info(f"交易次数: {detailed_result['total_trades']}")
        else:
            logger.error(f"详细回测失败: {detailed_result['error']}")
        
        # 批量回测
        logger.info(f"\n=== 批量回测结果 ===")
        batch_results = strategy.batch_backtest(test_stocks, start_date, end_date)
        
        if not batch_results.empty:
            logger.info(f"成功回测 {len(batch_results)} 只股票")
            
            # 统计结果
            avg_return = batch_results['total_return'].mean()
            avg_sharpe = batch_results['sharpe_ratio'].mean()
            avg_max_dd = batch_results['max_drawdown'].mean()
            positive_returns = (batch_results['total_return'] > 0).sum()
            
            logger.info(f"平均收益率: {avg_return:.2%}")
            logger.info(f"平均夏普比率: {avg_sharpe:.3f}")
            logger.info(f"平均最大回撤: {avg_max_dd:.2%}")
            logger.info(f"正收益股票数量: {positive_returns}/{len(batch_results)}")
            
            # 显示最佳和最差表现
            best_stock = batch_results.loc[batch_results['total_return'].idxmax()]
            worst_stock = batch_results.loc[batch_results['total_return'].idxmin()]
            
            logger.info(f"\n最佳表现: {best_stock['ts_code']} (收益: {best_stock['total_return']:.2%})")
            logger.info(f"最差表现: {worst_stock['ts_code']} (收益: {worst_stock['total_return']:.2%})")
        else:
            logger.warning("批量回测无结果")
        
        logger.info("✅ 策略示例运行完成")
        
    except Exception as e:
        logger.error(f"策略示例运行失败: {e}")

if __name__ == '__main__':
    run_strategy_example()


# .env.example
"""环境变量配置示例"""

# =============================================================================
# 数据源配置
# =============================================================================

# Tushare API Token (必须配置)
TUSHARE_TOKEN=your_tushare_token_here

# Tushare 配置
TUSHARE_TIMEOUT=60
TUSHARE_RETRY_COUNT=3
TUSHARE_RETRY_DELAY=1
TUSHARE_MAX_REQUESTS=200

# =============================================================================
# 数据库配置
# =============================================================================

# PostgreSQL 配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=quant_trading
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_postgres_password

# Redis 配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# =============================================================================
# 系统配置
# =============================================================================

# 运行环境 (development/production)
ENVIRONMENT=development

# 调试模式
DEBUG=True

# 日志级别 (DEBUG/INFO/WARNING/ERROR)
LOG_LEVEL=INFO

# 数据存储路径
DATA_PATH=./data

# 缓存过期时间（小时）
CACHE_EXPIRE_HOURS=24

# 系统性能配置
MAX_WORKERS=4
BATCH_SIZE=1000

# =============================================================================
# AI模型配置
# =============================================================================

# OpenAI API配置（可选）
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1

# 模型配置
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=2000
TEMPERATURE=0.7

# =============================================================================
# Web服务配置
# =============================================================================

# Flask配置
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_SECRET_KEY=your_secret_key_here

# API配置
API_VERSION=v1
API_PREFIX=/api

# =============================================================================
# 安全配置
# =============================================================================

# JWT密钥
JWT_SECRET_KEY=your_jwt_secret_key

# 加密密钥
ENCRYPTION_KEY=your_encryption_key

# =============================================================================
# 监控配置
# =============================================================================

# 性能监控
ENABLE_MONITORING=True
MONITORING_PORT=8080

# 告警配置
ALERT_EMAIL=your_email@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_smtp_user
SMTP_PASSWORD=your_smtp_password


# docker-compose.yml
"""Docker Compose 配置"""

version: '3.8'

services:
  # PostgreSQL 数据库
  postgres:
    image: postgres:15
    container_name: quant_postgres
    environment:
      POSTGRES_DB: quant_trading
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres123
      POSTGRES_INITDB_ARGS: "--encoding=UTF8 --locale=C"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    networks:
      - quant_network

  # Redis 缓存
  redis:
    image: redis:7-alpine
    container_name: quant_redis
    command: redis-server --appendonly yes --requirepass redis123
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    networks:
      - quant_network

  # 量化交易系统主应用
  app:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: quant_app
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_PASSWORD=postgres123
      - REDIS_HOST=redis
      - REDIS_PASSWORD=redis123
      - ENVIRONMENT=production
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    ports:
      - "5000:5000"
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    networks:
      - quant_network

  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    container_name: quant_nginx
    volumes:
      - ./docker/nginx.conf:/etc/nginx/nginx.conf
      - ./web_interface/static:/usr/share/nginx/html/static
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - app
    restart: unless-stopped
    networks:
      - quant_network

volumes:
  postgres_data:
  redis_data:

networks:
  quant_network:
    driver: bridge


# docker/Dockerfile
"""Docker 构建文件"""

FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 TA-Lib
RUN apt-get update && apt-get install -y \
    wget \
    && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xvzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib/ \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz \
    && apt-get remove -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p logs data

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 5000

# 启动命令
CMD ["python", "web_interface/app.py"]


# docker/nginx.conf
"""Nginx 配置文件"""

events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:5000;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # 静态文件
        location /static/ {
            alias /usr/share/nginx/html/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
        
        # API请求
        location /api/ {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # 其他请求
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}


# .gitignore
"""Git 忽略文件"""

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# 项目特定忽略
logs/
data/
*.db
*.sqlite
.DS_Store
.vscode/
.idea/
*.bak
*.tmp
*.temp

# 模型文件
models/*.pkl
models/*.h5
models/*.pt
models/*.pth

# 配置文件
config/local_*.py
config/production_*.py