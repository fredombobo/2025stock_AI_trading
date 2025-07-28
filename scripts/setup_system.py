# scripts/setup_system.py
"""系统初始化脚本"""
import sys
import os
from datetime import datetime, timedelta
import argparse

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.storage.database_manager import db_manager
from data.collectors.historical_collector import historical_collector
from data.processors.technical_indicators import technical_indicators
from config.system_config import system_config
from utils.logger import get_logger

logger = get_logger(__name__)

class SystemSetup:
    """系统初始化器"""
    
    def __init__(self):
        self.db_manager = db_manager
        self.historical_collector = historical_collector
        self.technical_indicators = technical_indicators
    
    def setup_database(self, recreate: bool = False):
        """初始化数据库"""
        logger.info("开始初始化数据库...")
        
        try:
            if recreate:
                logger.warning("重新创建数据库表...")
                self.db_manager.drop_tables()
            
            # 创建表
            self.db_manager.create_tables()
            logger.info("数据库表创建成功")
            
            # 测试连接
            count = self.db_manager.get_table_row_count('stock_basic')
            logger.info(f"数据库连接测试成功，股票基本信息表记录数: {count}")
            
            return True
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            return False
    
    def setup_basic_data(self, force_update: bool = False):
        """初始化基础数据"""
        logger.info("开始初始化基础数据...")
        
        try:
            # 1. 收集股票基本信息
            logger.info("收集股票基本信息...")
            success = self.historical_collector.collect_stock_basic_info(force_update)
            if not success:
                logger.error("股票基本信息收集失败")
                return False
            
            # 2. 获取股票列表
            from data.storage.database_manager import data_repository
            stock_basic = data_repository.get_stock_basic_info()
            
            if stock_basic.empty:
                logger.error("未找到股票基本信息")
                return False
            
            logger.info(f"共找到 {len(stock_basic)} 只股票")
            
            # 3. 收集最近一年的日线数据（限制股票数量以避免超时）
            sample_stocks = stock_basic.head(100)['ts_code'].tolist()  # 先处理100只股票
            
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            logger.info(f"收集样本股票日线数据: {start_date} 到 {end_date}")
            success = self.historical_collector.collect_daily_price_range(
                start_date, end_date, sample_stocks, parallel=True
            )
            
            if not success:
                logger.warning("部分日线数据收集失败，但继续执行")
            
            logger.info("基础数据初始化完成")
            return True
            
        except Exception as e:
            logger.error(f"基础数据初始化失败: {e}")
            return False
    
    def setup_technical_indicators(self, limit: int = 50):
        """初始化技术指标"""
        logger.info(f"开始计算技术指标，限制股票数量: {limit}")
        
        try:
            from data.storage.database_manager import data_repository
            
            # 获取有日线数据的股票
            sql = "SELECT DISTINCT ts_code FROM daily_price ORDER BY ts_code LIMIT :limit"
            result = self.db_manager.execute_sql(sql, {'limit': limit})
            ts_codes = [row[0] for row in result]
            
            if not ts_codes:
                logger.warning("未找到有日线数据的股票")
                return False
            
            logger.info(f"为 {len(ts_codes)} 只股票计算技术指标")
            
            # 批量计算技术指标
            results = self.technical_indicators.batch_calculate_indicators(ts_codes)
            
            success_count = sum(results.values())
            logger.info(f"技术指标计算完成: 成功 {success_count}/{len(ts_codes)}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"技术指标初始化失败: {e}")
            return False
    
    def run_full_setup(self, recreate_db: bool = False, force_data_update: bool = False):
        """完整系统初始化"""
        logger.info("开始完整系统初始化...")
        
        steps = [
            ("数据库初始化", lambda: self.setup_database(recreate_db)),
            ("基础数据初始化", lambda: self.setup_basic_data(force_data_update)),
            ("技术指标初始化", lambda: self.setup_technical_indicators(50))
        ]
        
        for step_name, step_func in steps:
            logger.info(f"执行步骤: {step_name}")
            try:
                success = step_func()
                if success:
                    logger.info(f"✓ {step_name} 完成")
                else:
                    logger.error(f"✗ {step_name} 失败")
                    return False
            except Exception as e:
                logger.error(f"✗ {step_name} 异常: {e}")
                return False
        
        logger.info("🎉 系统初始化完成！")
        return True
    
    def verify_system(self):
        """验证系统状态"""
        logger.info("验证系统状态...")
        
        try:
            # 检查表是否存在
            tables = [
                'stock_basic', 'daily_price', 'technical_indicators',
                'financial_data', 'market_news', 'data_update_log'
            ]
            
            for table in tables:
                try:
                    count = self.db_manager.get_table_row_count(table)
                    logger.info(f"表 {table}: {count} 条记录")
                except Exception as e:
                    logger.warning(f"表 {table} 检查失败: {e}")
            
            # 检查Redis连接
            try:
                redis_client = self.db_manager.get_redis_client()
                redis_client.ping()
                logger.info("✓ Redis连接正常")
            except Exception as e:
                logger.error(f"✗ Redis连接失败: {e}")
            
            # 检查Tushare连接
            try:
                from data.providers.tushare_provider import tushare_provider
                test_data = tushare_provider.get_stock_basic()
                if not test_data.empty:
                    logger.info("✓ Tushare连接正常")
                else:
                    logger.warning("⚠ Tushare连接可能有问题")
            except Exception as e:
                logger.error(f"✗ Tushare连接失败: {e}")
            
            logger.info("系统状态验证完成")
            
        except Exception as e:
            logger.error(f"系统验证失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='量化交易系统初始化脚本')
    parser.add_argument('--recreate-db', action='store_true', help='重新创建数据库表')
    parser.add_argument('--force-data-update', action='store_true', help='强制更新基础数据')
    parser.add_argument('--verify-only', action='store_true', help='仅验证系统状态')
    parser.add_argument('--setup-db-only', action='store_true', help='仅初始化数据库')
    
    args = parser.parse_args()
    
    # 检查必要的环境变量
    if not system_config.tushare.token:
        logger.error("请先设置TUSHARE_TOKEN environment variable")
        sys.exit(1)
    
    setup = SystemSetup()
    
    try:
        if args.verify_only:
            setup.verify_system()
        elif args.setup_db_only:
            success = setup.setup_database(args.recreate_db)
            sys.exit(0 if success else 1)
        else:
            success = setup.run_full_setup(args.recreate_db, args.force_data_update)
            
            if success:
                setup.verify_system()
                logger.info("✅ 系统初始化成功！你现在可以开始使用量化交易系统了。")
            else:
                logger.error("❌ 系统初始化失败！请检查错误日志。")
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("用户中断了初始化过程")
        sys.exit(1)
    except Exception as e:
        logger.error(f"初始化过程中发生未知错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()


# requirements.txt
"""项目依赖包列表"""

# 数据处理
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0

# 数据库
SQLAlchemy>=1.4.0
psycopg2-binary>=2.9.0
redis>=4.3.0
alembic>=1.8.0

# 金融数据
tushare>=1.2.89
akshare>=1.8.0
yfinance>=0.1.87

# 技术指标
TA-Lib>=0.4.25

# 机器学习
scikit-learn>=1.1.0
xgboost>=1.6.0
lightgbm>=3.3.0

# 深度学习
torch>=1.12.0
tensorflow>=2.9.0
transformers>=4.21.0

# Web框架
Flask>=2.2.0
Flask-SQLAlchemy>=2.5.1
Flask-CORS>=3.0.10
gunicorn>=20.1.0

# API框架
fastapi>=0.85.0
uvicorn>=0.18.0

# 异步处理
celery>=5.2.0
redis>=4.3.0

# 数据可视化
plotly>=5.10.0
matplotlib>=3.5.0
seaborn>=0.11.0

# 工具库
requests>=2.28.0
beautifulsoup4>=4.11.0
lxml>=4.9.0
schedule>=1.1.0
python-dotenv>=0.20.0
pydantic>=1.10.0
click>=8.1.0

# 数值计算和统计
statsmodels>=0.13.2
pykalman>=0.9.5

# 时间序列
pmdarima>=2.0.0

# 性能监控
psutil>=5.9.0
memory-profiler>=0.60.0

# 测试
pytest>=7.1.0
pytest-cov>=3.0.0
pytest-mock>=3.8.0

# 代码质量
black>=22.6.0
flake8>=5.0.0
mypy>=0.971

# 文档
sphinx>=5.1.0
sphinx-rtd-theme>=1.0.0

# 配置管理
python-decouple>=3.6

# 日志
colorlog>=6.6.0

# 加密
cryptography>=37.0.4

# 并发处理
concurrent-futures>=3.1.1

# 数据序列化
msgpack>=1.0.4
pickle5>=0.0.12

# HTTP客户端
httpx>=0.23.0

# 环境变量管理
python-dotenv>=0.20.0


# README.md
"""项目README文档"""

# A股量化交易系统

## 项目简介

这是一个专业级的A股量化交易系统，集成了现代AI技术和传统量化分析方法。系统采用模块化设计，支持多策略并行、实时数据处理、智能决策和风险管理。

## 主要特性

### 🚀 核心功能
- **多数据源支持**: Tushare、AKShare、Wind等主流数据源
- **实时数据处理**: 支持实时行情数据收集和处理
- **技术指标计算**: 内置50+种技术指标，支持自定义指标
- **AI增强决策**: 集成大语言模型进行市场分析和决策支持
- **多策略框架**: 支持技术分析、基本面分析、量价策略等
- **智能风控**: 实时风险监控、动态止损、组合风险管理
- **高性能回测**: 向量化回测引擎，支持多策略并行回测

### 🛠 技术架构
- **数据库**: PostgreSQL + Redis + ClickHouse(可选)
- **后端**: Python + FastAPI/Flask
- **前端**: React/Vue.js + ECharts/Plotly
- **AI**: PyTorch/TensorFlow + Transformers + OpenAI API
- **消息队列**: Celery + RabbitMQ/Redis
- **容器化**: Docker + Docker Compose

## 快速开始

### 环境要求

- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Node.js 16+ (前端开发)

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/your-repo/a-stock-quant-system.git
cd a-stock-quant-system
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，配置以下变量：
# TUSHARE_TOKEN=your_tushare_token
# POSTGRES_HOST=localhost
# POSTGRES_DB=quant_trading
# POSTGRES_USER=postgres
# POSTGRES_PASSWORD=your_password
# REDIS_HOST=localhost
```

5. **初始化系统**
```bash
python scripts/setup_system.py
```

### Docker 部署

1. **使用 Docker Compose**
```bash
docker-compose up -d
```

2. **初始化数据**
```bash
docker-compose exec app python scripts/setup_system.py
```

## 使用指南

### 数据收集

```python
from data.collectors.historical_collector import historical_collector

# 收集股票基本信息
historical_collector.collect_stock_basic_info()

# 收集日线数据
historical_collector.collect_daily_price_range(
    start_date='2023-01-01',
    end_date='2024-01-01',
    ts_codes=['000001.SZ', '000002.SZ']
)

# 增量数据更新
historical_collector.collect_incremental_daily_data(days_back=5)
```

### 技术指标计算

```python
from data.processors.technical_indicators import technical_indicators

# 计算单只股票的所有技术指标
indicators = technical_indicators.calculate_all_indicators('000001.SZ')

# 获取技术指标信号分析
signals = technical_indicators.get_signal_analysis('000001.SZ')

# 批量计算多只股票的技术指标
results = technical_indicators.batch_calculate_indicators(['000001.SZ', '000002.SZ'])
```

### 实时数据订阅

```python
from data.collectors.realtime_collector import realtime_collector

# 启动实时数据收集
realtime_collector.start_collection()

# 订阅股票实时数据
def price_callback(data):
    print(f"股票价格更新: {data}")

realtime_collector.subscribe_stock('000001.SZ', callback=price_callback)

# 获取最新价格
latest_price = realtime_collector.get_latest_price('000001.SZ')
```

### 策略开发示例

```python
from strategies.base.strategy_base import StrategyBase
from strategies.base.signal_generator import SignalGenerator

class MAStrategy(StrategyBase):
    """均线策略示例"""
    
    def __init__(self, short_window=5, long_window=20):
        super().__init__()
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        # 计算短期和长期均线
        data['ma_short'] = data['close_price'].rolling(self.short_window).mean()
        data['ma_long'] = data['close_price'].rolling(self.long_window).mean()
        
        # 生成交易信号
        signals = SignalGenerator.crossover_signals(
            data['ma_short'], 
            data['ma_long']
        )
        
        return signals
```

## 系统监控

### 启动Web界面
```bash
cd web_interface
python app.py
```

访问 http://localhost:5000 查看系统监控面板

### 主要监控指标
- 数据更新状态
- 策略运行状态
- 系统性能指标
- 风险控制状态

## 开发指南

### 项目结构
```
a_stock_quant_system/
├── config/              # 配置文件
├── data/               # 数据模块
│   ├── providers/      # 数据提供商
│   ├── collectors/     # 数据收集器
│   ├── processors/     # 数据处理器
│   └── storage/        # 数据存储
├── models/             # AI模型
├── strategies/         # 交易策略
├── execution/          # 交易执行
├── risk_management/    # 风险管理
├── backtesting/        # 回测系统
├── monitoring/         # 监控模块
├── web_interface/      # Web界面
├── utils/              # 工具模块
└── tests/              # 测试用例
```

### 添加新的数据提供商

1. 继承 `DataProviderBase` 类
2. 实现必要的接口方法
3. 在配置中注册新的提供商

### 开发新策略

1. 继承 `StrategyBase` 类
2. 实现 `generate_signals` 方法
3. 可选实现 `risk_management` 方法
4. 编写单元测试

### 代码规范

- 使用 Black 进行代码格式化
- 使用 Flake8 进行代码检查
- 编写详细的文档字符串
- 为核心功能编写单元测试

## 性能优化

### 数据处理优化
- 使用向量化操作替代循环
- 合理使用缓存机制
- 数据库查询优化
- 并行处理大批量数据

### 内存管理
- 及时释放大对象
- 使用生成器处理大数据集
- 监控内存使用情况

### 计算优化
- 使用NumPy/Pandas优化计算
- GPU加速深度学习模型
- 分布式计算支持

## 风险提示

⚠️ **重要提示**: 
- 本系统供学习和研究使用
- 实盘交易需要充分测试和风险评估
- 投资有风险，请谨慎操作
- 建议先在模拟环境中验证策略

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建功能分支
3. 提交变更
4. 推送到分支
5. 创建 Pull Request

## 联系方式

- 项目主页: https://github.com/your-repo/a-stock-quant-system
- 问题反馈: https://github.com/your-repo/a-stock-quant-system/issues
- 邮箱: your-email@example.com

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 完整的数据收集和处理系统
- 基础技术指标计算
- 简单策略框架

---

⭐ 如果这个项目对你有帮助，请给个 Star！