
# quick_test.py - 快速测试脚本
"""快速测试脚本"""
import os
import sys
sys.path.append('..')

# 设置环境变量
os.environ['DATABASE_TYPE'] = 'sqlite'
os.environ['SQLITE_DB_PATH'] = 'data/quant_trading.db'
os.environ['ENVIRONMENT'] = 'development'
os.environ['DEBUG'] = 'True'
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['TUSHARE_TOKEN'] = '你的token'  # 请替换为实际token

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# 导入我们的模块
try:
    from config.database_config import db_config, system_config
    from data.storage.sqlite_database_manager import sqlite_db_manager, sqlite_data_repository
    print("✅ 模块导入成功")
except ImportError as e:
    print(f"❌ 模块导入失败: {e}")
    print("请确保文件结构正确")
    exit(1)

def test_database_setup():
    """测试数据库设置"""
    print("\n=== 测试数据库设置 ===")

    try:
        # 检查配置
        print(f"数据库类型: {db_config.database_type}")
        print(f"数据库路径: {db_config.get_sqlite_config().database}")
        print(f"Tushare Token: {'已配置' if system_config.tushare.token and system_config.tushare.token != '你的token' else '未配置'}")

        # 创建数据表
        sqlite_db_manager.create_tables()
        print("✅ 数据表创建成功")

        # 获取数据库信息
        db_info = sqlite_db_manager.get_database_info()
        print(f"SQLite版本: {db_info.get('version')}")
        print(f"数据库大小: {db_info.get('database_size_mb', 0):.2f} MB")

        return True

    except Exception as e:
        print(f"❌ 数据库设置失败: {e}")
        return False

def test_data_collection():
    """测试数据收集"""
    print("\n=== 测试数据收集 ===")

    # 检查token
    if not system_config.tushare.token or system_config.tushare.token == '你的token':
        print("❌ 请先配置Tushare Token")
        print("获取地址: https://tushare.pro/register")
        return False

    try:
        import tushare as ts

        # 设置token
        ts.set_token(system_config.tushare.token)
        pro = ts.pro_api()

        print("正在获取股票基本信息...")

        # 获取少量股票用于测试
        stock_basic = pro.stock_basic(exchange='', list_status='L')
        test_stocks = stock_basic.head(5)  # 只取5只股票

        print(f"获取到 {len(test_stocks)} 只测试股票")

        # 保存到数据库
        test_stocks.to_sql('stock_basic', sqlite_db_manager._engine,
                          if_exists='replace', index=False)
        print("✅ 股票基本信息保存成功")

        # 获取日线数据
        print("正在获取日线数据...")
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

        all_daily_data = []

        for i, (_, stock) in enumerate(test_stocks.iterrows()):
            ts_code = stock['ts_code']
            print(f"获取 {ts_code} 数据... ({i+1}/{len(test_stocks)})")

            try:
                daily_data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if not daily_data.empty:
                    # 重命名列
                    daily_data = daily_data.rename(columns={
                        'open': 'open_price',
                        'high': 'high_price',
                        'low': 'low_price',
                        'close': 'close_price'
                    })
                    all_daily_data.append(daily_data)
                    print(f"  获取 {len(daily_data)} 条记录")

                # 避免请求过频
                time.sleep(0.2)

            except Exception as e:
                print(f"  获取 {ts_code} 失败: {e}")
                continue

        # 保存所有日线数据
        if all_daily_data:
            combined_data = pd.concat(all_daily_data, ignore_index=True)
            combined_data.to_sql('daily_price', sqlite_db_manager._engine,
                               if_exists='replace', index=False)
            print(f"✅ 保存 {len(combined_data)} 条日线数据")

        return True

    except Exception as e:
        print(f"❌ 数据收集失败: {e}")
        return False

def test_data_query():
    """测试数据查询"""
    print("\n=== 测试数据查询 ===")

    try:
        # 获取股票基本信息
        stock_basic = sqlite_data_repository.get_stock_basic_info()
        print(f"股票基本信息: {len(stock_basic)} 条记录")

        if not stock_basic.empty:
            # 显示前几只股票
            print("前3只股票:")
            for _, stock in stock_basic.head(3).iterrows():
                print(f"  {stock['ts_code']} - {stock['name']}")

            # 获取第一只股票的日线数据
            first_stock = stock_basic.iloc[0]['ts_code']
            daily_data = sqlite_data_repository.get_daily_price(first_stock)

            if not daily_data.empty:
                print(f"\n{first_stock} 日线数据: {len(daily_data)} 条记录")
                latest = daily_data.iloc[-1]
                print(f"最新价格: {latest['close_price']:.2f}")
                print(f"涨跌幅: {latest.get('pct_chg', 0):.2f}%")

                # 测试缓存
                print("测试缓存功能...")
                start_time = time.time()
                cached_data = sqlite_data_repository.get_daily_price(first_stock)
                cache_time = time.time() - start_time
                print(f"缓存查询耗时: {cache_time:.4f}秒")

                return True
            else:
                print("❌ 没有日线数据")
                return False
        else:
            print("❌ 没有股票基本信息")
            return False

    except Exception as e:
        print(f"❌ 数据查询失败: {e}")
        return False

def test_technical_indicators():
    """测试技术指标计算"""
    print("\n=== 测试技术指标计算 ===")

    try:
        # 获取一只股票的数据
        stock_basic = sqlite_data_repository.get_stock_basic_info()
        if stock_basic.empty:
            print("❌ 没有股票数据")
            return False

        first_stock = stock_basic.iloc[0]['ts_code']
        daily_data = sqlite_data_repository.get_daily_price(first_stock)

        if daily_data.empty or len(daily_data) < 20:
            print("❌ 数据不足，无法计算技术指标")
            return False

        print(f"为 {first_stock} 计算技术指标...")

        # 计算简单移动平均
        daily_data['ma5'] = daily_data['close_price'].rolling(5).mean()
        daily_data['ma10'] = daily_data['close_price'].rolling(10).mean()
        daily_data['ma20'] = daily_data['close_price'].rolling(20).mean()

        # 计算RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        daily_data['rsi'] = calculate_rsi(daily_data['close_price'])

        # 计算布林带
        def calculate_bollinger_bands(prices, period=20, std_dev=2):
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, sma, lower_band

        daily_data['boll_upper'], daily_data['boll_mid'], daily_data['boll_lower'] = \
            calculate_bollinger_bands(daily_data['close_price'])

        # 显示最新指标
        latest = daily_data.iloc[-1]
        print(f"股票: {first_stock}")
        print(f"最新价格: {latest['close_price']:.2f}")
        print(f"MA5: {latest['ma5']:.2f}")
        print(f"MA10: {latest['ma10']:.2f}")
        print(f"MA20: {latest['ma20']:.2f}")
        print(f"RSI: {latest['rsi']:.2f}")
        print(f"布林带上轨: {latest['boll_upper']:.2f}")
        print(f"布林带下轨: {latest['boll_lower']:.2f}")

        # 简单交易信号
        if latest['ma5'] > latest['ma10'] > latest['ma20']:
            signal = "强烈买入"
        elif latest['ma5'] > latest['ma10']:
            signal = "买入"
        elif latest['ma5'] < latest['ma10'] < latest['ma20']:
            signal = "强烈卖出"
        elif latest['ma5'] < latest['ma10']:
            signal = "卖出"
        else:
            signal = "持有"

        print(f"交易信号: {signal}")

        # 保存技术指标到数据库
        tech_data = daily_data[['ts_code', 'trade_date', 'ma5', 'ma10', 'ma20', 'rsi',
                               'boll_upper', 'boll_mid', 'boll_lower']].copy()
        tech_data = tech_data.dropna()

        if not tech_data.empty:
            tech_data.to_sql('technical_indicators', sqlite_db_manager._engine,
                           if_exists='replace', index=False)
            print(f"✅ 保存 {len(tech_data)} 条技术指标数据")

        return True

    except Exception as e:
        print(f"❌ 技术指标计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_strategy():
    """测试简单策略"""
    print("\n=== 测试简单策略 ===")

    try:
        # 获取数据
        stock_basic = sqlite_data_repository.get_stock_basic_info()
        first_stock = stock_basic.iloc[0]['ts_code']
        daily_data = sqlite_data_repository.get_daily_price(first_stock)

        if len(daily_data) < 30:
            print("❌ 数据不足，无法进行策略测试")
            return False

        print(f"策略测试股票: {first_stock}")
        print(f"数据期间: {daily_data['trade_date'].min()} 到 {daily_data['trade_date'].max()}")

        # 计算移动平均线
        daily_data['ma5'] = daily_data['close_price'].rolling(5).mean()
        daily_data['ma20'] = daily_data['close_price'].rolling(20).mean()

        # 生成交易信号（均线交叉策略）
        daily_data['signal'] = 0
        daily_data.loc[daily_data['ma5'] > daily_data['ma20'], 'signal'] = 1  # 买入
        daily_data.loc[daily_data['ma5'] < daily_data['ma20'], 'signal'] = -1  # 卖出

        # 计算收益率
        daily_data['returns'] = daily_data['close_price'].pct_change()
        daily_data['strategy_returns'] = daily_data['returns'] * daily_data['signal'].shift(1)

        # 去除NaN
        clean_data = daily_data.dropna()

        if len(clean_data) < 10:
            print("❌ 清理后数据不足")
            return False

        # 计算累积收益
        cumulative_returns = (1 + clean_data['strategy_returns']).cumprod()
        benchmark_returns = (1 + clean_data['returns']).cumprod()

        # 计算绩效指标
        total_strategy_return = cumulative_returns.iloc[-1] - 1
        total_benchmark_return = benchmark_returns.iloc[-1] - 1
        excess_return = total_strategy_return - total_benchmark_return

        # 计算夏普比率
        strategy_std = clean_data['strategy_returns'].std()
        sharpe_ratio = clean_data['strategy_returns'].mean() / strategy_std * np.sqrt(252) if strategy_std > 0 else 0

        # 计算最大回撤
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 计算胜率
        winning_trades = (clean_data['strategy_returns'] > 0).sum()
        total_trades = (clean_data['strategy_returns'] != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # 显示结果
        print(f"策略总收益: {total_strategy_return:.2%}")
        print(f"基准收益: {total_benchmark_return:.2%}")
        print(f"超额收益: {excess_return:.2%}")
        print(f"夏普比率: {sharpe_ratio:.3f}")
        print(f"最大回撤: {abs(max_drawdown):.2%}")
        print(f"胜率: {win_rate:.2%}")
        print(f"交易次数: {total_trades}")

        if total_strategy_return > 0:
            print("✅ 策略产生正收益")
        else:
            print("⚠️ 策略产生负收益")

        return True

    except Exception as e:
        print(f"❌ 策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 A股量化交易系统 - 完整测试")
    print("=" * 50)

    # 运行所有测试
    tests = [
        ("数据库设置", test_database_setup),
        ("数据收集", test_data_collection),
        ("数据查询", test_data_query),
        ("技术指标", test_technical_indicators),
        ("简单策略", test_simple_strategy)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                print(f"✅ {test_name} 测试通过")
                passed += 1
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")

    # 测试总结
    print(f"\n{'='*50}")
    print(f"测试总结: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有测试通过！系统运行正常！")
        print("\n下一步你可以:")
        print("1. 运行更多股票的数据收集")
        print("2. 开发更复杂的交易策略")
        print("3. 搭建Web监控界面")
        print("4. 连接实盘交易接口")
    else:
        print("⚠️ 部分测试失败，请检查配置和依赖")

        if not system_config.tushare.token or system_config.tushare.token == '你的token':
            print("\n❗ 重要提醒: 请配置Tushare Token")
            print("1. 访问 https://tushare.pro/register 注册账号")
            print("2. 获取token后，修改 .env 文件或代码中的TUSHARE_TOKEN")

if __name__ == "__main__":
    main()


# requirements_sqlite.txt - SQLite版本的依赖文件
"""
# 核心数据处理
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0

# 数据库 (SQLite版本)
SQLAlchemy>=1.4.0
# psycopg2-binary>=2.9.0  # PostgreSQL不需要了
# 可选Redis支持
redis>=4.3.0

# 金融数据
tushare>=1.2.89

# 技术指标 (可选，如果安装困难可以跳过)
# TA-Lib>=0.4.25
pandas-ta>=0.3.14b0  # TA-Lib的替代品

# 基础工具
requests>=2.28.0
python-dotenv>=0.20.0
colorlog>=6.6.0

# Web框架 (后续需要)
Flask>=2.2.0

# 测试和开发工具
pytest>=7.1.0
"""