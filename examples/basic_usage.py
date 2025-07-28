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


