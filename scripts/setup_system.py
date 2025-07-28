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

