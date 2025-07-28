# utils/logger.py
"""日志工具模块"""
import logging
import logging.handlers
import sys
import os
from datetime import datetime
from typing import Optional
from config.system_config import system_config

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'ENDC': '\033[0m'       # 结束颜色
    }
    
    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['ENDC']}"
        
        return super().format(record)

def setup_logger(log_dir: str = "logs"):
    """设置日志系统"""
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, system_config.log_level.upper()))
    
    # 清除现有处理器
    root_logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    colored_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(colored_formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器 - 所有日志
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'quant_system.log'),
        maxBytes=50*1024*1024,  # 50MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # 错误日志文件处理器
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, 'error.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)
    
def get_logger(name: str) -> logging.Logger:
    """获取日志器"""
    return logging.getLogger(name)

# 初始化日志系统
setup_logger()


# utils/decorators.py
"""装饰器模块"""
import functools
import time
import threading
from typing import Any, Callable, Optional, Type, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def retry(max_attempts: int = 3, delay: float = 1.0, 
          backoff: float = 2.0, exceptions: Tuple[Type[Exception], ...] = (Exception,)):
    """重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟倍增因子
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        logger.error(f"函数 {func.__name__} 重试 {max_attempts} 次后仍然失败: {e}")
                        raise
                    
                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败，{current_delay:.2f}秒后重试: {e}")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator

def log_execution_time(func: Callable) -> Callable:
    """记录函数执行时间装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.info(f"函数 {func.__name__} 执行完成，耗时: {execution_time:.2f}秒")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"函数 {func.__name__} 执行失败，耗时: {execution_time:.2f}秒，错误: {e}")
            raise
    
    return wrapper

def rate_limit(calls_per_minute: int = 60):
    """速率限制装饰器"""
    def decorator(func: Callable) -> Callable:
        call_times = []
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with lock:
                now = datetime.now()
                
                # 清理超过一分钟的调用记录
                call_times[:] = [t for t in call_times if now - t < timedelta(minutes=1)]
                
                # 检查是否超过限制
                if len(call_times) >= calls_per_minute:
                    wait_time = 60 - (now - call_times[0]).total_seconds()
                    if wait_time > 0:
                        logger.warning(f"函数 {func.__name__} 达到速率限制，等待 {wait_time:.2f} 秒")
                        time.sleep(wait_time + 0.1)
                
                # 记录当前调用时间
                call_times.append(now)
                
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def cache_result(expire_seconds: int = 3600):
    """结果缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_times = {}
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 生成缓存键
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            with lock:
                now = time.time()
                
                # 检查缓存是否存在且未过期
                if (cache_key in cache and 
                    cache_key in cache_times and 
                    now - cache_times[cache_key] < expire_seconds):
                    
                    logger.debug(f"函数 {func.__name__} 使用缓存结果")
                    return cache[cache_key]
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                cache[cache_key] = result
                cache_times[cache_key] = now
                
                # 清理过期缓存
                expired_keys = [k for k, t in cache_times.items() if now - t >= expire_seconds]
                for key in expired_keys:
                    cache.pop(key, None)
                    cache_times.pop(key, None)
                
                return result
        
        return wrapper
    return decorator

def validate_params(**param_validators):
    """参数验证装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 获取函数参数名
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # 验证参数
            for param_name, validator in param_validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"参数 {param_name} 验证失败: {value}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def singleton(cls):
    """单例模式装饰器"""
    instances = {}
    lock = threading.Lock()
    
    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return wrapper


# utils/exceptions.py
"""自定义异常模块"""

class QuantSystemError(Exception):
    """量化系统基础异常"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details,
            'timestamp': self.timestamp
        }

class DatabaseError(QuantSystemError):
    """数据库相关异常"""
    pass

class ConnectionError(QuantSystemError):
    """连接相关异常"""
    pass

class DataProviderError(QuantSystemError):
    """数据提供商异常"""
    pass

class DataCollectionError(QuantSystemError):
    """数据收集异常"""
    pass

class CalculationError(QuantSystemError):
    """计算相关异常"""
    pass

class ValidationError(QuantSystemError):
    """数据验证异常"""
    pass

class ConfigurationError(QuantSystemError):
    """配置相关异常"""
    pass

class RateLimitError(QuantSystemError):
    """速率限制异常"""
    pass

class AuthenticationError(QuantSystemError):
    """认证异常"""
    pass

class StrategyError(QuantSystemError):
    """策略相关异常"""
    pass

class RiskManagementError(QuantSystemError):
    """风险管理异常"""
    pass

class ExecutionError(QuantSystemError):
    """交易执行异常"""
    pass

class BacktestError(QuantSystemError):
    """回测相关异常"""
    pass


# utils/validators.py
"""数据验证工具"""
import re
import pandas as pd
from datetime import datetime
from typing import Any, Union, List, Optional

def is_valid_ts_code(ts_code: str) -> bool:
    """验证股票代码格式"""
    if not isinstance(ts_code, str):
        return False
    
    # A股代码格式: 6位数字.交易所代码
    pattern = r'^\d{6}\.(SH|SZ|BJ)
    return bool(re.match(pattern, ts_code))

def is_valid_date(date_str: str) -> bool:
    """验证日期格式"""
    if not isinstance(date_str, str):
        return False
    
    try:
        # 支持多种日期格式
        formats = ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']
        for fmt in formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
        return False
    except:
        return False

def is_positive_number(value: Any) -> bool:
    """验证是否为正数"""
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False

def is_non_negative_number(value: Any) -> bool:
    """验证是否为非负数"""
    try:
        return float(value) >= 0
    except (ValueError, TypeError):
        return False

def is_valid_price(price: Any) -> bool:
    """验证价格是否有效"""
    try:
        price_val = float(price)
        return price_val > 0 and price_val < 10000  # 价格应该在合理范围内
    except (ValueError, TypeError):
        return False

def is_valid_volume(volume: Any) -> bool:
    """验证成交量是否有效"""
    try:
        vol_val = int(volume)
        return vol_val >= 0
    except (ValueError, TypeError):
        return False

def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """验证DataFrame是否包含必需的列"""
    if not isinstance(df, pd.DataFrame):
        return False
    
    missing_columns = set(required_columns) - set(df.columns)
    return len(missing_columns) == 0

def validate_price_data(df: pd.DataFrame) -> List[str]:
    """验证价格数据的完整性和逻辑性"""
    errors = []
    
    if df.empty:
        errors.append("数据为空")
        return errors
    
    # 检查必需列
    required_columns = ['ts_code', 'trade_date', 'open_price', 'high_price', 'low_price', 'close_price']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"缺少必需列: {missing_columns}")
        return errors
    
    # 检查空值
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        errors.append(f"存在空值: {null_counts[null_counts > 0].to_dict()}")
    
    # 检查价格逻辑
    invalid_prices = df[
        ~((df['high_price'] >= df[['open_price', 'close_price', 'low_price']].max(axis=1)) &
          (df['low_price'] <= df[['open_price', 'close_price', 'high_price']].min(axis=1)))
    ]
    
    if not invalid_prices.empty:
        errors.append(f"存在 {len(invalid_prices)} 条价格逻辑错误的记录")
    
    # 检查价格范围
    price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
    for col in price_columns:
        invalid_price_count = ((df[col] <= 0) | (df[col] > 10000)).sum()
        if invalid_price_count > 0:
            errors.append(f"{col} 存在 {invalid_price_count} 个异常值")
    
    return errors

def normalize_date_format(date_str: str, output_format: str = '%Y-%m-%d') -> Optional[str]:
    """标准化日期格式"""
    if not isinstance(date_str, str):
        return None
    
    input_formats = ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d', '%m/%d/%Y']
    
    for fmt in input_formats:
        try:
            date_obj = datetime.strptime(date_str, fmt)
            return date_obj.strftime(output_format)
        except ValueError:
            continue
    
    return None

def clean_numeric_data(series: pd.Series, fill_method: str = 'forward') -> pd.Series:
    """清理数值数据"""
    # 转换为数值类型
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # 处理无穷大值
    numeric_series = numeric_series.replace([float('inf'), float('-inf')], pd.NA)
    
    # 填充缺失值
    if fill_method == 'forward':
        numeric_series = numeric_series.fillna(method='ffill')
    elif fill_method == 'backward':
        numeric_series = numeric_series.fillna(method='bfill')
    elif fill_method == 'zero':
        numeric_series = numeric_series.fillna(0)
    elif fill_method == 'mean':
        numeric_series = numeric_series.fillna(numeric_series.mean())
    
    return numeric_series

def detect_outliers(series: pd.Series, method: str = 'iqr', threshold: float = 1.5) -> pd.Series:
    """检测异常值"""
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        mean = series.mean()
        std = series.std()
        z_scores = abs((series - mean) / std)
        return z_scores > threshold
    
    else:
        raise ValueError(f"不支持的异常值检测方法: {method}")


# utils/helpers.py
"""辅助函数模块"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json
import hashlib
import os

def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
    """计算收益率"""
    if method == 'simple':
        return prices.pct_change()
    elif method == 'log':
        return np.log(prices / prices.shift(1))
    else:
        raise ValueError(f"不支持的收益率计算方法: {method}")

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """计算夏普比率"""
    if returns.empty or returns.std() == 0:
        return 0.0
    
    # 年化收益率
    annual_return = returns.mean() * 252
    
    # 年化波动率
    annual_vol = returns.std() * np.sqrt(252)
    
    # 夏普比率
    sharpe = (annual_return - risk_free_rate) / annual_vol
    
    return sharpe

def calculate_max_drawdown(cumulative_returns: pd.Series) -> Dict[str, Any]:
    """计算最大回撤"""
    if cumulative_returns.empty:
        return {'max_drawdown': 0, 'start_date': None, 'end_date': None}
    
    # 计算累积最高值
    running_max = cumulative_returns.expanding().max()
    
    # 计算回撤
    drawdown = (cumulative_returns - running_max) / running_max
    
    # 找到最大回撤
    max_drawdown = drawdown.min()
    max_drawdown_idx = drawdown.idxmin()
    
    # 找到最大回撤的开始时间（peak）
    peak_idx = running_max.loc[:max_drawdown_idx].idxmax()
    
    return {
        'max_drawdown': abs(max_drawdown),
        'start_date': peak_idx,
        'end_date': max_drawdown_idx,
        'duration_days': (max_drawdown_idx - peak_idx).days if isinstance(peak_idx, datetime) else None
    }

def get_trading_dates(start_date: str, end_date: str, 
                     exchange: str = 'SSE') -> List[str]:
    """获取交易日列表"""
    # 这里应该从数据库获取交易日历，简化实现
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    dates = []
    current = start
    
    while current <= end:
        # 简单排除周末（实际应该查询交易日历）
        if current.weekday() < 5:  # 0-4表示周一到周五
            dates.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return dates

def get_next_trading_date(date: str, days: int = 1) -> str:
    """获取下一个交易日"""
    current = datetime.strptime(date, '%Y-%m-%d')
    
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:  # 排除周末
            days -= 1
    
    return current.strftime('%Y-%m-%d')

def format_number(number: Union[int, float], decimals: int = 2) -> str:
    """格式化数字显示"""
    if pd.isna(number):
        return 'N/A'
    
    if abs(number) >= 1e8:  # 亿
        return f"{number/1e8:.{decimals}f}亿"
    elif abs(number) >= 1e4:  # 万
        return f"{number/1e4:.{decimals}f}万"
    else:
        return f"{number:.{decimals}f}"

def format_percentage(number: Union[int, float], decimals: int = 2) -> str:
    """格式化百分比显示"""
    if pd.isna(number):
        return 'N/A'
    
    return f"{number:.{decimals}f}%"

def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """计算相关性矩阵"""
    return data.corr()

def generate_hash(data: Any) -> str:
    """生成数据哈希值"""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode('utf-8')).hexdigest()

def ensure_directory(path: str) -> None:
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法，避免除零错误"""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default

def chunks(lst: List[Any], n: int) -> List[List[Any]]:
    """将列表分成指定大小的块"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """展平嵌套字典"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_memory_usage() -> Dict[str, str]:
    """获取内存使用情况"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss': format_number(memory_info.rss / 1024 / 1024, 2) + ' MB',
        'vms': format_number(memory_info.vms / 1024 / 1024, 2) + ' MB',
        'percent': f"{process.memory_percent():.2f}%"
    }

def timing_context(name: str):
    """计时上下文管理器"""
    class TimingContext:
        def __init__(self, name: str):
            self.name = name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            logger.info(f"开始执行: {self.name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            if exc_type is None:
                logger.info(f"完成执行: {self.name}，耗时: {duration:.2f}秒")
            else:
                logger.error(f"执行失败: {self.name}，耗时: {duration:.2f}秒，错误: {exc_val}")
    
    return TimingContext(name)