"""General helper functions used across the project."""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json
import hashlib
import os
import time
import logging

logger = logging.getLogger(__name__)


def calculate_returns(prices: pd.Series, method: str = "simple") -> pd.Series:
    """Calculate simple or log returns."""
    if method == "simple":
        return prices.pct_change()
    if method == "log":
        return np.log(prices / prices.shift(1))
    raise ValueError(f"不支持的收益率计算方法: {method}")


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.03) -> float:
    """Compute annualised Sharpe ratio from returns."""
    if returns.empty or returns.std() == 0:
        return 0.0

    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    return (annual_return - risk_free_rate) / annual_vol


def calculate_max_drawdown(cumulative_returns: pd.Series) -> Dict[str, Any]:
    """Calculate max drawdown statistics."""
    if cumulative_returns.empty:
        return {"max_drawdown": 0, "start_date": None, "end_date": None}

    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    max_idx = drawdown.idxmin()
    peak_idx = running_max.loc[:max_idx].idxmax()

    return {
        "max_drawdown": abs(max_drawdown),
        "start_date": peak_idx,
        "end_date": max_idx,
        "duration_days": (max_idx - peak_idx).days if isinstance(peak_idx, datetime) else None,
    }


def get_trading_dates(start_date: str, end_date: str, exchange: str = "SSE") -> List[str]:
    """Generate a list of trading dates. This is a simplified implementation."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def get_next_trading_date(date: str, days: int = 1) -> str:
    """Return the next trading date after *days* trading days."""
    current = datetime.strptime(date, "%Y-%m-%d")
    while days > 0:
        current += timedelta(days=1)
        if current.weekday() < 5:
            days -= 1
    return current.strftime("%Y-%m-%d")


def format_number(number: Union[int, float], decimals: int = 2) -> str:
    """Format large numbers using Chinese units."""
    if pd.isna(number):
        return "N/A"
    if abs(number) >= 1e8:
        return f"{number/1e8:.{decimals}f}亿"
    if abs(number) >= 1e4:
        return f"{number/1e4:.{decimals}f}万"
    return f"{number:.{decimals}f}"


def format_percentage(number: Union[int, float], decimals: int = 2) -> str:
    if pd.isna(number):
        return "N/A"
    return f"{number:.{decimals}f}%"


def calculate_correlation_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """Return correlation matrix for the given DataFrame."""
    return data.corr()


def generate_hash(data: Any) -> str:
    """Generate an MD5 hash for the given data."""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
    else:
        data_str = str(data)
    return hashlib.md5(data_str.encode("utf-8")).hexdigest()


def ensure_directory(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Perform division and guard against ZeroDivisionError."""
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def chunks(lst: List[Any], n: int):
    """Yield successive n-sized chunks from *lst*."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten nested dictionaries."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_memory_usage() -> Dict[str, str]:
    """Return current process memory usage information."""
    import psutil

    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss": format_number(memory_info.rss / 1024 / 1024, 2) + " MB",
        "vms": format_number(memory_info.vms / 1024 / 1024, 2) + " MB",
        "percent": f"{process.memory_percent():.2f}%",
    }


def timing_context(name: str):
    """Context manager for measuring a code block's execution time."""

    class TimingContext:
        def __init__(self, context_name: str):
            self.name = context_name
            self.start_time: Optional[float] = None

        def __enter__(self):
            self.start_time = time.time()
            logger.info("开始执行: %s", self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - (self.start_time or time.time())
            if exc_type is None:
                logger.info("完成执行: %s，耗时: %.2f秒", self.name, duration)
            else:
                logger.error("执行失败: %s，耗时: %.2f秒，错误: %s", self.name, duration, exc_val)

    return TimingContext(name)
