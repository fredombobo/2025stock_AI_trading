"""Validation helper functions."""
from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Union, List, Optional
import pandas as pd


def is_valid_ts_code(ts_code: str) -> bool:
    """Validate stock code format e.g. '000001.SZ'."""
    if not isinstance(ts_code, str):
        return False
    pattern = r'^\d{6}\.(SH|SZ|BJ)$'
    return bool(re.match(pattern, ts_code))


def is_valid_date(date_str: str) -> bool:
    """Validate several common date formats."""
    if not isinstance(date_str, str):
        return False
    formats = ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']
    for fmt in formats:
        try:
            datetime.strptime(date_str, fmt)
            return True
        except ValueError:
            continue
    return False


def is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0
    except (ValueError, TypeError):
        return False


def is_non_negative_number(value: Any) -> bool:
    try:
        return float(value) >= 0
    except (ValueError, TypeError):
        return False


def is_valid_price(price: Any) -> bool:
    try:
        price_val = float(price)
        return 0 < price_val < 10000
    except (ValueError, TypeError):
        return False


def is_valid_volume(volume: Any) -> bool:
    try:
        vol_val = int(volume)
        return vol_val >= 0
    except (ValueError, TypeError):
        return False


def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
    if not isinstance(df, pd.DataFrame):
        return False
    missing_columns = set(required_columns) - set(df.columns)
    return len(missing_columns) == 0


def validate_price_data(df: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    if df.empty:
        errors.append("数据为空")
        return errors
    required_columns = ['ts_code', 'trade_date', 'open_price', 'high_price', 'low_price', 'close_price']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"缺少必需列: {missing_columns}")
        return errors
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        errors.append(f"存在空值: {null_counts[null_counts > 0].to_dict()}")
    invalid_prices = df[
        ~((df['high_price'] >= df[['open_price', 'close_price', 'low_price']].max(axis=1)) &
          (df['low_price'] <= df[['open_price', 'close_price', 'high_price']].min(axis=1)))
    ]
    if not invalid_prices.empty:
        errors.append(f"存在 {len(invalid_prices)} 条价格逻辑错误的记录")
    price_columns = ['open_price', 'high_price', 'low_price', 'close_price']
    for col in price_columns:
        invalid_price_count = ((df[col] <= 0) | (df[col] > 10000)).sum()
        if invalid_price_count > 0:
            errors.append(f"{col} 存在 {invalid_price_count} 个异常值")
    return errors


def normalize_date_format(date_str: str, output_format: str = '%Y-%m-%d') -> Optional[str]:
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
    numeric_series = pd.to_numeric(series, errors='coerce')
    numeric_series = numeric_series.replace([float('inf'), float('-inf')], pd.NA)
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
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (series < lower) | (series > upper)
    if method == 'zscore':
        mean = series.mean()
        std = series.std()
        z_scores = abs((series - mean) / std)
        return z_scores > threshold
    raise ValueError(f"不支持的异常值检测方法: {method}")
