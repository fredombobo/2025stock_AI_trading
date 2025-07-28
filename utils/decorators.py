"""Common decorator utilities."""
from __future__ import annotations
import functools
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Type, Tuple

logger = logging.getLogger(__name__)


def retry(max_attempts: int = 3, delay: float = 1.0,
          backoff: float = 2.0,
          exceptions: Tuple[Type[Exception], ...] = (Exception,)) -> Callable:
    """Retry calling *func* in case of specified exceptions."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # pragma: no cover - just logging
                    if attempt == max_attempts - 1:
                        logger.error(
                            "函数 %s 重试 %s 次后仍然失败: %s",
                            func.__name__, max_attempts, exc,
                        )
                        raise

                    logger.warning(
                        "函数 %s 第 %s 次尝试失败，%.2f秒后重试: %s",
                        func.__name__, attempt + 1, current_delay, exc,
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """Decorator that logs the execution time of *func*."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info("函数 %s 执行完成，耗时: %.2f秒", func.__name__, duration)
            return result
        except Exception as exc:  # pragma: no cover - just logging
            duration = time.time() - start_time
            logger.error("函数 %s 执行失败，耗时: %.2f秒，错误: %s", func.__name__, duration, exc)
            raise
    return wrapper


def rate_limit(calls_per_minute: int = 60) -> Callable:
    """Rate-limit decorator."""

    def decorator(func: Callable) -> Callable:
        call_times: list[datetime] = []
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = datetime.now()
                call_times[:] = [t for t in call_times if now - t < timedelta(minutes=1)]
                if len(call_times) >= calls_per_minute:
                    wait_time = 60 - (now - call_times[0]).total_seconds()
                    if wait_time > 0:
                        logger.warning("函数 %s 达到速率限制，等待 %.2f 秒", func.__name__, wait_time)
                        time.sleep(wait_time + 0.1)
                call_times.append(now)
                return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_result(expire_seconds: int = 3600) -> Callable:
    """Cache return value of *func* for given seconds."""

    def decorator(func: Callable) -> Callable:
        cache: dict[str, Any] = {}
        cache_times: dict[str, float] = {}
        lock = threading.Lock()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = str(args) + str(sorted(kwargs.items()))
            with lock:
                now = time.time()
                if (
                    cache_key in cache
                    and cache_key in cache_times
                    and now - cache_times[cache_key] < expire_seconds
                ):
                    logger.debug("函数 %s 使用缓存结果", func.__name__)
                    return cache[cache_key]

                result = func(*args, **kwargs)
                cache[cache_key] = result
                cache_times[cache_key] = now

                expired = [k for k, t in cache_times.items() if now - t >= expire_seconds]
                for key in expired:
                    cache.pop(key, None)
                    cache_times.pop(key, None)
                return result
        return wrapper
    return decorator


def validate_params(**param_validators: Callable[[Any], bool]) -> Callable:
    """Validate parameters of *func* using provided validators."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for name, validator in param_validators.items():
                if name in bound_args.arguments:
                    value = bound_args.arguments[name]
                    if not validator(value):
                        raise ValueError(f"参数 {name} 验证失败: {value}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def singleton(cls: type) -> Callable:
    """Simple singleton decorator."""
    instances: dict[type, Any] = {}
    lock = threading.Lock()

    @functools.wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return wrapper
