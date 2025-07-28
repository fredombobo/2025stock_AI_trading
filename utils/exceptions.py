"""Custom exception classes for the trading system."""
from __future__ import annotations
from datetime import datetime
from typing import Optional


class QuantSystemError(Exception):
    """Base exception for the quant trading system."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class DatabaseError(QuantSystemError):
    """Database related error."""


class ConnectionError(QuantSystemError):
    """Connection related error."""


class DataProviderError(QuantSystemError):
    """Data provider error."""


class DataCollectionError(QuantSystemError):
    """Data collection error."""


class CalculationError(QuantSystemError):
    """Calculation related error."""


class ValidationError(QuantSystemError):
    """Data validation error."""


class ConfigurationError(QuantSystemError):
    """Configuration error."""


class RateLimitError(QuantSystemError):
    """Rate limit exceeded error."""


class AuthenticationError(QuantSystemError):
    """Authentication failure."""


class StrategyError(QuantSystemError):
    """Strategy related error."""


class RiskManagementError(QuantSystemError):
    """Risk management error."""


class ExecutionError(QuantSystemError):
    """Trade execution error."""


class BacktestError(QuantSystemError):
    """Backtest error."""
