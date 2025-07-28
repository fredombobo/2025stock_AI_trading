import numpy as np
import pandas as pd

class VaRCalculator:
    """Simple historical Value at Risk (VaR) calculator."""

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def calculate(self, returns: pd.Series) -> float:
        """Calculate the VaR for a series of returns."""
        if returns is None or len(returns) == 0:
            return 0.0
        quantile = np.quantile(returns, 1 - self.confidence_level)
        return float(-quantile)
