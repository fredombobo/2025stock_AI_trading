class DrawdownControl:
    """Monitor and enforce maximum drawdown limits."""

    def __init__(self, max_drawdown: float = 0.2):
        self.max_drawdown = max_drawdown

    def check_drawdown(self, current_drawdown: float) -> bool:
        """Return True if the current drawdown is within the allowed range."""
        return current_drawdown <= self.max_drawdown
