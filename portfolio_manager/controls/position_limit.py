class PositionLimitControl:
    """Check position and sector weight limits."""

    def __init__(self, max_position_weight: float = 0.1, max_sector_weight: float = 0.3):
        self.max_position_weight = max_position_weight
        self.max_sector_weight = max_sector_weight

    def check_position_weight(self, weight: float) -> bool:
        """Return True if the given position weight is allowed."""
        return weight <= self.max_position_weight

    def check_sector_weight(self, weight: float) -> bool:
        """Return True if the given sector weight is allowed."""
        return weight <= self.max_sector_weight
