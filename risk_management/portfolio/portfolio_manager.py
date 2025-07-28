# risk_management/portfolio/portfolio_manager.py
"""Simple portfolio management utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from data.storage.database_manager import data_repository
from utils.logger import get_logger
from utils.helpers import calculate_sharpe_ratio, calculate_max_drawdown

logger = get_logger(__name__)


@dataclass
class Position:
    """Represents a single stock position."""

    ts_code: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight: float
    entry_date: datetime

    def to_dict(self) -> Dict[str, object]:
        return {
            "ts_code": self.ts_code,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "weight": self.weight,
            "entry_date": self.entry_date.isoformat(),
        }


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a given time."""

    timestamp: datetime
    total_value: float
    cash: float
    positions_value: float
    total_pnl: float
    daily_return: float
    positions: List[Position]
    risk_metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_value": self.total_value,
            "cash": self.cash,
            "positions_value": self.positions_value,
            "total_pnl": self.total_pnl,
            "daily_return": self.daily_return,
            "positions": [p.to_dict() for p in self.positions],
            "risk_metrics": self.risk_metrics,
        }


class PortfolioManager:
    """Manage current portfolio holdings and historical snapshots."""

    def __init__(
        self,
        initial_capital: float = 1_000_000,
        max_position_weight: float = 0.1,
        max_sector_weight: float = 0.3,
        cash_reserve_ratio: float = 0.05,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.max_position_weight = max_position_weight
        self.max_sector_weight = max_sector_weight
        self.cash_reserve_ratio = cash_reserve_ratio

        self.positions: Dict[str, Position] = {}
        self.snapshots: List[PortfolioSnapshot] = []
        self._last_value: Optional[float] = None

    # ------------------------------------------------------------------
    # Basic portfolio operations
    # ------------------------------------------------------------------
    def add_position(
        self,
        ts_code: str,
        quantity: int,
        price: float,
        entry_date: Optional[datetime] = None,
    ) -> None:
        """Open or increase a position."""
        if quantity <= 0 or price <= 0:
            raise ValueError("quantity and price must be positive")

        cost = quantity * price
        min_cash = self.initial_capital * self.cash_reserve_ratio
        if self.cash - cost < min_cash:
            raise ValueError("not enough available cash to open position")

        if ts_code in self.positions:
            pos = self.positions[ts_code]
            new_qty = pos.quantity + quantity
            new_avg = (pos.avg_cost * pos.quantity + cost) / new_qty
            pos.quantity = new_qty
            pos.avg_cost = new_avg
            pos.entry_date = pos.entry_date or entry_date or datetime.now()
        else:
            self.positions[ts_code] = Position(
                ts_code=ts_code,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                weight=0.0,
                entry_date=entry_date or datetime.now(),
            )

        self.cash -= cost
        self.update_market_values({ts_code: price})

    def reduce_position(
        self,
        ts_code: str,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
    ) -> None:
        """Reduce or close a position."""
        if ts_code not in self.positions:
            raise ValueError("position does not exist")

        pos = self.positions[ts_code]
        quantity = quantity or pos.quantity
        if quantity <= 0 or quantity > pos.quantity:
            raise ValueError("invalid quantity to reduce")

        if price is None:
            self.update_market_values({ts_code: None})
            price = self.positions[ts_code].current_price

        proceeds = price * quantity
        pos.quantity -= quantity
        self.cash += proceeds

        if pos.quantity == 0:
            del self.positions[ts_code]
        self.update_market_values({ts_code: price})

    # ------------------------------------------------------------------
    def update_market_values(self, price_map: Optional[Dict[str, Optional[float]]] = None) -> None:
        """Update market value of all holdings using provided or repository prices."""
        if not self.positions:
            return

        price_map = price_map or {}
        symbols = [ts for ts in self.positions.keys() if price_map.get(ts) is None]
        if symbols:
            try:
                df = data_repository.get_latest_prices(symbols)
            except Exception as exc:  # pragma: no cover - data source may not exist
                logger.error(f"failed to fetch latest prices: {exc}")
                df = pd.DataFrame()
            for ts in symbols:
                row = df[df["ts_code"] == ts].head(1)
                if not row.empty:
                    price_map[ts] = float(row.iloc[0]["close_price"])

        total_value = self.cash
        for ts, pos in self.positions.items():
            price = price_map.get(ts, pos.current_price)
            pos.current_price = price
            pos.market_value = price * pos.quantity
            pos.unrealized_pnl = (price - pos.avg_cost) * pos.quantity
            pos.unrealized_pnl_pct = (price - pos.avg_cost) / pos.avg_cost if pos.avg_cost else 0.0
            total_value += pos.market_value

        for pos in self.positions.values():
            pos.weight = pos.market_value / total_value if total_value > 0 else 0.0

    # ------------------------------------------------------------------
    def portfolio_value(self) -> float:
        """Current total portfolio value."""
        return self.cash + sum(p.market_value for p in self.positions.values())

    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    # ------------------------------------------------------------------
    def create_snapshot(self, timestamp: Optional[datetime] = None) -> PortfolioSnapshot:
        """Create a snapshot of the current portfolio state."""
        self.update_market_values()
        timestamp = timestamp or datetime.now()
        total_value = self.portfolio_value()
        positions_value = self.positions_value()
        prev_value = self._last_value or total_value
        daily_ret = (total_value - prev_value) / prev_value if prev_value else 0.0
        self._last_value = total_value

        returns = pd.Series([s.daily_return for s in self.snapshots if s])
        risk_metrics: Dict[str, float] = {}
        if not returns.empty:
            risk_metrics["sharpe"] = calculate_sharpe_ratio(returns)
            cumulative = (1 + returns).cumprod()
            risk_metrics["max_drawdown"] = calculate_max_drawdown(cumulative)["max_drawdown"]
        else:
            risk_metrics["sharpe"] = 0.0
            risk_metrics["max_drawdown"] = 0.0

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash=self.cash,
            positions_value=positions_value,
            total_pnl=sum(p.unrealized_pnl for p in self.positions.values()),
            daily_return=daily_ret,
            positions=list(self.positions.values()),
            risk_metrics=risk_metrics,
        )

        self.snapshots.append(snapshot)
        return snapshot

    # Convenience ------------------------------------------------------------------
    def to_json(self) -> str:
        """Serialize the latest snapshot to JSON."""
        snap = self.create_snapshot()
        return json.dumps(snap.to_dict(), ensure_ascii=False, indent=2)

