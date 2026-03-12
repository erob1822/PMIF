"""
Data models for the PMIF system.

Core abstractions:
  Market       – a single predictive market contract (binary YES/NO event)
  MarketPrice  – a timestamped price observation for a market
  Position     – the fund's holding in one market
  Trade        – an executed buy or sell order
  Portfolio    – the full fund state at a point in time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class MarketStatus(str, Enum):
    OPEN = "open"
    RESOLVED_YES = "resolved_yes"
    RESOLVED_NO = "resolved_no"
    VOIDED = "voided"


class TradeDirection(str, Enum):
    BUY = "buy"
    SELL = "sell"


class MarketCategory(str, Enum):
    POLITICS = "politics"
    ECONOMICS = "economics"
    SPORTS = "sports"
    CRYPTO = "crypto"
    WEATHER = "weather"
    SCIENCE = "science"
    ENTERTAINMENT = "entertainment"
    OTHER = "other"


@dataclass
class Market:
    """
    Represents a single predictive market contract.

    In a binary predictive market a contract trades between $0.01 and $0.99.
    If the event resolves YES the holder of one YES-contract receives $1.00.
    If it resolves NO the holder receives $0.00.

    The market price is therefore a direct estimate of the crowd's
    probability that the event occurs.
    """

    market_id: str
    name: str
    category: MarketCategory
    open_date: datetime
    close_date: datetime
    status: MarketStatus = MarketStatus.OPEN
    resolution: Optional[int] = None        # 1 = YES, 0 = NO, None = open
    total_volume: float = 0.0               # cumulative $ volume traded
    last_price: float = 0.5                 # most-recent mid-price (probability)
    description: str = ""

    @property
    def days_to_close(self) -> float:
        """Remaining calendar days until resolution."""
        now = datetime.utcnow()
        if self.close_date <= now:
            return 0.0
        return (self.close_date - now).total_seconds() / 86_400

    @property
    def is_open(self) -> bool:
        return self.status == MarketStatus.OPEN

    @property
    def is_resolved(self) -> bool:
        return self.status in (MarketStatus.RESOLVED_YES, MarketStatus.RESOLVED_NO)

    def __repr__(self) -> str:
        return (
            f"Market(id={self.market_id!r}, name={self.name!r}, "
            f"price={self.last_price:.3f}, status={self.status.value})"
        )


@dataclass
class MarketPrice:
    """A single price tick for a market."""

    market_id: str
    timestamp: datetime
    price: float            # mid-price (0–1 probability scale)
    volume: float = 0.0     # volume in this tick window

    def __post_init__(self) -> None:
        if not (0.0 <= self.price <= 1.0):
            raise ValueError(
                f"Price must be in [0, 1]; got {self.price} for {self.market_id}"
            )


@dataclass
class Position:
    """
    The fund's current holding in one market.

    Contracts are denominated in *shares* (i.e. number of $1-face-value
    YES-contracts).  Negative shares mean a SHORT position (selling YES
    is equivalent to buying NO).
    """

    market_id: str
    shares: float           # positive = long YES, negative = short YES / long NO
    avg_cost: float         # weighted average entry price per share
    market_value: float = 0.0   # marked-to-market value

    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.shares * self.avg_cost

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost


@dataclass
class Trade:
    """An executed order."""

    trade_id: str
    market_id: str
    timestamp: datetime
    direction: TradeDirection
    shares: float
    price: float
    commission: float = 0.0

    @property
    def gross_value(self) -> float:
        return self.shares * self.price

    @property
    def net_value(self) -> float:
        return self.gross_value + self.commission  # commission is negative cost


@dataclass
class Portfolio:
    """
    Snapshot of the fund's complete state.

    The portfolio tracks cash, open positions, and the full trade log.
    Portfolio value = cash + sum of position market values.
    """

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    trade_log: list[Trade] = field(default_factory=list)
    timestamp: Optional[datetime] = None

    @property
    def position_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_value(self) -> float:
        return self.cash + self.position_value

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    def get_position(self, market_id: str) -> Optional[Position]:
        return self.positions.get(market_id)

    def __repr__(self) -> str:
        return (
            f"Portfolio(value=${self.total_value:,.2f}, "
            f"cash=${self.cash:,.2f}, "
            f"positions={self.num_positions})"
        )
