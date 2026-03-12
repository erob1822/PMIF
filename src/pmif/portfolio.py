"""
Portfolio management for PMIF.

The PortfolioManager handles:
  • Executing trades (buy/sell) and updating cash & positions
  • Marking positions to market on each price tick
  • Processing market resolutions (cash settlement)
  • Tracking high-water mark and generating snapshots for the backtester
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Optional

from pmif.models import (
    Market,
    MarketStatus,
    Portfolio,
    Position,
    Trade,
    TradeDirection,
)


class PortfolioManager:
    """
    Manages the fund's portfolio state throughout a simulation.

    Parameters
    ----------
    initial_capital : float
        Starting cash balance (USD).
    """

    def __init__(self, initial_capital: float = 1_000_000.0) -> None:
        self._portfolio = Portfolio(
            cash=initial_capital,
            timestamp=datetime.now(timezone.utc),
        )
        self._high_water_mark: float = initial_capital
        self._initial_capital: float = initial_capital

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def high_water_mark(self) -> float:
        return self._high_water_mark

    @property
    def nav(self) -> float:
        return self._portfolio.total_value

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------

    def execute_trade(self, trade: Trade) -> bool:
        """
        Apply a trade to the portfolio.

        Returns True if the trade was executed, False if rejected (e.g.
        insufficient cash).
        """
        port = self._portfolio
        price = trade.price
        direction = trade.direction
        shares = trade.shares
        commission = trade.commission   # negative dollar cost

        if direction == TradeDirection.BUY:
            cost = shares * price - abs(commission)     # cash out
            if cost > port.cash + 1e-6:                 # allow rounding tolerance
                # Scale down to what we can afford
                affordable = max(0, port.cash - abs(commission))
                shares = affordable / price if price > 0 else 0
                if shares < 1e-6:
                    return False
                cost = shares * price

            port.cash -= cost
            pos = port.positions.get(trade.market_id)
            if pos is None:
                port.positions[trade.market_id] = Position(
                    market_id=trade.market_id,
                    shares=shares,
                    avg_cost=price,
                    market_value=shares * price,
                )
            else:
                new_shares = pos.shares + shares
                if new_shares > 0:
                    pos.avg_cost = (
                        (pos.shares * pos.avg_cost + shares * price) / new_shares
                    )
                pos.shares = new_shares
                pos.market_value = pos.shares * price

        else:  # SELL
            pos = port.positions.get(trade.market_id)
            if pos is None or pos.shares < shares - 1e-6:
                # Can only sell what we own
                if pos is None:
                    return False
                shares = pos.shares
            if shares < 1e-6:
                return False

            proceeds = shares * price + abs(commission)  # cash in (commission is cost)
            port.cash += proceeds - 2 * abs(commission)   # pay commission both sides

            pos.shares -= shares
            pos.market_value = pos.shares * price
            if pos.shares < 1e-6:
                del port.positions[trade.market_id]

        port.trade_log.append(trade)
        return True

    def execute_trades(self, trades: list[Trade]) -> int:
        """Execute a list of trades.  Returns number successfully executed."""
        return sum(1 for t in trades if self.execute_trade(t))

    # ------------------------------------------------------------------
    # Mark-to-market
    # ------------------------------------------------------------------

    def mark_to_market(
        self,
        price_snapshot: dict[str, float],
        timestamp: datetime,
    ) -> None:
        """Update all position market values with latest prices."""
        port = self._portfolio
        port.timestamp = timestamp

        for market_id, pos in port.positions.items():
            price = price_snapshot.get(market_id, pos.avg_cost)
            pos.market_value = pos.shares * price

        # Update high-water mark
        if port.total_value > self._high_water_mark:
            self._high_water_mark = port.total_value

    # ------------------------------------------------------------------
    # Market resolution settlement
    # ------------------------------------------------------------------

    def settle_resolved_markets(self, resolved_markets: list[Market]) -> dict:
        """
        Cash-settle all positions in resolved markets.

        Parameters
        ----------
        resolved_markets : list[Market]
            Markets that have reached their resolution date.

        Returns
        -------
        dict
            Summary: {market_id: {'pnl': float, 'resolution': int}}
        """
        port = self._portfolio
        settlements: dict = {}

        for market in resolved_markets:
            pos = port.positions.get(market.market_id)
            if pos is None:
                continue

            payout_per_share = float(market.resolution or 0)
            total_payout = pos.shares * payout_per_share
            cost_basis = pos.cost_basis
            pnl = total_payout - cost_basis

            port.cash += total_payout
            del port.positions[market.market_id]

            settlements[market.market_id] = {
                "pnl": pnl,
                "resolution": market.resolution,
                "shares": pos.shares,
                "avg_cost": pos.avg_cost,
                "payout": total_payout,
            }

        return settlements

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> Portfolio:
        """Return a deep copy of the current portfolio state."""
        return deepcopy(self._portfolio)
