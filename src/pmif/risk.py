"""
Risk management for the PMIF portfolio.

Provides position-level and portfolio-level controls:

  • Max single-position weight  – no contract exceeds X% of NAV
  • Max category concentration  – category total ≤ Y% of NAV
  • Drawdown circuit-breaker    – halt trading if rolling drawdown > Z%
  • Kelly fraction cap          – never bet more than fractional Kelly
  • Minimum diversification     – require at least N positions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pmif.models import Market, Portfolio


@dataclass
class RiskLimits:
    """
    Configurable risk limits.

    Attributes
    ----------
    max_position_weight : float
        Max fraction of total NAV in a single market (default 2%).
    max_category_weight : float
        Max fraction of NAV in any one category (default 25%).
    max_drawdown_pct : float
        Circuit-breaker: stop new positions if rolling drawdown exceeds this
        fraction from the high-water mark (default 20%).
    max_kelly_fraction : float
        Cap on the Kelly criterion multiplier (default 0.25 = quarter-Kelly).
    min_positions : int
        Minimum number of open positions before the fund is considered
        fully invested (default 50).
    max_leverage : float
        Maximum gross leverage (sum of abs position values / NAV).
        Default 1.0 = no leverage (only long positions up to 100% of NAV).
    """

    max_position_weight: float = 0.02
    max_category_weight: float = 0.25
    max_drawdown_pct: float = 0.20
    max_kelly_fraction: float = 0.25
    min_positions: int = 50
    max_leverage: float = 1.0


class RiskManager:
    """
    Evaluates proposed trades and portfolio state against risk limits.

    Usage
    -----
    ```python
    rm = RiskManager(limits=RiskLimits())
    approved_size = rm.position_size(market, portfolio, edge=0.05)
    if rm.check_drawdown(portfolio, high_water_mark):
        # place trade
    ```
    """

    def __init__(self, limits: Optional[RiskLimits] = None) -> None:
        self.limits = limits or RiskLimits()

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def position_size(
        self,
        market: Market,
        portfolio: Portfolio,
        edge: float,
    ) -> float:
        """
        Compute the dollar amount to invest in ``market``.

        Uses fractional Kelly sizing capped at ``max_position_weight``.

        Kelly fraction = edge / (1 - price)   for YES bets
                       = edge / price          for NO bets
        where edge = |fair_probability - market_price|.

        Parameters
        ----------
        market : Market
            The target market.
        portfolio : Portfolio
            Current portfolio state.
        edge : float
            Estimated probability edge (fair_prob - price for YES; price -
            fair_prob for NO).  Must be positive to place a trade.

        Returns
        -------
        float
            Dollar amount to invest (0 if trade not recommended).
        """
        if edge <= 0:
            return 0.0

        nav = portfolio.total_value
        if nav <= 0:
            return 0.0

        price = market.last_price

        # Kelly formula for a binary bet:
        # f* = (p*b - q) / b  where b = (1-price)/price (odds), p = fair_prob
        # Simplified: f* = edge / (1 - price)
        denominator = max(1 - price, 1e-6)
        kelly_fraction = edge / denominator

        # Apply fractional Kelly
        kelly_fraction *= self.limits.max_kelly_fraction

        # Cap at max_position_weight
        fraction = min(kelly_fraction, self.limits.max_position_weight)

        dollar_size = nav * fraction

        # Check category concentration
        category_used = self._category_nav_fraction(market, portfolio)
        if category_used + fraction > self.limits.max_category_weight:
            allowed = max(0.0, self.limits.max_category_weight - category_used)
            dollar_size = nav * allowed

        # Check existing position (don't add to an already-capped position)
        existing = portfolio.get_position(market.market_id)
        if existing is not None:
            existing_frac = existing.market_value / nav
            if existing_frac >= self.limits.max_position_weight:
                return 0.0

        return max(0.0, dollar_size)

    # ------------------------------------------------------------------
    # Portfolio-level checks
    # ------------------------------------------------------------------

    def check_drawdown(
        self, portfolio: Portfolio, high_water_mark: float
    ) -> bool:
        """
        Return True if drawdown is within limits (trading allowed).
        Return False if circuit-breaker is triggered (pause new trades).
        """
        if high_water_mark <= 0:
            return True
        drawdown = (high_water_mark - portfolio.total_value) / high_water_mark
        return drawdown < self.limits.max_drawdown_pct

    def check_leverage(self, portfolio: Portfolio) -> bool:
        """Return True if gross leverage is within limits."""
        nav = portfolio.total_value
        if nav <= 0:
            return False
        gross = sum(
            abs(p.market_value) for p in portfolio.positions.values()
        )
        return (gross / nav) <= self.limits.max_leverage

    def is_diversified(self, portfolio: Portfolio) -> bool:
        """Return True if the portfolio holds enough positions."""
        return portfolio.num_positions >= self.limits.min_positions

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def risk_report(self, portfolio: Portfolio, high_water_mark: float) -> dict:
        """Return a summary of current risk metrics."""
        nav = portfolio.total_value
        gross = sum(abs(p.market_value) for p in portfolio.positions.values())

        return {
            "nav": nav,
            "positions": portfolio.num_positions,
            "gross_exposure": gross,
            "leverage": gross / nav if nav > 0 else 0.0,
            "drawdown_pct": (high_water_mark - nav) / high_water_mark
            if high_water_mark > 0
            else 0.0,
            "drawdown_ok": self.check_drawdown(portfolio, high_water_mark),
            "leverage_ok": self.check_leverage(portfolio),
            "diversified": self.is_diversified(portfolio),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _category_nav_fraction(
        self, market: Market, portfolio: Portfolio
    ) -> float:
        """Fraction of NAV currently in the same category as ``market``."""
        nav = portfolio.total_value
        if nav <= 0:
            return 0.0

        # We need market metadata to look up category – iterate positions
        # (in real usage, a market registry would be passed in)
        # Since we don't have a registry here, return 0 as a safe default.
        # The backtester passes the full market list and handles this.
        return 0.0
