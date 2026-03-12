"""
Market universe selection.

Analogous to the S&P 500 index committee process, the UniverseSelector
picks the top-N most eligible markets at any given point in time based
on a set of configurable criteria:

  1. Minimum volume  – markets with too little trading are excluded
                       (thin markets have wider spreads and noisier prices)
  2. Minimum days to close – avoid markets that are about to resolve
                              (limited time to profit, high execution risk)
  3. Maximum days to close – avoid very long-dated markets with low liquidity
  4. Category caps   – no single category dominates the portfolio
                       (avoids correlated political-event concentration)
  5. Price band      – exclude markets near 0 or 1 (near-certainties offer
                       minimal alpha and tie up capital at low yield)
  6. Ranking metric  – rank surviving markets by volume (liquid markets have
                       better calibration, lower spreads, and more alpha)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from pmif.models import Market, MarketCategory


_DEFAULT_CATEGORY_CAPS: dict[MarketCategory, float] = {
    MarketCategory.POLITICS: 0.35,
    MarketCategory.ECONOMICS: 0.30,
    MarketCategory.SPORTS: 0.25,
    MarketCategory.CRYPTO: 0.20,
    MarketCategory.WEATHER: 0.15,
    MarketCategory.SCIENCE: 0.10,
    MarketCategory.ENTERTAINMENT: 0.10,
    MarketCategory.OTHER: 0.10,
}


class UniverseSelector:
    """
    Selects an eligible market universe from a candidate pool.

    Parameters
    ----------
    max_markets : int
        Maximum number of markets in the universe (analogous to S&P 500 size).
    min_volume : float
        Minimum cumulative dollar volume required for inclusion.
    min_days_to_close : float
        Minimum days remaining until a market resolves.
    max_days_to_close : float
        Maximum days remaining (exclude very long-dated contracts).
    min_price : float
        Minimum mid-price; excludes near-certain NO outcomes.
    max_price : float
        Maximum mid-price; excludes near-certain YES outcomes.
    category_caps : dict | None
        Maximum fraction of the universe each category may occupy.
        Defaults to ``_DEFAULT_CATEGORY_CAPS``.
    """

    def __init__(
        self,
        max_markets: int = 500,
        min_volume: float = 1_000.0,
        min_days_to_close: float = 3.0,
        max_days_to_close: float = 365.0,
        min_price: float = 0.03,
        max_price: float = 0.97,
        category_caps: Optional[dict[MarketCategory, float]] = None,
    ) -> None:
        self.max_markets = max_markets
        self.min_volume = min_volume
        self.min_days_to_close = min_days_to_close
        self.max_days_to_close = max_days_to_close
        self.min_price = min_price
        self.max_price = max_price
        self.category_caps = category_caps or _DEFAULT_CATEGORY_CAPS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        candidates: list[Market],
        as_of: datetime,
        price_snapshot: Optional[dict[str, float]] = None,
    ) -> list[Market]:
        """
        Return the eligible universe of markets as of ``as_of``.

        Parameters
        ----------
        candidates : list[Market]
            All known markets (open + resolved).
        as_of : datetime
            Evaluation timestamp (filters out closed/not-yet-open markets).
        price_snapshot : dict[str, float] | None
            Latest mid-prices keyed by market_id.  If provided, overrides
            ``market.last_price`` for filtering.

        Returns
        -------
        list[Market]
            The selected universe, sorted by descending volume.
        """
        # Step 1 – filter to open markets that existed at as_of
        eligible = [
            m for m in candidates
            if m.open_date <= as_of < m.close_date and m.is_open
        ]

        # Step 2 – apply price snapshot overrides
        if price_snapshot:
            for m in eligible:
                if m.market_id in price_snapshot:
                    m.last_price = price_snapshot[m.market_id]

        # Step 3 – apply eligibility filters
        eligible = self._apply_filters(eligible, as_of)

        # Step 4 – sort by volume (descending) – liquid markets rank higher
        eligible.sort(key=lambda m: m.total_volume, reverse=True)

        # Step 5 – apply category caps and select top-N
        selected = self._apply_category_caps(eligible)

        return selected[: self.max_markets]

    def eligibility_report(
        self,
        candidates: list[Market],
        as_of: datetime,
    ) -> pd.DataFrame:
        """
        Return a DataFrame summarising why each candidate was included/excluded.
        Useful for debugging and auditing.
        """
        rows = []
        for m in candidates:
            days_left = (m.close_date - as_of).total_seconds() / 86_400
            rows.append(
                {
                    "market_id": m.market_id,
                    "name": m.name,
                    "category": m.category.value,
                    "volume": m.total_volume,
                    "price": m.last_price,
                    "days_to_close": days_left,
                    "is_open": m.is_open,
                    "opened_before": m.open_date <= as_of,
                    "not_yet_closed": as_of < m.close_date,
                    "vol_ok": m.total_volume >= self.min_volume,
                    "days_ok": self.min_days_to_close <= days_left <= self.max_days_to_close,
                    "price_ok": self.min_price <= m.last_price <= self.max_price,
                }
            )
        df = pd.DataFrame(rows)
        df["eligible"] = (
            df["is_open"]
            & df["opened_before"]
            & df["not_yet_closed"]
            & df["vol_ok"]
            & df["days_ok"]
            & df["price_ok"]
        )
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_filters(self, markets: list[Market], as_of: datetime) -> list[Market]:
        kept = []
        for m in markets:
            days_left = (m.close_date - as_of).total_seconds() / 86_400

            if m.total_volume < self.min_volume:
                continue
            if not (self.min_days_to_close <= days_left <= self.max_days_to_close):
                continue
            if not (self.min_price <= m.last_price <= self.max_price):
                continue

            kept.append(m)
        return kept

    def _apply_category_caps(self, sorted_markets: list[Market]) -> list[Market]:
        """
        Greedily add markets in volume order while respecting category caps.
        Category caps are soft maximums – if a category is under-represented
        it can grow up to its cap fraction of the total selected universe.
        """
        # First pass: unlimited selection
        if not sorted_markets:
            return []

        total = min(len(sorted_markets), self.max_markets)
        category_counts: dict[MarketCategory, int] = {c: 0 for c in MarketCategory}
        selected: list[Market] = []

        for m in sorted_markets:
            cap_frac = self.category_caps.get(m.category, 0.15)
            current_size = len(selected) + 1
            cat_max = max(1, int(total * cap_frac))

            if category_counts[m.category] < cat_max:
                selected.append(m)
                category_counts[m.category] += 1

            if len(selected) >= self.max_markets:
                break

        return selected
