"""
Trading strategies for PMIF.

Three strategies are implemented:

1. LiquidityWeightedIndexStrategy  (main, passive-style)
   -------------------------------------------------------
   Mirrors the S&P 500 philosophy: hold the top-N most liquid predictive
   markets, weighted by trading volume.  No price-prediction required –
   we simply trust the aggregate crowd wisdom and harvest diversification.

   Expected alpha sources:
     • Diversification reduces idiosyncratic resolution variance
     • Passive participation in liquid, well-calibrated markets
     • Gradual rebalancing as new markets open and old ones resolve

2. LongshotFadeStrategy  (contrarian, exploits cognitive bias)
   ------------------------------------------------------------
   Research across predictive markets shows a consistent *longshot bias*:
   bettors systematically overprice low-probability events (< 15%) and
   underprice high-probability events (> 85%).  This strategy sells YES
   on high-priced events and sells NO on low-priced events to harvest
   this premium.

3. TimeLagArbitrageStrategy  (event-driven, exploits information lag)
   ------------------------------------------------------------------
   When a correlated high-volume market moves sharply (> threshold), there
   is often a brief window (minutes to hours) before related markets reprice.
   This strategy detects those correlated mis-pricings and trades the spread.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from pmif.models import Market, Portfolio, Position, Trade, TradeDirection


class BaseStrategy(ABC):
    """Abstract base class for all PMIF strategies."""

    name: str = "base"

    @abstractmethod
    def generate_orders(
        self,
        universe: list[Market],
        portfolio: Portfolio,
        price_snapshot: dict[str, float],
        timestamp: datetime,
    ) -> list[Trade]:
        """
        Given the current universe and portfolio state, produce a list of
        desired trades.  The backtester will execute them at the given prices.
        """
        ...

    @staticmethod
    def _make_trade(
        market_id: str,
        direction: TradeDirection,
        shares: float,
        price: float,
        timestamp: datetime,
        commission_rate: float = 0.001,
    ) -> Trade:
        gross = shares * price
        commission = -abs(gross) * commission_rate
        return Trade(
            trade_id=str(uuid.uuid4()),
            market_id=market_id,
            timestamp=timestamp,
            direction=direction,
            shares=shares,
            price=price,
            commission=commission,
        )


# ---------------------------------------------------------------------------
# Strategy 1: Liquidity-Weighted Index
# ---------------------------------------------------------------------------

class LiquidityWeightedIndexStrategy(BaseStrategy):
    """
    Core PMIF index strategy.

    Allocates capital proportionally to each market's share of total
    universe volume (exactly like market-cap weighting in a stock index).
    Rebalances on every call (caller controls frequency).

    Parameters
    ----------
    target_investment_pct : float
        Fraction of NAV to keep invested (default 0.95, keep 5% cash buffer).
    min_trade_size : float
        Ignore rebalancing trades smaller than this dollar amount.
    commission_rate : float
        Proportional transaction cost per trade.
    """

    name = "liquidity_weighted_index"

    def __init__(
        self,
        target_investment_pct: float = 0.95,
        min_trade_size: float = 10.0,
        commission_rate: float = 0.001,
    ) -> None:
        self.target_investment_pct = target_investment_pct
        self.min_trade_size = min_trade_size
        self.commission_rate = commission_rate

    def generate_orders(
        self,
        universe: list[Market],
        portfolio: Portfolio,
        price_snapshot: dict[str, float],
        timestamp: datetime,
    ) -> list[Trade]:
        if not universe:
            return []

        nav = portfolio.total_value
        investable = nav * self.target_investment_pct

        # Compute target dollar allocations (volume-weighted)
        total_vol = sum(m.total_volume for m in universe) or 1.0
        target_alloc: dict[str, float] = {
            m.market_id: (m.total_volume / total_vol) * investable
            for m in universe
        }

        trades: list[Trade] = []

        # For each market in universe, compute required trade
        for market in universe:
            mid = price_snapshot.get(market.market_id, market.last_price)
            if mid <= 0:
                continue

            target_dollars = target_alloc[market.market_id]
            target_shares = target_dollars / mid

            pos = portfolio.get_position(market.market_id)
            current_shares = pos.shares if pos else 0.0

            delta_shares = target_shares - current_shares
            delta_dollars = abs(delta_shares * mid)

            if delta_dollars < self.min_trade_size:
                continue

            direction = TradeDirection.BUY if delta_shares > 0 else TradeDirection.SELL
            trade = self._make_trade(
                market_id=market.market_id,
                direction=direction,
                shares=abs(delta_shares),
                price=mid,
                timestamp=timestamp,
                commission_rate=self.commission_rate,
            )
            trades.append(trade)

        # Close positions no longer in universe
        for market_id, pos in portfolio.positions.items():
            if market_id not in target_alloc and pos.shares > 0:
                mid = price_snapshot.get(market_id, pos.avg_cost)
                trade = self._make_trade(
                    market_id=market_id,
                    direction=TradeDirection.SELL,
                    shares=pos.shares,
                    price=mid,
                    timestamp=timestamp,
                    commission_rate=self.commission_rate,
                )
                trades.append(trade)

        return trades


# ---------------------------------------------------------------------------
# Strategy 2: Longshot Fade
# ---------------------------------------------------------------------------

class LongshotFadeStrategy(BaseStrategy):
    """
    Exploits the well-documented longshot bias in prediction markets.

    Research finding: bettors systematically overestimate the probability of
    unlikely events and underestimate the probability of likely events.
    A contract at 8% tends to resolve YES only ~6% of the time; a contract
    at 92% tends to resolve YES ~94% of the time.

    This strategy:
      • SHORT YES on markets priced below ``longshot_threshold``
        (sell overpriced longshots)
      • LONG YES on markets priced above ``favorite_threshold``
        (buy underpriced favorites)

    Risk controls:
      • Maximum position size per market
      • Only trade if price edge exceeds ``min_edge``
    """

    name = "longshot_fade"

    def __init__(
        self,
        longshot_threshold: float = 0.12,
        favorite_threshold: float = 0.88,
        min_edge: float = 0.02,
        max_position_pct: float = 0.005,
        commission_rate: float = 0.001,
    ) -> None:
        self.longshot_threshold = longshot_threshold
        self.favorite_threshold = favorite_threshold
        self.min_edge = min_edge
        self.max_position_pct = max_position_pct
        self.commission_rate = commission_rate

    # Empirical calibration: average observed edge from longshot bias studies
    # (Snowberg & Wolfers 2010, Rothschild 2009, etc.)
    _LONGSHOT_EDGE_COEFF = 0.15   # expected edge ≈ 15% of distance from threshold
    _FAVORITE_EDGE_COEFF = 0.10

    def generate_orders(
        self,
        universe: list[Market],
        portfolio: Portfolio,
        price_snapshot: dict[str, float],
        timestamp: datetime,
    ) -> list[Trade]:
        nav = portfolio.total_value
        if nav <= 0:
            return []

        trades: list[Trade] = []
        max_dollars = nav * self.max_position_pct

        for market in universe:
            mid = price_snapshot.get(market.market_id, market.last_price)
            pos = portfolio.get_position(market.market_id)

            # Longshot: overpriced at low probabilities → SHORT YES (fade)
            if mid <= self.longshot_threshold:
                edge = (self.longshot_threshold - mid) * self._LONGSHOT_EDGE_COEFF
                if edge >= self.min_edge:
                    current = pos.shares if pos else 0.0
                    target_short = -(max_dollars / mid)
                    delta = target_short - current
                    if abs(delta * mid) >= 10.0:
                        direction = TradeDirection.SELL if delta < 0 else TradeDirection.BUY
                        trades.append(
                            self._make_trade(
                                market_id=market.market_id,
                                direction=direction,
                                shares=abs(delta),
                                price=mid,
                                timestamp=timestamp,
                                commission_rate=self.commission_rate,
                            )
                        )

            # Favorite: underpriced at high probabilities → LONG YES
            elif mid >= self.favorite_threshold:
                edge = (mid - self.favorite_threshold) * self._FAVORITE_EDGE_COEFF
                if edge >= self.min_edge:
                    current = pos.shares if pos else 0.0
                    target_long = max_dollars / mid
                    delta = target_long - current
                    if abs(delta * mid) >= 10.0:
                        direction = TradeDirection.BUY if delta > 0 else TradeDirection.SELL
                        trades.append(
                            self._make_trade(
                                market_id=market.market_id,
                                direction=direction,
                                shares=abs(delta),
                                price=mid,
                                timestamp=timestamp,
                                commission_rate=self.commission_rate,
                            )
                        )

        return trades


# ---------------------------------------------------------------------------
# Strategy 3: Time-Lag Arbitrage
# ---------------------------------------------------------------------------

class TimeLagArbitrageStrategy(BaseStrategy):
    """
    Exploits information propagation delays across correlated markets.

    Predictive markets don't always reprice instantly when related news breaks.
    Example: If "Will Party A win?" drops sharply, "Will Candidate X win
    the primary?" may lag before repricing.

    Detection heuristic:
      1. Compute rolling 1-hour returns for all markets.
      2. If a market moves > ``shock_threshold`` in one hour, flag it as a
         "signal" market.
      3. Find other markets in the same category that have NOT moved yet.
      4. Trade the direction of the signal in the lagging market.
      5. Close the position after ``hold_hours`` hours.

    Parameters
    ----------
    shock_threshold : float
        Minimum logit-return magnitude to qualify as a shock (default 0.4).
    hold_hours : int
        Hold period in ticks after entering a lag trade (default 4).
    max_position_pct : float
        Maximum position size per arb trade as fraction of NAV.
    """

    name = "time_lag_arb"

    def __init__(
        self,
        shock_threshold: float = 0.40,
        hold_hours: int = 4,
        max_position_pct: float = 0.003,
        commission_rate: float = 0.001,
    ) -> None:
        self.shock_threshold = shock_threshold
        self.hold_hours = hold_hours
        self.max_position_pct = max_position_pct
        self.commission_rate = commission_rate
        self._arb_positions: dict[str, int] = {}   # market_id → ticks held

    def generate_orders(
        self,
        universe: list[Market],
        portfolio: Portfolio,
        price_snapshot: dict[str, float],
        timestamp: datetime,
        prev_snapshot: Optional[dict[str, float]] = None,
    ) -> list[Trade]:
        if prev_snapshot is None:
            return []

        nav = portfolio.total_value
        if nav <= 0:
            return []

        trades: list[Trade] = []
        max_dollars = nav * self.max_position_pct

        # Build category → markets mapping
        category_map: dict[str, list[Market]] = {}
        for m in universe:
            category_map.setdefault(m.category.value, []).append(m)

        # Detect shocks
        shocks: list[tuple[Market, float]] = []  # (market, logit_return)
        for m in universe:
            prev = prev_snapshot.get(m.market_id)
            curr = price_snapshot.get(m.market_id, m.last_price)
            if prev is None or prev <= 0 or prev >= 1:
                continue
            curr_c = np.clip(curr, 1e-6, 1 - 1e-6)
            prev_c = np.clip(prev, 1e-6, 1 - 1e-6)
            logit_ret = float(np.log(curr_c / (1 - curr_c)) - np.log(prev_c / (1 - prev_c)))
            if abs(logit_ret) >= self.shock_threshold:
                shocks.append((m, logit_ret))

        # For each shock, look for lagging correlated markets
        for signal_market, logit_ret in shocks:
            peers = [
                m for m in category_map.get(signal_market.category.value, [])
                if m.market_id != signal_market.market_id
            ]
            for peer in peers:
                prev_peer = prev_snapshot.get(peer.market_id)
                curr_peer = price_snapshot.get(peer.market_id, peer.last_price)
                if prev_peer is None:
                    continue
                peer_ret = abs(
                    float(
                        np.log(np.clip(curr_peer, 1e-6, 1 - 1e-6) /
                               (1 - np.clip(curr_peer, 1e-6, 1 - 1e-6)))
                        - np.log(np.clip(prev_peer, 1e-6, 1 - 1e-6) /
                                 (1 - np.clip(prev_peer, 1e-6, 1 - 1e-6)))
                    )
                )
                # Peer has NOT moved yet (small reaction) → lag opportunity
                if peer_ret < self.shock_threshold * 0.3:
                    direction = TradeDirection.BUY if logit_ret > 0 else TradeDirection.SELL
                    shares = (max_dollars / curr_peer) if curr_peer > 0 else 0
                    if shares > 0:
                        trades.append(
                            self._make_trade(
                                market_id=peer.market_id,
                                direction=direction,
                                shares=shares,
                                price=curr_peer,
                                timestamp=timestamp,
                                commission_rate=self.commission_rate,
                            )
                        )
                        self._arb_positions[peer.market_id] = 0

        # Age open arb positions; close expired ones
        expired = [
            mid for mid, age in self._arb_positions.items()
            if age >= self.hold_hours
        ]
        for mid in expired:
            pos = portfolio.get_position(mid)
            if pos and abs(pos.shares) > 0:
                curr = price_snapshot.get(mid, pos.avg_cost)
                close_dir = (
                    TradeDirection.SELL if pos.shares > 0 else TradeDirection.BUY
                )
                trades.append(
                    self._make_trade(
                        market_id=mid,
                        direction=close_dir,
                        shares=abs(pos.shares),
                        price=curr,
                        timestamp=timestamp,
                        commission_rate=self.commission_rate,
                    )
                )
            del self._arb_positions[mid]

        # Increment ages
        for mid in list(self._arb_positions.keys()):
            self._arb_positions[mid] += 1

        return trades
