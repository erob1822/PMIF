"""
Event-driven backtesting engine for PMIF.

The Backtester replays historical price ticks in chronological order,
triggering rebalancing, settlement, and risk checks at each step.

Simulation loop (per tick):
  1. Update price snapshot for all open markets.
  2. Mark portfolio to market.
  3. Settle any markets that have resolved since the last tick.
  4. If it is a rebalancing tick:
     a. Refresh the eligible universe via UniverseSelector.
     b. Ask each strategy for its desired trades.
     c. Filter trades through RiskManager.
     d. Execute approved trades via PortfolioManager.
  5. Record a NAV snapshot for the equity curve.

Results:
  • equity_curve – pd.Series of NAV by timestamp
  • trade_log    – list of all Trade objects
  • settlements  – dict of market resolution P&L
  • final metrics via PerformanceMetrics
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from pmif.data.loader import DataLoader
from pmif.metrics import PerformanceMetrics
from pmif.models import Market, MarketStatus, Portfolio, Trade
from pmif.portfolio import PortfolioManager
from pmif.risk import RiskLimits, RiskManager
from pmif.strategy import BaseStrategy
from pmif.universe import UniverseSelector


class BacktestResult:
    """Container for all backtest output."""

    def __init__(
        self,
        equity_curve: pd.Series,
        trade_log: list[Trade],
        settlements: list[dict],
        universe_sizes: pd.Series,
    ) -> None:
        self.equity_curve = equity_curve
        self.trade_log = trade_log
        self.settlements = settlements
        self.universe_sizes = universe_sizes

    @property
    def metrics(self) -> PerformanceMetrics:
        return PerformanceMetrics(self.equity_curve)

    @property
    def num_trades(self) -> int:
        return len(self.trade_log)

    @property
    def num_resolutions(self) -> int:
        return len(self.settlements)

    def print_summary(self) -> None:
        print(f"\nBacktest completed.")
        print(f"  Trades executed   : {self.num_trades:,}")
        print(f"  Markets settled   : {self.num_resolutions:,}")
        print(
            f"  Avg universe size : "
            f"{self.universe_sizes.mean():.0f} markets"
        )
        self.metrics.print_summary()


class Backtester:
    """
    Runs a full backtest of one or more strategies against historical
    (or synthetic) price data.

    Parameters
    ----------
    markets : list[Market]
        Full universe of known markets (open + resolved).
    price_history : pd.DataFrame
        Tick data – columns: market_id, timestamp, price, volume.
    initial_capital : float
        Starting NAV (default $1,000,000).
    rebalance_every_n_ticks : int
        How often to rebalance.  With hourly ticks, 24 = daily rebalance.
    strategies : list[BaseStrategy] | None
        Strategies to run (in order).  Defaults to LiquidityWeightedIndex.
    universe_selector : UniverseSelector | None
        Custom universe selector; uses defaults if None.
    risk_limits : RiskLimits | None
        Custom risk limits; uses defaults if None.
    verbose : bool
        Print progress every N steps.
    """

    def __init__(
        self,
        markets: list[Market],
        price_history: pd.DataFrame,
        initial_capital: float = 1_000_000.0,
        rebalance_every_n_ticks: int = 24,
        strategies: Optional[list[BaseStrategy]] = None,
        universe_selector: Optional[UniverseSelector] = None,
        risk_limits: Optional[RiskLimits] = None,
        verbose: bool = True,
    ) -> None:
        self.markets = markets
        self.price_history = price_history.sort_values("timestamp").reset_index(
            drop=True
        )
        self.initial_capital = initial_capital
        self.rebalance_every_n_ticks = rebalance_every_n_ticks

        # Set defaults
        if strategies is None:
            from pmif.strategy import LiquidityWeightedIndexStrategy
            strategies = [LiquidityWeightedIndexStrategy()]
        self.strategies = strategies

        self.selector = universe_selector or UniverseSelector()
        self.risk_manager = RiskManager(limits=risk_limits)
        self.verbose = verbose

        # Build a market lookup dict
        self._market_map: dict[str, Market] = {m.market_id: m for m in markets}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return results."""
        pm = PortfolioManager(initial_capital=self.initial_capital)

        equity_curve: dict[datetime, float] = {}
        all_settlements: list[dict] = []
        universe_sizes: dict[datetime, int] = {}
        prev_snapshot: Optional[dict[str, float]] = None

        # Group ticks by timestamp for efficient iteration
        ticks_by_time = (
            self.price_history.groupby("timestamp", sort=True)
        )

        timestamps = sorted(self.price_history["timestamp"].unique())
        n = len(timestamps)

        for step, ts in enumerate(timestamps):
            ts_dt = pd.Timestamp(ts).to_pydatetime()
            group = self.price_history[self.price_history["timestamp"] == ts]

            # 1. Build price snapshot
            snapshot: dict[str, float] = {
                row["market_id"]: float(row["price"])
                for _, row in group.iterrows()
            }

            # Update last_price on market objects
            for mid, price in snapshot.items():
                if mid in self._market_map:
                    self._market_map[mid].last_price = price

            # 2. Mark to market
            pm.mark_to_market(snapshot, ts_dt)

            # 3. Update market statuses (must happen before settlement check)
            self._update_market_statuses(ts_dt)

            # 4. Settle resolved markets that we hold
            newly_resolved = [
                m for m in self.markets
                if m.is_resolved and m.market_id in pm.portfolio.positions
            ]
            if newly_resolved:
                sett = pm.settle_resolved_markets(newly_resolved)
                for mid, info in sett.items():
                    all_settlements.append(
                        {"market_id": mid, "timestamp": ts_dt, **info}
                    )

            # 5. Rebalance on schedule
            if step % self.rebalance_every_n_ticks == 0:
                universe = self.selector.select(
                    candidates=list(self._market_map.values()),
                    as_of=ts_dt,
                    price_snapshot=snapshot,
                )
                universe_sizes[ts_dt] = len(universe)

                # Risk check: skip new trades if circuit-breaker fired
                if self.risk_manager.check_drawdown(
                    pm.portfolio, pm.high_water_mark
                ):
                    for strategy in self.strategies:
                        kwargs: dict = {
                            "universe": universe,
                            "portfolio": pm.portfolio,
                            "price_snapshot": snapshot,
                            "timestamp": ts_dt,
                        }
                        # TimeLagArb needs prev_snapshot
                        if hasattr(strategy, "_arb_positions"):
                            kwargs["prev_snapshot"] = prev_snapshot
                        orders = strategy.generate_orders(**kwargs)
                        pm.execute_trades(orders)

            # 6. Record equity curve
            equity_curve[ts_dt] = pm.nav

            prev_snapshot = snapshot

            if self.verbose and step % max(1, n // 20) == 0:
                pct = 100 * step / n
                print(
                    f"  [{pct:5.1f}%] {ts_dt.date()}  "
                    f"NAV=${pm.nav:>12,.0f}  "
                    f"Positions={pm.portfolio.num_positions}"
                )

        # Build results
        nav_series = pd.Series(equity_curve).sort_index()
        nav_series.index = pd.DatetimeIndex(nav_series.index)

        return BacktestResult(
            equity_curve=nav_series,
            trade_log=list(pm.portfolio.trade_log),
            settlements=all_settlements,
            universe_sizes=pd.Series(universe_sizes),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_market_statuses(self, as_of: datetime) -> None:
        """
        Mark markets as resolved if their close_date has passed.
        In a live system this would come from the exchange API.
        """
        for m in self.markets:
            if m.status == MarketStatus.OPEN and m.close_date <= as_of:
                if m.resolution == 1:
                    m.status = MarketStatus.RESOLVED_YES
                elif m.resolution == 0:
                    m.status = MarketStatus.RESOLVED_NO
