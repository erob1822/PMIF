"""
PMIF Demo Backtest
==================
Run a full backtest of the Liquidity-Weighted Index strategy (plus the
Longshot Fade overlay) on synthetic data and print performance metrics.

Usage:
    python scripts/run_backtest.py [--markets N] [--capital C] [--seed S]
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datetime import datetime

from pmif.backtester import Backtester
from pmif.data.generator import SyntheticDataGenerator
from pmif.data.loader import DataLoader
from pmif.risk import RiskLimits
from pmif.strategy import LiquidityWeightedIndexStrategy, LongshotFadeStrategy
from pmif.universe import UniverseSelector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PMIF Backtest Demo")
    p.add_argument("--markets", type=int, default=200, help="Number of synthetic markets")
    p.add_argument("--capital", type=float, default=1_000_000.0, help="Starting capital ($)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--start", type=str, default="2021-01-01", help="Sim start date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--end", type=str, default="2023-12-31", help="Sim end date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--save-data", action="store_true", help="Save synthetic data to data/sample/"
    )
    p.add_argument(
        "--strategy",
        choices=["index", "longshot", "combined"],
        default="combined",
        help="Strategy to run",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  PMIF – Predictive Market Index Fund  |  Demo Backtest")
    print(f"{'='*60}")
    print(f"  Markets      : {args.markets}")
    print(f"  Capital      : ${args.capital:,.0f}")
    print(f"  Period       : {start.date()} → {end.date()}")
    print(f"  Seed         : {args.seed}")
    print(f"  Strategy     : {args.strategy}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # 1. Generate synthetic data
    # ------------------------------------------------------------------ #
    print("Generating synthetic market data …")
    gen = SyntheticDataGenerator(
        num_markets=args.markets,
        start_date=start,
        end_date=end,
        tick_interval_hours=1.0,
        seed=args.seed,
    )
    markets, price_df = gen.generate()
    print(f"  {len(markets)} markets | {len(price_df):,} price ticks")

    if args.save_data:
        out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample")
        os.makedirs(out_dir, exist_ok=True)
        DataLoader.save_markets_csv(markets, os.path.join(out_dir, "markets.csv"))
        DataLoader.save_prices_csv(price_df, os.path.join(out_dir, "prices.csv"))
        print(f"  Saved to data/sample/")

    # ------------------------------------------------------------------ #
    # 2. Configure strategies
    # ------------------------------------------------------------------ #
    if args.strategy == "index":
        strategies = [LiquidityWeightedIndexStrategy()]
    elif args.strategy == "longshot":
        strategies = [LongshotFadeStrategy()]
    else:  # combined
        strategies = [
            LiquidityWeightedIndexStrategy(),
            LongshotFadeStrategy(),
        ]

    # ------------------------------------------------------------------ #
    # 3. Run backtest
    # ------------------------------------------------------------------ #
    selector = UniverseSelector(
        max_markets=500,
        min_volume=0.0,   # include all markets in synthetic data
        min_days_to_close=2.0,
    )
    risk = RiskLimits(
        max_position_weight=0.02,
        max_drawdown_pct=0.25,
        max_kelly_fraction=0.25,
    )

    print("\nRunning backtest …")
    bt = Backtester(
        markets=markets,
        price_history=price_df,
        initial_capital=args.capital,
        rebalance_every_n_ticks=24,   # daily rebalance
        strategies=strategies,
        universe_selector=selector,
        risk_limits=risk,
        verbose=True,
    )
    result = bt.run()

    # ------------------------------------------------------------------ #
    # 4. Print results
    # ------------------------------------------------------------------ #
    result.print_summary()

    # Category breakdown of settlements
    print("\nSample resolved markets (first 5):")
    for s in result.settlements[:5]:
        pnl_str = f"+${s['pnl']:,.2f}" if s["pnl"] >= 0 else f"-${abs(s['pnl']):,.2f}"
        res_str = "YES" if s["resolution"] == 1 else "NO"
        print(f"  {s['market_id']}  resolved={res_str}  P&L={pnl_str}")


if __name__ == "__main__":
    main()
