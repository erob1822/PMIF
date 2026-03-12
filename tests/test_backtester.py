"""
Integration test: run a full end-to-end backtest on synthetic data.
Validates that the backtester runs without errors and produces sane output.
"""

import pytest
from datetime import datetime
from pmif.data.generator import SyntheticDataGenerator
from pmif.backtester import Backtester
from pmif.strategy import LiquidityWeightedIndexStrategy, LongshotFadeStrategy
from pmif.universe import UniverseSelector
from pmif.risk import RiskLimits


@pytest.fixture(scope="module")
def synthetic_data():
    """Generate a small dataset for integration tests."""
    gen = SyntheticDataGenerator(
        num_markets=30,
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2022, 6, 30),
        tick_interval_hours=24,    # daily ticks for speed
        seed=99,
    )
    return gen.generate()


def test_backtest_runs(synthetic_data):
    markets, price_df = synthetic_data
    assert len(markets) > 0
    assert not price_df.empty

    bt = Backtester(
        markets=markets,
        price_history=price_df,
        initial_capital=100_000.0,
        rebalance_every_n_ticks=7,   # weekly rebalance
        strategies=[LiquidityWeightedIndexStrategy()],
        universe_selector=UniverseSelector(max_markets=20, min_volume=0.0),
        verbose=False,
    )
    result = bt.run()

    # NAV series should be non-empty and positive
    assert not result.equity_curve.empty
    assert (result.equity_curve > 0).all()


def test_final_nav_reasonable(synthetic_data):
    """Final NAV should be within plausible range of initial capital."""
    markets, price_df = synthetic_data
    bt = Backtester(
        markets=markets,
        price_history=price_df,
        initial_capital=100_000.0,
        rebalance_every_n_ticks=7,
        strategies=[LiquidityWeightedIndexStrategy()],
        universe_selector=UniverseSelector(max_markets=20, min_volume=0.0),
        verbose=False,
    )
    result = bt.run()
    final_nav = result.equity_curve.iloc[-1]
    # Should not blow up or go to zero in a 6-month sim
    assert 10_000 < final_nav < 500_000


def test_metrics_available(synthetic_data):
    markets, price_df = synthetic_data
    bt = Backtester(
        markets=markets,
        price_history=price_df,
        initial_capital=100_000.0,
        rebalance_every_n_ticks=7,
        strategies=[LiquidityWeightedIndexStrategy()],
        universe_selector=UniverseSelector(max_markets=20, min_volume=0.0),
        verbose=False,
    )
    result = bt.run()
    summary = result.metrics.summary()
    assert "sharpe_ratio" in summary
    assert "max_drawdown_pct" in summary
    assert "cagr_pct" in summary


def test_longshot_strategy_runs(synthetic_data):
    markets, price_df = synthetic_data
    bt = Backtester(
        markets=markets,
        price_history=price_df,
        initial_capital=100_000.0,
        rebalance_every_n_ticks=7,
        strategies=[LongshotFadeStrategy()],
        universe_selector=UniverseSelector(max_markets=20, min_volume=0.0),
        verbose=False,
    )
    result = bt.run()
    assert not result.equity_curve.empty
