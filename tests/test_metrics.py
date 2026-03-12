"""
Tests for PerformanceMetrics.
"""

from datetime import datetime, timedelta
import pytest
import pandas as pd
import numpy as np
from pmif.metrics import PerformanceMetrics


def _make_nav(
    returns: list[float],
    start_nav: float = 1_000_000.0,
    start_date: datetime = datetime(2021, 1, 1),
    freq_hours: int = 24,
) -> pd.Series:
    """Build a NAV series from a list of period returns."""
    nav = start_nav
    data = {start_date: nav}
    dt = start_date
    for r in returns:
        nav *= (1 + r)
        dt += timedelta(hours=freq_hours)
        data[dt] = nav
    return pd.Series(data)


def test_total_return_positive():
    nav = _make_nav([0.01, 0.02, 0.01])
    m = PerformanceMetrics(nav)
    assert m.total_return > 0


def test_total_return_flat():
    nav = _make_nav([0.0, 0.0, 0.0])
    m = PerformanceMetrics(nav)
    assert m.total_return == pytest.approx(0.0)


def test_total_return_negative():
    nav = _make_nav([-0.10, -0.05, -0.02])
    m = PerformanceMetrics(nav)
    assert m.total_return < 0


def test_max_drawdown_no_drawdown():
    """Monotonically rising NAV → zero drawdown."""
    nav = _make_nav([0.01] * 100)
    m = PerformanceMetrics(nav)
    assert m.max_drawdown == pytest.approx(0.0, abs=1e-9)


def test_max_drawdown_known_value():
    """NAV peaks at 110, then drops to 99 → drawdown ≈ 10%."""
    nav = pd.Series(
        {
            datetime(2023, 1, 1): 100.0,
            datetime(2023, 1, 2): 110.0,
            datetime(2023, 1, 3): 99.0,
        }
    )
    m = PerformanceMetrics(nav)
    expected_dd = (110.0 - 99.0) / 110.0
    assert m.max_drawdown == pytest.approx(expected_dd, rel=1e-4)


def test_sharpe_positive_for_strong_uptrend():
    """
    A noisy uptrend with positive expected return >> risk-free rate
    should yield a positive Sharpe ratio.
    """
    rng = np.random.default_rng(0)
    # 0.3%/day drift + small noise → strong uptrend
    returns = list(rng.normal(loc=0.003, scale=0.005, size=252))
    nav = _make_nav(returns, freq_hours=24)
    m = PerformanceMetrics(nav, risk_free_rate=0.04)
    assert m.sharpe_ratio > 0


def test_sharpe_zero_for_flat():
    nav = _make_nav([0.0] * 100)
    m = PerformanceMetrics(nav)
    assert m.sharpe_ratio == pytest.approx(0.0)


def test_empty_nav_raises():
    with pytest.raises(ValueError):
        PerformanceMetrics(pd.Series(dtype=float))


def test_summary_keys():
    nav = _make_nav([0.001] * 50)
    m = PerformanceMetrics(nav)
    s = m.summary()
    expected_keys = {
        "total_return_pct",
        "cagr_pct",
        "annualised_volatility_pct",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown_pct",
        "calmar_ratio",
        "avg_drawdown_pct",
        "num_periods",
    }
    assert set(s.keys()) == expected_keys


def test_cagr_two_years_10pct():
    """Starting $100, ending $121 over exactly 2 years → CAGR ≈ 10%."""
    start = datetime(2021, 1, 1)
    end = datetime(2023, 1, 1)
    nav = pd.Series({start: 100.0, end: 121.0})
    m = PerformanceMetrics(nav)
    assert m.cagr == pytest.approx(0.10, abs=0.005)
