"""
Performance metrics for PMIF backtests.

Computes standard quantitative finance metrics from a time series of
portfolio NAV values:

  • Total return
  • Annualised return (CAGR)
  • Annualised volatility
  • Sharpe ratio (risk-adjusted return over risk-free rate)
  • Sortino ratio (downside risk only)
  • Maximum drawdown
  • Calmar ratio (CAGR / max drawdown)
  • Win rate (fraction of resolved markets with positive P&L)
  • Average profit/loss per trade
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """
    Compute performance metrics from a NAV time series.

    Parameters
    ----------
    nav_series : pd.Series
        Indexed by datetime, values are portfolio NAV at each timestamp.
    risk_free_rate : float
        Annualised risk-free rate (default 4.5% ≈ current T-bill yield).
    """

    TRADING_DAYS_PER_YEAR: int = 252
    HOURS_PER_YEAR: float = 252 * 6.5   # approximate market hours

    def __init__(
        self,
        nav_series: pd.Series,
        risk_free_rate: float = 0.045,
    ) -> None:
        if nav_series.empty:
            raise ValueError("NAV series is empty.")
        self.nav = nav_series.sort_index().copy()
        self.risk_free_rate = risk_free_rate
        self._returns: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def returns(self) -> pd.Series:
        """Period-over-period percentage returns."""
        if self._returns is None:
            self._returns = self.nav.pct_change().dropna()
        return self._returns

    @property
    def total_return(self) -> float:
        """Total percentage return over the full period."""
        return float((self.nav.iloc[-1] / self.nav.iloc[0]) - 1)

    @property
    def cagr(self) -> float:
        """
        Compound Annual Growth Rate.
        Uses calendar years derived from the index timestamps.
        """
        start = self.nav.index[0]
        end = self.nav.index[-1]
        years = (end - start).total_seconds() / (365.25 * 86_400)
        if years <= 0:
            return 0.0
        return float((self.nav.iloc[-1] / self.nav.iloc[0]) ** (1 / years) - 1)

    @property
    def annualised_volatility(self) -> float:
        """Annualised standard deviation of returns."""
        if self.returns.empty:
            return 0.0
        # Determine periods per year from index frequency
        periods_per_year = self._infer_periods_per_year()
        return float(self.returns.std() * np.sqrt(periods_per_year))

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio."""
        vol = self.annualised_volatility
        if vol == 0:
            return 0.0
        return float((self.cagr - self.risk_free_rate) / vol)

    @property
    def sortino_ratio(self) -> float:
        """
        Annualised Sortino ratio (penalises only downside volatility).
        """
        downside = self.returns[self.returns < 0]
        if downside.empty:
            return float("inf")
        periods_per_year = self._infer_periods_per_year()
        downside_vol = float(downside.std() * np.sqrt(periods_per_year))
        if downside_vol == 0:
            return 0.0
        return float((self.cagr - self.risk_free_rate) / downside_vol)

    @property
    def max_drawdown(self) -> float:
        """Maximum peak-to-trough decline (as a positive fraction)."""
        cummax = self.nav.cummax()
        drawdowns = (self.nav - cummax) / cummax
        return float(abs(drawdowns.min()))

    @property
    def calmar_ratio(self) -> float:
        """CAGR divided by maximum drawdown."""
        mdd = self.max_drawdown
        if mdd == 0:
            return float("inf")
        return float(self.cagr / mdd)

    @property
    def avg_drawdown(self) -> float:
        """Average drawdown across all underwater periods."""
        cummax = self.nav.cummax()
        drawdowns = (self.nav - cummax) / cummax
        underwater = drawdowns[drawdowns < 0]
        return float(abs(underwater.mean())) if not underwater.empty else 0.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return all metrics in a single dictionary."""
        return {
            "total_return_pct": round(self.total_return * 100, 2),
            "cagr_pct": round(self.cagr * 100, 2),
            "annualised_volatility_pct": round(self.annualised_volatility * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "avg_drawdown_pct": round(self.avg_drawdown * 100, 2),
            "num_periods": len(self.returns),
        }

    def print_summary(self) -> None:
        """Pretty-print the metrics table."""
        s = self.summary()
        print("\n" + "=" * 50)
        print("  PMIF Performance Summary")
        print("=" * 50)
        print(f"  Total Return       : {s['total_return_pct']:>8.2f}%")
        print(f"  CAGR               : {s['cagr_pct']:>8.2f}%")
        print(f"  Annualised Vol     : {s['annualised_volatility_pct']:>8.2f}%")
        print(f"  Sharpe Ratio       : {s['sharpe_ratio']:>8.3f}")
        print(f"  Sortino Ratio      : {s['sortino_ratio']:>8.3f}")
        print(f"  Max Drawdown       : {s['max_drawdown_pct']:>8.2f}%")
        print(f"  Calmar Ratio       : {s['calmar_ratio']:>8.3f}")
        print(f"  Avg Drawdown       : {s['avg_drawdown_pct']:>8.2f}%")
        print(f"  Observation periods: {s['num_periods']:>8d}")
        print("=" * 50 + "\n")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _infer_periods_per_year(self) -> float:
        """
        Infer number of return periods per calendar year from the index.
        Falls back to daily (252) if inference fails.
        """
        if len(self.nav) < 2:
            return self.TRADING_DAYS_PER_YEAR
        try:
            deltas = pd.Series(self.nav.index).diff().dropna()
            median_seconds = deltas.median().total_seconds()
            if median_seconds <= 0:
                return self.TRADING_DAYS_PER_YEAR
            seconds_per_year = 365.25 * 86_400
            return seconds_per_year / median_seconds
        except Exception:
            return float(self.TRADING_DAYS_PER_YEAR)
