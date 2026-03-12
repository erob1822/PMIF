"""
PMIF – Predictive Market Index Fund
====================================
A Python library for building, backtesting, and running a diversified
index strategy across predictive markets (Kalshi, Polymarket, PredictIt, etc.).
"""

from pmif.models import Market, MarketPrice, Position, Trade, Portfolio
from pmif.universe import UniverseSelector
from pmif.portfolio import PortfolioManager
from pmif.backtester import Backtester
from pmif.metrics import PerformanceMetrics

__version__ = "0.1.0"
__all__ = [
    "Market",
    "MarketPrice",
    "Position",
    "Trade",
    "Portfolio",
    "UniverseSelector",
    "PortfolioManager",
    "Backtester",
    "PerformanceMetrics",
]
