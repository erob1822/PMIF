"""
Tests for PortfolioManager.
"""

from datetime import datetime, timedelta
import pytest
from pmif.models import Market, MarketCategory, MarketStatus, Trade, TradeDirection
from pmif.portfolio import PortfolioManager


def ts() -> datetime:
    return datetime(2023, 6, 1, 12, 0, 0)


def make_trade(
    market_id: str = "MKT_1",
    direction: TradeDirection = TradeDirection.BUY,
    shares: float = 100.0,
    price: float = 0.50,
    commission: float = -0.05,
) -> Trade:
    return Trade(
        trade_id="T001",
        market_id=market_id,
        timestamp=ts(),
        direction=direction,
        shares=shares,
        price=price,
        commission=commission,
    )


# ---------------------------------------------------------------------------
# Buy / sell basics
# ---------------------------------------------------------------------------

def test_buy_increases_position():
    pm = PortfolioManager(initial_capital=10_000.0)
    trade = make_trade(shares=100, price=0.50)
    ok = pm.execute_trade(trade)
    assert ok
    pos = pm.portfolio.get_position("MKT_1")
    assert pos is not None
    assert pos.shares == pytest.approx(100.0)


def test_buy_decreases_cash():
    pm = PortfolioManager(initial_capital=10_000.0)
    before = pm.portfolio.cash
    trade = make_trade(shares=100, price=0.50, commission=-0.0)
    pm.execute_trade(trade)
    after = pm.portfolio.cash
    assert after < before
    assert after == pytest.approx(before - 100 * 0.50)


def test_sell_decreases_position():
    pm = PortfolioManager(initial_capital=10_000.0)
    pm.execute_trade(make_trade(shares=100, price=0.50, commission=0.0))
    sell = make_trade(direction=TradeDirection.SELL, shares=50, price=0.55, commission=0.0)
    ok = pm.execute_trade(sell)
    assert ok
    pos = pm.portfolio.get_position("MKT_1")
    assert pos is not None
    assert pos.shares == pytest.approx(50.0)


def test_sell_all_removes_position():
    pm = PortfolioManager(initial_capital=10_000.0)
    pm.execute_trade(make_trade(shares=100, price=0.50, commission=0.0))
    sell = make_trade(direction=TradeDirection.SELL, shares=100, price=0.55, commission=0.0)
    pm.execute_trade(sell)
    assert pm.portfolio.get_position("MKT_1") is None


def test_oversell_capped():
    """Selling more than we own should be capped to current shares."""
    pm = PortfolioManager(initial_capital=10_000.0)
    pm.execute_trade(make_trade(shares=100, price=0.50, commission=0.0))
    sell = make_trade(direction=TradeDirection.SELL, shares=200, price=0.55, commission=0.0)
    ok = pm.execute_trade(sell)
    assert ok
    assert pm.portfolio.get_position("MKT_1") is None   # all sold


def test_sell_with_no_position_rejected():
    pm = PortfolioManager(initial_capital=10_000.0)
    sell = make_trade(direction=TradeDirection.SELL, shares=100, price=0.50)
    ok = pm.execute_trade(sell)
    assert not ok


# ---------------------------------------------------------------------------
# Mark-to-market
# ---------------------------------------------------------------------------

def test_mark_to_market_updates_value():
    pm = PortfolioManager(initial_capital=10_000.0)
    pm.execute_trade(make_trade(shares=100, price=0.50, commission=0.0))
    pm.mark_to_market({"MKT_1": 0.70}, ts())
    pos = pm.portfolio.get_position("MKT_1")
    assert pos.market_value == pytest.approx(100 * 0.70)


def test_high_water_mark_updates():
    pm = PortfolioManager(initial_capital=10_000.0)
    assert pm.high_water_mark == 10_000.0
    # Buy a position that gains value
    pm.execute_trade(make_trade(shares=100, price=0.50, commission=0.0))
    pm.mark_to_market({"MKT_1": 1.0}, ts())
    assert pm.high_water_mark > 10_000.0


# ---------------------------------------------------------------------------
# Resolution settlement
# ---------------------------------------------------------------------------

def make_market(
    market_id: str,
    resolution: int,
    status: MarketStatus,
) -> Market:
    return Market(
        market_id=market_id,
        name="Test",
        category=MarketCategory.POLITICS,
        open_date=datetime(2023, 1, 1),
        close_date=datetime(2023, 6, 1),
        status=status,
        resolution=resolution,
        total_volume=1000.0,
        last_price=0.5,
    )


def test_resolution_yes_pays_full():
    pm = PortfolioManager(initial_capital=10_000.0)
    pm.execute_trade(make_trade(shares=1000, price=0.60, commission=0.0))
    cash_before = pm.portfolio.cash

    market = make_market("MKT_1", resolution=1, status=MarketStatus.RESOLVED_YES)
    sett = pm.settle_resolved_markets([market])

    assert "MKT_1" in sett
    assert sett["MKT_1"]["payout"] == pytest.approx(1000.0)  # 1000 shares * $1
    assert pm.portfolio.get_position("MKT_1") is None
    assert pm.portfolio.cash > cash_before


def test_resolution_no_pays_zero():
    pm = PortfolioManager(initial_capital=10_000.0)
    pm.execute_trade(make_trade(shares=1000, price=0.60, commission=0.0))
    cash_before = pm.portfolio.cash

    market = make_market("MKT_1", resolution=0, status=MarketStatus.RESOLVED_NO)
    sett = pm.settle_resolved_markets([market])

    assert sett["MKT_1"]["payout"] == pytest.approx(0.0)
    assert pm.portfolio.cash == pytest.approx(cash_before)  # no payout received
