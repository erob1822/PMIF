"""
Tests for data models.
"""

from datetime import datetime
import pytest
from pmif.models import (
    Market,
    MarketCategory,
    MarketPrice,
    MarketStatus,
    Portfolio,
    Position,
    Trade,
    TradeDirection,
)


# ---------------------------------------------------------------------------
# Market
# ---------------------------------------------------------------------------

def make_market(**kwargs) -> Market:
    defaults = dict(
        market_id="TEST_001",
        name="Will X happen?",
        category=MarketCategory.POLITICS,
        open_date=datetime(2023, 1, 1),
        close_date=datetime(2023, 12, 31),
        status=MarketStatus.OPEN,
        total_volume=10_000.0,
        last_price=0.55,
    )
    defaults.update(kwargs)
    return Market(**defaults)


def test_market_is_open():
    m = make_market(status=MarketStatus.OPEN)
    assert m.is_open
    assert not m.is_resolved


def test_market_is_resolved_yes():
    m = make_market(status=MarketStatus.RESOLVED_YES, resolution=1)
    assert m.is_resolved
    assert not m.is_open


def test_market_is_resolved_no():
    m = make_market(status=MarketStatus.RESOLVED_NO, resolution=0)
    assert m.is_resolved


def test_market_repr():
    m = make_market()
    r = repr(m)
    assert "TEST_001" in r
    assert "0.550" in r


# ---------------------------------------------------------------------------
# MarketPrice
# ---------------------------------------------------------------------------

def test_market_price_valid():
    mp = MarketPrice(
        market_id="TEST",
        timestamp=datetime(2023, 6, 1),
        price=0.72,
        volume=100.0,
    )
    assert mp.price == 0.72


def test_market_price_invalid_raises():
    with pytest.raises(ValueError):
        MarketPrice(
            market_id="TEST",
            timestamp=datetime(2023, 6, 1),
            price=1.5,   # out of range
        )

    with pytest.raises(ValueError):
        MarketPrice(
            market_id="TEST",
            timestamp=datetime(2023, 6, 1),
            price=-0.1,  # out of range
        )


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def test_portfolio_total_value_empty():
    port = Portfolio(cash=100_000.0)
    assert port.total_value == 100_000.0
    assert port.position_value == 0.0
    assert port.num_positions == 0


def test_portfolio_with_position():
    port = Portfolio(cash=90_000.0)
    port.positions["MKT_1"] = Position(
        market_id="MKT_1",
        shares=100,
        avg_cost=0.60,
        market_value=65.0,
    )
    assert port.total_value == pytest.approx(90_065.0)
    assert port.num_positions == 1


def test_portfolio_repr():
    port = Portfolio(cash=50_000.0)
    r = repr(port)
    assert "$50,000.00" in r


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

def test_position_unrealized_pnl():
    pos = Position(
        market_id="MKT_X",
        shares=200,
        avg_cost=0.50,
        market_value=200 * 0.60,
    )
    # cost = 200 * 0.50 = 100; value = 200 * 0.60 = 120; pnl = 20
    assert pos.unrealized_pnl == pytest.approx(20.0)
    assert pos.cost_basis == pytest.approx(100.0)
