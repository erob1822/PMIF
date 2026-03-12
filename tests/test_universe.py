"""
Tests for UniverseSelector.
"""

from datetime import datetime, timedelta
import pytest
from pmif.models import Market, MarketCategory, MarketStatus
from pmif.universe import UniverseSelector


AS_OF = datetime(2023, 6, 1)


def make_mkt(
    mid: str = "M1",
    price: float = 0.50,
    volume: float = 5_000.0,
    days_left: int = 60,
    category: MarketCategory = MarketCategory.ECONOMICS,
    status: MarketStatus = MarketStatus.OPEN,
) -> Market:
    now = datetime(2023, 6, 1)
    return Market(
        market_id=mid,
        name=f"Market {mid}",
        category=category,
        open_date=now - timedelta(days=10),
        close_date=now + timedelta(days=days_left),
        status=status,
        total_volume=volume,
        last_price=price,
    )


def test_basic_selection():
    selector = UniverseSelector(max_markets=10, min_volume=1_000)
    markets = [make_mkt(f"M{i}") for i in range(20)]
    selected = selector.select(markets, as_of=AS_OF)
    assert len(selected) <= 10
    assert all(m.is_open for m in selected)


def test_volume_filter():
    selector = UniverseSelector(min_volume=5_000.0)
    markets = [
        make_mkt("LOW_VOL", volume=100.0),
        make_mkt("HIGH_VOL", volume=10_000.0),
    ]
    selected = selector.select(markets, as_of=AS_OF)
    ids = [m.market_id for m in selected]
    assert "HIGH_VOL" in ids
    assert "LOW_VOL" not in ids


def test_price_filter():
    selector = UniverseSelector(min_price=0.05, max_price=0.95)
    markets = [
        make_mkt("NEAR_ZERO", price=0.01),
        make_mkt("NEAR_ONE", price=0.99),
        make_mkt("MID", price=0.50),
    ]
    selected = selector.select(markets, as_of=AS_OF)
    ids = [m.market_id for m in selected]
    assert "MID" in ids
    assert "NEAR_ZERO" not in ids
    assert "NEAR_ONE" not in ids


def test_days_to_close_filter():
    selector = UniverseSelector(min_days_to_close=5, max_days_to_close=180)
    markets = [
        make_mkt("TOO_CLOSE", days_left=2),
        make_mkt("TOO_FAR", days_left=400),
        make_mkt("JUST_RIGHT", days_left=30),
    ]
    selected = selector.select(markets, as_of=AS_OF)
    ids = [m.market_id for m in selected]
    assert "JUST_RIGHT" in ids
    assert "TOO_CLOSE" not in ids
    assert "TOO_FAR" not in ids


def test_resolved_markets_excluded():
    selector = UniverseSelector()
    markets = [
        make_mkt("OPEN", status=MarketStatus.OPEN),
        make_mkt("RESOLVED", status=MarketStatus.RESOLVED_YES),
    ]
    selected = selector.select(markets, as_of=AS_OF)
    ids = [m.market_id for m in selected]
    assert "OPEN" in ids
    assert "RESOLVED" not in ids


def test_sorted_by_volume_descending():
    selector = UniverseSelector(max_markets=5)
    markets = [
        make_mkt(f"M{i}", volume=float(i * 1000 + 2000)) for i in range(5)
    ]
    selected = selector.select(markets, as_of=AS_OF)
    volumes = [m.total_volume for m in selected]
    assert volumes == sorted(volumes, reverse=True)


def test_eligibility_report():
    selector = UniverseSelector()
    markets = [make_mkt("M1"), make_mkt("M2", volume=0)]
    report = selector.eligibility_report(markets, as_of=AS_OF)
    assert "eligible" in report.columns
    assert len(report) == 2
