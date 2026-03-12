"""
Data loader utilities.

Provides adapters to load price history from:
  • CSV files (default for backtesting with local data)
  • Pandas DataFrames (in-memory, used by the synthetic generator)

Future extensions: Kalshi REST API, Polymarket GraphQL, PredictIt JSON feeds.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from pmif.models import Market, MarketCategory, MarketPrice, MarketStatus


class DataLoader:
    """
    Load market metadata and price history from flat files or DataFrames.

    Expected CSV schemas
    --------------------
    markets.csv  : market_id, name, category, open_date, close_date,
                   status, resolution, total_volume, last_price, description

    prices.csv   : market_id, timestamp, price, volume
    """

    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    # ------------------------------------------------------------------
    # CSV loaders
    # ------------------------------------------------------------------

    @classmethod
    def load_markets_csv(cls, path: str | Path) -> list[Market]:
        """Load a list of Market objects from a CSV file."""
        df = pd.read_csv(path, parse_dates=["open_date", "close_date"])
        return [cls._row_to_market(row) for _, row in df.iterrows()]

    @classmethod
    def load_prices_csv(cls, path: str | Path) -> pd.DataFrame:
        """
        Load price history from CSV.

        Returns a DataFrame with columns:
            market_id (str), timestamp (datetime), price (float), volume (float)
        """
        df = pd.read_csv(path, parse_dates=["timestamp"])
        cls._validate_prices_df(df)
        return df.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # DataFrame converters
    # ------------------------------------------------------------------

    @staticmethod
    def prices_df_to_ticks(df: pd.DataFrame, market_id: str) -> list[MarketPrice]:
        """Convert a price DataFrame to a list of MarketPrice objects for one market."""
        subset = df[df["market_id"] == market_id].copy()
        ticks: list[MarketPrice] = []
        for _, row in subset.iterrows():
            ticks.append(
                MarketPrice(
                    market_id=market_id,
                    timestamp=row["timestamp"],
                    price=float(row["price"]),
                    volume=float(row.get("volume", 0.0)),
                )
            )
        return ticks

    # ------------------------------------------------------------------
    # CSV writers (persist synthetic data for reproducibility)
    # ------------------------------------------------------------------

    @staticmethod
    def save_markets_csv(markets: list[Market], path: str | Path) -> None:
        """Serialize a list of Market objects to CSV."""
        rows = []
        for m in markets:
            rows.append(
                {
                    "market_id": m.market_id,
                    "name": m.name,
                    "category": m.category.value,
                    "open_date": m.open_date,
                    "close_date": m.close_date,
                    "status": m.status.value,
                    "resolution": m.resolution,
                    "total_volume": m.total_volume,
                    "last_price": m.last_price,
                    "description": m.description,
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    @staticmethod
    def save_prices_csv(df: pd.DataFrame, path: str | Path) -> None:
        """Save price history DataFrame to CSV."""
        df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _row_to_market(cls, row: pd.Series) -> Market:
        try:
            category = MarketCategory(row["category"])
        except ValueError:
            category = MarketCategory.OTHER

        try:
            status = MarketStatus(row["status"])
        except ValueError:
            status = MarketStatus.OPEN

        resolution = None
        if pd.notna(row.get("resolution")):
            resolution = int(row["resolution"])

        return Market(
            market_id=str(row["market_id"]),
            name=str(row["name"]),
            category=category,
            open_date=pd.Timestamp(row["open_date"]).to_pydatetime(),
            close_date=pd.Timestamp(row["close_date"]).to_pydatetime(),
            status=status,
            resolution=resolution,
            total_volume=float(row.get("total_volume", 0.0)),
            last_price=float(row.get("last_price", 0.5)),
            description=str(row.get("description", "")),
        )

    @staticmethod
    def _validate_prices_df(df: pd.DataFrame) -> None:
        required = {"market_id", "timestamp", "price"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Price CSV missing required columns: {missing}")
        out_of_range = df[(df["price"] < 0) | (df["price"] > 1)]
        if not out_of_range.empty:
            raise ValueError(
                f"Found {len(out_of_range)} rows with price outside [0, 1]."
            )
