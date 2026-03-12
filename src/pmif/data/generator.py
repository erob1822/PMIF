"""
Synthetic data generator for backtesting PMIF strategies.

Simulates a realistic universe of predictive market contracts including:
  • Diverse categories (politics, economics, sports, crypto, etc.)
  • Brownian-motion price paths that drift toward the eventual resolution
  • Volume ramp-up near resolution (markets get more attention near close)
  • Occasional news-shock jumps
  • Calibrated resolution: a contract priced at p resolves YES roughly p
    of the time (crowd-calibrated markets), with optional longshot-bias.
"""

from __future__ import annotations

import random
import string
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from pmif.models import Market, MarketCategory, MarketPrice, MarketStatus


class SyntheticDataGenerator:
    """
    Generates a synthetic universe of predictive market contracts together
    with full price histories.

    Parameters
    ----------
    num_markets : int
        Number of markets to generate.
    start_date : datetime
        Simulation start date.
    end_date : datetime
        Simulation end date.
    tick_interval_hours : float
        Granularity of price ticks (default 1 hour).
    seed : int | None
        Random seed for reproducibility.
    longshot_bias : float
        Fraction by which longshot probabilities are inflated (0 = calibrated).
        E.g. 0.10 means a true-10%-probability event trades at ~11%.
    """

    CATEGORY_WEIGHTS: dict[MarketCategory, float] = {
        MarketCategory.POLITICS: 0.25,
        MarketCategory.ECONOMICS: 0.20,
        MarketCategory.SPORTS: 0.20,
        MarketCategory.CRYPTO: 0.15,
        MarketCategory.WEATHER: 0.10,
        MarketCategory.SCIENCE: 0.05,
        MarketCategory.ENTERTAINMENT: 0.05,
    }

    def __init__(
        self,
        num_markets: int = 200,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        tick_interval_hours: float = 1.0,
        seed: Optional[int] = 42,
        longshot_bias: float = 0.08,
    ) -> None:
        self.num_markets = num_markets
        self.start_date = start_date or datetime(2021, 1, 1)
        self.end_date = end_date or datetime(2023, 12, 31)
        self.tick_interval_hours = tick_interval_hours
        self.longshot_bias = longshot_bias
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> tuple[list[Market], pd.DataFrame]:
        """
        Generate the full synthetic dataset.

        Returns
        -------
        markets : list[Market]
            All generated market objects (with final resolution and status set).
        price_history : pd.DataFrame
            Columns: market_id, timestamp, price, volume.
        """
        markets: list[Market] = []
        all_ticks: list[dict] = []

        total_days = (self.end_date - self.start_date).days

        for _ in range(self.num_markets):
            market, ticks = self._generate_single_market(total_days)
            markets.append(market)
            all_ticks.extend(ticks)

        price_df = pd.DataFrame(all_ticks)
        if not price_df.empty:
            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
            price_df = price_df.sort_values(["market_id", "timestamp"]).reset_index(
                drop=True
            )

        return markets, price_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_single_market(
        self, total_sim_days: int
    ) -> tuple[Market, list[dict]]:
        """Generate one market and its full price history."""
        market_id = "MKT_" + "".join(
            random.choices(string.ascii_uppercase + string.digits, k=8)
        )
        category = self._sample_category()
        true_prob = float(self._rng.beta(a=2, b=2))   # centred around 0.5

        # Market duration: 14–365 days, starting at a random offset
        duration_days = int(self._rng.integers(14, 366))
        offset_days = int(self._rng.integers(0, max(1, total_sim_days - duration_days)))
        open_date = self.start_date + timedelta(days=offset_days)
        close_date = open_date + timedelta(days=duration_days)

        # Initial price: noisy estimate of true probability
        initial_noise = float(self._rng.normal(0, 0.12))
        initial_price = float(
            np.clip(true_prob + initial_noise, 0.02, 0.98)
        )

        # Generate price path
        ticks = self._simulate_price_path(
            market_id=market_id,
            open_date=open_date,
            close_date=close_date,
            true_prob=true_prob,
            initial_price=initial_price,
            duration_days=duration_days,
        )

        # Resolution (predetermined but not yet public – backtester reveals it at close_date)
        resolve_yes = bool(self._rng.random() < true_prob)
        # Markets start OPEN; the backtester transitions them to resolved at close_date
        status = MarketStatus.OPEN

        # Base volume proportional to duration and "interest" (random)
        interest = float(self._rng.lognormal(mean=8, sigma=1.5))
        total_volume = interest * duration_days

        market = Market(
            market_id=market_id,
            name=self._make_name(category, market_id),
            category=category,
            open_date=open_date,
            close_date=close_date,
            status=status,
            resolution=1 if resolve_yes else 0,
            total_volume=total_volume,
            last_price=ticks[-1]["price"] if ticks else initial_price,
        )
        return market, ticks

    def _simulate_price_path(
        self,
        market_id: str,
        open_date: datetime,
        close_date: datetime,
        true_prob: float,
        initial_price: float,
        duration_days: int,
    ) -> list[dict]:
        """
        Simulate an hour-by-hour price path using a mean-reverting random walk.

        Price dynamics:
          logit(p_{t+1}) = logit(p_t)
                         + κ * (logit(true_p) - logit(p_t)) * dt
                         + σ * sqrt(dt) * ε
                         + jump (rare news shock)
        where κ is mean-reversion speed and σ is volatility.
        """
        total_hours = int(duration_days * 24 / self.tick_interval_hours)
        if total_hours < 2:
            return []

        dt = self.tick_interval_hours / (duration_days * 24)
        kappa = 1.5          # mean-reversion speed
        sigma = 0.35         # diffusion volatility
        jump_prob = 0.002    # probability of news shock per tick

        def logit(p: float) -> float:
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return float(np.log(p / (1 - p)))

        def sigmoid(x: float) -> float:
            return float(1 / (1 + np.exp(-x)))

        # Introduce longshot bias in the observable price
        biased_true = self._apply_longshot_bias(true_prob)

        logit_true = logit(biased_true)
        logit_price = logit(initial_price)

        ticks: list[dict] = []
        current_time = open_date
        interval = timedelta(hours=self.tick_interval_hours)

        # Volume curve: low at open, peaks ~3 days before close
        base_volume = 50.0
        noise_arr = self._rng.standard_normal(total_hours)
        jump_arr = self._rng.random(total_hours)

        for i in range(total_hours):
            price = sigmoid(logit_price)

            # Volume ramp: increases toward close
            frac = i / total_hours
            volume_mult = 0.5 + 3.0 * (frac ** 2)
            volume = base_volume * volume_mult * abs(float(self._rng.lognormal(0, 0.3)))

            ticks.append(
                {
                    "market_id": market_id,
                    "timestamp": current_time,
                    "price": round(float(np.clip(price, 0.01, 0.99)), 4),
                    "volume": round(volume, 2),
                }
            )

            # Evolve logit price
            drift = kappa * (logit_true - logit_price) * dt
            diffusion = sigma * np.sqrt(dt) * noise_arr[i]
            logit_price += drift + diffusion

            # Occasional jump (news shock)
            if jump_arr[i] < jump_prob:
                jump_size = float(self._rng.normal(0, 0.5))
                logit_price += jump_size

            current_time += interval
            if current_time >= close_date:
                break

        return ticks

    def _apply_longshot_bias(self, true_prob: float) -> float:
        """
        Apply longshot bias: inflate probabilities near extremes.
        Markets systematically overrate unlikely events.
        """
        if self.longshot_bias == 0:
            return true_prob
        # Pull extreme probs toward 0.5 (markets overestimate longshots)
        distance_from_edge = min(true_prob, 1 - true_prob)
        bias_strength = self.longshot_bias * (1 - 2 * distance_from_edge)
        return float(np.clip(true_prob + bias_strength, 0.01, 0.99))

    def _sample_category(self) -> MarketCategory:
        categories = list(self.CATEGORY_WEIGHTS.keys())
        weights = list(self.CATEGORY_WEIGHTS.values())
        return random.choices(categories, weights=weights, k=1)[0]

    @staticmethod
    def _make_name(category: MarketCategory, market_id: str) -> str:
        templates: dict[MarketCategory, list[str]] = {
            MarketCategory.POLITICS: [
                "Will Party A win the next election?",
                "Will the incumbent be re-elected?",
                "Will the bill pass the Senate?",
                "Will the candidate drop out by month-end?",
            ],
            MarketCategory.ECONOMICS: [
                "Will GDP grow above 2% this quarter?",
                "Will the Fed raise rates in Q3?",
                "Will CPI exceed 4% year-over-year?",
                "Will unemployment drop below 4%?",
            ],
            MarketCategory.SPORTS: [
                "Will Team X win the championship?",
                "Will the underdog cover the spread?",
                "Will the game go to overtime?",
                "Will the top seed reach the finals?",
            ],
            MarketCategory.CRYPTO: [
                "Will BTC exceed $50,000 by month-end?",
                "Will ETH flip BTC by market cap?",
                "Will the SEC approve a spot ETF?",
                "Will the altcoin season peak this quarter?",
            ],
            MarketCategory.WEATHER: [
                "Will the hurricane reach Category 4?",
                "Will average temps exceed historic highs?",
                "Will there be above-average rainfall?",
                "Will the wildfire season be declared extreme?",
            ],
            MarketCategory.SCIENCE: [
                "Will the clinical trial show efficacy?",
                "Will the mission successfully land?",
                "Will the new model beat the benchmark?",
                "Will the merger close before year-end?",
            ],
            MarketCategory.ENTERTAINMENT: [
                "Will the film gross over $200M opening weekend?",
                "Will the album debut at #1?",
                "Will the show be renewed for another season?",
                "Will the awards ceremony viewership beat last year?",
            ],
        }
        options = templates.get(category, [f"Event {market_id}"])
        return random.choice(options)
