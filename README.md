# PMIF – Predictive Market Index Fund

> *Apply the wisdom of crowds to predictive markets the same way the S&P 500 applied it to equities.*

---

## Table of Contents
1. [What Is PMIF?](#1-what-is-pmif)
2. [How It Works](#2-how-it-works)
3. [Repository Structure](#3-repository-structure)
4. [Quick Start](#4-quick-start)
5. [Strategies](#5-strategies)
6. [Backtesting](#6-backtesting)
7. [Risk Management](#7-risk-management)
8. [Realistic Return Expectations](#8-realistic-return-expectations)
9. [Roadmap](#9-roadmap)
10. [Contributing](#10-contributing)

---

## 1. What Is PMIF?

The **Predictive Market Index Fund** is a systematic, diversified investment fund that participates in prediction markets (Kalshi, Polymarket, PredictIt, etc.) the way an index fund participates in equities: by holding a broad basket of the most liquid contracts, weighted by trading volume.

### The Core Insight

The S&P 500's genius was not stock-picking — it was **trusting the wisdom of crowds** (market prices reflect aggregate information) and **diversifying away idiosyncratic risk**. PMIF applies the same two principles to prediction markets:

| S&P 500 | PMIF |
|---|---|
| Tracks top 500 companies by market cap | Tracks top 500–5,000 markets by trading volume |
| Weights by market capitalisation | Weights by cumulative trading volume |
| Rebalances quarterly | Rebalances daily/weekly |
| Expected return ≈ economy growth | Expected return ≈ liquidity premium + bias harvest |

In a prediction market, a contract trading at **$0.65 pays $1.00 if the event occurs** and $0.00 if it doesn't. The price *is* the crowd's probability estimate. A diversified PMIF therefore holds a portfolio whose expected payout mirrors the crowd's aggregate calibration.

---

## 2. How It Works

### Signal

Prices in prediction markets are probabilities. A contract priced at 0.60 means the market consensus is 60% chance of YES. We don't need to know whether an event will happen — we trust the crowd, diversify broadly, and harvest two structural edges:

1. **Longshot Bias** — Documented across all prediction markets, people systematically *overbet* unlikely events (< 12%) and *underprice* near-certainties (> 88%). We fade this bias.

2. **Time-Lag Arbitrage** — When breaking news reprices one market in a category, related markets in the same category often lag by minutes to hours. We detect these correlated mispricings and trade them.

### Portfolio Construction

```
All available markets
        ↓
  Universe Selector  (volume ≥ min, 3–365 days to close, 3%–97% price band)
        ↓
  Top-N by volume  (with category diversification caps)
        ↓
  Volume-weighted allocation  (higher volume → better calibrated → larger weight)
        ↓
  Kelly-capped position sizing  (no single position > 2% NAV)
        ↓
  Risk Manager check  (drawdown circuit breaker, leverage limit)
        ↓
  Execute trades
```

---

## 3. Repository Structure

```
PMIF/
├── README.md                  ← You are here
├── Requirements               ← Original project requirements
├── notes.txt                  ← Design rationale and recommendations
├── requirements.txt           ← Python dependencies
├── .gitignore
│
├── src/
│   └── pmif/
│       ├── __init__.py
│       ├── models.py          ← Data classes: Market, Position, Portfolio, Trade
│       ├── universe.py        ← Market universe selection (S&P-committee equivalent)
│       ├── strategy.py        ← Trading strategies (index, longshot fade, time-lag arb)
│       ├── portfolio.py       ← Portfolio execution and position management
│       ├── risk.py            ← Kelly sizing, drawdown circuit breaker
│       ├── backtester.py      ← Event-driven backtesting engine
│       ├── metrics.py         ← Sharpe, drawdown, CAGR, Sortino, Calmar
│       └── data/
│           ├── generator.py   ← Synthetic market data for backtesting
│           └── loader.py      ← CSV / DataFrame data loading utilities
│
├── scripts/
│   └── run_backtest.py        ← Main demo: generate data + run backtest
│
└── tests/
    ├── conftest.py
    ├── test_models.py
    ├── test_universe.py
    ├── test_portfolio.py
    ├── test_metrics.py
    └── test_backtester.py     ← End-to-end integration tests
```

---

## 4. Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Demo Backtest

```bash
python scripts/run_backtest.py
```

This generates 200 synthetic prediction markets over 3 years and runs the combined index + longshot fade strategy.

**Options:**

```
--markets N        Number of synthetic markets (default: 200)
--capital C        Starting capital in USD (default: 1,000,000)
--start YYYY-MM-DD Simulation start date
--end   YYYY-MM-DD Simulation end date
--seed  S          Random seed for reproducibility
--strategy         index | longshot | combined (default: combined)
--save-data        Save synthetic data to data/sample/ as CSV
```

**Example:**
```bash
python scripts/run_backtest.py --markets 500 --capital 1000000 \
    --start 2021-01-01 --end 2023-12-31 --strategy combined
```

### Run the Tests

```bash
python -m pytest tests/ -v
```

---

## 5. Strategies

### 5.1 Liquidity-Weighted Index (`LiquidityWeightedIndexStrategy`)

The **primary** PMIF strategy. Passive, systematic, low-turnover.

- Allocates capital proportionally to each market's share of total universe volume
- Exactly mirrors market-cap weighting used by S&P 500 / Vanguard
- Rebalances on each call (caller controls frequency — default: daily)
- Exits positions that drop out of the eligible universe (too close to resolution, too illiquid)

```python
from pmif.strategy import LiquidityWeightedIndexStrategy
strategy = LiquidityWeightedIndexStrategy(
    target_investment_pct=0.95,  # keep 5% cash buffer
    min_trade_size=10.0,         # ignore tiny rebalancing trades
    commission_rate=0.001,       # 0.1% transaction cost
)
```

### 5.2 Longshot Fade (`LongshotFadeStrategy`)

Harvests the well-documented favourite-longshot bias.

- **SHORT YES** (buy NO) on contracts priced < 12% — these resolve YES only ~6% of the time, creating a ~2% structural edge
- **LONG YES** on contracts priced > 88% — these resolve YES ~94% of the time
- Small positions (0.5% of NAV max) to remain diversified

```python
from pmif.strategy import LongshotFadeStrategy
strategy = LongshotFadeStrategy(
    longshot_threshold=0.12,
    favorite_threshold=0.88,
    min_edge=0.02,
    max_position_pct=0.005,
)
```

### 5.3 Time-Lag Arbitrage (`TimeLagArbitrageStrategy`)

Exploits information propagation delays across correlated markets.

- Detects large logit-return moves (> 0.40 in one tick) as "shocks"
- Finds correlated markets in the same category that haven't repriced yet
- Trades in the direction of the shock, closes after 4 hours

```python
from pmif.strategy import TimeLagArbitrageStrategy
strategy = TimeLagArbitrageStrategy(
    shock_threshold=0.40,
    hold_hours=4,
    max_position_pct=0.003,
)
```

---

## 6. Backtesting

The `Backtester` class runs an **event-driven simulation** against historical (or synthetic) price data.

### Using Synthetic Data (for development)

```python
from pmif.data.generator import SyntheticDataGenerator
from pmif.backtester import Backtester
from pmif.strategy import LiquidityWeightedIndexStrategy

gen = SyntheticDataGenerator(num_markets=200, seed=42)
markets, price_df = gen.generate()

bt = Backtester(
    markets=markets,
    price_history=price_df,
    initial_capital=1_000_000,
    rebalance_every_n_ticks=24,   # daily rebalance with hourly ticks
    strategies=[LiquidityWeightedIndexStrategy()],
)
result = bt.run()
result.print_summary()
```

### Using Real CSV Data

```python
from pmif.data.loader import DataLoader
from pmif.backtester import Backtester

markets = DataLoader.load_markets_csv("data/kalshi_markets.csv")
prices  = DataLoader.load_prices_csv("data/kalshi_prices.csv")

bt = Backtester(markets=markets, price_history=prices, ...)
```

### Interpreting Results

```
PMIF Performance Summary
==================================================
  Total Return       :    55.94%    ← cumulative return over simulation
  CAGR               :    56.77%    ← annualised compound growth rate
  Annualised Vol     :    40.13%    ← standard deviation of returns (ann.)
  Sharpe Ratio       :     1.302    ← risk-adjusted return (>1 = good)
  Sortino Ratio      :     1.547    ← Sharpe using downside vol only
  Max Drawdown       :    18.61%    ← worst peak-to-trough decline
  Calmar Ratio       :     3.050    ← CAGR / max drawdown
  Avg Drawdown       :     6.00%    ← average underwater period size
  Observation periods:     8663     ← number of hourly ticks
```

> ⚠️ **Important**: These results use synthetic data with 0.1% commissions.  Real prediction markets have bid-ask spreads of 2-5% and platform fees of 5-7% of winnings.  Apply a realistic -15 to -20% annual drag before comparing to real-world expectations.  See [notes.txt](notes.txt) for a detailed discussion.

---

## 7. Risk Management

The `RiskManager` implements several controls:

| Control | Default | Purpose |
|---|---|---|
| `max_position_weight` | 2% of NAV | No single market dominates |
| `max_category_weight` | 25% of NAV | No single category dominates |
| `max_drawdown_pct` | 20% | Circuit breaker: halt new trades if drawdown > 20% |
| `max_kelly_fraction` | 0.25 | Quarter-Kelly sizing (conservative) |
| `min_positions` | 50 | Fund must hold ≥ 50 positions to be considered diversified |
| `max_leverage` | 1.0 | No leverage by default |

### Position Sizing

Position sizes use the **Kelly Criterion** capped at quarter-Kelly:

```
Kelly fraction = edge / (1 - price)
Capped at:  min(quarter_kelly, max_position_weight)
```

Where `edge` = estimated probability edge (fair probability minus market price).  In the index strategy, we assume zero edge (we trust the crowd) so positions are sized purely by the volume-weighting.

---

## 8. Realistic Return Expectations

| Scenario | Annual Return | Notes |
|---|---|---|
| Synthetic backtest | ~50-60% | Synthetic data, minimal costs |
| After realistic spreads (-15%) | ~35-45% | Still overstated |
| After platform fees (-8%) | ~27-37% | Approaching realistic |
| Conservative real-world estimate | **5-12%** | Liquidity-constrained, real calibration |
| Moderate (bias harvesting works) | **12-20%** | If systematic edges persist |
| Optimistic (+ fast time-lag arb) | **20-30%** | Requires low latency execution |

For context: best-in-class systematic prediction market funds (as of 2024) report 15-25% annual returns with Sharpe ratios of 0.8-1.5.

---

## 9. Roadmap

- [x] Core algorithm design and implementation
- [x] Synthetic data generator for backtesting
- [x] Liquidity-weighted index strategy
- [x] Longshot fade strategy
- [x] Time-lag arbitrage strategy
- [x] Event-driven backtester
- [x] Performance metrics (Sharpe, drawdown, CAGR, Sortino, Calmar)
- [x] Risk management (Kelly sizing, circuit breakers, category caps)
- [ ] **Phase 1**: Kalshi API connector (live data)
- [ ] **Phase 1**: Paper trading mode
- [ ] **Phase 1**: Realistic bid-ask spread simulation
- [ ] **Phase 2**: Bayesian calibration prior
- [ ] **Phase 2**: Multi-platform aggregation (Polymarket, Manifold)
- [ ] **Phase 2**: Market-maker mode (post limit orders to earn the spread)
- [ ] **Phase 3**: Legal/fund structure review
- [ ] **Phase 3**: Formal benchmark index construction

---

## 10. Contributing

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest tests/ -v`
4. See [notes.txt](notes.txt) for detailed design rationale

---

*For detailed design rationale, expected returns analysis, risk factors, and next steps see [notes.txt](notes.txt).*

