# Portfolio Optimization Research Pipeline

Institutional-grade quantitative portfolio construction implementing Black-Litterman, ML alpha, CVaR optimization, and robust methods.

## Setup

```bash
pip install -r requirements.txt
```

## Data Requirements

- **Bloomberg CSV**: Place at `C:\Users\manoj\Downloads\SPX as of Oct 08 20251 (1).csv`
  - Must contain: Dates, PX_LAST, PX_VOLUME, PX_OPEN, PX_HIGH, PX_LOW per equity
- **Market cap & fundamentals**: Fetched from yfinance for the 30 tickers selected by `portfo77.py`

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `data_loader.py` | Price (OHLCV) from Bloomberg, market cap from yfinance |
| 2 | `feature_engineering.py` | Returns, momentum, risk, liquidity, composite alpha |
| 3 | `expected_returns.py` | Equilibrium prior, Black-Litterman posterior |
| 4 | `covariance_estimation.py` | Ledoit-Wolf shrinkage |
| 5 | `optimization_engines.py` | MV, Mean-CVaR, Robust CVaR, Risk Parity |
| 6 | `constraints.py` | Sector, turnover, liquidity constraints |

## Run

```bash
python main_pipeline.py
```

Or run individual stages:
```bash
python data_loader.py      # Load data, get 30 tickers, fetch market cap
python portfo77.py         # Get selected tickers only
```

## Output

- `data/portfolio_weights.csv` — Weights from MV, CVaR, and Risk Parity

## Notes

- **Fundamental data**: yfinance uses fiscal period end dates, not earnings release dates — creates look-ahead bias. For production, use Bloomberg/FactSet with point-in-time release dates.
- **Sentiment**: Excluded per spec; to be combined later by another team.
- **Config**: Edit `config.py` for date ranges, constraints, and parameters.
