# SMA Momentum Backtest

A simple SMA momentum strategy backtester using `yfinance`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python backtest.py --ticker SPY --start 2000-01-01 --end 2025-12-31 --sma-fast 50 --sma-slow 200 --fee-bps 5
```

Outputs an `equity_curve.csv` by default. Set `--output-csv ""` to skip writing.
Use `--plot` to display the equity curves, or `--plot-path plot.png` to save a chart.

## Grid search (PyTorch)

```bash
python backtest.py --ticker SPY --start 2000-01-01 --end 2025-12-31 --grid-search \
  --grid-fast 10:100:10 --grid-slow 50:300:25 --grid-metric sharpe
```

- Range format: `start:end:step` (inclusive)
- Or provide a list: `--grid-fast 10,20,50`
- Saves full results to `grid_search.csv` by default

## Strategy logic

- Compute fast and slow SMAs of price
- Go long when fast SMA > slow SMA
- Otherwise hold cash
- Apply fees on position changes (round-trip basis)
- Compare equity curve to buy & hold benchmark

## Notes

This is a minimal example intended for experimentation, not live trading.
