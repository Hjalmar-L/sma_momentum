from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


@dataclass
class BacktestConfig:
    ticker: str
    start: str
    end: str
    sma_fast: int
    sma_slow: int
    fee_bps: float
    initial_cash: float
    output_csv: Optional[str]
    plot: bool
    plot_path: Optional[str]
    grid_search: bool
    grid_fast: str
    grid_slow: str
    grid_metric: str
    grid_top: int
    grid_output_csv: Optional[str]


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            raise ValueError("Expected 'Close' in downloaded data.")
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            if ticker in close.columns:
                close = close[ticker]
            elif close.shape[1] == 1:
                close = close.iloc[:, 0]
            else:
                raise ValueError("Multiple Close columns returned; specify a single ticker.")
        return close.rename("price")
    if "Close" not in df.columns:
        raise ValueError("Expected 'Close' in downloaded data.")
    return df["Close"].rename("price")


def run_backtest(prices: pd.Series, cfg: BacktestConfig) -> pd.DataFrame:
    data = pd.DataFrame(prices).copy()
    data["sma_fast"] = data["price"].rolling(cfg.sma_fast).mean()
    data["sma_slow"] = data["price"].rolling(cfg.sma_slow).mean()
    data["signal"] = (data["sma_fast"] > data["sma_slow"]).astype(int)

    data["position"] = data["signal"].shift(1).fillna(0).astype(int)
    data["daily_return"] = data["price"].pct_change().fillna(0)

    fee = cfg.fee_bps / 10000.0
    trades = data["position"].diff().abs().fillna(0)
    data["strategy_return"] = data["position"] * data["daily_return"] - trades * fee

    data["equity"] = cfg.initial_cash * (1 + data["strategy_return"]).cumprod()
    data["buy_hold_equity"] = cfg.initial_cash * (1 + data["daily_return"]).cumprod()

    return data


def torch_sma(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("SMA window must be positive.")
    if window > values.size:
        return np.full_like(values, np.nan, dtype=np.float32)
    t = torch.tensor(values, dtype=torch.float32).view(1, 1, -1)
    kernel = torch.ones(1, 1, window, dtype=torch.float32) / float(window)
    out = F.conv1d(t, kernel)
    sma = np.full(values.shape[0], np.nan, dtype=np.float32)
    sma[window - 1 :] = out.view(-1).numpy()
    return sma


def parse_range(text: str) -> list[int]:
    if not text:
        return []
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError("Range must be formatted as start:end:step")
        start, end, step = (int(p) for p in parts)
        if step <= 0:
            raise ValueError("Range step must be positive.")
        if end < start:
            raise ValueError("Range end must be >= start.")
        return list(range(start, end + 1, step))
    return [int(p.strip()) for p in text.split(",") if p.strip()]


def grid_search(prices: pd.Series, cfg: BacktestConfig, fast_list: list[int], slow_list: list[int]) -> pd.DataFrame:
    values = prices.values.astype(np.float32)
    daily_returns = pd.Series(values).pct_change().fillna(0).values.astype(np.float32)
    fee = cfg.fee_bps / 10000.0

    results = []
    sma_cache: dict[int, np.ndarray] = {}

    for fast in fast_list:
        if fast not in sma_cache:
            sma_cache[fast] = torch_sma(values, fast)
        sma_fast = sma_cache[fast]
        for slow in slow_list:
            if fast >= slow:
                continue
            if slow not in sma_cache:
                sma_cache[slow] = torch_sma(values, slow)
            sma_slow = sma_cache[slow]

            signal = (sma_fast > sma_slow).astype(np.float32)
            signal = np.nan_to_num(signal, nan=0.0)
            position = np.roll(signal, 1)
            position[0] = 0.0
            trades = np.abs(np.diff(position, prepend=position[0]))

            strategy_return = position * daily_returns - trades * fee
            equity = cfg.initial_cash * np.cumprod(1 + strategy_return)

            total_return = equity[-1] / cfg.initial_cash - 1
            days = (prices.index[-1] - prices.index[0]).days
            years = days / 365.25 if days > 0 else 0
            cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

            vol = strategy_return.std() * np.sqrt(252)
            sharpe = (strategy_return.mean() * 252) / vol if vol > 0 else 0

            running_max = np.maximum.accumulate(equity)
            drawdown = equity / running_max - 1
            max_drawdown = float(drawdown.min())

            results.append(
                {
                    "sma_fast": fast,
                    "sma_slow": slow,
                    "total_return": float(total_return),
                    "cagr": float(cagr),
                    "sharpe": float(sharpe),
                    "max_drawdown": max_drawdown,
                    "trades": int(trades.sum()),
                }
            )

    return pd.DataFrame(results)


def calc_metrics(bt: pd.DataFrame, cfg: BacktestConfig) -> dict:
    total_return = bt["equity"].iloc[-1] / cfg.initial_cash - 1
    days = (bt.index[-1] - bt.index[0]).days
    years = days / 365.25 if days > 0 else 0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    daily = bt["strategy_return"]
    vol = daily.std() * np.sqrt(252)
    sharpe = (daily.mean() * 252) / vol if vol > 0 else 0

    rolling_max = bt["equity"].cummax()
    drawdown = bt["equity"] / rolling_max - 1
    max_drawdown = drawdown.min()

    trades = bt["position"].diff().abs().sum()

    bench_total = bt["buy_hold_equity"].iloc[-1] / cfg.initial_cash - 1
    bench_cagr = (1 + bench_total) ** (1 / years) - 1 if years > 0 else 0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annual Vol": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Trades": int(trades),
        "Benchmark Total Return": bench_total,
        "Benchmark CAGR": bench_cagr,
    }


def plot_results(bt: pd.DataFrame, cfg: BacktestConfig, output_path: Optional[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bt.index, bt["equity"], label="SMA Strategy", linewidth=2)
    ax.plot(bt.index, bt["buy_hold_equity"], label="Buy & Hold", linewidth=2, alpha=0.8)
    ax.set_title(f"{cfg.ticker} SMA Crossover ({cfg.sma_fast}/{cfg.sma_slow}) Long-only vs Buy & Hold")
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
    else:
        plt.show()


def parse_args(argv: list[str]) -> BacktestConfig:
    parser = argparse.ArgumentParser(description="SMA crossover backtest.")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol")
    parser.add_argument("--start", default="2000-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    parser.add_argument("--sma-fast", type=int, default=50, help="Fast SMA lookback window")
    parser.add_argument("--sma-slow", type=int, default=200, help="Slow SMA lookback window")
    parser.add_argument("--fee-bps", type=float, default=5.0, help="Round-trip fee in bps per position change")
    parser.add_argument("--initial-cash", type=float, default=100000.0, help="Initial equity")
    parser.add_argument("--output-csv", default="equity_curve.csv", help="Output CSV path (or empty to skip)")
    parser.add_argument("--plot", action="store_true", help="Show an equity curve plot")
    parser.add_argument("--plot-path", default="", help="Save plot to a file instead of showing it")
    parser.add_argument("--grid-search", action="store_true", help="Run a grid search for SMA windows")
    parser.add_argument("--grid-fast", default="10:100:10", help="Fast SMA list or range start:end:step")
    parser.add_argument("--grid-slow", default="50:300:25", help="Slow SMA list or range start:end:step")
    parser.add_argument("--grid-metric", default="sharpe", help="Metric to rank: sharpe, cagr, total_return")
    parser.add_argument("--grid-top", type=int, default=10, help="Number of top grid results to show")
    parser.add_argument("--grid-output-csv", default="grid_search.csv", help="Grid search output CSV path (or empty to skip)")

    args = parser.parse_args(argv)
    output_csv = args.output_csv or None

    cfg = BacktestConfig(
        ticker=args.ticker.upper(),
        start=args.start,
        end=args.end,
        sma_fast=args.sma_fast,
        sma_slow=args.sma_slow,
        fee_bps=args.fee_bps,
        initial_cash=args.initial_cash,
        output_csv=output_csv,
        plot=args.plot,
        plot_path=args.plot_path or None,
        grid_search=args.grid_search,
        grid_fast=args.grid_fast,
        grid_slow=args.grid_slow,
        grid_metric=args.grid_metric,
        grid_top=args.grid_top,
        grid_output_csv=args.grid_output_csv or None,
    )
    return cfg


def main(argv: list[str]) -> int:
    cfg = parse_args(argv)
    prices = download_prices(cfg.ticker, cfg.start, cfg.end)

    if cfg.grid_search:
        fast_list = parse_range(cfg.grid_fast)
        slow_list = parse_range(cfg.grid_slow)
        if not fast_list or not slow_list:
            raise ValueError("Grid ranges must produce at least one value each.")

        metric = cfg.grid_metric.lower()
        if metric not in {"sharpe", "cagr", "total_return"}:
            raise ValueError("--grid-metric must be one of: sharpe, cagr, total_return.")

        results = grid_search(prices, cfg, fast_list, slow_list)
        if results.empty:
            raise ValueError("Grid search produced no valid fast/slow pairs.")

        results = results.sort_values(metric, ascending=False)
        top_n = results.head(cfg.grid_top)

        print("SMA Grid Search")
        print(f"Ticker: {cfg.ticker}")
        print(f"Date Range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Fast Range: {cfg.grid_fast}")
        print(f"Slow Range: {cfg.grid_slow}")
        print("Mode: Long-only")
        print(f"Rank Metric: {metric}")
        print("")
        print(top_n.to_string(index=False))

        if cfg.grid_output_csv:
            results.to_csv(cfg.grid_output_csv, index=False)
            print(f"\nSaved grid results to {cfg.grid_output_csv}")

        best = results.iloc[0]
        cfg.sma_fast = int(best["sma_fast"])
        cfg.sma_slow = int(best["sma_slow"])

    if cfg.sma_fast >= cfg.sma_slow:
        raise ValueError("--sma-fast must be less than --sma-slow for a crossover strategy.")

    bt = run_backtest(prices, cfg)
    metrics = calc_metrics(bt, cfg)

    print("\nSMA Momentum Backtest")
    print(f"Ticker: {cfg.ticker}")
    print(f"Date Range: {bt.index[0].date()} to {bt.index[-1].date()}")
    print(f"SMA Windows: {cfg.sma_fast}/{cfg.sma_slow}")
    print("Mode: Long-only")
    print(f"Fees (bps): {cfg.fee_bps}")
    print("")
    for k, v in metrics.items():
        if k == "Trades":
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.2%}")

    if cfg.output_csv:
        bt.to_csv(cfg.output_csv)
        print(f"\nSaved equity curve to {cfg.output_csv}")

    if cfg.plot or cfg.plot_path:
        plot_results(bt, cfg, cfg.plot_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
