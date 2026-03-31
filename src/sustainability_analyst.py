"""
Sustainability & ESG Analyst Agent
====================================
Part of the Human-in-Command AI Workforce for the CRE Intelligence Platform.

This agent monitors the clean-energy and green-building investment landscape
to gauge ESG momentum relevant to commercial real estate:

  - ICLN: iShares Global Clean Energy ETF
  - TAN:  Invesco Solar ETF
  - QCLN: First Trust NASDAQ Clean Edge Green Energy ETF

Green REIT proxies (REITs with strong sustainability profiles):
  - PROLOGIS (PLD)  — logistics/industrial, LEED-certified portfolio
  - EQUINIX  (EQIX) — data centres with renewable-energy commitments
  - ALEXANDRIA (ARE) — life-science campuses, carbon-neutral goals

Outputs an "ESG Momentum Signal" (STRONG / NEUTRAL / WEAK) based on
trailing performance of clean-energy ETFs vs the broad market (SPY).

Cache: cache/sustainability_data.json
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

CLEAN_ENERGY_TICKERS = {
    "ICLN": "Global Clean Energy (ICLN)",
    "TAN": "Solar (TAN)",
    "QCLN": "Clean Tech (QCLN)",
}

GREEN_REIT_TICKERS = {
    "PLD": "Prologis (PLD)",
    "EQIX": "Equinix (EQIX)",
    "ARE": "Alexandria Real Estate (ARE)",
}

BENCHMARK = "SPY"
LOOKBACK = "6mo"
SMA_WINDOW = 60


def _write_cache(data: dict, signal: str):
    """Write sustainability data to cache in Chief-of-Staff-compatible format."""
    payload = {
        "agent_name": "sustainability_analyst",
        "timestamp": datetime.now().isoformat(),
        "signals": {
            "esg_momentum_signal": signal,
        },
        "data": data,
    }
    with open(CACHE_DIR / "sustainability_data.json", "w") as f:
        json.dump(payload, f, default=str, indent=2)


def _fetch_group(tickers: dict, group_name: str) -> pd.DataFrame:
    """Download closing prices for a group of tickers."""
    print(f"[Sustainability Analyst] Fetching {group_name} ...")
    frames = []
    for ticker, label in tickers.items():
        print(f"  -> {label}")
        try:
            hist = yf.Ticker(ticker).history(period=LOOKBACK)
            if hist.empty:
                print(f"     WARNING: no data for {ticker}")
                continue
            close = hist[["Close"]].rename(columns={"Close": ticker})
            frames.append(close)
        except Exception as e:
            print(f"     ERROR fetching {ticker}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).dropna()


def _ticker_stats(df: pd.DataFrame, tickers: dict) -> list[dict]:
    """Compute trailing return and SMA stats for each ticker."""
    records = []
    for ticker, label in tickers.items():
        if ticker not in df.columns:
            continue
        series = df[ticker]
        latest = float(series.iloc[-1])
        start = float(series.iloc[0])
        period_return = (latest - start) / start * 100
        sma = float(series.rolling(SMA_WINDOW, min_periods=20).mean().iloc[-1])
        pct_above = (latest - sma) / sma * 100 if sma else 0.0
        records.append({
            "ticker": ticker,
            "label": label,
            "latest_price": round(latest, 2),
            "period_return_pct": round(period_return, 2),
            "sma_60": round(sma, 2),
            "pct_above_sma": round(pct_above, 2),
        })
    return records


def _benchmark_return() -> float:
    """Get the trailing return for SPY over the same lookback period."""
    print(f"[Sustainability Analyst] Fetching benchmark ({BENCHMARK}) ...")
    try:
        hist = yf.Ticker(BENCHMARK).history(period=LOOKBACK)
        if hist.empty:
            return 0.0
        start = float(hist["Close"].iloc[0])
        latest = float(hist["Close"].iloc[-1])
        return (latest - start) / start * 100
    except Exception as e:
        print(f"     ERROR fetching benchmark: {e}")
        return 0.0


def esg_momentum_signal(clean_stats: list[dict], bench_return: float) -> str:
    """
    Derive an ESG momentum signal for CRE decision-making.

    Logic:
      - Average the period return of clean-energy ETFs.
      - Compare to the SPY benchmark return.
      - Outperforming by >2 pp  -> STRONG  (green capital flowing in)
      - Within +/- 2 pp         -> NEUTRAL
      - Underperforming by >2 pp -> WEAK   (green momentum fading)
    """
    if not clean_stats:
        return "UNKNOWN"
    avg_clean = sum(r["period_return_pct"] for r in clean_stats) / len(clean_stats)
    spread = avg_clean - bench_return
    if spread > 2:
        signal = "STRONG"
    elif spread < -2:
        signal = "WEAK"
    else:
        signal = "NEUTRAL"
    print(f"[Sustainability Analyst] ESG Momentum Signal: {signal}")
    print(f"    Clean-energy avg return: {avg_clean:+.1f}%  |  SPY return: {bench_return:+.1f}%  |  Spread: {spread:+.1f} pp")
    return signal


def run_sustainability_analyst():
    """Main entry point — fetch, analyse, cache."""
    print("=" * 60)
    print("[Sustainability Analyst] Starting run ...")
    print("=" * 60)

    clean_df = _fetch_group(CLEAN_ENERGY_TICKERS, "Clean Energy ETFs")
    reit_df = _fetch_group(GREEN_REIT_TICKERS, "Green REITs")

    clean_stats = _ticker_stats(clean_df, CLEAN_ENERGY_TICKERS)
    reit_stats = _ticker_stats(reit_df, GREEN_REIT_TICKERS)
    bench_return = _benchmark_return()

    signal = esg_momentum_signal(clean_stats, bench_return)

    data = {
        "clean_energy": clean_stats,
        "green_reits": reit_stats,
        "benchmark_return_pct": round(bench_return, 2),
        "esg_momentum_signal": signal,
        "avg_clean_energy_return_pct": round(
            sum(r["period_return_pct"] for r in clean_stats) / len(clean_stats), 2
        ) if clean_stats else None,
        "trading_days_analysed": max(len(clean_df), len(reit_df)),
    }

    _write_cache(data, signal)
    print(f"[Sustainability Analyst] Results saved to cache/sustainability_data.json")
    print("=" * 60)
    return data


if __name__ == "__main__":
    result = run_sustainability_analyst()
    print("\n--- Clean Energy ETFs ---")
    for c in result["clean_energy"]:
        print(f"  {c['label']:30s}  ${c['latest_price']:>8.2f}  Return: {c['period_return_pct']:+.1f}%")
    print("\n--- Green REITs ---")
    for r in result["green_reits"]:
        print(f"  {r['label']:30s}  ${r['latest_price']:>8.2f}  Return: {r['period_return_pct']:+.1f}%")
    print(f"\n  SPY Benchmark Return: {result['benchmark_return_pct']:+.1f}%")
    print(f"  ESG Momentum Signal:  {result['esg_momentum_signal']}")
