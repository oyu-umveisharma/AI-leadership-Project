"""
Energy & Construction Cost Analyst Agent
=========================================
Part of the Human-in-Command AI Workforce for the CRE Intelligence Platform.

This agent tracks energy commodities and industrial metals that directly impact
commercial real estate construction and operating costs:

  - Oil (USO) & Natural Gas (UNG): Building heating/cooling and transportation costs
  - Energy Sector (XLE): Broad energy market sentiment
  - Copper (CPER): Key indicator for electrical and plumbing costs
  - Steel (SLX): Structural construction cost driver

Outputs a "Construction Cost Signal" (HIGH / MODERATE / LOW) based on
trailing price momentum of these commodities relative to their 60-day averages.

Cache: cache/energy_data.json
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

TICKERS = {
    "USO": "Oil (USO)",
    "UNG": "Natural Gas (UNG)",
    "XLE": "Energy Sector (XLE)",
    "CPER": "Copper (CPER)",
    "SLX": "Steel (SLX)",
}

LOOKBACK = "6mo"
SMA_WINDOW = 60  # trading days for the moving average baseline


def _write_cache(data: dict, signal: str):
    """Write energy data to cache in Chief-of-Staff-compatible format."""
    payload = {
        "agent_name": "energy_analyst",
        "timestamp": datetime.now().isoformat(),
        "signals": {
            "construction_cost_signal": signal,
        },
        "data": data,
    }
    with open(CACHE_DIR / "energy_data.json", "w") as f:
        json.dump(payload, f, default=str, indent=2)


def fetch_energy_prices() -> pd.DataFrame:
    """Download recent closing prices for all tracked tickers."""
    print("[Energy Analyst] Fetching energy & metals prices ...")
    frames = []
    for ticker, label in TICKERS.items():
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
        raise RuntimeError("No energy data could be fetched")
    df = pd.concat(frames, axis=1).dropna()
    print(f"[Energy Analyst] Received {len(df)} trading days of data")
    return df


def compute_momentum(df: pd.DataFrame) -> list[dict]:
    """Compute price vs SMA-60 momentum for each ticker."""
    print("[Energy Analyst] Computing momentum signals ...")
    records = []
    for ticker, label in TICKERS.items():
        if ticker not in df.columns:
            continue
        series = df[ticker]
        latest = float(series.iloc[-1])
        sma = float(series.rolling(SMA_WINDOW, min_periods=20).mean().iloc[-1])
        pct_above = (latest - sma) / sma * 100 if sma else 0.0
        records.append({
            "ticker": ticker,
            "label": label,
            "latest_price": round(latest, 2),
            "sma_60": round(sma, 2),
            "pct_above_sma": round(pct_above, 2),
        })
    return records


def construction_cost_signal(momentum: list[dict]) -> str:
    """
    Derive a construction-cost signal from commodity momentum.

    Logic:
      - Average the pct_above_sma across all tickers.
      - > +5 %  -> HIGH   (costs rising notably)
      -  -5 % to +5 % -> MODERATE
      - < -5 %  -> LOW    (costs easing)
    """
    if not momentum:
        return "UNKNOWN"
    avg = sum(r["pct_above_sma"] for r in momentum) / len(momentum)
    if avg > 5:
        signal = "HIGH"
    elif avg < -5:
        signal = "LOW"
    else:
        signal = "MODERATE"
    print(f"[Energy Analyst] Construction Cost Signal: {signal}  (avg momentum {avg:+.1f}%)")
    return signal


_FALLBACK_MOMENTUM = [
    {"ticker": "USO",  "label": "Oil (USO)",          "latest_price": 73.50, "sma_60": 73.50, "pct_above_sma": 0.0},
    {"ticker": "UNG",  "label": "Natural Gas (UNG)",   "latest_price": 14.20, "sma_60": 14.20, "pct_above_sma": 0.0},
    {"ticker": "XLE",  "label": "Energy Sector (XLE)", "latest_price": 88.00, "sma_60": 88.00, "pct_above_sma": 0.0},
    {"ticker": "CPER", "label": "Copper (CPER)",        "latest_price": 25.10, "sma_60": 25.10, "pct_above_sma": 0.0},
    {"ticker": "SLX",  "label": "Steel (SLX)",          "latest_price": 62.00, "sma_60": 62.00, "pct_above_sma": 0.0},
]


def run_energy_analyst():
    """Main entry point — fetch, analyse, cache. Falls back to neutral values if market data is unavailable."""
    print("=" * 60)
    print("[Energy Analyst] Starting run ...")
    print("=" * 60)

    fallback_used = False
    try:
        df = fetch_energy_prices()
        momentum = compute_momentum(df)
        trading_days = len(df)
    except Exception as e:
        print(f"[Energy Analyst] Market data unavailable ({e}) — using neutral fallback values")
        momentum     = _FALLBACK_MOMENTUM
        trading_days = 0
        fallback_used = True

    signal = construction_cost_signal(momentum)

    data = {
        "commodities":              momentum,
        "construction_cost_signal": signal,
        "avg_momentum_pct":         round(
            sum(r["pct_above_sma"] for r in momentum) / len(momentum), 2
        ) if momentum else None,
        "trading_days_analysed":    trading_days,
        "fallback_used":            fallback_used,
    }

    _write_cache(data, signal)
    print(f"[Energy Analyst] Results saved to cache/energy_data.json (fallback={fallback_used})")
    print("=" * 60)
    return data


if __name__ == "__main__":
    result = run_energy_analyst()
    print("\n--- Summary ---")
    for c in result["commodities"]:
        print(f"  {c['label']:25s}  ${c['latest_price']:>8.2f}  SMA-60 ${c['sma_60']:>8.2f}  ({c['pct_above_sma']:+.1f}%)")
    print(f"\n  Construction Cost Signal: {result['construction_cost_signal']}")
