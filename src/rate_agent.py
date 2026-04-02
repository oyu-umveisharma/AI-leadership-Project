"""
Agent 6 · Interest Rate & Debt Markets
Fetches rate data from FRED API and computes dynamic cap rate adjustments.

Series tracked:
  DGS10        — 10-Year Treasury Constant Maturity
  DGS2         — 2-Year Treasury
  DGS3MO       — 3-Month Treasury
  DGS5         — 5-Year Treasury
  DGS30        — 30-Year Treasury
  FEDFUNDS     — Federal Funds Effective Rate
  MORTGAGE30US — 30-Year Fixed Mortgage Average (weekly)
  SOFR         — Secured Overnight Financing Rate
  DPRIME       — Bank Prime Loan Rate
  BAMLC0A0CM   — ICE BofA IG Corporate OAS (CMBS proxy, in pct points)

Cap rate adjustment model:
  adjusted_cap_rate = benchmark + (10Y - BASELINE_10Y) × beta
  BASELINE_10Y = 4.0%  (rate at which static benchmarks were calibrated)
"""

import os
import json
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────

FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
BASELINE_10Y = 4.0   # % — calibration point for static cap rate benchmarks

# FRED series to fetch: (series_id, display_name, unit_label, multiply_factor)
FRED_SERIES = [
    ("DGS10",        "10Y Treasury",     "%",   1.0),
    ("DGS2",         "2Y Treasury",      "%",   1.0),
    ("DGS3MO",       "3M Treasury",      "%",   1.0),
    ("DGS5",         "5Y Treasury",      "%",   1.0),
    ("DGS30",        "30Y Treasury",     "%",   1.0),
    ("FEDFUNDS",     "Fed Funds Rate",   "%",   1.0),
    ("MORTGAGE30US", "30Y Mortgage",     "%",   1.0),
    ("SOFR",         "SOFR",             "%",   1.0),
    ("DPRIME",       "Prime Rate",       "%",   1.0),
    ("BAMLC0A0CM",   "IG Corp Spread",   "bps", 100.0),  # pct → bps
]

# Cap rate adjustment: per-type pass-through betas and historical spreads over 10Y
# Spread = long-run average (cap_rate - 10Y Treasury) in percentage points
CAP_RATE_SPREADS = {
    "Industrial / Logistics":    {"spread": 1.60, "beta": 0.60},
    "Multifamily / Residential": {"spread": 1.20, "beta": 0.70},
    "Retail":                    {"spread": 2.80, "beta": 0.50},
    "Office":                    {"spread": 4.30, "beta": 0.35},
    "Healthcare / Medical":      {"spread": 1.80, "beta": 0.60},
    "Self-Storage":              {"spread": 1.60, "beta": 0.65},
    "Data Centers":              {"spread": 0.70, "beta": 0.50},
}

# REIT universe from cre_pricing (duplicated here to avoid circular import)
_REIT_TICKERS = [
    "PLD", "STAG", "EGP", "FR",          # Industrial
    "EQR", "AVB", "MAA", "CPT",          # Multifamily
    "O",   "NNN", "SPG", "KIM",          # Retail
    "BXP", "HIW", "CUZ", "PDM",          # Office
    "WELL","VTR", "HR",  "DOC",          # Healthcare
    "PSA", "EXR", "CUBE","LSI",          # Self-Storage
    "EQIX","DLR",                         # Data Centers
]

_REIT_NAMES = {
    "PLD": "Prologis",      "STAG": "STAG Industrial", "EGP": "EastGroup",    "FR": "First Industrial",
    "EQR": "Equity Residential", "AVB": "AvalonBay", "MAA": "Mid-America",  "CPT": "Camden Property",
    "O":   "Realty Income", "NNN": "NNN REIT",      "SPG": "Simon Property","KIM": "Kimco Realty",
    "BXP": "Boston Props",  "HIW": "Highwoods",     "CUZ": "Cousins Props", "PDM": "Piedmont Office",
    "WELL":"Welltower",     "VTR": "Ventas",        "HR":  "Healthcare Realty","DOC":"Healthpeak",
    "PSA": "Public Storage","EXR": "Extra Space",   "CUBE":"CubeSmart",     "LSI": "Life Storage",
    "EQIX":"Equinix",       "DLR": "Digital Realty",
}

_REIT_TYPES = {
    **{t: "Industrial / Logistics"    for t in ["PLD","STAG","EGP","FR"]},
    **{t: "Multifamily / Residential" for t in ["EQR","AVB","MAA","CPT"]},
    **{t: "Retail"                    for t in ["O","NNN","SPG","KIM"]},
    **{t: "Office"                    for t in ["BXP","HIW","CUZ","PDM"]},
    **{t: "Healthcare / Medical"      for t in ["WELL","VTR","HR","DOC"]},
    **{t: "Self-Storage"              for t in ["PSA","EXR","CUBE","LSI"]},
    **{t: "Data Centers"              for t in ["EQIX","DLR"]},
}


# ── FRED API helpers ───────────────────────────────────────────────────────────

def _load_fred_key() -> str:
    """Load FRED API key from environment or .env file."""
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    return os.getenv("FRED_API_KEY", "")


def _fetch_fred(series_id: str, api_key: str, limit: int = 500) -> list[dict]:
    """
    Fetch observations for a FRED series.
    Returns list of {"date": "YYYY-MM-DD", "value": float} sorted oldest→newest.
    Missing values (FRED returns ".") are dropped.
    """
    params = urllib.parse.urlencode({
        "series_id":    series_id,
        "api_key":      api_key,
        "file_type":    "json",
        "limit":        limit,
        "sort_order":   "desc",
    })
    url = f"{FRED_BASE}?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cre-rate-agent/1.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        obs = [
            {"date": o["date"], "value": float(o["value"])}
            for o in data.get("observations", [])
            if o.get("value") not in (".", "", None)
        ]
        return sorted(obs, key=lambda x: x["date"])   # oldest first
    except Exception as e:
        return []


def _latest(series: list[dict]) -> float | None:
    """Most recent non-null value."""
    return series[-1]["value"] if series else None


def _value_n_days_ago(series: list[dict], n: int) -> float | None:
    """Value approximately n trading days ago."""
    if not series:
        return None
    cutoff = (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d")
    past = [o for o in series if o["date"] <= cutoff]
    return past[-1]["value"] if past else None


# ── Rate environment classifier ────────────────────────────────────────────────

def _classify_environment(rates: dict) -> dict:
    """
    Return signal (Bullish / Cautious / Bearish) and explanation bullets.
    Rates dict keys: display names → current value in %.
    """
    t10   = rates.get("10Y Treasury")
    t2    = rates.get("2Y Treasury")
    ff    = rates.get("Fed Funds Rate")
    t10_1m = rates.get("10Y Treasury_1m")
    sofr  = rates.get("SOFR")
    corp  = rates.get("IG Corp Spread")     # bps

    bullets = []
    score   = 0   # positive = bullish, negative = bearish

    if t10 is not None:
        if t10 < 3.5:
            score += 2
            bullets.append(f"10Y at {t10:.2f}% — well below 4% neutral zone, supportive for cap rate compression")
        elif t10 < 4.5:
            score += 0
            bullets.append(f"10Y at {t10:.2f}% — near neutral zone (4%), limited directional pressure on cap rates")
        elif t10 < 5.5:
            score -= 1
            bullets.append(f"10Y at {t10:.2f}% — elevated, adding moderate upward pressure on cap rates")
        else:
            score -= 2
            bullets.append(f"10Y at {t10:.2f}% — high, materially pressuring cap rates and property valuations")

    # Yield curve shape
    if t10 is not None and t2 is not None:
        spread = t10 - t2
        if spread > 0.5:
            score += 1
            bullets.append(f"Yield curve normal (+{spread:.0f}bps): 10Y > 2Y — positive economic growth signal")
        elif spread > 0:
            bullets.append(f"Yield curve flat (+{spread:.0f}bps): muted growth expectations")
        else:
            score -= 1
            bullets.append(f"Yield curve inverted ({spread:.0f}bps): 2Y > 10Y — recession risk signal")

    # Rate direction (is 10Y rising or falling?)
    if t10 is not None and t10_1m is not None:
        delta = t10 - t10_1m
        if delta < -0.25:
            score += 1
            bullets.append(f"10Y falling {abs(delta):.2f}% over past month — improving CRE valuation environment")
        elif delta > 0.25:
            score -= 1
            bullets.append(f"10Y rising {delta:.2f}% over past month — headwind for CRE valuations")

    # Credit spreads
    if corp is not None:
        if corp < 100:
            score += 1
            bullets.append(f"IG corporate spreads tight at {corp:.0f}bps — credit markets confident, CMBS cheap")
        elif corp > 175:
            score -= 1
            bullets.append(f"IG corporate spreads wide at {corp:.0f}bps — credit stress, CMBS financing more costly")
        else:
            bullets.append(f"IG corporate spreads at {corp:.0f}bps — neutral credit environment")

    if score >= 2:
        signal = "BULLISH"
        color  = "#1b5e20"
        icon   = "🟢"
        summary = "Rate environment is supportive for CRE — cap rates under compression pressure, financing accessible."
    elif score <= -2:
        signal = "BEARISH"
        color  = "#b71c1c"
        icon   = "🔴"
        summary = "Rate environment is challenging for CRE — elevated rates expanding cap rates, compressing valuations."
    else:
        signal = "CAUTIOUS"
        color  = "#e65100"
        icon   = "🟡"
        summary = "Mixed rate signals — monitor direction; selective opportunities in rate-resilient sectors."

    return {
        "signal":  signal,
        "color":   color,
        "icon":    icon,
        "summary": summary,
        "bullets": bullets,
        "score":   score,
    }


# ── Cap rate adjustments ───────────────────────────────────────────────────────

def compute_cap_rate_adjustments(current_10y: float) -> list[dict]:
    """
    For each property type, compute the rate-adjusted cap rate and resulting
    change in effective profit margin vs. the static benchmark.

    adjustment = (current_10y - BASELINE_10Y) × beta
    adjusted_cap = benchmark_cap + adjustment
    """
    from src.cre_pricing import CAP_RATE_BENCHMARKS

    rows = []
    delta_10y = current_10y - BASELINE_10Y

    for pt, bench in CAP_RATE_BENCHMARKS.items():
        sp = CAP_RATE_SPREADS.get(pt, {"spread": 2.0, "beta": 0.5})
        beta        = sp["beta"]
        adjustment  = delta_10y * beta / 100   # convert bps→ decimal
        base_cap    = bench["cap_rate"]
        adj_cap     = base_cap + adjustment
        noi         = bench["noi_margin"]
        vacancy     = bench["vacancy"]
        rent_growth = bench["rent_growth"]

        base_margin = noi * (1 - vacancy) * (1 + rent_growth)
        # Effective profit normalised by cap rate (higher cap → lower multiple → lower margin proxy)
        # Simple impact: higher cap rate compresses the price/earnings multiple
        cap_ratio   = base_cap / adj_cap if adj_cap > 0 else 1.0
        adj_margin  = base_margin * cap_ratio

        rows.append({
            "Property Type":      pt,
            "Baseline Cap Rate":  round(base_cap * 100, 2),
            "Rate Adjustment bps": round(adjustment * 10000, 1),
            "Adjusted Cap Rate":  round(adj_cap * 100, 2),
            "Static Margin %":    round(base_margin * 100, 2),
            "Adj Margin %":       round(adj_margin * 100, 2),
            "Margin Delta bps":   round((adj_margin - base_margin) * 10000, 1),
            "Beta":               beta,
        })

    return sorted(rows, key=lambda x: x["Rate Adjustment bps"], reverse=True)


# ── REIT debt / refinancing risk ───────────────────────────────────────────────

def fetch_reit_debt_risk() -> list[dict]:
    """
    For each REIT, pull balance sheet data to estimate near-term refinancing risk.
    Uses quarterly_balance_sheet rows: CurrentDebt, LongTermDebt, TotalDebt.
    Risk % = near-term (current) debt / total debt.
    """
    rows = []
    _DEBT_ROWS    = ["Current Debt", "Current Debt And Capital Lease Obligation",
                     "Short Term Debt", "Short Long Term Debt"]
    _LT_DEBT_ROWS = ["Long Term Debt", "Long Term Debt And Capital Lease Obligation",
                     "Long Term Debt Non Current"]

    for ticker in _REIT_TICKERS:
        try:
            t  = yf.Ticker(ticker)
            bs = t.quarterly_balance_sheet

            def _get_row(candidates):
                for name in candidates:
                    if name in bs.index:
                        val = bs.loc[name].iloc[0]
                        if pd.notna(val):
                            return float(val)
                return None

            current_debt = _get_row(_DEBT_ROWS)
            lt_debt      = _get_row(_LT_DEBT_ROWS)

            if current_debt is None and lt_debt is None:
                continue

            cd = current_debt or 0.0
            ld = lt_debt      or 0.0
            total = cd + ld
            risk_pct = (cd / total * 100) if total > 0 else 0.0

            risk_level = (
                "High"   if risk_pct >= 25 else
                "Medium" if risk_pct >= 10 else
                "Low"
            )
            rows.append({
                "Ticker":            ticker,
                "Name":              _REIT_NAMES.get(ticker, ticker),
                "Property Type":     _REIT_TYPES.get(ticker, "Unknown"),
                "Total Debt $B":     round(total / 1e9, 2),
                "Near-Term Debt $B": round(cd    / 1e9, 2),
                "Risk %":            round(risk_pct, 1),
                "Risk Level":        risk_level,
            })
            time.sleep(0.15)   # gentle rate limiting
        except Exception:
            pass

    return sorted(rows, key=lambda x: x["Risk %"], reverse=True)


# ── Main agent runner ──────────────────────────────────────────────────────────

def run_rate_agent() -> dict:
    """
    Fetch all rate data, classify environment, compute cap rate adjustments,
    analyse REIT debt risk. Returns the cache payload dict.
    """
    api_key = _load_fred_key()
    if not api_key:
        return {
            "error": "FRED_API_KEY not set. Add it to .env to enable the Rate Environment agent.",
            "rates": {}, "environment": {}, "cap_rate_adjustments": [],
            "reit_debt_risk": [], "yield_curve": {}, "cached_at": datetime.now().isoformat(),
        }

    # ── Fetch all FRED series ─────────────────────────────────────────────────
    raw = {}
    for sid, name, unit, factor in FRED_SERIES:
        series = _fetch_fred(sid, api_key, limit=500)
        if factor != 1.0:
            for o in series:
                o["value"] = round(o["value"] * factor, 2)
        raw[name] = series
        time.sleep(0.1)   # respect FRED rate limits

    # ── Build rates summary dict (current + deltas) ───────────────────────────
    rates_summary = {}
    for _, name, unit, _ in FRED_SERIES:
        series = raw.get(name, [])
        curr   = _latest(series)
        w1     = _value_n_days_ago(series, 7)
        m1     = _value_n_days_ago(series, 30)
        y1     = _value_n_days_ago(series, 365)
        if curr is None:
            continue
        rates_summary[name] = {
            "current":    round(curr, 4),
            "unit":       unit,
            "delta_1w":   round(curr - w1, 4) if w1 is not None else None,
            "delta_1m":   round(curr - m1, 4) if m1 is not None else None,
            "delta_1y":   round(curr - y1, 4) if y1 is not None else None,
            "series":     series[-365:],    # last 365 data points for charting
        }

    # ── Yield curve snapshot ──────────────────────────────────────────────────
    yield_curve = {}
    for label, series_name in [
        ("3M",  "3M Treasury"),
        ("2Y",  "2Y Treasury"),
        ("5Y",  "5Y Treasury"),
        ("10Y", "10Y Treasury"),
        ("30Y", "30Y Treasury"),
    ]:
        s = raw.get(series_name, [])
        v = _latest(s)
        if v is not None:
            yield_curve[label] = round(v, 4)

    # ── Environment signal ────────────────────────────────────────────────────
    flat_rates = {name: d["current"] for name, d in rates_summary.items()}
    t10 = rates_summary.get("10Y Treasury", {})
    flat_rates["10Y Treasury_1m"] = t10.get("current", 0) - (t10.get("delta_1m") or 0)
    environment = _classify_environment(flat_rates)

    # ── Cap rate adjustments ──────────────────────────────────────────────────
    current_10y = rates_summary.get("10Y Treasury", {}).get("current")
    cap_rate_adjustments = (
        compute_cap_rate_adjustments(current_10y)
        if current_10y is not None else []
    )

    # ── REIT debt risk ────────────────────────────────────────────────────────
    reit_debt_risk = fetch_reit_debt_risk()

    return {
        "rates":               rates_summary,
        "yield_curve":         yield_curve,
        "environment":         environment,
        "cap_rate_adjustments": cap_rate_adjustments,
        "reit_debt_risk":      reit_debt_risk,
        "baseline_10y":        BASELINE_10Y,
        "current_10y":         current_10y,
        "cached_at":           datetime.now().isoformat(),
        "error":               None,
    }
