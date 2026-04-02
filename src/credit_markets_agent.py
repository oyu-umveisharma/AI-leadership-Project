"""
Agent 12 · Credit & Capital Markets
=====================================
Monitors the availability and cost of CRE debt by tracking corporate credit
spreads, market volatility, and bank lending standards — all leading indicators
of whether capital is flowing into or out of commercial real estate.

FRED Series:
  BAMLC0A0CM      — ICE BofA IG Corporate OAS (Investment Grade spread, bps)
  BAMLH0A0HYM2    — ICE BofA HY Corporate OAS (High Yield spread, bps)
  BAMLC0A4CBBB    — ICE BofA BBB Corporate OAS (BBB-rated, CRE-relevant)
  BAA             — Moody's Baa Corporate Bond Yield (%)
  AAA             — Moody's Aaa Corporate Bond Yield (%)
  VIXCLS          — CBOE VIX (market fear gauge)
  DRTSCILM        — Banks tightening C&I loan standards (net %, quarterly)
  DRTSCLCC        — Banks tightening CRE loan standards (net %, quarterly)
  TERMCBCCALLNS   — Commercial & Industrial loan rate (%)

CRE Implications:
  Wide spreads / tight lending standards → Harder to finance CRE → cap rate expansion
  Narrow spreads / loose standards       → Cheap debt → cap rate compression, deal flow rises
  High VIX                               → Risk-off; CRE transactions freeze
  BAA–AAA spread widening                → Credit stress, CMBS market pressure

Cache: cache/credit_data.json
Schedule: every 6 hours
"""

import json
import os
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_DIR    = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

FRED_SERIES = [
    ("BAMLC0A0CM",    "IG Corporate Spread",       "bps", 100.0),  # pct → bps
    ("BAMLH0A0HYM2",  "HY Corporate Spread",       "bps", 100.0),
    ("BAMLC0A4CBBB",  "BBB Corporate Spread",      "bps", 100.0),
    ("BAA",           "Moody's BAA Yield",          "%",   1.0),
    ("AAA",           "Moody's AAA Yield",          "%",   1.0),
    ("VIXCLS",        "VIX",                        "pts", 1.0),
    ("DRTSCILM",      "C&I Loan Tightening",        "%",   1.0),
    ("DRTSCLCC",      "CRE Loan Tightening",        "%",   1.0),
    ("TERMCBCCALLNS", "C&I Loan Rate",              "%",   1.0),
]


def _fred_fetch(series_id: str, lookback_years: int = 3, retries: int = 3) -> list[dict]:
    if not FRED_API_KEY:
        return []
    import time
    start = (datetime.now() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    params = urllib.parse.urlencode({
        "series_id":         series_id,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "observation_start": start,
        "sort_order":        "asc",
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{FRED_BASE}?{params}", timeout=15) as resp:
                data = json.loads(resp.read())
            return [{"date": o["date"], "value": float(o["value"])}
                    for o in data.get("observations", []) if o["value"] not in (".", "")]
        except Exception as e:
            print(f"  [CreditAgent] FRED error {series_id} (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return []


def _summarize(series: list[dict], unit: str, multiply: float = 1.0) -> dict:
    if not series:
        return {"current": None, "delta_1m": None, "delta_1y": None, "series": [], "unit": unit}
    vals  = [{"date": o["date"], "value": round(o["value"] * multiply, 2)} for o in series]
    cur   = vals[-1]["value"]
    today = datetime.fromisoformat(vals[-1]["date"])
    d1m = next((o["value"] for o in reversed(vals)
                if (today - datetime.fromisoformat(o["date"])).days >= 25), None)
    d1y = next((o["value"] for o in vals
                if (today - datetime.fromisoformat(o["date"])).days >= 330), None)
    return {
        "current":  cur,
        "delta_1m": round(cur - d1m, 2) if d1m is not None else None,
        "delta_1y": round(cur - d1y, 2) if d1y is not None else None,
        "series":   vals[-24:],
        "unit":     unit,
    }


def _classify_credit(data: dict) -> dict:
    """
    Derive a Credit Conditions signal for CRE.
    LOOSE  → cheap debt, deal flow rising
    TIGHT  → expensive debt, transaction market slowing
    STRESS → spread blow-out, lending freeze risk
    """
    score = 50
    bullets = []

    ig = data.get("IG Corporate Spread", {}).get("current")
    hy = data.get("HY Corporate Spread", {}).get("current")
    if ig is not None:
        if ig < 100:
            score += 15; bullets.append(f"IG spreads tight at {ig:.0f}bps — cheap investment-grade debt available")
        elif ig < 150:
            score += 5;  bullets.append(f"IG spreads moderate at {ig:.0f}bps — financing conditions manageable")
        else:
            score -= 15; bullets.append(f"IG spreads wide at {ig:.0f}bps — credit stress, CRE financing costs elevated")
    if hy is not None:
        if hy > 600:
            score -= 10; bullets.append(f"HY spreads at {hy:.0f}bps — risk-off environment, mezzanine debt expensive")
        elif hy < 350:
            score += 8;  bullets.append(f"HY spreads tight at {hy:.0f}bps — risk appetite strong, deal flow supported")

    vix = data.get("VIX", {}).get("current")
    if vix is not None:
        if vix > 30:
            score -= 15; bullets.append(f"VIX elevated at {vix:.1f} — market volatility suppressing transaction activity")
        elif vix < 15:
            score += 10; bullets.append(f"VIX low at {vix:.1f} — calm markets, favorable for CRE deal-making")
        else:
            bullets.append(f"VIX at {vix:.1f} — moderate market uncertainty")

    cre_tight = data.get("CRE Loan Tightening", {}).get("current")
    if cre_tight is not None:
        if cre_tight > 30:
            score -= 15; bullets.append(f"{cre_tight:.0f}% of banks tightening CRE loan standards — lending supply shrinking")
        elif cre_tight > 10:
            score -= 5;  bullets.append(f"{cre_tight:.0f}% of banks tightening CRE standards — modest headwind")
        elif cre_tight < -10:
            score += 10; bullets.append(f"Banks easing CRE lending standards — credit supply expanding")

    baa = data.get("Moody's BAA Yield", {}).get("current")
    aaa = data.get("Moody's AAA Yield", {}).get("current")
    if baa and aaa:
        spread = round(baa - aaa, 2)
        if spread > 2.0:
            score -= 10; bullets.append(f"BAA–AAA spread wide at {spread:.2f}% — corporate credit stress elevated")
        else:
            bullets.append(f"BAA–AAA spread at {spread:.2f}% — corporate credit quality healthy")

    score = max(0, min(100, score))
    if score >= 65:
        label   = "LOOSE"
        summary = "Credit conditions favorable. Debt is accessible and relatively cheap — supports CRE transaction volume and cap rate compression."
    elif score >= 40:
        label   = "NEUTRAL"
        summary = "Credit conditions moderate. Financing available but at higher cost. Selective deal-making favored over broad-market exposure."
    else:
        label   = "TIGHT"
        summary = "Credit conditions restrictive. Lenders pulling back, spreads widening. CRE transactions slowing; focus on well-capitalized assets with low leverage."

    return {"label": label, "score": score, "bullets": bullets, "summary": summary}


def run_credit_markets_agent() -> dict:
    print("=" * 60)
    print("[CreditAgent] Starting run ...")
    print("=" * 60)

    series_data = {}
    multiply_map = {label: mult for _, label, _, mult in FRED_SERIES}
    for series_id, label, unit, mult in FRED_SERIES:
        print(f"  -> {label}")
        raw = _fred_fetch(series_id)
        series_data[label] = _summarize(raw, unit, multiply=mult)

    # Derived: BAA-AAA spread series
    baa_s = series_data.get("Moody's BAA Yield", {}).get("series", [])
    aaa_s = series_data.get("Moody's AAA Yield", {}).get("series", [])
    if baa_s and aaa_s:
        aaa_d = {o["date"]: o["value"] for o in aaa_s}
        spread_series = [{"date": o["date"], "value": round(o["value"] - aaa_d[o["date"]], 3)}
                         for o in baa_s if o["date"] in aaa_d]
        baa_cur = series_data["Moody's BAA Yield"].get("current")
        aaa_cur = series_data["Moody's AAA Yield"].get("current")
        series_data["BAA-AAA Spread"] = {
            "current":  round(baa_cur - aaa_cur, 3) if baa_cur and aaa_cur else None,
            "delta_1m": None,
            "delta_1y": None,
            "series":   spread_series[-24:],
            "unit":     "%",
        }

    signal = _classify_credit(series_data)
    print(f"[CreditAgent] Credit Conditions: {signal['label']} (score {signal['score']})")
    print("=" * 60)

    return {
        "series":    series_data,
        "signal":    signal,
        "cached_at": datetime.now().isoformat(),
        "error":     None,
    }


if __name__ == "__main__":
    result = run_credit_markets_agent()
    for k, v in result["series"].items():
        print(f"  {k:35s}  {v.get('current')}  {v.get('unit', '')}")
    print(f"\n  Credit Conditions: {result['signal']['label']} (score {result['signal']['score']})")
