"""
Cross-Agent Signal Correlator
==============================
Reads from multiple agent caches and synthesizes a single
"Composite Market Signal" that tells the Investment Advisor
whether multiple independent data sources are confirming or
contradicting each other.

Signal levels:
  STRONG BUY  — 4-5 signals aligned positively
  BUY         — 3 signals positive, none strongly negative
  HOLD        — mixed signals, no clear direction
  CAUTION     — 2+ signals negative
  AVOID       — 4-5 signals aligned negatively

Output includes:
  - Overall signal + confidence (0-100)
  - Per-dimension verdict with plain-English reason
  - Aligned vs. conflicting signal count
  - Top 3 reasons to act + top 3 risks
"""

import json
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"


def _read(key: str) -> dict:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
        return raw.get("data") or raw
    except Exception:
        return {}


# ── Individual signal extractors ─────────────────────────────────────────────

def _labor_signal(data: dict) -> dict:
    """STRONG/MODERATE/SOFT tenant demand from labor market agent."""
    ds = data.get("demand_signal", {})
    score = ds.get("score", 50)
    label = ds.get("label", "MODERATE")
    if score >= 65:
        verdict, direction = "POSITIVE", 1
        reason = f"Labor demand STRONG (score {score}) — payroll growth + low unemployment driving leasing"
    elif score <= 40:
        verdict, direction = "NEGATIVE", -1
        reason = f"Labor demand SOFT (score {score}) — hiring slowing, tenant demand may weaken"
    else:
        verdict, direction = "NEUTRAL", 0
        reason = f"Labor demand MODERATE (score {score}) — stable but not accelerating"
    return {"dimension": "Labor / Tenant Demand", "verdict": verdict, "direction": direction,
            "score": score, "reason": reason, "label": label}


def _migration_signal(data: dict) -> dict:
    """Population inflow from migration agent."""
    rows = data.get("migration", [])
    if not rows:
        return {"dimension": "Population Migration", "verdict": "NEUTRAL", "direction": 0,
                "score": 50, "reason": "No migration data available", "label": "N/A"}
    avg = sum(r.get("composite_score", 50) for r in rows) / len(rows)
    if avg >= 65:
        verdict, direction = "POSITIVE", 1
        reason = f"Strong population inflow (avg score {avg:.0f}) — expanding renter/occupier base"
    elif avg <= 40:
        verdict, direction = "NEGATIVE", -1
        reason = f"Weak migration (avg score {avg:.0f}) — population outflow may pressure occupancy"
    else:
        verdict, direction = "NEUTRAL", 0
        reason = f"Moderate migration (avg score {avg:.0f}) — steady but not high-growth"
    return {"dimension": "Population Migration", "verdict": verdict, "direction": direction,
            "score": round(avg, 1), "reason": reason, "label": ""}


def _vacancy_signal(data: dict) -> dict:
    """Vacancy and absorption health."""
    rows = data.get("market_rows", [])
    if not rows:
        return {"dimension": "Vacancy & Absorption", "verdict": "NEUTRAL", "direction": 0,
                "score": 50, "reason": "No vacancy data available", "label": "N/A"}
    vacs = [r.get("vacancy_rate", 10) for r in rows if r.get("vacancy_rate")]
    avg_vac = sum(vacs) / len(vacs) if vacs else 10
    if avg_vac < 7:
        verdict, direction = "POSITIVE", 1
        reason = f"Tight vacancy ({avg_vac:.1f}% avg) — landlords have pricing power, low supply risk"
    elif avg_vac > 14:
        verdict, direction = "NEGATIVE", -1
        reason = f"High vacancy ({avg_vac:.1f}% avg) — oversupply, rent growth at risk"
    else:
        verdict, direction = "NEUTRAL", 0
        reason = f"Balanced vacancy ({avg_vac:.1f}% avg) — normal market conditions"
    return {"dimension": "Vacancy & Absorption", "verdict": verdict, "direction": direction,
            "score": round(max(0, min(100, (14 - avg_vac) / (14 - 4) * 100)), 1),
            "reason": reason, "label": f"{avg_vac:.1f}%"}


def _rent_signal(data: dict) -> dict:
    """Rent growth momentum."""
    mrg = data.get("market_rent_growth", {})
    if not mrg:
        return {"dimension": "Rent Growth", "verdict": "NEUTRAL", "direction": 0,
                "score": 50, "reason": "No rent growth data available", "label": "N/A"}
    all_vals = []
    for mkt_data in mrg.values():
        for k in ("industrial_psf", "multifamily", "retail_psf", "office_psf"):
            v = mkt_data.get(k)
            if v is not None:
                all_vals.append(v)
    if not all_vals:
        return {"dimension": "Rent Growth", "verdict": "NEUTRAL", "direction": 0,
                "score": 50, "reason": "Rent growth data incomplete", "label": "N/A"}
    avg_rg = sum(all_vals) / len(all_vals)
    if avg_rg > 4:
        verdict, direction = "POSITIVE", 1
        reason = f"Rent growth strong ({avg_rg:.1f}% avg) — income rising faster than inflation"
    elif avg_rg < 0:
        verdict, direction = "NEGATIVE", -1
        reason = f"Rents declining ({avg_rg:.1f}% avg) — income erosion risk for investors"
    else:
        verdict, direction = "NEUTRAL", 0
        reason = f"Rent growth moderate ({avg_rg:.1f}% avg) — keeping pace with inflation"
    return {"dimension": "Rent Growth", "verdict": verdict, "direction": direction,
            "score": round(max(0, min(100, (avg_rg + 5) / 17 * 100)), 1),
            "reason": reason, "label": f"{avg_rg:+.1f}%"}


def _rates_signal(data: dict) -> dict:
    """Interest rate environment."""
    rates = data.get("rates", data)
    t10y_data = rates.get("10Y Treasury", {})
    t10y = t10y_data.get("current") if isinstance(t10y_data, dict) else None
    if t10y is None:
        # try alternate key formats
        for k, v in rates.items():
            if "10y" in k.lower() or "10-year" in k.lower():
                t10y = v.get("current") if isinstance(v, dict) else v
                break
    if t10y is None:
        return {"dimension": "Interest Rates", "verdict": "NEUTRAL", "direction": 0,
                "score": 50, "reason": "Rate data unavailable", "label": "N/A"}
    if t10y < 4.0:
        verdict, direction = "POSITIVE", 1
        reason = f"10Y Treasury at {t10y:.2f}% — low rates reduce cap rate pressure, cheaper debt"
    elif t10y > 5.5:
        verdict, direction = "NEGATIVE", -1
        reason = f"10Y Treasury at {t10y:.2f}% — high rates compress CRE valuations, expensive debt"
    else:
        verdict, direction = "NEUTRAL", 0
        reason = f"10Y Treasury at {t10y:.2f}% — moderate rate environment, manageable financing costs"
    return {"dimension": "Interest Rates", "verdict": verdict, "direction": direction,
            "score": round(max(0, min(100, (6.5 - t10y) / 3.5 * 100)), 1),
            "reason": reason, "label": f"{t10y:.2f}%"}


def _credit_signal(data: dict) -> dict:
    """Credit conditions (lending availability)."""
    sig = data.get("signal", {})
    score = sig.get("score", 50)
    label = sig.get("label", "NEUTRAL")
    if score >= 60:
        verdict, direction = "POSITIVE", 1
        reason = f"Credit conditions supportive (score {score}) — lenders active, spreads tight"
    elif score <= 40:
        verdict, direction = "NEGATIVE", -1
        reason = f"Credit conditions tight (score {score}) — lenders cautious, spreads widening"
    else:
        verdict, direction = "NEUTRAL", 0
        reason = f"Credit conditions neutral (score {score}) — selective lending environment"
    return {"dimension": "Credit Conditions", "verdict": verdict, "direction": direction,
            "score": score, "reason": reason, "label": label}


def _gdp_signal(data: dict) -> dict:
    """GDP cycle position."""
    cycle = str(data.get("gdp_cycle") or data.get("cycle") or "").lower()
    gdp_growth = data.get("gdp_growth") or data.get("real_gdp_growth")
    if cycle in ("expansion", "recovery"):
        verdict, direction = "POSITIVE", 1
        reason = f"GDP in {cycle} phase — economic growth supports occupier demand and valuations"
    elif cycle in ("contraction", "recession"):
        verdict, direction = "NEGATIVE", -1
        reason = f"GDP in {cycle} phase — economic weakness may soften CRE demand"
    elif gdp_growth is not None:
        if gdp_growth > 2.0:
            verdict, direction = "POSITIVE", 1
            reason = f"GDP growth {gdp_growth:.1f}% — solid economic expansion"
        elif gdp_growth < 0:
            verdict, direction = "NEGATIVE", -1
            reason = f"GDP contraction ({gdp_growth:.1f}%) — recession risk elevated"
        else:
            verdict, direction = "NEUTRAL", 0
            reason = f"GDP growth {gdp_growth:.1f}% — slow but positive economic expansion"
    else:
        return {"dimension": "GDP / Economy", "verdict": "NEUTRAL", "direction": 0,
                "score": 50, "reason": "GDP data unavailable", "label": "N/A"}
    score = 75 if direction == 1 else (25 if direction == -1 else 50)
    return {"dimension": "GDP / Economy", "verdict": verdict, "direction": direction,
            "score": score, "reason": reason, "label": cycle.title() or ""}


# ── Main correlator ───────────────────────────────────────────────────────────

_SIGNAL_LABELS = {
    (4, 5):  ("STRONG BUY",  "#4caf50"),
    (3, 3):  ("BUY",         "#8bc34a"),
    (2, 2):  ("HOLD",        "#ffb74d"),
    (1, 1):  ("CAUTION",     "#ff9800"),
    (0, 0):  ("AVOID",       "#f44336"),
}


def run_signal_correlator() -> dict:
    """
    Read all relevant caches and produce a composite market signal.
    Returns a dict suitable for display in the Investment Advisor.
    """
    labor_data  = _read("labor_market")
    mig_data    = _read("migration")
    vac_data    = _read("vacancy")
    rent_data   = _read("rent_growth")
    rate_data   = _read("rates")
    credit_data = _read("credit_data")
    gdp_data    = _read("gdp_data")

    signals = [
        _labor_signal(labor_data),
        _migration_signal(mig_data),
        _vacancy_signal(vac_data),
        _rent_signal(rent_data),
        _rates_signal(rate_data),
        _credit_signal(credit_data),
        _gdp_signal(gdp_data),
    ]

    positives   = [s for s in signals if s["direction"] == 1]
    negatives   = [s for s in signals if s["direction"] == -1]
    neutrals    = [s for s in signals if s["direction"] == 0]
    n_pos       = len(positives)
    n_neg       = len(negatives)
    n_available = sum(1 for s in signals if s["label"] != "N/A")

    # Net signal score: +1 per positive, -1 per negative, weighted by available data
    net = n_pos - n_neg
    confidence = round(min(100, (abs(net) / max(n_available, 1)) * 100 + (n_available / 7 * 20)))

    if net >= 4:
        overall, color = "STRONG BUY", "#4caf50"
    elif net >= 2:
        overall, color = "BUY", "#8bc34a"
    elif net >= 0 and n_neg == 0:
        overall, color = "HOLD", "#ffb74d"
    elif net < -2:
        overall, color = "AVOID", "#f44336"
    elif net < 0:
        overall, color = "CAUTION", "#ff9800"
    else:
        overall, color = "HOLD", "#ffb74d"

    # Top reasons and risks
    top_reasons = [s["reason"] for s in positives[:3]]
    top_risks   = [s["reason"] for s in negatives[:3]]
    if not top_risks and neutrals:
        top_risks = [f"No strong catalysts: {s['dimension']} is neutral" for s in neutrals[:2]]

    # Alignment summary
    if n_pos >= 4:
        alignment = f"{n_pos} of {n_available} signals confirm a positive market environment"
    elif n_neg >= 4:
        alignment = f"{n_neg} of {n_available} signals warn of a deteriorating environment"
    elif n_pos > n_neg:
        alignment = f"Lean positive — {n_pos} signals bullish, {n_neg} cautionary, {len(neutrals)} neutral"
    elif n_neg > n_pos:
        alignment = f"Lean negative — {n_neg} signals cautionary, {n_pos} bullish, {len(neutrals)} neutral"
    else:
        alignment = f"Mixed — {n_pos} bullish vs {n_neg} cautionary ({len(neutrals)} neutral)"

    return {
        "overall":      overall,
        "color":        color,
        "confidence":   confidence,
        "net_score":    net,
        "n_positive":   n_pos,
        "n_negative":   n_neg,
        "n_neutral":    len(neutrals),
        "n_available":  n_available,
        "alignment":    alignment,
        "signals":      signals,
        "top_reasons":  top_reasons,
        "top_risks":    top_risks,
        "generated_at": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    result = run_signal_correlator()
    print(f"\nComposite Signal: {result['overall']}  (confidence {result['confidence']}%)")
    print(f"Alignment: {result['alignment']}")
    print("\nSignals:")
    for s in result["signals"]:
        icon = "+" if s["direction"] == 1 else ("-" if s["direction"] == -1 else "~")
        print(f"  [{icon}] {s['dimension']:30s} {s['verdict']:10s} — {s['reason'][:80]}")
    if result["top_reasons"]:
        print("\nTop reasons:")
        for r in result["top_reasons"]:
            print(f"  + {r}")
    if result["top_risks"]:
        print("\nTop risks:")
        for r in result["top_risks"]:
            print(f"  - {r}")
