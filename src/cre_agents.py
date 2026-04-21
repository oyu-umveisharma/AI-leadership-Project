"""
Background Agent Runner — nine independent agents updating on schedules.

Agent 1 · Migration Tracker    — every 6 hours
Agent 2 · REIT Pricing          — every 1 hour
Agent 3 · Company Predictions   — every 24 hours (LLM)
Agent 4 · Debugger / Monitor    — every 30 minutes
Agent 5 · News & Announcements  — every 4 hours
Agent 6 · Interest Rate & Debt  — every 1 hour  (requires FRED_API_KEY)
Agent 7 · Energy & Construction — every 6 hours
Agent 8 · Sustainability & ESG  — every 6 hours
Agent 9 · Labor Market & Tenant Demand — every 6 hours

Uses APScheduler + file-based JSON cache so results survive Streamlit reruns.
"""

import json
import os
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

try:
    from dotenv import load_dotenv
    from pathlib import Path as _P
    load_dotenv(_P(__file__).parent.parent / ".env", override=True)
except ImportError:
    pass

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Pre-populate predictions cache at import time if missing so the UI
# never shows "never updated" — the full agent enriches it later.
def _prepopulate_predictions():
    p = CACHE_DIR / "predictions.json"
    if not p.exists():
        from datetime import datetime
        baseline = [
            {"company":"TSMC","ticker":"TSM","type":"Semiconductor Fab","location":"Phoenix, AZ","investment":"$65B","jobs":"6,000+","detail":"TSMC building three semiconductor fabs in Phoenix; first fab began production in 2024.","cre_impact":"Drives demand for industrial, multifamily, and retail in greater Phoenix metro.","source":"Public Record"},
            {"company":"Intel","ticker":"INTC","type":"Semiconductor Fab","location":"New Albany, OH","investment":"$20B","jobs":"3,000+","detail":"Intel's Ohio One campus — two fabs under construction as part of CHIPS Act investment.","cre_impact":"Creates significant industrial and multifamily demand in Columbus suburban corridor.","source":"Public Record"},
            {"company":"Samsung","ticker":"SSNLF","type":"Semiconductor Fab","location":"Taylor, TX","investment":"$17B","jobs":"2,000+","detail":"Samsung semiconductor fab in Taylor, TX expanding US chip manufacturing footprint.","cre_impact":"Boosts industrial and housing demand in Taylor and greater Austin area.","source":"Public Record"},
            {"company":"Hyundai / Kia","ticker":"HYMTF","type":"Manufacturing Plant","location":"Savannah, GA","investment":"$7.6B","jobs":"8,500+","detail":"Metaplant America EV assembly plant in Bryan County, GA began production in 2024.","cre_impact":"Strong industrial, logistics, and multifamily demand in Savannah/Brunswick corridor.","source":"Public Record"},
            {"company":"Rivian","ticker":"RIVN","type":"Manufacturing Plant","location":"Stanton Springs, GA","investment":"$5B","jobs":"7,500+","detail":"Rivian's second manufacturing plant in Morgan County, GA targeting 2026 opening.","cre_impact":"Industrial and workforce housing demand in Atlanta exurban markets.","source":"Public Record"},
            {"company":"Mercedes-Benz","ticker":"MBG","type":"Manufacturing Plant","location":"Tuscaloosa, AL","investment":"$1B+","jobs":"1,000+","detail":"Mercedes expanding Vance, AL plant to produce all-electric EQ-class SUVs.","cre_impact":"Supports industrial supplier parks and multifamily growth near Tuscaloosa.","source":"Public Record"},
            {"company":"Toyota","ticker":"TM","type":"Battery Plant","location":"Liberty, NC","investment":"$13.9B","jobs":"5,000+","detail":"Toyota battery manufacturing plant in Randolph County, NC to supply EV production.","cre_impact":"Creates industrial and housing demand in Triad region (Greensboro/High Point).","source":"Public Record"},
            {"company":"Microsoft","ticker":"MSFT","type":"Data Center","location":"Multiple US Markets","investment":"$80B","jobs":"","detail":"Microsoft investing $80B in new AI-capable data centers across the US in fiscal 2025.","cre_impact":"Drives demand for large industrial/data center campuses near power infrastructure.","source":"Public Record"},
            {"company":"Amazon Web Services","ticker":"AMZN","type":"Data Center","location":"Multiple US Markets","investment":"$150B+","jobs":"","detail":"AWS expanding data center footprint in Virginia, Ohio, Oregon, and Texas.","cre_impact":"Significant industrial land demand in suburban markets with reliable power grid.","source":"Public Record"},
            {"company":"Scout Motors","ticker":"VWAGY","type":"Manufacturing Plant","location":"Blythewood, SC","investment":"$2B","jobs":"4,000+","detail":"Scout Motors building an EV truck and SUV plant in South Carolina, opening ~2027.","cre_impact":"Industrial supplier and workforce housing demand in Columbia, SC metro.","source":"Public Record"},
            {"company":"Eli Lilly","ticker":"LLY","type":"Manufacturing Plant","location":"Lebanon, IN","investment":"$9B","jobs":"1,000+","detail":"Eli Lilly building four new manufacturing sites in Indiana for weight-loss and diabetes drugs.","cre_impact":"Industrial and office demand in Indianapolis suburban corridor.","source":"Public Record"},
            {"company":"Nucor","ticker":"NUE","type":"Manufacturing Plant","location":"Mason County, WV","investment":"$3B","jobs":"800+","detail":"Nucor steel sheet mill in Mason County, WV serving automotive and construction markets.","cre_impact":"Industrial growth opportunity in Ohio River Valley corridor.","source":"Public Record"},
        ]
        import json as _j
        p.write_text(_j.dumps({"updated_at": datetime.now().isoformat(), "data": {
            "confirmed_announcements": baseline,
            "top5_states": [], "listings": {}, "top3_abbr": [],
        }}, default=str))

_prepopulate_predictions()

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def write_cache(key: str, data: Any):
    payload = {"updated_at": datetime.now().isoformat(), "data": data}
    # Data quality: reject if data is unexpectedly None
    if data is not None:
        _validate_cache_data(key, data)
    with open(_cache_path(key), "w") as f:
        json.dump(payload, f, default=str)


def _validate_cache_data(key: str, data: Any):
    """Lightweight schema validation and outlier detection on cache writes."""
    warnings = []
    if key == "migration" and isinstance(data, dict):
        mig = data.get("migration", [])
        for rec in mig:
            score = rec.get("composite_score")
            if score is not None and (score < 0 or score > 100):
                warnings.append(f"migration: {rec.get('state_abbr')} composite_score={score} out of [0,100]")
            pop = rec.get("population")
            if pop is not None and pop <= 0:
                warnings.append(f"migration: {rec.get('state_abbr')} population <= 0")
    elif key == "pricing" and isinstance(data, dict):
        reits = data.get("reits", [])
        null_prices = sum(1 for r in reits if not r.get("price") or r.get("price", 0) <= 0)
        if reits and null_prices / max(len(reits), 1) > 0.10:
            warnings.append(f"pricing: {null_prices}/{len(reits)} REITs have null/zero prices (>10%)")
        for r in reits:
            p = r.get("price", 0)
            if p and p > 1000:
                warnings.append(f"pricing: {r.get('ticker')} price ${p} exceeds $1000 sanity limit")
    elif key == "energy_data" and isinstance(data, dict):
        sig = data.get("construction_cost_signal", "")
        if sig and sig not in ("HIGH", "MODERATE", "LOW"):
            warnings.append(f"energy: invalid construction_cost_signal '{sig}'")
        for c in data.get("commodities", []):
            if abs(c.get("pct_above_sma", 0)) > 50:
                warnings.append(f"energy: {c.get('ticker')} pct_above_sma={c.get('pct_above_sma'):.1f}% (extreme)")
    elif key == "rates" and isinstance(data, dict):
        yc = data.get("yield_curve", {})
        for mat in ["3M", "2Y", "5Y", "10Y", "30Y"]:
            val = yc.get(mat)
            if val is not None and (val < 0 or val > 15):
                warnings.append(f"rates: yield_curve {mat}={val}% out of range")
    # Log warnings via audit logger if any
    if warnings:
        try:
            from src.audit_logger import log_agent_run
            log_agent_run(key, "warning", 0, "; ".join(warnings[:3]))
        except Exception:
            pass


def read_cache(key: str) -> dict:
    p = _cache_path(key)
    if not p.exists():
        return {"updated_at": None, "data": None, "stale": True}
    try:
        with open(p) as f:
            payload = json.load(f)
        # Support both old format (updated_at) and Chief-of-Staff format (timestamp)
        ts = payload.get("updated_at") or payload.get("timestamp")
        if not ts:
            return {"updated_at": None, "data": None, "stale": True}
        updated = datetime.fromisoformat(ts)
        payload["updated_at"] = ts
        age_hours = (datetime.now() - updated).total_seconds() / 3600
        payload["stale"] = age_hours > 25
        payload["age_minutes"] = round((datetime.now() - updated).total_seconds() / 60, 1)
        return payload
    except Exception:
        return {"updated_at": None, "data": None, "stale": True}


def cache_age_label(key: str) -> str:
    c = read_cache(key)
    if not c["updated_at"]:
        return "Never updated"
    mins = c.get("age_minutes", 0)
    if mins < 2:   return "Just now"
    if mins < 60:  return f"{int(mins)}m ago"
    if mins < 1440: return f"{int(mins/60)}h ago"
    return f"{int(mins/1440)}d ago"


# ── Agent Status Log ──────────────────────────────────────────────────────────

_status_lock = threading.Lock()
_agent_status = {
    "migration":      {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "pricing":        {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "predictions":    {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "debugger":       {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "news":           {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "rates":          {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "energy":         {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "sustainability": {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "labor_market":   {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "gdp":            {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "inflation":      {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "credit":         {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "land_market":     {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "cap_rate":        {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "rent_growth":     {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "opportunity_zone":{"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "distressed":      {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "market_score":    {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "vacancy":         {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "property_tax":    {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "climate_risk":    {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
}

def get_status() -> dict:
    with _status_lock:
        return {k: dict(v) for k, v in _agent_status.items()}

def _set_status(agent: str, status: str, error: str = None):
    with _status_lock:
        _agent_status[agent]["status"] = status
        if status == "running":
            _agent_status[agent]["last_run"] = datetime.now().isoformat()
            _agent_status[agent]["_start_time"] = datetime.now()
        if error:
            _agent_status[agent]["last_error"] = error
        else:
            _agent_status[agent]["last_error"] = None
            _agent_status[agent]["runs"] += 1
        # Audit logging on completion (ok or error)
        if status in ("ok", "error"):
            try:
                from src.audit_logger import log_agent_run
                start = _agent_status[agent].get("_start_time")
                latency = (datetime.now() - start).total_seconds() * 1000 if start else 0
                log_agent_run(agent, status, latency, error=error or "")
            except Exception:
                pass


# ── Agent 1 · Migration ───────────────────────────────────────────────────────

def run_migration_agent():
    _set_status("migration", "running")
    try:
        from src.cre_population import fetch_migration_scores, get_top_metros
        mig_df   = fetch_migration_scores()
        metro_df = get_top_metros()
        write_cache("migration", {
            "migration": mig_df.to_dict(orient="records"),
            "metros":    metro_df.to_dict(orient="records"),
            "top3_cities": mig_df.head(3)["state_name"].tolist(),
        })
        _set_status("migration", "ok")
    except Exception as e:
        _set_status("migration", "error", str(e))


# ── Agent 2 · REIT Pricing ────────────────────────────────────────────────────

def run_pricing_agent():
    _set_status("pricing", "running")
    try:
        from src.cre_pricing import fetch_reit_prices, get_top_opportunities, get_property_type_summary
        reit_df  = fetch_reit_prices()
        top_opps = get_top_opportunities(10)
        pt_sum   = get_property_type_summary()
        write_cache("pricing", {
            "reits":        reit_df.to_dict(orient="records"),
            "top_opps":     top_opps.to_dict(orient="records"),
            "pt_summary":   pt_sum.to_dict(orient="records"),
        })
        _set_status("pricing", "ok")
    except Exception as e:
        _set_status("pricing", "error", str(e))


# ── Agent 3 · Company Predictions + Listings ─────────────────────────────────

def run_predictions_agent():
    _set_status("predictions", "running")

    # ── Write baseline immediately so UI never shows "never updated" ──────────
    existing = read_cache("predictions")
    if existing["data"] is None:
        baseline_confirmed = _extract_confirmed_announcements([])
        write_cache("predictions", {
            "confirmed_announcements": baseline_confirmed,
            "top5_states": [],
            "listings": {},
            "top3_abbr": [],
        })
        _set_status("predictions", "ok")

    # ── Now do the slow external calls to enrich the cache ────────────────────
    top5, top3_cities_abbr, listings = [], [], {}
    try:
        from src.cre_population import fetch_migration_scores
        mig_df = fetch_migration_scores()
        top5 = mig_df.head(5)[["state_name","state_abbr","pop_growth_pct","biz_score","key_companies","growth_drivers"]].to_dict(orient="records")
        top3_cities_abbr = mig_df.head(3)["state_abbr"].tolist()
    except Exception:
        pass

    try:
        from src.cre_listings import get_cheapest_buildings
        for abbr in top3_cities_abbr:
            try:
                listings[abbr] = get_cheapest_buildings(abbr)
            except Exception:
                listings[abbr] = []
    except Exception:
        pass

    try:
        from src.cre_news import fetch_facility_announcements
        articles = fetch_facility_announcements()
    except Exception:
        articles = []

    confirmed = _extract_confirmed_announcements(articles)

    write_cache("predictions", {
        "confirmed_announcements": confirmed,
        "top5_states":             top5,
        "listings":                listings,
        "top3_abbr":               top3_cities_abbr,
    })
    _set_status("predictions", "ok")


def _extract_confirmed_announcements(articles: list) -> list:
    """
    Uses Groq to build a list of confirmed US company facility announcements.
    Two-pass approach:
      1. Extract any confirmed announcements from the provided live news articles.
      2. Supplement with Groq's knowledge of major recent announcements (2023-2025)
         so the tab always shows useful data even when RSS feeds are sparse.
    Returns a deduplicated list of announcement dicts.
    """
    import json as _json

    # ── Hardcoded baseline — always shown even without API key ────────────────
    BASELINE = [
        {"company":"TSMC","ticker":"TSM","type":"Semiconductor Fab","location":"Phoenix, AZ","investment":"$65B","jobs":"6,000+","detail":"TSMC building three semiconductor fabs in Phoenix; first fab began production in 2024.","cre_impact":"Drives demand for industrial, multifamily, and retail in greater Phoenix metro.","source":"Public Record"},
        {"company":"Intel","ticker":"INTC","type":"Semiconductor Fab","location":"New Albany, OH","investment":"$20B","jobs":"3,000+","detail":"Intel's Ohio One campus — two fabs under construction as part of CHIPS Act investment.","cre_impact":"Creates significant industrial and multifamily demand in Columbus suburban corridor.","source":"Public Record"},
        {"company":"Samsung","ticker":"SSNLF","type":"Semiconductor Fab","location":"Taylor, TX","investment":"$17B","jobs":"2,000+","detail":"Samsung's semiconductor fab in Taylor, TX expanding US chip manufacturing footprint.","cre_impact":"Boosts industrial and housing demand in Taylor and greater Austin area.","source":"Public Record"},
        {"company":"Hyundai / Kia","ticker":"HYMTF","type":"Manufacturing Plant","location":"Savannah, GA","investment":"$7.6B","jobs":"8,500+","detail":"Metaplant America EV assembly plant in Bryan County, GA began production in 2024.","cre_impact":"Strong industrial, logistics, and multifamily demand in Savannah/Brunswick corridor.","source":"Public Record"},
        {"company":"Rivian","ticker":"RIVN","type":"Manufacturing Plant","location":"Stanton Springs, GA","investment":"$5B","jobs":"7,500+","detail":"Rivian's second manufacturing plant in Morgan County, GA targeting 2026 opening.","cre_impact":"Industrial and workforce housing demand in Atlanta exurban markets.","source":"Public Record"},
        {"company":"Mercedes-Benz","ticker":"MBG","type":"Manufacturing Plant","location":"Tuscaloosa, AL","investment":"$1B+","jobs":"1,000+","detail":"Mercedes expanding Vance, AL plant to produce all-electric EQ-class SUVs.","cre_impact":"Supports industrial supplier parks and multifamily growth near Tuscaloosa.","source":"Public Record"},
        {"company":"Toyota","ticker":"TM","type":"Battery Plant","location":"Liberty, NC","investment":"$13.9B","jobs":"5,000+","detail":"Toyota battery manufacturing plant in Randolph County, NC to supply EV production.","cre_impact":"Creates industrial and housing demand in Triad region (Greensboro/High Point).","source":"Public Record"},
        {"company":"Microsoft","ticker":"MSFT","type":"Data Center","location":"Multiple US Markets","investment":"$80B (2025)","jobs":"","detail":"Microsoft investing $80B in new AI-capable data centers across the US in fiscal 2025.","cre_impact":"Drives demand for large industrial/data center campuses near power infrastructure.","source":"Public Record"},
        {"company":"Amazon Web Services","ticker":"AMZN","type":"Data Center","location":"Multiple US Markets","investment":"$150B+","jobs":"","detail":"AWS expanding data center footprint in Virginia, Ohio, Oregon, and Texas.","cre_impact":"Significant industrial land demand in suburban markets with reliable power grid.","source":"Public Record"},
        {"company":"Volkswagen / Scout Motors","ticker":"VWAGY","type":"Manufacturing Plant","location":"Blythewood, SC","investment":"$2B","jobs":"4,000+","detail":"Scout Motors building an EV truck and SUV plant in South Carolina, opening ~2027.","cre_impact":"Industrial supplier and workforce housing demand in Columbia, SC metro.","source":"Public Record"},
        {"company":"Eli Lilly","ticker":"LLY","type":"Manufacturing Plant","location":"Lebanon, IN","investment":"$9B","jobs":"1,000+","detail":"Eli Lilly building four new manufacturing sites in Indiana for weight-loss and diabetes drugs.","cre_impact":"Industrial and office demand in Indianapolis suburban corridor.","source":"Public Record"},
        {"company":"Nucor","ticker":"NUE","type":"Manufacturing Plant","location":"West Virginia","investment":"$3B","jobs":"800+","detail":"Nucor steel sheet mill in Mason County, WV serving automotive and construction markets.","cre_impact":"Industrial growth opportunity in Ohio River Valley corridor.","source":"Public Record"},
    ]

    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return BASELINE

    try:
        from groq import Groq
        client = Groq(api_key=key)

        SCHEMA = (
            'Each element must have these exact keys:\n'
            '  "company"    — company name\n'
            '  "ticker"     — stock ticker if public, else ""\n'
            '  "type"       — "Manufacturing Plant" | "Warehouse / Distribution" | "Data Center" |\n'
            '                 "Training Center" | "Headquarters" | "Semiconductor Fab" |\n'
            '                 "Battery Plant" | "Research & Development" | "Other"\n'
            '  "location"   — "City, State"\n'
            '  "investment" — dollar amount (e.g. "$1.2B") or ""\n'
            '  "jobs"       — job count (e.g. "2,000+") or ""\n'
            '  "detail"     — one sentence describing the announcement\n'
            '  "cre_impact" — one sentence on CRE demand this creates\n'
            '  "source"     — news source or "Public Record"\n'
            'Return raw JSON array only — no markdown, no explanation.'
        )

        results = []

        # ── Pass 1: extract from live news articles ────────────────────────────
        if articles:
            lines = [
                f"{i}. [{a['source']}] {a['title']}\n   {a.get('description','')[:200]}"
                for i, a in enumerate(articles[:50], 1)
            ]
            article_block = "\n".join(lines)
            p1 = f"""Today is {datetime.now().strftime('%B %d, %Y')}.

Below are news headlines about US facility announcements:

{article_block}

Extract every CONFIRMED company announcement of a new or expanded US facility.
Only include what is clearly stated in the articles above.
{SCHEMA}"""
            try:
                r1 = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": "You are a data extraction assistant. Return a JSON object with key 'announcements' containing an array of facility objects. Only include what is explicitly stated in the articles — do not hallucinate details."},
                        {"role": "user", "content": p1},
                    ],
                    max_tokens=1400, temperature=0.1,
                    response_format={"type": "json_object"},
                )
                raw1 = r1.choices[0].message.content.strip()
                parsed1 = _json.loads(raw1)
                # Handle JSON object wrapper from structured output
                if isinstance(parsed1, dict):
                    results = parsed1.get("announcements", parsed1.get("facilities", []))
                    if not isinstance(results, list):
                        results = [parsed1] if parsed1.get("company") else []
                else:
                    results = parsed1 if isinstance(parsed1, list) else []
                # Validate each record has required schema keys
                valid = []
                for rec in results:
                    if isinstance(rec, dict) and rec.get("company"):
                        valid.append(rec)
                results = valid
            except Exception:
                results = []

        # ── Pass 2: supplement with known announcements from Groq knowledge ───
        p2 = f"""Today is {datetime.now().strftime('%B %d, %Y')}.

List 15 real, confirmed US company facility announcements from 2023 through early 2026.
Include a diverse mix: automotive plants, semiconductor fabs, battery plants, data centers,
distribution centers, training centers, and HQ moves.
Include well-known examples such as:
- Mercedes-Benz Alabama plant expansion
- TSMC Arizona semiconductor fab
- Intel Ohio fabs
- Samsung Texas semiconductor fab
- Hyundai/Kia Georgia EV plant
- Rivian Georgia EV plant
- Toyota North Carolina battery plant
- Amazon / Microsoft / Google data center expansions
- Any other major confirmed US facility announcements you know about

{SCHEMA}"""
        try:
            r2 = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a corporate real estate analyst with deep knowledge of US facility announcements. Return a JSON object with key 'announcements' containing an array of facility objects."},
                    {"role": "user", "content": p2},
                ],
                max_tokens=2400, temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw2 = r2.choices[0].message.content.strip()
            parsed2 = _json.loads(raw2)
            if isinstance(parsed2, dict):
                known = parsed2.get("announcements", parsed2.get("facilities", []))
                if not isinstance(known, list):
                    known = [parsed2] if parsed2.get("company") else []
            else:
                known = parsed2 if isinstance(parsed2, list) else []
            # Validate records
            known = [r for r in known if isinstance(r, dict) and r.get("company")]
        except Exception:
            known = []

        # Deduplicate: live news takes priority; supplement with known
        seen = {a.get("company", "").lower() for a in results}
        for item in known:
            if item.get("company", "").lower() not in seen:
                results.append(item)
                seen.add(item.get("company", "").lower())

        return results

    except Exception as e:
        return [{"company": "Error", "ticker": "", "type": "Other", "location": "",
                 "investment": "", "jobs": "", "detail": str(e),
                 "cre_impact": "", "source": ""}]


# ── Agent 6 · Interest Rate & Debt Markets ───────────────────────────────────

def run_rate_agent():
    _set_status("rates", "running")
    try:
        from src.rate_agent import run_rate_agent as _run
        result = _run()
        write_cache("rates", result)
        if result.get("error"):
            _set_status("rates", "error", result["error"])
        else:
            _set_status("rates", "ok")
    except Exception as e:
        _set_status("rates", "error", str(e))


# ── Agent 4 · Debugger / Monitor ─────────────────────────────────────────────

def run_debugger_agent():
    _set_status("debugger", "running")
    try:
        issues  = []
        healthy = []

        # Check each cache
        for key, max_age_h in [("migration", 7), ("pricing", 2), ("predictions", 25), ("energy_data", 7), ("sustainability_data", 7)]:
            c = read_cache(key)
            if c["data"] is None:
                issues.append(f"❌ {key}: no data in cache — agent has not run yet")
            elif c.get("stale"):
                issues.append(f"⚠️  {key}: data is stale ({c.get('age_minutes',0):.0f}m old, max {max_age_h*60}m)")
            else:
                healthy.append(f"✅ {key}: fresh ({c.get('age_minutes',0):.0f}m old)")

        # Check yfinance connectivity
        try:
            import yfinance as yf
            t = yf.Ticker("VNQ")
            p = t.info.get("currentPrice") or t.info.get("regularMarketPrice")
            healthy.append(f"✅ yfinance: live (VNQ = ${p:.2f})" if p else "⚠️  yfinance: price not returned")
        except Exception as e:
            issues.append(f"❌ yfinance: {e}")

        # Check Census API
        try:
            import requests
            r = requests.get(
                "https://api.census.gov/data/2023/pep/population",
                params={"get": "NAME,POP_2023", "for": "state:48"},   # Texas only
                timeout=8,
            )
            if r.status_code == 200:
                healthy.append("✅ Census API: reachable")
            else:
                issues.append(f"⚠️  Census API: HTTP {r.status_code}")
        except Exception as e:
            issues.append(f"❌ Census API: {e}")

        # Check Groq
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            healthy.append("✅ Groq API key: present")
        else:
            issues.append("⚠️  Groq API key: not set (.env missing GROQ_API_KEY)")

        # Check FRED API key
        fred_key = os.getenv("FRED_API_KEY", "")
        if fred_key:
            healthy.append("✅ FRED API key: present")
        else:
            issues.append("⚠️  FRED API key: not set (.env missing FRED_API_KEY — required for Rate Environment agent)")

        # Check rates cache
        rc = read_cache("rates")
        if rc["data"] is None:
            issues.append("❌ rates: no data in cache — agent has not run yet")
        elif rc.get("stale"):
            issues.append(f"⚠️  rates: data is stale ({rc.get('age_minutes',0):.0f}m old)")
        else:
            healthy.append(f"✅ rates: fresh ({rc.get('age_minutes',0):.0f}m old)")

        write_cache("debugger", {
            "issues":   issues,
            "healthy":  healthy,
            "agent_status": get_status(),
            "checked_at": datetime.now().isoformat(),
        })
        _set_status("debugger", "ok")
    except Exception as e:
        _set_status("debugger", "error", traceback.format_exc())


# ── Agent 5 · News & Facility Announcements ──────────────────────────────────

def run_news_agent():
    _set_status("news", "running")
    try:
        from src.cre_news import run_news_fetch
        result = run_news_fetch()
        write_cache("news", result)
        _set_status("news", "ok")
    except Exception as e:
        _set_status("news", "error", str(e))


# ── Agent 6 · Energy & Construction Costs ────────────────────────────────────

def run_energy_agent():
    _set_status("energy", "running")
    try:
        from src.energy_analyst import run_energy_analyst
        run_energy_analyst()
        _set_status("energy", "ok")
    except Exception as e:
        _set_status("energy", "error", str(e))


# ── Agent 7 · Sustainability & ESG ──────────────────────────────────────────

def run_sustainability_agent():
    _set_status("sustainability", "running")
    try:
        from src.sustainability_analyst import run_sustainability_analyst
        run_sustainability_analyst()
        _set_status("sustainability", "ok")
    except Exception as e:
        _set_status("sustainability", "error", str(e))


# ── Agent 9 · Labor Market & Tenant Demand ───────────────────────────────────

def run_labor_market_agent():
    _set_status("labor_market", "running")
    try:
        from src.labor_market_agent import run_labor_market_agent as _run
        result = _run()
        write_cache("labor_market", result)
        if result.get("error"):
            _set_status("labor_market", "error", result["error"])
        else:
            _set_status("labor_market", "ok")
    except Exception as e:
        _set_status("labor_market", "error", str(e))


# ── Agent 10 · GDP & Economic Growth ─────────────────────────────────────────

def run_gdp_agent():
    _set_status("gdp", "running")
    try:
        from src.gdp_agent import run_gdp_agent as _run
        result = _run()
        write_cache("gdp_data", result)
        if result.get("error"):
            _set_status("gdp", "error", result["error"])
        else:
            _set_status("gdp", "ok")
    except Exception as e:
        _set_status("gdp", "error", str(e))


# ── Agent 11 · Inflation & Construction Costs ─────────────────────────────────

def run_inflation_agent():
    _set_status("inflation", "running")
    try:
        from src.inflation_agent import run_inflation_agent as _run
        result = _run()
        write_cache("inflation_data", result)
        if result.get("error"):
            _set_status("inflation", "error", result["error"])
        else:
            _set_status("inflation", "ok")
    except Exception as e:
        _set_status("inflation", "error", str(e))


# ── Agent 12 · Credit & Capital Markets ──────────────────────────────────────

def run_credit_markets_agent():
    _set_status("credit", "running")
    try:
        from src.credit_markets_agent import run_credit_markets_agent as _run
        result = _run()
        write_cache("credit_data", result)
        if result.get("error"):
            _set_status("credit", "error", result["error"])
        else:
            _set_status("credit", "ok")
    except Exception as e:
        _set_status("credit", "error", str(e))


def run_vacancy_agent():
    _set_status("vacancy", "running")
    try:
        from src.vacancy_agent import run_vacancy_agent as _run
        result = _run()
        write_cache("vacancy", result)
        _set_status("vacancy", "ok")
    except Exception as e:
        _set_status("vacancy", "error", str(e))


# ── Agent 13 · Land Market ────────────────────────────────────────────────────

def run_land_market_agent():
    _set_status("land_market", "running")
    try:
        from src.land_market_agent import run_land_market_agent as _run
        result = _run()
        write_cache("land_market", result)
        _set_status("land_market", "ok")
    except Exception as e:
        _set_status("land_market", "error", str(e))


# ── Agent 14 · Cap Rate Monitor ───────────────────────────────────────────────

def run_cap_rate_agent():
    _set_status("cap_rate", "running")
    try:
        from src.cap_rate_agent import run_cap_rate_agent as _run
        result = _run()
        write_cache("cap_rate", result)
        _set_status("cap_rate", "ok")
    except Exception as e:
        _set_status("cap_rate", "error", str(e))


# ── Agent 15 · Rent Growth Tracker ───────────────────────────────────────────

def run_rent_growth_agent():
    _set_status("rent_growth", "running")
    try:
        from src.rent_growth_agent import run_rent_growth_agent as _run
        result = _run()
        write_cache("rent_growth", result)
        _set_status("rent_growth", "ok")
    except Exception as e:
        _set_status("rent_growth", "error", str(e))


# ── Agent 16 · Opportunity Zone & Incentives ─────────────────────────────────

def run_opportunity_zone_agent():
    _set_status("opportunity_zone", "running")
    try:
        from src.opportunity_zone_agent import run_opportunity_zone_agent as _run
        result = _run()
        write_cache("opportunity_zone", result)
        _set_status("opportunity_zone", "ok")
    except Exception as e:
        _set_status("opportunity_zone", "error", str(e))


# ── Agent 17 · CMBS & Distressed Assets ──────────────────────────────────────

def run_distressed_agent():
    _set_status("distressed", "running")
    try:
        from src.distressed_asset_agent import run_distressed_asset_agent as _run
        result = _run()
        write_cache("distressed", result)
        _set_status("distressed", "ok")
    except Exception as e:
        _set_status("distressed", "error", str(e))


# ── Manager Agent · System Health & Self-Healing ─────────────────────────────

def run_manager_agent_job():
    _set_status("manager", "running")
    try:
        from src.manager_agent import run_manager_agent
        result = run_manager_agent()
        import json as _json
        (CACHE_DIR / "manager_report.json").write_text(
            _json.dumps({"updated_at": datetime.now().isoformat(), "data": result}, default=str)
        )
        _set_status("manager", "ok")
    except Exception as e:
        _set_status("manager", "error", str(e))


# ── Agent 18 · Market Opportunity Score ──────────────────────────────────────

def run_market_score_agent():
    _set_status("market_score", "running")
    try:
        from src.market_score_agent import run_market_score_agent as _run
        result = _run()
        write_cache("market_score", result)
        _set_status("market_score", "ok")
    except Exception as e:
        _set_status("market_score", "error", str(e))


# ── Agent 19 · Climate Risk ───────────────────────────────────────────────────

def run_climate_risk_agent():
    _set_status("climate_risk", "running")
    try:
        from src.climate_risk_agent import run_climate_risk_agent as _run
        result = _run()
        write_cache("climate_risk", result)
        _set_status("climate_risk", "ok")
    except Exception as e:
        _set_status("climate_risk", "error", str(e))


# ── Scheduler Singleton ───────────────────────────────────────────────────────

_scheduler: BackgroundScheduler = None
_scheduler_lock = threading.Lock()


def start_scheduler():
    global _scheduler
    with _scheduler_lock:
        if _scheduler is not None and _scheduler.running:
            return  # already running

        _scheduler = BackgroundScheduler(daemon=True)

        _scheduler.add_job(run_migration_agent,   IntervalTrigger(hours=6),        id="migration",   replace_existing=True)
        _scheduler.add_job(run_pricing_agent,      IntervalTrigger(hours=1),        id="pricing",     replace_existing=True)
        _scheduler.add_job(run_predictions_agent,  IntervalTrigger(hours=24),       id="predictions", replace_existing=True)
        _scheduler.add_job(run_debugger_agent,     IntervalTrigger(minutes=30),     id="debugger",    replace_existing=True)
        _scheduler.add_job(run_news_agent,         IntervalTrigger(hours=4),        id="news",        replace_existing=True)
        _scheduler.add_job(run_rate_agent,         IntervalTrigger(hours=1),        id="rates",       replace_existing=True)
        _scheduler.add_job(run_energy_agent,       IntervalTrigger(hours=6),        id="energy",      replace_existing=True)
        _scheduler.add_job(run_sustainability_agent, IntervalTrigger(hours=6),      id="sustainability",  replace_existing=True)
        _scheduler.add_job(run_labor_market_agent,   IntervalTrigger(hours=6),      id="labor_market",    replace_existing=True)
        _scheduler.add_job(run_gdp_agent,            IntervalTrigger(hours=6),      id="gdp",             replace_existing=True)
        _scheduler.add_job(run_inflation_agent,      IntervalTrigger(hours=6),      id="inflation",       replace_existing=True)
        _scheduler.add_job(run_credit_markets_agent, IntervalTrigger(hours=6),      id="credit",          replace_existing=True)
        _scheduler.add_job(run_vacancy_agent,          IntervalTrigger(hours=12),     id="vacancy",          replace_existing=True)
        _scheduler.add_job(run_land_market_agent,      IntervalTrigger(hours=12),     id="land_market",      replace_existing=True)
        _scheduler.add_job(run_cap_rate_agent,         IntervalTrigger(hours=6),      id="cap_rate",         replace_existing=True)
        _scheduler.add_job(run_rent_growth_agent,      IntervalTrigger(hours=6),      id="rent_growth",      replace_existing=True)
        _scheduler.add_job(run_opportunity_zone_agent, IntervalTrigger(hours=24),     id="opportunity_zone", replace_existing=True)
        _scheduler.add_job(run_distressed_agent,       IntervalTrigger(hours=6),      id="distressed",       replace_existing=True)
        _scheduler.add_job(run_market_score_agent,     IntervalTrigger(hours=6),      id="market_score",     replace_existing=True)
        _scheduler.add_job(run_climate_risk_agent,     IntervalTrigger(hours=24),     id="climate_risk",     replace_existing=True)
        _scheduler.add_job(run_manager_agent_job,       IntervalTrigger(minutes=15),   id="manager",          replace_existing=True)

        _scheduler.start()

        # Run all agents immediately on first start (in background threads)
        for fn in [run_debugger_agent, run_migration_agent, run_pricing_agent, run_predictions_agent,
                   run_news_agent, run_rate_agent, run_energy_agent, run_sustainability_agent,
                   run_labor_market_agent, run_gdp_agent, run_inflation_agent, run_credit_markets_agent,
                   run_vacancy_agent, run_land_market_agent, run_cap_rate_agent, run_rent_growth_agent,
                   run_opportunity_zone_agent, run_distressed_agent, run_market_score_agent,
                   run_climate_risk_agent, run_manager_agent_job]:
            t = threading.Thread(target=fn, daemon=True)
            t.start()


def force_run(agent_name: str):
    """Manually trigger a specific agent immediately."""
    agents = {
        "migration":      run_migration_agent,
        "pricing":        run_pricing_agent,
        "predictions":    run_predictions_agent,
        "debugger":       run_debugger_agent,
        "news":           run_news_agent,
        "rates":          run_rate_agent,
        "energy":         run_energy_agent,
        "sustainability": run_sustainability_agent,
        "labor_market":  run_labor_market_agent,
        "gdp":           run_gdp_agent,
        "inflation":     run_inflation_agent,
        "credit":           run_credit_markets_agent,
        "vacancy":          run_vacancy_agent,
        "land_market":      run_land_market_agent,
        "cap_rate":         run_cap_rate_agent,
        "rent_growth":      run_rent_growth_agent,
        "opportunity_zone": run_opportunity_zone_agent,
        "distressed":       run_distressed_agent,
        "market_score":     run_market_score_agent,
        "manager":          run_manager_agent_job,
        "climate_risk":     run_climate_risk_agent,
    }
    fn = agents.get(agent_name)
    if fn:
        t = threading.Thread(target=fn, daemon=True)
        t.start()
