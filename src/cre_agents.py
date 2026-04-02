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

CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def write_cache(key: str, data: Any):
    payload = {"updated_at": datetime.now().isoformat(), "data": data}
    with open(_cache_path(key), "w") as f:
        json.dump(payload, f, default=str)


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
}

def get_status() -> dict:
    with _status_lock:
        return {k: dict(v) for k, v in _agent_status.items()}

def _set_status(agent: str, status: str, error: str = None):
    with _status_lock:
        _agent_status[agent]["status"] = status
        if status == "running":
            _agent_status[agent]["last_run"] = datetime.now().isoformat()
        if error:
            _agent_status[agent]["last_error"] = error
        else:
            _agent_status[agent]["runs"] += 1


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
    if confirmed:
        _set_status("predictions", "ok")
    else:
        _set_status("predictions", "error", "No announcements extracted — check GROQ_API_KEY")


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

    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return []

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
                        {"role": "system", "content": "You are a data extraction assistant. Return only valid JSON arrays."},
                        {"role": "user", "content": p1},
                    ],
                    max_tokens=1400, temperature=0.1,
                )
                raw1 = r1.choices[0].message.content.strip()
                if raw1.startswith("```"):
                    raw1 = raw1.split("```")[1]
                    if raw1.startswith("json"):
                        raw1 = raw1[4:]
                results = _json.loads(raw1.strip())
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
                    {"role": "system", "content": "You are a corporate real estate analyst with deep knowledge of US facility announcements. Return only valid JSON arrays."},
                    {"role": "user", "content": p2},
                ],
                max_tokens=2400, temperature=0.2,
            )
            raw2 = r2.choices[0].message.content.strip()
            if raw2.startswith("```"):
                raw2 = raw2.split("```")[1]
                if raw2.startswith("json"):
                    raw2 = raw2[4:]
            known = _json.loads(raw2.strip())
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

        _scheduler.start()

        # Run all agents immediately on first start (in background threads)
        for fn in [run_debugger_agent, run_migration_agent, run_pricing_agent, run_predictions_agent,
                   run_news_agent, run_rate_agent, run_energy_agent, run_sustainability_agent,
                   run_labor_market_agent, run_gdp_agent, run_inflation_agent, run_credit_markets_agent]:
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
        "credit":        run_credit_markets_agent,
    }
    fn = agents.get(agent_name)
    if fn:
        t = threading.Thread(target=fn, daemon=True)
        t.start()
