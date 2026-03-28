"""
Background Agent Runner — six independent agents updating on schedules.

Agent 1 · Migration Tracker    — every 6 hours
Agent 2 · REIT Pricing          — every 1 hour
Agent 3 · Company Predictions   — every 24 hours (LLM)
Agent 4 · Debugger / Monitor    — every 30 minutes
Agent 5 · News & Announcements  — every 4 hours
Agent 6 · Interest Rate & Debt  — every 1 hour  (requires FRED_API_KEY)

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
        updated = datetime.fromisoformat(payload["updated_at"])
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
    "migration":    {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "pricing":      {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "predictions":  {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "debugger":     {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "news":         {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
    "rates":        {"status": "idle",    "last_run": None, "last_error": None, "runs": 0},
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
    try:
        import os
        from groq import Groq
        from src.cre_population import fetch_migration_scores
        from src.cre_listings import get_cheapest_buildings

        mig_df = fetch_migration_scores()
        top5   = mig_df.head(5)[["state_name","state_abbr","pop_growth_pct","biz_score","key_companies","growth_drivers"]].to_dict(orient="records")
        top3_cities_abbr = mig_df.head(3)["state_abbr"].tolist()

        # Predicted company moves via LLM
        predictions_text = _generate_predictions(top5)

        # Cheapest buildings in top 3 cities
        listings = {}
        for abbr in top3_cities_abbr:
            try:
                listings[abbr] = get_cheapest_buildings(abbr)
            except Exception:
                listings[abbr] = []

        write_cache("predictions", {
            "predictions_text": predictions_text,
            "top5_states":      top5,
            "listings":         listings,
            "top3_abbr":        top3_cities_abbr,
        })
        _set_status("predictions", "ok")
    except Exception as e:
        _set_status("predictions", "error", str(e))


def _generate_predictions(top5_states: list) -> str:
    try:
        import os
        from groq import Groq
        key = os.getenv("GROQ_API_KEY", "")
        if not key:
            return "_GROQ_API_KEY not set. Add it to .env to enable AI predictions._"
        client = Groq(api_key=key)
        states_str = "\n".join([
            f"- {s['state_name']}: pop growth {s['pop_growth_pct']:+.1f}%, biz score {s['biz_score']}, "
            f"recent moves: {s['key_companies']}, drivers: {s['growth_drivers']}"
            for s in top5_states
        ])
        prompt = f"""
Today is {datetime.now().strftime('%B %d, %Y')}.

Top 5 US states by migration attractiveness:
{states_str}

Based on current migration trends, tax policy, labor markets, and recent corporate announcements:

1. Name 8 specific companies (real, public companies) most likely to announce HQ relocation
   or major expansion in the next 12 months. For each give:
   - Company name and ticker
   - Current HQ state
   - Predicted destination state
   - Probability (High/Medium/Low)
   - Primary reason (taxes, talent, supply chain, incentives)

2. Name 3 industries most likely to cluster in each of the top 3 states.

Format as a clean table then a brief paragraph for each top state.
"""
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a corporate real estate strategist tracking Fortune 500 and S&P 500 company relocations. Base predictions on publicly available signals."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=900,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"_Prediction generation failed: {e}_"


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
        for key, max_age_h in [("migration", 7), ("pricing", 2), ("predictions", 25)]:
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

        _scheduler.start()

        # Run all agents immediately on first start (in background threads)
        for fn in [run_debugger_agent, run_migration_agent, run_pricing_agent, run_predictions_agent, run_news_agent, run_rate_agent]:
            t = threading.Thread(target=fn, daemon=True)
            t.start()


def force_run(agent_name: str):
    """Manually trigger a specific agent immediately."""
    agents = {
        "migration":   run_migration_agent,
        "pricing":     run_pricing_agent,
        "predictions": run_predictions_agent,
        "debugger":    run_debugger_agent,
        "news":        run_news_agent,
        "rates":       run_rate_agent,
    }
    fn = agents.get(agent_name)
    if fn:
        t = threading.Thread(target=fn, daemon=True)
        t.start()
