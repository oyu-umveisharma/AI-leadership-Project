"""
Manager Agent — System Health & Self-Healing Supervisor
=========================================================
Runs every 15 minutes. Checks every other agent for:
  1. API key availability (FRED, GROQ)
  2. Cache file presence and freshness
  3. Error status in in-memory agent_status dict
  4. Stale caches that exceeded their refresh window
  5. Auto-restarts any agent whose cache is missing or errored

Writes a health report to cache/manager_report.json.
"""

import os
import threading
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

CACHE_DIR = Path(__file__).parent.parent / "cache"

# ── API keys to verify ────────────────────────────────────────────────────────
API_KEYS = {
    "FRED_API_KEY":  "FRED (rates, labor, macro data)",
    "GROQ_API_KEY":  "Groq AI (news summaries, investment advisor)",
}

# ── Agent registry: key → (cache_file, max_stale_hours, restart_fn_name) ─────
AGENT_REGISTRY = {
    "migration":       ("migration",         7),
    "pricing":         ("pricing",           2),
    "predictions":     ("predictions",       25),
    "news":            ("news",              5),
    "rates":           ("rates",             2),
    "energy":          ("energy_data",       7),
    "sustainability":  ("sustainability_data", 7),
    "labor_market":    ("labor_market",      7),
    "gdp":             ("gdp_data",          7),
    "inflation":       ("inflation_data",    7),
    "credit":          ("credit_data",       7),
    "vacancy":         ("vacancy",           13),
    "land_market":     ("land_market",       13),
    "cap_rate":        ("cap_rate",          7),
    "rent_growth":     ("rent_growth",       7),
    "opportunity_zone":("opportunity_zone",  25),
    "distressed":      ("distressed",        7),
    "market_score":    ("market_score",      7),
    "climate_risk":    ("climate_risk",      25),
}


def _check_api_keys() -> list[dict]:
    """Verify all required API keys are present and non-empty."""
    results = []
    for key, description in API_KEYS.items():
        val = os.getenv(key, "").strip()
        if not val or val.startswith("your_"):
            results.append({
                "key":         key,
                "description": description,
                "status":      "MISSING",
                "message":     f"{key} is not set in .env — {description} will be disabled.",
            })
        else:
            results.append({
                "key":         key,
                "description": description,
                "status":      "OK",
                "message":     f"{key} is set ({val[:6]}…)",
            })
    return results


def _check_cache(cache_key: str, max_stale_hours: int) -> dict:
    """Check a cache file for presence and freshness."""
    path = CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return {"status": "MISSING", "age_hours": None, "path": str(path)}

    try:
        import json
        with open(path) as f:
            data = json.load(f)
        # Try several timestamp fields
        ts_raw = (
            data.get("updated_at") or
            data.get("timestamp") or
            data.get("cached_at") or
            (data.get("data") or {}).get("cached_at") or
            (data.get("data") or {}).get("updated_at")
        )
        if ts_raw:
            ts = datetime.fromisoformat(str(ts_raw)[:19])
            age_hours = (datetime.now() - ts).total_seconds() / 3600
            stale = age_hours > max_stale_hours
            return {
                "status":     "STALE" if stale else "OK",
                "age_hours":  round(age_hours, 1),
                "updated_at": ts.isoformat(),
            }
        return {"status": "OK", "age_hours": None, "updated_at": None}
    except Exception as e:
        return {"status": "ERROR", "age_hours": None, "error": str(e)}


def _restart_agent(agent_key: str) -> str:
    """Import and fire the agent's run function in a background thread."""
    try:
        from src.cre_agents import force_run
        t = threading.Thread(target=force_run, args=(agent_key,), daemon=True)
        t.start()
        return "RESTARTED"
    except Exception as e:
        return f"RESTART_FAILED: {e}"


def run_manager_agent() -> dict:
    """
    Main entry — audit all agents, auto-heal where possible, return health report.
    """
    print("=" * 60)
    print("[Manager] Starting health check ...")
    print("=" * 60)

    report_time = datetime.now()

    # ── 1. API key audit ──────────────────────────────────────────────────────
    key_checks = _check_api_keys()
    key_issues = [k for k in key_checks if k["status"] == "MISSING"]
    print(f"[Manager] API keys: {len(key_checks) - len(key_issues)}/{len(key_checks)} OK")
    for ki in key_issues:
        print(f"  ⚠  {ki['message']}")

    # ── 2. Cache + status audit ───────────────────────────────────────────────
    try:
        from src.cre_agents import get_status
        mem_status = get_status()
    except Exception:
        mem_status = {}

    agent_checks = []
    healed = []
    issues = []

    for agent_key, (cache_key, max_stale_h) in AGENT_REGISTRY.items():
        mem  = mem_status.get(agent_key, {})
        mem_err = mem.get("last_error") or ""
        mem_st  = mem.get("status", "idle")

        cache_result = _check_cache(cache_key, max_stale_h)
        cache_status = cache_result["status"]

        # Determine overall health
        if mem_st == "running":
            health = "RUNNING"
        elif cache_status == "MISSING":
            health = "MISSING"
        elif cache_status == "ERROR":
            health = "CACHE_ERROR"
        elif cache_status == "STALE":
            health = "STALE"
        elif mem_st == "error" and cache_status != "OK":
            health = "ERROR"
        else:
            health = "OK"

        needs_heal = health in ("MISSING", "STALE", "ERROR", "CACHE_ERROR")
        heal_result = None

        if needs_heal:
            print(f"  [Manager] {agent_key} → {health} — attempting restart ...")
            heal_result = _restart_agent(agent_key)
            if "RESTARTED" in heal_result:
                healed.append(agent_key)
                print(f"  ✓ {agent_key} restarted")
            else:
                issues.append({"agent": agent_key, "health": health, "error": heal_result})
                print(f"  ✗ {agent_key} restart failed: {heal_result}")
        else:
            print(f"  [Manager] {agent_key} → {health}")

        agent_checks.append({
            "agent":       agent_key,
            "cache_key":   cache_key,
            "health":      health,
            "cache_status":cache_status,
            "age_hours":   cache_result.get("age_hours"),
            "updated_at":  cache_result.get("updated_at"),
            "mem_status":  mem_st,
            "last_error":  mem_err[:120] if mem_err else "",
            "healed":      heal_result,
        })

    # ── 3. Summary ────────────────────────────────────────────────────────────
    n_total   = len(agent_checks)
    n_ok      = sum(1 for a in agent_checks if a["health"] == "OK")
    n_healed  = len(healed)
    n_issues  = sum(1 for a in agent_checks if a["health"] not in ("OK", "RUNNING"))
    health_pct = round(n_ok / n_total * 100)

    report = {
        "checked_at":    report_time.isoformat(),
        "health_pct":    health_pct,
        "total_agents":  n_total,
        "ok":            n_ok,
        "healed":        n_healed,
        "issues":        n_issues - n_healed,
        "api_keys":      key_checks,
        "key_issues":    len(key_issues),
        "agents":        agent_checks,
        "healed_agents": healed,
        "unresolved":    [a["agent"] for a in agent_checks
                          if a["health"] not in ("OK", "RUNNING") and a.get("healed") and "FAILED" in str(a["healed"])],
    }

    print(f"[Manager] Health: {health_pct}%  OK:{n_ok}  Healed:{n_healed}  Issues:{n_issues - n_healed}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    result = run_manager_agent()
    print(f"\nHealth: {result['health_pct']}%")
    print(f"Healed agents: {result['healed_agents']}")
    for ki in result["api_keys"]:
        print(f"  {ki['key']}: {ki['status']} — {ki['message']}")
