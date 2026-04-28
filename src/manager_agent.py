"""
Manager Agent — System Health & Self-Healing Supervisor
=========================================================
Runs every 15 minutes. Checks every other agent for:
  1. API key availability (FRED, GROQ)
  2. Cache file presence and freshness
  3. Required field validation inside cache files
  4. Error status and consecutive-failure tracking
  5. Agent dependency chain for Investment Advisor
  6. Auto-restarts agents — but backs off after 3 consecutive failures
  7. Verification pass: confirms restart actually wrote fresh data

Writes a health report to cache/manager_report.json.
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH, override=True)

CACHE_DIR = Path(__file__).parent.parent / "cache"

# ── API keys to verify ────────────────────────────────────────────────────────
API_KEYS = {
    "FRED_API_KEY": "FRED (rates, labor, macro data)",
    "GROQ_API_KEY": "Groq AI (news summaries, investment advisor)",
}

# ── Known error patterns → actionable messages ────────────────────────────────
ERROR_HINTS = {
    "FRED_API_KEY":  "Set FRED_API_KEY in your .env file — get a free key at fred.stlouisfed.org",
    "GROQ_API_KEY":  "Set GROQ_API_KEY in your .env file — get a free key at console.groq.com",
    "ConnectionError":   "Network unreachable — check internet connection",
    "Timeout":           "API request timed out — server may be overloaded, will retry next cycle",
    "401":               "API key rejected — verify the key is correct and active",
    "403":               "API access forbidden — check plan limits or key permissions",
    "429":               "Rate limit hit — too many requests, will back off automatically",
    "JSONDecodeError":   "API returned malformed data — likely a temporary outage",
    "KeyError":          "Cache schema changed — agent needs a full refresh",
}

# ── Required fields per cache — validates data completeness ───────────────────
# Format: cache_key -> list of top-level keys that must be present and non-empty
REQUIRED_FIELDS = {
    "labor_market":    ["fred_labor", "sector_etfs", "demand_signal"],
    "rates":           ["data"],
    "cap_rate":        ["data"],
    "rent_growth":     ["data"],
    "vacancy":         ["data"],
    "market_score":    ["data"],
    "migration":       ["migration"],
    "gdp_data":        ["data"],
    "inflation_data":  ["data"],
    "credit_data":     ["data"],
    "news":            ["articles"],
    "pricing":         ["data"],
    "energy_data":     ["data"],
    "sustainability_data": ["data"],
    "climate_risk":    ["data"],
    "opportunity_zone":["data"],
    "distressed":      ["data"],
    "rentcast":        ["data"],
    "forecast":        ["data"],
}

# ── Investment Advisor dependency chain ───────────────────────────────────────
# These agents feed the Investment Advisor — if degraded, advisor output suffers
ADVISOR_DEPENDENCIES = [
    "labor_market", "rates", "cap_rate", "rent_growth",
    "vacancy", "market_score", "migration", "gdp",
    "inflation", "credit", "climate_risk", "forecast",
]

# ── Agent registry: key → (cache_file, max_stale_hours) ──────────────────────
AGENT_REGISTRY = {
    "migration":        ("migration",          7),
    "pricing":          ("pricing",            2),
    "predictions":      ("predictions",        25),
    "news":             ("news",               5),
    "rates":            ("rates",              2),
    "energy":           ("energy_data",        7),
    "sustainability":   ("sustainability_data", 7),
    "labor_market":     ("labor_market",       7),
    "gdp":              ("gdp_data",           7),
    "inflation":        ("inflation_data",     7),
    "credit":           ("credit_data",        7),
    "vacancy":          ("vacancy",            13),
    "land_market":      ("land_market",        13),
    "cap_rate":         ("cap_rate",           7),
    "rent_growth":      ("rent_growth",        7),
    "opportunity_zone": ("opportunity_zone",   25),
    "distressed":       ("distressed",         7),
    "market_score":     ("market_score",       7),
    "climate_risk":     ("climate_risk",       25),
    "rentcast":         ("rentcast",           25),
    "forecast":         ("forecast",           7),
}

# Max consecutive failures before backing off restarts
MAX_FAILURES_BEFORE_BACKOFF = 8

# If cache is older than this many hours, bypass backoff and force a retry
FORCE_RETRY_AFTER_HOURS = 24


# ── Pending verification state (persisted across runs) ────────────────────────
_PENDING_FILE = CACHE_DIR / "manager_pending.json"

def _load_pending() -> dict:
    try:
        if _PENDING_FILE.exists():
            return json.loads(_PENDING_FILE.read_text())
    except Exception:
        pass
    return {}

def _save_pending(pending: dict):
    try:
        _PENDING_FILE.write_text(json.dumps(pending))
    except Exception:
        pass


# ── API key audit ─────────────────────────────────────────────────────────────
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
                "hint":        ERROR_HINTS.get(key, ""),
            })
        else:
            results.append({
                "key":         key,
                "description": description,
                "status":      "OK",
                "message":     f"{key} is set ({val[:6]}…)",
                "hint":        "",
            })
    return results


# ── Cache freshness + field validation ────────────────────────────────────────
def _check_cache(cache_key: str, max_stale_hours: int) -> dict:
    """Check a cache file for presence, freshness, and required field completeness."""
    path = CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return {"status": "MISSING", "age_hours": None, "path": str(path)}

    try:
        with open(path) as f:
            data = json.load(f)

        # Timestamp check
        ts_raw = (
            data.get("updated_at") or
            data.get("timestamp") or
            data.get("cached_at") or
            (data.get("data") or {}).get("cached_at") or
            (data.get("data") or {}).get("updated_at")
        )
        age_hours = None
        stale = False
        if ts_raw:
            ts = datetime.fromisoformat(str(ts_raw)[:19])
            age_hours = (datetime.now() - ts).total_seconds() / 3600
            stale = age_hours > max_stale_hours

        # Check for error payload written into the cache
        inner = data.get("data") or data
        cached_error = data.get("error") or inner.get("error")
        if cached_error:
            return {
                "status":    "INVALID",
                "age_hours": round(age_hours, 1) if age_hours else None,
                "updated_at": ts_raw,
                "missing_fields": [],
                "error":     str(cached_error)[:200],
            }

        # Required field validation
        missing_fields = []
        for field in REQUIRED_FIELDS.get(cache_key, []):
            val = data.get(field)
            if val is None or val == [] or val == {}:
                missing_fields.append(field)

        if missing_fields:
            return {
                "status":         "INVALID",
                "age_hours":      round(age_hours, 1) if age_hours else None,
                "updated_at":     ts_raw,
                "missing_fields": missing_fields,
            }

        if stale:
            return {
                "status":    "STALE",
                "age_hours": round(age_hours, 1),
                "updated_at": ts_raw,
            }

        return {
            "status":    "OK",
            "age_hours": round(age_hours, 1) if age_hours else None,
            "updated_at": ts_raw,
        }

    except Exception as e:
        return {"status": "ERROR", "age_hours": None, "error": str(e)}


# ── Restart with backoff ──────────────────────────────────────────────────────
def _restart_agent(agent_key: str, consecutive_failures: int, age_hours: float = None) -> str:
    """
    Fire the agent's run function in a background thread.
    Backs off after MAX_FAILURES_BEFORE_BACKOFF, but always retries
    if the cache is older than FORCE_RETRY_AFTER_HOURS.
    """
    force_retry = age_hours is not None and age_hours >= FORCE_RETRY_AFTER_HOURS
    if consecutive_failures >= MAX_FAILURES_BEFORE_BACKOFF and not force_retry:
        return f"SKIPPED_BACKOFF (failed {consecutive_failures}x — manual check needed)"

    try:
        from src.cre_agents import force_run
        t = threading.Thread(target=force_run, args=(agent_key,), daemon=True)
        t.start()
        return "RESTARTED"
    except Exception as e:
        return f"RESTART_FAILED: {e}"


# ── Actionable hint from error text ──────────────────────────────────────────
def _get_error_hint(error_text: str) -> str:
    if not error_text:
        return ""
    for pattern, hint in ERROR_HINTS.items():
        if pattern.lower() in error_text.lower():
            return hint
    return ""


# ── Verify pending restarts from previous run ─────────────────────────────────
def _verify_pending(pending: dict) -> dict:
    """
    For agents restarted in the previous cycle, check if the cache was
    actually updated after the restart time. Returns verification results.
    """
    results = {}
    for agent_key, info in pending.items():
        cache_key    = info.get("cache_key", agent_key)
        restarted_at = info.get("restarted_at")
        if not restarted_at:
            continue
        path = CACHE_DIR / f"{cache_key}.json"
        if not path.exists():
            results[agent_key] = "NOT_WRITTEN"
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            ts_raw = (
                data.get("updated_at") or data.get("timestamp") or
                data.get("cached_at") or
                (data.get("data") or {}).get("cached_at") or
                (data.get("data") or {}).get("updated_at")
            )
            if ts_raw:
                ts = datetime.fromisoformat(str(ts_raw)[:19])
                restart_dt = datetime.fromisoformat(restarted_at[:19])
                results[agent_key] = "CONFIRMED" if ts > restart_dt else "UNCONFIRMED"
            else:
                results[agent_key] = "UNCONFIRMED"
        except Exception:
            results[agent_key] = "UNCONFIRMED"
    return results


# ── Main health check ─────────────────────────────────────────────────────────
def run_manager_agent() -> dict:
    """
    Main entry — audit all agents, auto-heal where possible, return health report.
    """
    print("=" * 60)
    print("[Manager] Starting health check ...")
    print("=" * 60)

    report_time = datetime.now()

    # ── 1. API key audit ──────────────────────────────────────────────────────
    key_checks  = _check_api_keys()
    key_issues  = [k for k in key_checks if k["status"] == "MISSING"]
    print(f"[Manager] API keys: {len(key_checks) - len(key_issues)}/{len(key_checks)} OK")
    for ki in key_issues:
        print(f"  ⚠  {ki['message']}")

    # ── 2. Verify previous restarts ───────────────────────────────────────────
    pending = _load_pending()
    verification = _verify_pending(pending) if pending else {}
    if verification:
        confirmed   = sum(1 for v in verification.values() if v == "CONFIRMED")
        unconfirmed = sum(1 for v in verification.values() if v != "CONFIRMED")
        print(f"[Manager] Restart verification: {confirmed} confirmed, {unconfirmed} unconfirmed")

    # ── 3. Get in-memory status (consecutive failure counts) ──────────────────
    try:
        from src.cre_agents import get_status
        mem_status = get_status()
    except Exception:
        mem_status = {}

    # ── 4. Cache + field audit ────────────────────────────────────────────────
    agent_checks = []
    healed       = []
    backed_off   = []
    issues       = []
    new_pending  = {}

    for agent_key, (cache_key, max_stale_h) in AGENT_REGISTRY.items():
        mem       = mem_status.get(agent_key, {})
        mem_err   = mem.get("last_error") or ""
        mem_st    = mem.get("status", "idle")
        consec_f  = mem.get("consecutive_failures", 0)

        cache_result = _check_cache(cache_key, max_stale_h)
        cache_status = cache_result["status"]

        # Verify previous restart if applicable
        verify_status = verification.get(agent_key)

        # Detect API key config errors — no point restarting, need user action
        _api_key_error = any(
            kw in mem_err.lower() for kw in ("api_key not set", "api key not set", "_api_key")
        ) if mem_err else False

        # Overall health
        if mem_st == "running":
            health = "RUNNING"
        elif cache_status == "MISSING":
            health = "MISSING"
        elif cache_status == "ERROR":
            health = "CACHE_ERROR"
        elif cache_status == "INVALID":
            health = "INVALID"
        elif cache_status == "STALE":
            health = "STALE"
        elif mem_st == "error":
            health = "NEEDS_CONFIG" if _api_key_error else "ERROR"
        else:
            health = "OK"

        needs_heal  = health in ("MISSING", "STALE", "ERROR", "CACHE_ERROR", "INVALID")
        # Never restart agents that need a config change — restart won't help
        if health == "NEEDS_CONFIG":
            needs_heal = False
        heal_result = None
        hint        = _get_error_hint(mem_err)

        if needs_heal:
            print(f"  [Manager] {agent_key} → {health} (failures: {consec_f}) — attempting restart ...")
            heal_result = _restart_agent(agent_key, consec_f, cache_result.get("age_hours"))
            if heal_result == "RESTARTED":
                healed.append(agent_key)
                new_pending[agent_key] = {
                    "cache_key":    cache_key,
                    "restarted_at": report_time.isoformat(),
                }
                print(f"  ✓ {agent_key} restarted")
            elif "BACKOFF" in heal_result:
                backed_off.append(agent_key)
                print(f"  — {agent_key} skipped (backoff): {heal_result}")
            else:
                issues.append({"agent": agent_key, "health": health, "error": heal_result})
                print(f"  ✗ {agent_key} restart failed: {heal_result}")
        else:
            print(f"  [Manager] {agent_key} → {health}")

        agent_checks.append({
            "agent":               agent_key,
            "cache_key":           cache_key,
            "health":              health,
            "cache_status":        cache_status,
            "age_hours":           cache_result.get("age_hours"),
            "updated_at":          cache_result.get("updated_at"),
            "missing_fields":      cache_result.get("missing_fields", []),
            "mem_status":          mem_st,
            "last_error":          mem_err[:200] if mem_err else "",
            "consecutive_failures":consec_f,
            "hint":                hint,
            "healed":              heal_result,
            "verify_status":       verify_status,
        })

    # Save pending verifications for the next run
    _save_pending(new_pending)

    # ── 5. Investment Advisor dependency health ────────────────────────────────
    advisor_dep_checks = []
    for dep in ADVISOR_DEPENDENCIES:
        entry = next((a for a in agent_checks if a["agent"] == dep), None)
        if entry:
            advisor_dep_checks.append({
                "agent":  dep,
                "health": entry["health"],
                "hint":   entry["hint"],
            })

    advisor_degraded = [d for d in advisor_dep_checks if d["health"] not in ("OK", "RUNNING")]
    advisor_ok       = len(advisor_dep_checks) - len(advisor_degraded)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    n_total    = len(agent_checks)
    n_ok       = sum(1 for a in agent_checks if a["health"] == "OK")
    n_healed   = len(healed)
    n_backed   = len(backed_off)
    n_issues   = sum(1 for a in agent_checks if a["health"] not in ("OK", "RUNNING"))
    health_pct = round(n_ok / n_total * 100)

    report = {
        "checked_at":           report_time.isoformat(),
        "health_pct":           health_pct,
        "total_agents":         n_total,
        "ok":                   n_ok,
        "healed":               n_healed,
        "backed_off":           n_backed,
        "issues":               max(0, n_issues - n_healed - n_backed),
        "api_keys":             key_checks,
        "key_issues":           len(key_issues),
        "agents":               agent_checks,
        "healed_agents":        healed,
        "backed_off_agents":    backed_off,
        "verification":         verification,
        "advisor_dependencies": advisor_dep_checks,
        "advisor_ok":           advisor_ok,
        "advisor_degraded":     advisor_degraded,
        "unresolved":           [a["agent"] for a in agent_checks
                                 if a["health"] not in ("OK", "RUNNING")
                                 and a.get("healed") and "FAILED" in str(a["healed"])],
    }

    print(f"[Manager] Health: {health_pct}%  OK:{n_ok}  Healed:{n_healed}  "
          f"Backed-off:{n_backed}  Issues:{report['issues']}")
    if advisor_degraded:
        print(f"[Manager] Advisor inputs degraded: {[d['agent'] for d in advisor_degraded]}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    result = run_manager_agent()
    print(f"\nHealth: {result['health_pct']}%")
    print(f"Healed: {result['healed_agents']}")
    print(f"Backed-off: {result['backed_off_agents']}")
    if result["advisor_degraded"]:
        print(f"Advisor inputs degraded: {[d['agent'] for d in result['advisor_degraded']]}")
    for ki in result["api_keys"]:
        print(f"  {ki['key']}: {ki['status']} — {ki['message']}")
