"""
Chief of Staff Agent
====================
Autonomous platform overseer. Runs every 5 minutes.

Responsibilities:
  1. Staleness checks    — flag agents whose cache hasn't updated on schedule
  2. Sanity checks       — validate output values are within expected ranges
  3. Cross-agent consistency — detect contradictory signals across agents
  4. Auto-fix            — restart stale agents in background threads
  5. Task management     — persist an actionable task list to cos_tasks.json
  6. Health score        — 0-100 platform health metric written to chief_of_staff.json
"""

import json
import uuid
import threading
from datetime import datetime, timedelta
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"
TASKS_FILE = CACHE_DIR / "cos_tasks.json"

# ── Staleness thresholds (hours) — 2× the normal schedule ────────────────────
_STALE_THRESHOLDS = {
    "migration":       12,
    "pricing":          2,
    "predictions":     48,
    "debugger":         1,
    "news":             8,
    "rates":            2,
    "energy_data":     12,
    "sustainability_data": 12,
    "labor_market":    12,
    "gdp_data":        12,
    "inflation_data":  12,
    "credit_data":     12,
    "vacancy":         24,
    "land_market":     24,
    "cap_rate":        12,
    "rent_growth":     12,
    "opportunity_zone": 48,
    "distressed":      12,
    "market_score":    12,
    "climate_risk":    48,
}

# Maps cache key → agent key used in force_run()
_CACHE_TO_AGENT = {
    "migration":           "migration",
    "pricing":             "pricing",
    "predictions":         "predictions",
    "debugger":            "debugger",
    "news":                "news",
    "rates":               "rates",
    "energy_data":         "energy",
    "sustainability_data": "sustainability",
    "labor_market":        "labor_market",
    "gdp_data":            "gdp",
    "inflation_data":      "inflation",
    "credit_data":         "credit",
    "vacancy":             "vacancy",
    "land_market":         "land_market",
    "cap_rate":            "cap_rate",
    "rent_growth":         "rent_growth",
    "opportunity_zone":    "opportunity_zone",
    "distressed":          "distressed",
    "market_score":        "market_score",
    "climate_risk":        "climate_risk",
}

# ── Cache helpers ─────────────────────────────────────────────────────────────

def _read(key: str) -> dict:
    p = CACHE_DIR / f"{key}.json"
    if not p.exists():
        return {"updated_at": None, "data": None}
    try:
        with open(p) as f:
            d = json.load(f)
        # Normalise timestamp field
        ts = d.get("updated_at") or d.get("timestamp")
        d["updated_at"] = ts
        d["data"] = d.get("data")
        return d
    except Exception:
        return {"updated_at": None, "data": None}


def _age_hours(cache: dict) -> float | None:
    ts = cache.get("updated_at")
    if not ts:
        return None
    try:
        updated = datetime.fromisoformat(ts)
        return (datetime.now() - updated).total_seconds() / 3600
    except Exception:
        return None


# ── Task list helpers ─────────────────────────────────────────────────────────

def _load_tasks() -> list:
    if not TASKS_FILE.exists():
        return []
    try:
        with open(TASKS_FILE) as f:
            return json.load(f).get("tasks", [])
    except Exception:
        return []


def _save_tasks(tasks: list):
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TASKS_FILE, "w") as f:
        json.dump({"updated_at": datetime.now().isoformat(), "tasks": tasks}, f, indent=2)


def _dedup_task_title(tasks: list, title: str) -> bool:
    """Return True if an open task with this title already exists."""
    return any(t["title"] == title and t["status"] in ("open", "in_progress") for t in tasks)


def _add_task(tasks: list, title: str, description: str, priority: str,
              task_type: str, agent: str = None, auto_fixable: bool = False) -> dict | None:
    if _dedup_task_title(tasks, title):
        return None
    task = {
        "id":           str(uuid.uuid4())[:8],
        "created_at":   datetime.now().isoformat(),
        "priority":     priority,          # critical / high / medium / low
        "type":         task_type,         # auto_fix / manual_review / improvement
        "title":        title,
        "description":  description,
        "status":       "open",            # open / in_progress / resolved / dismissed
        "resolved_at":  None,
        "agent":        agent,
        "auto_fixable": auto_fixable,
    }
    tasks.append(task)
    return task


def resolve_task(task_id: str):
    """Mark a task resolved. Called externally (CLI or dashboard)."""
    tasks = _load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            t["status"] = "resolved"
            t["resolved_at"] = datetime.now().isoformat()
            break
    _save_tasks(tasks)


def dismiss_task(task_id: str):
    tasks = _load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            t["status"] = "dismissed"
            t["resolved_at"] = datetime.now().isoformat()
            break
    _save_tasks(tasks)


def add_manual_task(title: str, description: str, priority: str = "medium") -> dict | None:
    """Add a task manually from the dashboard or CLI."""
    tasks = _load_tasks()
    task = _add_task(tasks, title, description, priority,
                     task_type="manual_review", auto_fixable=False)
    _save_tasks(tasks)
    return task


# ── Step 1: Staleness checks ──────────────────────────────────────────────────

def _check_staleness(tasks: list) -> list[dict]:
    issues = []
    for cache_key, threshold_h in _STALE_THRESHOLDS.items():
        cache = _read(cache_key)
        age   = _age_hours(cache)
        agent_key = _CACHE_TO_AGENT.get(cache_key, cache_key)

        if age is None:
            issues.append({
                "severity": "critical",
                "agent":    agent_key,
                "cache":    cache_key,
                "message":  f"{cache_key} cache missing — agent has never run.",
                "auto_fix": "restart",
            })
            _add_task(tasks,
                      title=f"Cache missing: {cache_key}",
                      description=f"The {cache_key} cache does not exist. Agent '{agent_key}' may not have run yet.",
                      priority="critical", task_type="auto_fix",
                      agent=agent_key, auto_fixable=True)
        elif age > threshold_h:
            issues.append({
                "severity": "warning",
                "agent":    agent_key,
                "cache":    cache_key,
                "message":  f"{cache_key} is stale ({age:.1f}h old, threshold {threshold_h}h).",
                "auto_fix": "restart",
            })
            _add_task(tasks,
                      title=f"Stale cache: {cache_key}",
                      description=f"{cache_key} last updated {age:.1f}h ago (threshold: {threshold_h}h). Triggering restart.",
                      priority="high", task_type="auto_fix",
                      agent=agent_key, auto_fixable=True)

    return issues


# ── Step 2: Sanity checks ─────────────────────────────────────────────────────

def _check_sanity(tasks: list) -> list[dict]:
    issues = []

    # ── Cap rates ──────────────────────────────────────────────────────────
    cap_data = _read("cap_rate").get("data") or {}
    mkt_caps = cap_data.get("market_cap_rates", {})
    for market, types in mkt_caps.items():
        if not isinstance(types, dict):
            continue
        for pt, rate in types.items():
            if rate is not None and not (1.5 <= rate <= 25):
                issues.append({
                    "severity": "warning",
                    "agent":    "cap_rate",
                    "message":  f"Cap rate out of range: {market} {pt} = {rate}% (expected 1.5–25%)",
                })
                _add_task(tasks,
                          title=f"Cap rate anomaly: {market} {pt}",
                          description=f"Cap rate of {rate}% for {pt} in {market} is outside expected range (1.5–25%). Verify data source.",
                          priority="medium", task_type="manual_review",
                          agent="cap_rate", auto_fixable=False)

    # ── Rent growth ────────────────────────────────────────────────────────
    rg_data   = _read("rent_growth").get("data") or {}
    rg_market = rg_data.get("market_rent_growth", {})
    for market, vals in rg_market.items():
        if not isinstance(vals, dict):
            continue
        for metric, val in vals.items():
            if isinstance(val, (int, float)) and not (-40 <= val <= 60):
                issues.append({
                    "severity": "warning",
                    "agent":    "rent_growth",
                    "message":  f"Rent growth out of range: {market} {metric} = {val}% (expected -40 to +60%)",
                })
                _add_task(tasks,
                          title=f"Rent growth anomaly: {market}",
                          description=f"{metric} rent growth of {val}% in {market} is outside expected range.",
                          priority="medium", task_type="manual_review",
                          agent="rent_growth", auto_fixable=False)

    # ── Vacancy rates ──────────────────────────────────────────────────────
    vac_data = _read("vacancy").get("data") or {}
    vac_rows = vac_data.get("market_rows", [])
    for row in vac_rows:
        rate = row.get("vacancy_rate")
        if rate is not None and not (0 <= rate <= 65):
            issues.append({
                "severity": "warning",
                "agent":    "vacancy",
                "message":  f"Vacancy rate out of range: {row.get('market')} {row.get('property_type')} = {rate}%",
            })
            _add_task(tasks,
                      title=f"Vacancy anomaly: {row.get('market')} {row.get('property_type')}",
                      description=f"Vacancy rate of {rate}% is outside expected range (0–65%).",
                      priority="medium", task_type="manual_review",
                      agent="vacancy", auto_fixable=False)

    # ── Climate risk scores ────────────────────────────────────────────────
    cr_cache = _read("climate_risk")
    cr_data  = (cr_cache.get("data") or {})
    for metro in cr_data.get("metros", []):
        score = metro.get("composite_score")
        if score is not None and not (0 <= score <= 100):
            issues.append({
                "severity": "warning",
                "agent":    "climate_risk",
                "message":  f"Climate score out of range: {metro.get('metro')} = {score}",
            })

    # ── Market scores ──────────────────────────────────────────────────────
    ms_data = _read("market_score").get("data") or {}
    for row in ms_data.get("rankings", []):
        score = row.get("composite")
        if score is not None and not (0 <= score <= 100):
            issues.append({
                "severity": "warning",
                "agent":    "market_score",
                "message":  f"Market composite score out of range: {row.get('market')} = {score}",
            })
            _add_task(tasks,
                      title=f"Market score anomaly: {row.get('market')}",
                      description=f"Composite score of {score} is outside 0–100 range.",
                      priority="medium", task_type="manual_review",
                      agent="market_score", auto_fixable=False)

    # ── Interest rates ─────────────────────────────────────────────────────
    rates_data = _read("rates").get("data") or {}
    fed_funds  = rates_data.get("fed_funds_rate")
    t10y       = rates_data.get("treasury_10y")
    if fed_funds is not None and not (0 <= fed_funds <= 25):
        issues.append({
            "severity": "warning",
            "agent":    "rates",
            "message":  f"Fed funds rate out of range: {fed_funds}%",
        })
    if t10y is not None and not (0 <= t10y <= 20):
        issues.append({
            "severity": "warning",
            "agent":    "rates",
            "message":  f"10Y Treasury yield out of range: {t10y}%",
        })

    # ── Migration scores ───────────────────────────────────────────────────
    mig_data = _read("migration").get("data") or {}
    for row in mig_data.get("migration", []):
        score = row.get("migration_score")
        if score is not None and not (0 <= float(score) <= 100):
            issues.append({
                "severity": "warning",
                "agent":    "migration",
                "message":  f"Migration score out of range: {row.get('state_abbr')} = {score}",
            })

    return issues


# ── Step 3: Cross-agent consistency checks ────────────────────────────────────

def _check_consistency(tasks: list) -> list[dict]:
    issues = []

    # Build lookup tables
    cr_cache   = _read("climate_risk")
    cr_data    = cr_cache.get("data") or {}
    cr_metros  = {m["metro"]: m for m in cr_data.get("metros", [])}

    ms_data    = _read("market_score").get("data") or {}
    ms_by_mkt  = {r["market"]: r for r in ms_data.get("rankings", [])}

    rg_data    = _read("rent_growth").get("data") or {}
    rg_market  = rg_data.get("market_rent_growth", {})

    vac_data   = _read("vacancy").get("data") or {}
    vac_rows   = vac_data.get("market_rows", [])
    vac_by_key = {}
    for row in vac_rows:
        vac_by_key[(row["market"], row["property_type"])] = row.get("vacancy_rate")

    cap_data   = _read("cap_rate").get("data") or {}
    t10y       = _read("rates").get("data", {}).get("treasury_10y")

    credit_data  = _read("credit_data").get("data") or {}
    credit_label = (credit_data.get("signal") or {}).get("label", "NEUTRAL").upper()

    # ── 1. High climate risk + very high market score ──────────────────────
    for market, ms_row in ms_by_mkt.items():
        city = market.split(", ")[0]
        cr_metro = cr_metros.get(city) or next(
            (v for k, v in cr_metros.items() if city.lower() in k.lower()), None
        )
        if cr_metro and ms_row:
            cr_score = cr_metro.get("composite_score", 0)
            ms_score = ms_row.get("composite", 0)
            if cr_score >= 65 and ms_score >= 78:
                issues.append({
                    "severity": "warning",
                    "agent":    "cross_agent",
                    "message":  (
                        f"Consistency: {market} has high climate risk ({cr_score:.0f}/100) "
                        f"but very high market score ({ms_score:.0f}/100). "
                        "Climate factor may be underweighted in market score."
                    ),
                })
                _add_task(tasks,
                          title=f"Consistency: climate vs market score — {market}",
                          description=(
                              f"{market} scores {ms_score:.0f}/100 on market opportunity but "
                              f"{cr_score:.0f}/100 on climate risk. Review whether climate "
                              f"factor is adequately reflected in the market score agent."
                          ),
                          priority="medium", task_type="manual_review",
                          agent="market_score", auto_fixable=False)

    # ── 2. High vacancy + high positive rent growth (same market+type) ─────
    for (market, pt), vac_rate in vac_by_key.items():
        rg = rg_market.get(market, {})
        rent_key = {"Industrial": "industrial_psf", "Office": "office_psf",
                    "Retail": "retail_psf", "Multifamily": "multifamily"}.get(pt)
        if rent_key and rg and vac_rate is not None:
            growth = rg.get(rent_key)
            if growth is not None and vac_rate > 18 and growth > 8:
                issues.append({
                    "severity": "info",
                    "agent":    "cross_agent",
                    "message":  (
                        f"Consistency: {market} {pt} vacancy={vac_rate}% "
                        f"but rent growth={growth}% — high vacancy usually suppresses rents."
                    ),
                })

    # ── 3. Cap rate below 10Y Treasury (negative spread) ──────────────────
    if t10y and t10y > 0:
        mkt_caps = cap_data.get("market_cap_rates", {})
        below_treasury = []
        for market, types in mkt_caps.items():
            if not isinstance(types, dict):
                continue
            for pt, rate in types.items():
                if rate is not None and rate < t10y - 0.5:
                    below_treasury.append(f"{market} {pt}: {rate:.2f}% (vs T10Y {t10y:.2f}%)")
        if len(below_treasury) >= 3:
            issues.append({
                "severity": "warning",
                "agent":    "cross_agent",
                "message":  (
                    f"Consistency: {len(below_treasury)} market/type cap rates are below "
                    f"the 10Y Treasury ({t10y:.2f}%). Negative spreads may indicate "
                    "stale cap rate data or extraordinary market conditions."
                ),
            })
            _add_task(tasks,
                      title="Cap rates below 10Y Treasury in multiple markets",
                      description=(
                          f"{len(below_treasury)} market/property type combinations show "
                          f"cap rates below the 10Y Treasury yield ({t10y:.2f}%). "
                          "Review cap rate data freshness and source accuracy."
                      ),
                      priority="high", task_type="manual_review",
                      agent="cap_rate", auto_fixable=False)

    # ── 4. Tight credit but cap rates not adjusting upward ────────────────
    if credit_label == "TIGHT":
        national_caps = cap_data.get("national", {})
        low_caps = [
            f"{pt}: {v.get('rate', 0):.2f}%"
            for pt, v in national_caps.items()
            if isinstance(v, dict) and v.get("rate", 10) < 5.0
        ]
        if low_caps:
            issues.append({
                "severity": "info",
                "agent":    "cross_agent",
                "message":  (
                    f"Consistency: credit conditions are TIGHT but some national cap rates "
                    f"are below 5% ({', '.join(low_caps[:3])}). "
                    "Tight credit typically pressures cap rates upward."
                ),
            })

    return issues


# ── Step 4: Auto-fix — restart stale agents ───────────────────────────────────

def _auto_restart(stale_issues: list[dict], restart_fn, agent_status: dict) -> list[str]:
    """
    For each stale/missing cache, restart the agent if it's not currently running
    and hasn't been restarted in the last 10 minutes.
    Returns list of restart action descriptions.
    """
    actions = []
    already_restarted = set()

    for issue in stale_issues:
        if issue.get("auto_fix") != "restart":
            continue
        agent_key = issue.get("agent")
        if not agent_key or agent_key in already_restarted:
            continue

        # Don't restart if already running
        st = agent_status.get(agent_key, {})
        if st.get("status") == "running":
            continue

        # Don't restart if run very recently (within 10 min)
        last_run = st.get("last_run")
        if last_run:
            try:
                lr = datetime.fromisoformat(last_run)
                if (datetime.now() - lr).total_seconds() < 600:
                    continue
            except Exception:
                pass

        try:
            restart_fn(agent_key)
            already_restarted.add(agent_key)
            actions.append(f"Restarted agent '{agent_key}' (cache was stale/missing).")
        except Exception as e:
            actions.append(f"Failed to restart '{agent_key}': {e}")

    return actions


# ── Step 5: Health score ──────────────────────────────────────────────────────

def _compute_health_score(stale_issues: list, sanity_issues: list,
                           consistency_issues: list) -> int:
    score = 100
    for issue in stale_issues:
        if issue["severity"] == "critical": score -= 8
        elif issue["severity"] == "warning": score -= 4
    for issue in sanity_issues:
        if issue["severity"] == "warning": score -= 3
        elif issue["severity"] == "info": score -= 1
    for issue in consistency_issues:
        if issue["severity"] == "warning": score -= 2
        elif issue["severity"] == "info": score -= 1
    return max(0, min(100, score))


# ── Main entry point ──────────────────────────────────────────────────────────

def run_chief_of_staff(restart_fn=None, agent_status: dict = None) -> dict:
    """
    Run a full oversight sweep. Called by cre_agents every 5 minutes.

    Args:
        restart_fn:    Callable(agent_key) — triggers an agent restart (force_run).
        agent_status:  Current agent status dict from get_status().
    """
    now       = datetime.now().isoformat()
    tasks     = _load_tasks()
    actions   = []

    # Prune resolved/dismissed tasks older than 7 days
    cutoff = datetime.now() - timedelta(days=7)
    tasks = [
        t for t in tasks
        if t["status"] in ("open", "in_progress")
        or (t.get("resolved_at") and datetime.fromisoformat(t["resolved_at"]) > cutoff)
    ]

    # Run all checks
    stale_issues       = _check_staleness(tasks)
    sanity_issues      = _check_sanity(tasks)
    consistency_issues = _check_consistency(tasks)

    all_issues = stale_issues + sanity_issues + consistency_issues

    # Auto-restart stale agents
    if restart_fn and agent_status:
        actions = _auto_restart(stale_issues, restart_fn, agent_status)
        # Mark auto-fix tasks as in_progress for restarted agents
        restarted_agents = {
            a.split("'")[1] for a in actions if a.startswith("Restarted")
        }
        for t in tasks:
            if t.get("agent") in restarted_agents and t["status"] == "open" and t.get("auto_fixable"):
                t["status"] = "in_progress"

    # Compute health score
    health_score = _compute_health_score(stale_issues, sanity_issues, consistency_issues)

    # Critical alerts = issues that need immediate attention
    critical_alerts = [i for i in all_issues if i["severity"] == "critical"]
    warnings        = [i for i in all_issues if i["severity"] == "warning"]

    # Save updated task list
    _save_tasks(tasks)

    result = {
        "timestamp":           now,
        "health_score":        health_score,
        "total_issues":        len(all_issues),
        "critical_count":      len(critical_alerts),
        "warning_count":       len(warnings),
        "info_count":          len([i for i in all_issues if i["severity"] == "info"]),
        "stale_issues":        stale_issues,
        "sanity_issues":       sanity_issues,
        "consistency_issues":  consistency_issues,
        "actions_taken":       actions,
        "open_tasks":          len([t for t in tasks if t["status"] == "open"]),
        "all_issues":          all_issues,
    }

    return result
