"""
cos platform — Chief of Staff platform oversight commands
==========================================================
Surfaces agent health, tasks, and sweep results from the terminal.

Commands (dispatched from cos.py):
  status               Print platform health, issue summary, agent statuses
  tasks [--filter]     List tasks (open / all)
  resolve <id>         Mark a task resolved
  dismiss <id>         Mark a task dismissed
  add "<title>"        Add a manual task
  sweep                Run a full CoS sweep now and print results
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Resolve repo root relative to this file
_ROOT = Path(__file__).resolve().parent.parent.parent

# ── Colour helpers (ANSI) ──────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_GOLD   = "\033[33m"
_DIM    = "\033[2m"

def _c(text, *codes): return "".join(codes) + str(text) + _RESET
def _ok(t):    return _c(t, _GREEN)
def _warn(t):  return _c(t, _YELLOW)
def _err(t):   return _c(t, _RED)
def _head(t):  return _c(t, _BOLD, _GOLD)
def _dim(t):   return _c(t, _DIM)


# ── Cache readers ─────────────────────────────────────────────────────────────

def _read_cache(key: str) -> dict:
    p = _ROOT / "cache" / f"{key}.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            d = json.load(f)
        return d
    except Exception:
        return {}


def _cos_data() -> dict:
    c = _read_cache("chief_of_staff")
    return c.get("data") or {}


def _load_tasks() -> list:
    p = _ROOT / "cache" / "cos_tasks.json"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            return json.load(f).get("tasks", [])
    except Exception:
        return []


def _save_tasks(tasks: list):
    p = _ROOT / "cache" / "cos_tasks.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"updated_at": datetime.now().isoformat(), "tasks": tasks}, f, indent=2)


# ── Subcommands ───────────────────────────────────────────────────────────────

def status():
    """Print platform health score, issue summary, and agent statuses."""
    data = _cos_data()

    if not data:
        print(_warn("Chief of Staff has not run yet. Start the dashboard or run `cos platform sweep`."))
        return

    ts = _read_cache("chief_of_staff").get("updated_at", "")
    ts_str = ts[:16].replace("T", " ") + " UTC" if ts else "unknown"

    health   = data.get("health_score", 0)
    critical = data.get("critical_count", 0)
    warnings = data.get("warning_count", 0)
    info_c   = data.get("info_count", 0)
    tasks_n  = data.get("open_tasks", 0)
    actions  = data.get("actions_taken", [])

    health_str = (
        _ok(f"{health}/100") if health >= 80
        else _warn(f"{health}/100") if health >= 55
        else _err(f"{health}/100")
    )

    print()
    print(_head("═" * 58))
    print(_head(f"  CRE Platform — Chief of Staff Status"))
    print(_head("═" * 58))
    print(f"  {'Last sweep:':<18} {_dim(ts_str)}")
    print(f"  {'Health score:':<18} {health_str}")
    print(f"  {'Critical issues:':<18} {_err(critical) if critical else _ok(critical)}")
    print(f"  {'Warnings:':<18} {_warn(warnings) if warnings else _ok(warnings)}")
    print(f"  {'Info:':<18} {_dim(info_c)}")
    print(f"  {'Open tasks:':<18} {_c(tasks_n, _CYAN)}")
    print(f"  {'Actions taken:':<18} {_ok(len(actions)) if actions else _dim(0)}")
    print()

    if actions:
        print(_head("  Actions taken this sweep:"))
        for a in actions:
            print(f"    {_ok('✓')} {a}")
        print()

    all_issues = data.get("all_issues", [])
    if all_issues:
        print(_head("  Issues detected:"))
        _sev_fn = {"critical": _err, "warning": _warn, "info": _dim}
        for iss in all_issues:
            sev    = iss.get("severity", "info")
            agent  = iss.get("agent", "")
            msg    = iss.get("message", "")
            fn     = _sev_fn.get(sev, _dim)
            badge  = fn(f"[{sev.upper():8}]")
            print(f"    {badge}  {_dim(f'[{agent}]')}")
            print(f"             {msg}")
        print()
    else:
        print(f"  {_ok('✓ No issues detected — platform is healthy.')}")
        print()


def tasks(filter_by: str = "open"):
    """List tasks from the CoS task list."""
    all_tasks = _load_tasks()

    if filter_by == "open":
        rows = [t for t in all_tasks if t["status"] in ("open", "in_progress")]
    else:
        rows = all_tasks

    if not rows:
        print(_ok("  No tasks found.") if filter_by == "open" else _dim("  No tasks in list."))
        return

    _prio_fn = {"critical": _err, "high": _warn, "medium": _c, "low": _dim}

    print()
    print(_head("═" * 72))
    print(_head(f"  CoS Tasks — {filter_by.title()} ({len(rows)})"))
    print(_head("═" * 72))
    print(f"  {'ID':<8} {'PRIO':<9} {'TYPE':<14} {'STATUS':<12} TITLE")
    print(_dim("  " + "─" * 68))

    for t in rows:
        tid    = t.get("id", "?")
        prio   = t.get("priority", "medium")
        ttype  = t.get("type", "")[:13]
        status = t.get("status", "open")
        title  = t.get("title", "")[:48]
        agent  = t.get("agent") or ""

        prio_fn = _prio_fn.get(prio, _dim)
        stat_fn = _ok if status == "resolved" else (_warn if status == "in_progress" else (_dim if status == "dismissed" else str))

        print(f"  {_c('#' + tid, _DIM):<8} {prio_fn(prio):<9} {_dim(ttype):<14} {stat_fn(status):<12} {title}")
        desc = t.get("description", "")
        if desc:
            print(f"  {'':<8} {_dim(desc[:70])}{'...' if len(desc) > 70 else ''}")
        if agent:
            print(f"  {'':<8} {_dim(f'Agent: {agent}')}")
        print()


def resolve(task_id: str):
    """Mark a task resolved."""
    ts = _load_tasks()
    matched = False
    for t in ts:
        if t["id"] == task_id:
            t["status"] = "resolved"
            t["resolved_at"] = datetime.now().isoformat()
            matched = True
            break
    if matched:
        _save_tasks(ts)
        print(_ok(f"  ✓ Task #{task_id} marked as resolved."))
    else:
        print(_err(f"  Task #{task_id} not found."))


def dismiss(task_id: str):
    """Mark a task dismissed."""
    ts = _load_tasks()
    matched = False
    for t in ts:
        if t["id"] == task_id:
            t["status"] = "dismissed"
            t["resolved_at"] = datetime.now().isoformat()
            matched = True
            break
    if matched:
        _save_tasks(ts)
        print(_warn(f"  Task #{task_id} dismissed."))
    else:
        print(_err(f"  Task #{task_id} not found."))


def add(title: str, description: str = "", priority: str = "medium"):
    """Add a manual task to the CoS task list."""
    import uuid
    ts = _load_tasks()

    # Dedup check
    for t in ts:
        if t["title"] == title and t["status"] in ("open", "in_progress"):
            print(_warn(f"  Task already exists: '{title}' (#{t['id']})"))
            return

    task = {
        "id":           str(uuid.uuid4())[:8],
        "created_at":   datetime.now().isoformat(),
        "priority":     priority,
        "type":         "manual",
        "title":        title,
        "description":  description,
        "status":       "open",
        "resolved_at":  None,
        "agent":        None,
        "auto_fixable": False,
    }
    ts.append(task)
    _save_tasks(ts)
    print(_ok(f"  ✓ Task added: '{title}' (#{task['id']}, priority: {priority})"))


def sweep():
    """Run a full Chief of Staff sweep now and print results."""
    print(_dim("  Running Chief of Staff sweep..."))
    sys.path.insert(0, str(_ROOT))

    try:
        # Load env vars for GROQ etc.
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env", override=True)
    except Exception:
        pass

    try:
        from src.chief_of_staff_agent import run_chief_of_staff
        result = run_chief_of_staff(restart_fn=None, agent_status={})
    except Exception as e:
        print(_err(f"  Sweep failed: {e}"))
        return

    # Write to cache
    try:
        with open(_ROOT / "cache" / "chief_of_staff.json", "w") as f:
            json.dump({"updated_at": datetime.now().isoformat(), "data": result}, f, indent=2)
    except Exception:
        pass

    health   = result.get("health_score", 0)
    critical = result.get("critical_count", 0)
    warnings = result.get("warning_count", 0)
    info_c   = result.get("info_count", 0)
    actions  = result.get("actions_taken", [])
    all_iss  = result.get("all_issues", [])

    health_str = (
        _ok(f"{health}/100") if health >= 80
        else _warn(f"{health}/100") if health >= 55
        else _err(f"{health}/100")
    )

    print()
    print(_head("═" * 58))
    print(_head("  Sweep Complete"))
    print(_head("═" * 58))
    print(f"  {'Health score:':<18} {health_str}")
    print(f"  {'Critical:':<18} {_err(critical) if critical else _ok(critical)}")
    print(f"  {'Warnings:':<18} {_warn(warnings) if warnings else _ok(warnings)}")
    print(f"  {'Info:':<18} {_dim(info_c)}")
    print(f"  {'Open tasks:':<18} {_c(result.get('open_tasks', 0), _CYAN)}")
    print()

    if actions:
        print(_head("  Actions taken:"))
        for a in actions:
            print(f"    {_ok('✓')} {a}")
        print()

    if all_iss:
        print(_head("  Issues found:"))
        _sev_fn = {"critical": _err, "warning": _warn, "info": _dim}
        for iss in all_iss:
            sev   = iss.get("severity", "info")
            agent = iss.get("agent", "")
            msg   = iss.get("message", "")
            fn    = _sev_fn.get(sev, _dim)
            print(f"    {fn(f'[{sev.upper()}]')} {_dim(f'[{agent}]')} {msg}")
        print()
    else:
        print(f"  {_ok('✓ No issues — platform is healthy.')}")
        print()
