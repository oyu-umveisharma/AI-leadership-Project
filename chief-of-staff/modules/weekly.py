"""
Weekly Digest — cos weekly
Summarizes the past 7 days: what shipped, in progress, blocked, next week.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import llm
from . import followups as fu

REPO_DIR   = Path(__file__).parent.parent.parent
CACHE_DIR  = REPO_DIR / "cache"
STATE_DIR  = Path(__file__).parent.parent / "state"


def _git(*args):
    r = subprocess.run(["git"] + list(args), cwd=REPO_DIR, capture_output=True, text=True)
    return r.stdout.strip()


def _commits_this_week() -> list[str]:
    return [l.strip() for l in _git("log", "--since=7 days ago", "--oneline", "--no-merges").splitlines() if l.strip()]


def _files_changed_this_week() -> list[str]:
    raw = _git("diff", "--name-only", "HEAD~7", "HEAD")
    return [f.strip() for f in raw.splitlines() if f.strip()]


def _commits_by_day() -> dict[str, int]:
    raw = _git("log", "--since=7 days ago", "--format=%ad", "--date=format:%a %b %-d")
    counts: dict[str, int] = {}
    for line in raw.splitlines():
        day = line.strip()
        if day:
            counts[day] = counts.get(day, 0) + 1
    return counts


def _open_tasks() -> tuple[list[str], list[str]]:
    tasks_file = STATE_DIR / "tasks.md"
    if not tasks_file.exists():
        return [], []
    import re
    open_tasks, blocked = [], []
    for line in tasks_file.read_text().splitlines():
        m = re.match(r"\s*-\s*\[\s*\]\s*(.+)", line)
        if m:
            body = m.group(1).strip()
            if any(kw in body.upper() for kw in ["BLOCKER", "BLOCKED", "DEPENDS"]):
                blocked.append(body)
            else:
                open_tasks.append(body)
    return open_tasks, blocked


def _overdue_followups() -> list[dict]:
    fu._ensure_file()
    text = fu.FOLLOWUPS_FILE.read_text()
    rows = fu._parse_rows(text)
    return [r for r in rows if fu._is_overdue(r["due"], r["status"])]


def _agent_health() -> list[str]:
    now   = datetime.now(timezone.utc).timestamp()
    items = []
    if not CACHE_DIR.exists():
        return ["  Cache directory not found."]
    for f in sorted(CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            ts   = data.get("timestamp") or data.get("last_updated") or data.get("cached_at")
            if ts:
                age_h = (now - datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()) / 3600
                flag  = " ⚠️  STALE" if age_h > 8 else " ✅"
                items.append(f"  `{f.stem}` — last ran {age_h:.1f}h ago{flag}")
        except Exception:
            items.append(f"  `{f.stem}` — unreadable")
    return items or ["  No cache files found."]


def run():
    commits   = _commits_this_week()
    by_day    = _commits_by_day()
    changed   = _files_changed_this_week()
    open_t, blocked = _open_tasks()
    overdue   = _overdue_followups()
    health    = _agent_health()

    week_end  = datetime.now().strftime("%B %-d, %Y")
    week_start = _git("log", "--since=7 days ago", "--format=%ad", "--date=format:%B %-d", "--reverse").splitlines()
    week_start = week_start[0].strip() if week_start else "7 days ago"

    lines = [
        "# 📰 Weekly Digest",
        f"_Week of {week_start} → {week_end}_",
        "",
        "---",
        "",
        f"## ✅ What Shipped — {len(commits)} commit(s)",
    ]

    if commits:
        lines += [f"  - {c}" for c in commits[:20]]
        if len(commits) > 20:
            lines.append(f"  _…and {len(commits)-20} more commits._")
    else:
        lines.append("  _No commits this week._")

    if by_day:
        lines += ["", "**Activity by day:**"]
        for day, count in sorted(by_day.items()):
            bar = "█" * min(count, 20)
            lines.append(f"  {day:<12} {bar} {count}")

    lines += ["", f"## 📁 Files Changed ({len(changed)} file(s))"]
    if changed:
        lines += [f"  - `{f}`" for f in changed[:15]]
        if len(changed) > 15:
            lines.append(f"  _…and {len(changed)-15} more._")
    else:
        lines.append("  _No file changes detected._")

    lines += ["", f"## 🚧 In Progress — {len(open_t)} open task(s)"]
    if open_t:
        lines += [f"  - {t[:100]}" for t in open_t[:10]]
        if len(open_t) > 10:
            lines.append(f"  _…and {len(open_t)-10} more._")
    else:
        lines.append("  _No open tasks. Add some to `tasks.md`._")

    lines += ["", f"## 🔴 Blocked — {len(blocked)} blocker(s)"]
    if blocked:
        lines += [f"  - {t[:100]}" for t in blocked]
    else:
        lines.append("  _No blockers. 🟢_")

    lines += ["", f"## ⏰ Overdue Follow-ups — {len(overdue)} item(s)"]
    if overdue:
        for r in overdue:
            lines.append(f"  - **#{r['id']}** {r['item']} | Owner: {r['owner']} | Due: {r['due']}")
    else:
        lines.append("  _No overdue follow-ups. 🟢_")

    lines += ["", "## 🤖 Agent Health"]
    lines += health

    # Next week focus — LLM or heuristic
    lines += ["", "## 🔭 Focus for Next Week"]
    if llm.available() and (commits or open_t):
        shipped_text = "\n".join(commits[:10])
        open_text    = "\n".join(open_t[:10])
        blocked_text = "\n".join(blocked[:5]) or "None"
        digest_prompt = (
            f"Generate a concise executive weekly digest for a CRE intelligence platform.\n\n"
            f"What shipped (git commits):\n{shipped_text}\n\n"
            f"Open tasks:\n{open_text}\n\n"
            f"Blocked items:\n{blocked_text}\n\n"
            f"Write: (1) a 2-sentence summary of the week, "
            f"(2) top 3 priorities for next week, "
            f"(3) one key risk or concern to watch."
        )
        digest = llm.ask(digest_prompt, system="You are an executive chief of staff writing a weekly digest.", max_tokens=500)
        if digest:
            lines += [digest]
    else:
        if open_t:
            lines += [f"  - {t[:90]}" for t in open_t[:5]]
        else:
            lines.append("  _Update tasks.md to populate this section._")

    lines += ["", "---", f"_Chief of Staff Weekly Digest — {week_end}_"]

    report = "\n".join(lines)
    print(report)

    out = STATE_DIR / f"weekly_{datetime.now().strftime('%Y-%m-%d')}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(f"\n[cos] Saved to {out.relative_to(REPO_DIR)}", flush=True)
