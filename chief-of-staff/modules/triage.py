"""
Task Triage & Prioritization — cos triage
Reads tasks.md, ranks items by urgency/impact, flags blockers and dependencies.
"""

import re
from datetime import datetime
from pathlib import Path

from . import llm

STATE_DIR = Path(__file__).parent.parent / "state"
TASKS_FILE = STATE_DIR / "tasks.md"

TASKS_TEMPLATE = """\
# Tasks

> Add tasks below as `- [ ] description`. Use tags to help triage:
> `[BLOCKER]` `[BUG]` `[CRITICAL]` `[DEPENDS: task]` `[OWNER: name]` `[DUE: YYYY-MM-DD]`

## High Priority
- [ ] [EXAMPLE] Replace synthetic listings with real LoopNet/Crexi API data
- [ ] [BUG] Census API fallback sometimes returns stale 2022 data

## Medium Priority
- [ ] Add unit tests for cre_pricing.py profit matrix calculation
- [ ] Improve RSS deduplication logic in cre_news.py

## Low Priority
- [ ] Add dark mode toggle to Streamlit UI
- [ ] Export agent cache to CSV for offline analysis
"""

_URGENCY_KEYWORDS = {
    "critical": 30, "blocker": 25, "blocking": 25, "bug": 20,
    "broken": 20, "urgent": 20, "security": 20, "crash": 18,
    "fail": 15, "error": 15, "asap": 15, "hotfix": 15,
    "due today": 25, "overdue": 28,
}

_PRIORITY_SCORE = {"high priority": 15, "medium priority": 8, "low priority": 2}


def _parse_tasks(text: str) -> list[dict]:
    tasks = []
    current_section = "Uncategorized"
    idx = 0
    for line in text.splitlines():
        # Section headers
        if line.startswith("## "):
            current_section = line[3:].strip()
            continue
        # Open task
        m = re.match(r"\s*-\s*\[\s*\]\s*(.+)", line)
        if m:
            body = m.group(1).strip()
            idx += 1
            task = {
                "id":      idx,
                "body":    body,
                "section": current_section,
                "score":   0,
                "flags":   [],
                "owner":   None,
                "due":     None,
            }
            # Extract tags
            owner_m = re.search(r"\[OWNER:\s*([^\]]+)\]", body, re.I)
            due_m   = re.search(r"\[DUE:\s*([^\]]+)\]",   body, re.I)
            if owner_m:
                task["owner"] = owner_m.group(1).strip()
            if due_m:
                task["due"] = due_m.group(1).strip()
                try:
                    days_left = (datetime.strptime(task["due"], "%Y-%m-%d") - datetime.now()).days
                    if days_left < 0:
                        task["flags"].append("OVERDUE")
                        task["score"] += 28
                    elif days_left <= 2:
                        task["flags"].append("DUE SOON")
                        task["score"] += 18
                    elif days_left <= 7:
                        task["score"] += 8
                except ValueError:
                    pass
            # Section base score
            for sec_key, pts in _PRIORITY_SCORE.items():
                if sec_key in current_section.lower():
                    task["score"] += pts
                    break
            # Keyword scoring
            body_lower = body.lower()
            for kw, pts in _URGENCY_KEYWORDS.items():
                if kw in body_lower:
                    task["score"] += pts
                    task["flags"].append(kw.upper())
            # Blocker / dependency detection
            if re.search(r"\[BLOCKER\]|\[DEPENDS:", body, re.I):
                task["flags"].append("BLOCKER")
                task["score"] += 25
            tasks.append(task)

    # Done tasks (for context, not shown in triage)
    return tasks


def _fmt_task(t: dict, rank: int) -> str:
    flags = f" `{'` `'.join(set(t['flags']))}`" if t['flags'] else ""
    owner = f" — Owner: **{t['owner']}**" if t['owner'] else ""
    due   = f" — Due: `{t['due']}`" if t['due'] else ""
    body  = re.sub(r"\[(OWNER|DUE|BLOCKER|DEPENDS|BUG|CRITICAL|EXAMPLE):[^\]]*\]", "", t["body"], flags=re.I).strip()
    body  = re.sub(r"\[(BLOCKER|BUG|CRITICAL|URGENT)\]", "", body, flags=re.I).strip()
    return f"  **{rank}.** {body}{flags}{owner}{due}"


def run():
    STATE_DIR.mkdir(exist_ok=True)

    if not TASKS_FILE.exists():
        TASKS_FILE.write_text(TASKS_TEMPLATE)
        print(f"[cos] Created tasks file: {TASKS_FILE.relative_to(TASKS_FILE.parent.parent.parent)}")
        print("      Edit it to add your tasks, then run `cos triage` again.\n")

    text  = TASKS_FILE.read_text()
    tasks = _parse_tasks(text)

    if not tasks:
        print("No open tasks found in tasks.md. Add some tasks and try again.")
        return

    ranked   = sorted(tasks, key=lambda t: t["score"], reverse=True)
    blockers = [t for t in ranked if "BLOCKER" in t["flags"]]
    overdue  = [t for t in ranked if "OVERDUE" in t["flags"]]

    now = datetime.now().strftime("%A, %B %-d %Y")
    lines = [
        "# 🗂️  Task Triage Report",
        f"_{now} — {len(tasks)} open task(s)_",
        "",
    ]

    if blockers:
        lines += ["## 🚧 Blockers — Resolve First"]
        for i, t in enumerate(blockers, 1):
            lines.append(_fmt_task(t, i))
        lines.append("")

    if overdue:
        lines += ["## ⏰ Overdue"]
        for i, t in enumerate(overdue, 1):
            lines.append(_fmt_task(t, i))
        lines.append("")

    lines += [f"## 📊 Prioritized Task List (score-ranked)"]
    for i, t in enumerate(ranked, 1):
        lines.append(_fmt_task(t, i))
        lines.append(f"     _Section: {t['section']} | Priority score: {t['score']}_")

    # Dependency detection
    dep_tasks = [t for t in ranked if re.search(r"\[DEPENDS:", t["body"], re.I)]
    if dep_tasks:
        lines += ["", "## 🔗 Tasks with Dependencies"]
        for t in dep_tasks:
            dep_m = re.search(r"\[DEPENDS:\s*([^\]]+)\]", t["body"], re.I)
            dep   = dep_m.group(1) if dep_m else "unknown"
            lines.append(f"  - **{t['body'][:60]}…** depends on: _{dep}_")

    # Suggested next action
    top = ranked[0]
    top_clean = re.sub(r"\[[^\]]*\]", "", top["body"]).strip()
    lines += ["", "## ✅ Suggested Next Action"]
    lines.append(f"  **Start with:** {top_clean}")
    if len(ranked) > 1:
        lines.append(f"  **Then:** {re.sub(r'[^]]*]','',ranked[1]['body']).strip()[:80]}")

    # Optional LLM enhancement
    if llm.available():
        task_list = "\n".join(f"- {t['body']}" for t in ranked[:15])
        advice = llm.ask(
            f"Given these project tasks, identify the top 3 most strategic to work on this week "
            f"and briefly explain why (2 sentences each):\n\n{task_list}",
            system="You are a sharp product manager and executive advisor.",
            max_tokens=400,
        )
        if advice:
            lines += ["", "## 🤖 AI Strategic Recommendation"]
            lines.append(advice)

    lines += ["", "---", f"_Tasks file: `{TASKS_FILE.relative_to(TASKS_FILE.parent.parent.parent)}`_"]

    report = "\n".join(lines)
    print(report)

    out = STATE_DIR / "last_triage.md"
    out.write_text(report)
