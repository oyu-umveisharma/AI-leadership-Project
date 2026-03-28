"""
Decision Log — cos decide / cos decisions
Maintains decisions.md with structured ADR-style entries.
"""

import re
from datetime import datetime
from pathlib import Path

STATE_DIR      = Path(__file__).parent.parent / "state"
DECISIONS_FILE = STATE_DIR / "decisions.md"

_HEADER = """\
# Decision Log

> Architectural and product decisions — recorded for institutional memory.
> Format: `cos decide "Title" --context "..." --options "..." --decision "..." --rationale "..."`

---

"""


def _ensure_file():
    STATE_DIR.mkdir(exist_ok=True)
    if not DECISIONS_FILE.exists():
        DECISIONS_FILE.write_text(_HEADER)


def add(title: str, context: str, options: str, decision: str, rationale: str):
    _ensure_file()
    date = datetime.now().strftime("%Y-%m-%d")
    time = datetime.now().strftime("%H:%M")

    entry_lines = [
        f"## [{date}] {title}",
        "",
        f"**Date:** {date} at {time}",
        "",
        "**Context:**",
        context.strip(),
        "",
    ]

    if options:
        entry_lines += ["**Options Considered:**"]
        for i, opt in enumerate(re.split(r"\s*[|;,]\s*", options), 1):
            opt = opt.strip()
            if opt:
                entry_lines.append(f"{i}. {opt}")
        entry_lines.append("")

    entry_lines += [
        "**Decision:**",
        decision.strip(),
        "",
        "**Rationale:**",
        rationale.strip(),
        "",
        "---",
        "",
    ]

    entry = "\n".join(entry_lines)

    # Append after header
    existing = DECISIONS_FILE.read_text()
    DECISIONS_FILE.write_text(existing + entry)

    print(f"✅ Decision logged: [{date}] {title}")
    print(f"   File: {DECISIONS_FILE.relative_to(DECISIONS_FILE.parent.parent.parent)}")


def _parse_decisions(text: str) -> list[dict]:
    """Parse decisions.md into a list of dicts."""
    entries = []
    current = None
    for line in text.splitlines():
        m = re.match(r"^## \[(\d{4}-\d{2}-\d{2})\]\s+(.+)", line)
        if m:
            if current:
                entries.append(current)
            current = {"date": m.group(1), "title": m.group(2), "lines": []}
        elif current is not None:
            current["lines"].append(line)
    if current:
        entries.append(current)
    return entries


def list_decisions(n: int = 5):
    _ensure_file()
    text    = DECISIONS_FILE.read_text()
    entries = _parse_decisions(text)

    if not entries:
        print("No decisions logged yet. Use `cos decide` to record one.")
        return

    recent  = entries[-n:][::-1]  # most recent first
    print(f"# 📜 Decision Log — Last {len(recent)} Entr{'y' if len(recent)==1 else 'ies'}\n")
    for e in recent:
        print(f"## [{e['date']}] {e['title']}")
        # Print first few meaningful lines
        shown = 0
        for line in e["lines"]:
            if line.strip() in ("---", ""):
                continue
            print(f"  {line}")
            shown += 1
            if shown >= 8:
                print("  _…_")
                break
        print()

    print(f"_Total decisions: {len(entries)} | File: {DECISIONS_FILE.relative_to(DECISIONS_FILE.parent.parent.parent)}_")
