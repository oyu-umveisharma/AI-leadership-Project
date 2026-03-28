"""
Follow-up Tracker — cos follow-up add / list / done
Tracks action items with owners and due dates in follow-ups.md.
"""

import re
from datetime import datetime
from pathlib import Path

STATE_DIR     = Path(__file__).parent.parent / "state"
FOLLOWUPS_FILE = STATE_DIR / "follow-ups.md"

_HEADER = """\
# Follow-ups

> Action items with owners and due dates.
> Commands:
>   cos follow-up add "item" --owner "name" --due "YYYY-MM-DD"
>   cos follow-up list [--filter open|overdue|done|all]
>   cos follow-up done <id>

| ID | Item | Owner | Due | Status | Added |
|----|------|-------|-----|--------|-------|
"""


def _ensure_file():
    STATE_DIR.mkdir(exist_ok=True)
    if not FOLLOWUPS_FILE.exists():
        FOLLOWUPS_FILE.write_text(_HEADER)


def _parse_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        # Match table data rows (not header or separator)
        if not line.startswith("|") or "---" in line or "ID" in line[:10]:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 6:
            continue
        try:
            rows.append({
                "id":     int(parts[0]),
                "item":   parts[1],
                "owner":  parts[2],
                "due":    parts[3],
                "status": parts[4],
                "added":  parts[5],
            })
        except (ValueError, IndexError):
            pass
    return rows


def _next_id(rows: list[dict]) -> int:
    return max((r["id"] for r in rows), default=0) + 1


def _rewrite(header_text: str, rows: list[dict]):
    """Rewrite the file preserving the header and writing all rows."""
    table_rows = []
    for r in rows:
        table_rows.append(f"| {r['id']} | {r['item']} | {r['owner']} | {r['due']} | {r['status']} | {r['added']} |")
    FOLLOWUPS_FILE.write_text(header_text + "\n".join(table_rows) + "\n")


def _header_text() -> str:
    text = FOLLOWUPS_FILE.read_text()
    # Everything up to and including the header separator line
    lines = text.splitlines()
    header_end = 0
    for i, line in enumerate(lines):
        if "|----" in line:
            header_end = i + 1
            break
    return "\n".join(lines[:header_end]) + "\n"


def _is_overdue(due_str: str, status: str) -> bool:
    if status.lower() == "done":
        return False
    try:
        return datetime.strptime(due_str, "%Y-%m-%d") < datetime.now()
    except ValueError:
        return False


def add(item: str, owner: str, due: str):
    _ensure_file()

    # Validate date format
    if due:
        try:
            datetime.strptime(due, "%Y-%m-%d")
        except ValueError:
            print(f"[cos] Error: --due must be in YYYY-MM-DD format (got: '{due}')")
            return

    text   = FOLLOWUPS_FILE.read_text()
    rows   = _parse_rows(text)
    new_id = _next_id(rows)
    today  = datetime.now().strftime("%Y-%m-%d")

    rows.append({
        "id":     new_id,
        "item":   item,
        "owner":  owner or "—",
        "due":    due or "—",
        "status": "Open",
        "added":  today,
    })

    _rewrite(_header_text(), rows)
    print(f"✅ Follow-up #{new_id} added: {item}")
    if owner:
        print(f"   Owner: {owner}")
    if due:
        print(f"   Due:   {due}")


def complete(fid: int):
    _ensure_file()
    text = FOLLOWUPS_FILE.read_text()
    rows = _parse_rows(text)
    matched = [r for r in rows if r["id"] == fid]
    if not matched:
        print(f"[cos] No follow-up with ID {fid}.")
        return
    for r in rows:
        if r["id"] == fid:
            r["status"] = f"Done {datetime.now().strftime('%Y-%m-%d')}"
    _rewrite(_header_text(), rows)
    print(f"✅ Follow-up #{fid} marked done: {matched[0]['item']}")


def list_followups(filter_by: str = "open"):
    _ensure_file()
    text  = FOLLOWUPS_FILE.read_text()
    rows  = _parse_rows(text)

    if not rows:
        print("No follow-ups yet. Use `cos follow-up add` to create one.")
        return

    if filter_by == "open":
        display = [r for r in rows if "done" not in r["status"].lower()]
    elif filter_by == "overdue":
        display = [r for r in rows if _is_overdue(r["due"], r["status"])]
    elif filter_by == "done":
        display = [r for r in rows if "done" in r["status"].lower()]
    else:
        display = rows

    overdue_ids = {r["id"] for r in rows if _is_overdue(r["due"], r["status"])}

    now = datetime.now().strftime("%Y-%m-%d")
    print(f"# 📌 Follow-ups — {filter_by.capitalize()} ({len(display)} item(s))  [{now}]\n")

    if not display:
        print(f"  No {filter_by} follow-ups.")
        return

    # Group: overdue first, then by due date
    overdue_items = [r for r in display if r["id"] in overdue_ids]
    other_items   = [r for r in display if r["id"] not in overdue_ids]

    def _row_str(r):
        flag = " ⚠️  OVERDUE" if r["id"] in overdue_ids else ""
        done = "~~" if "done" in r["status"].lower() else ""
        return (
            f"  **#{r['id']}** {done}{r['item']}{done}{flag}\n"
            f"       Owner: {r['owner']} | Due: {r['due']} | Status: {r['status']}"
        )

    if overdue_items:
        print("## ⏰ Overdue")
        for r in overdue_items:
            print(_row_str(r))
            print()

    if other_items:
        label = "Open" if filter_by != "done" else "Done"
        print(f"## {label}")
        for r in sorted(other_items, key=lambda x: x["due"]):
            print(_row_str(r))
            print()

    total_open = sum(1 for r in rows if "done" not in r["status"].lower())
    total_done = sum(1 for r in rows if "done" in r["status"].lower())
    print(f"_Total: {len(rows)} | Open: {total_open} | Done: {total_done} | Overdue: {len(overdue_ids)}_")
