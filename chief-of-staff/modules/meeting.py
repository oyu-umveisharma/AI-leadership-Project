"""
Meeting Prep — cos prep "<topic>"
Pulls codebase context relevant to the topic and generates a prep doc.
"""

import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from . import llm

REPO_DIR  = Path(__file__).parent.parent.parent
STATE_DIR = Path(__file__).parent.parent / "state"
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".pipeline", "chief-of-staff", "cache"}


def _git(*args):
    r = subprocess.run(["git"] + list(args), cwd=REPO_DIR, capture_output=True, text=True)
    return r.stdout.strip()


def _search_codebase(keywords: list[str]) -> list[dict]:
    """Find files and lines matching any keyword."""
    hits = []
    extensions = {".py", ".md", ".txt", ".json"}
    seen_files = set()

    for root, dirs, files in os.walk(REPO_DIR):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if Path(fname).suffix not in extensions:
                continue
            fpath = Path(root) / fname
            try:
                content = fpath.read_text(errors="ignore")
                matches = []
                for i, line in enumerate(content.splitlines(), 1):
                    if any(kw.lower() in line.lower() for kw in keywords):
                        matches.append((i, line.strip()))
                if matches:
                    rel = str(fpath.relative_to(REPO_DIR))
                    hits.append({"file": rel, "matches": matches[:5]})
                    seen_files.add(rel)
            except Exception:
                pass
    return hits[:20]


def _commits_for_topic(keywords: list[str], n=20) -> list[str]:
    log = _git("log", f"-{n}", "--oneline", "--no-merges")
    results = []
    for line in log.splitlines():
        if any(kw.lower() in line.lower() for kw in keywords):
            results.append(line.strip())
    return results


def _read_open_tasks(keywords: list[str]) -> list[str]:
    tasks_file = STATE_DIR / "tasks.md"
    if not tasks_file.exists():
        return []
    results = []
    for line in tasks_file.read_text().splitlines():
        m = re.match(r"\s*-\s*\[\s*\]\s*(.+)", line)
        if m and any(kw.lower() in line.lower() for kw in keywords):
            results.append(m.group(1).strip())
    return results


def run(topic: str):
    keywords = [w for w in re.split(r"[\s,/]+", topic) if len(w) > 2]
    if not keywords:
        print("Please provide a topic with at least one meaningful word.")
        return

    hits    = _search_codebase(keywords)
    commits = _commits_for_topic(keywords)
    tasks   = _read_open_tasks(keywords)

    now = datetime.now().strftime("%A, %B %-d %Y")
    lines = [
        f"# 📅 Meeting Prep — {topic}",
        f"_{now}_",
        "",
        "---",
        "",
        "## 🗂️  Relevant Files & Code",
    ]

    if hits:
        for h in hits[:12]:
            lines.append(f"\n### `{h['file']}`")
            for lineno, text in h["matches"]:
                lines.append(f"  - L{lineno}: `{text[:100]}`")
    else:
        lines.append("  _No matching code found for this topic._")

    lines += ["", "## 🔀 Related Commits"]
    if commits:
        lines += [f"  - {c}" for c in commits[:10]]
    else:
        lines.append("  _No recent commits matching this topic._")

    lines += ["", "## 📝 Open Tasks Related to This Topic"]
    if tasks:
        lines += [f"  - {t}" for t in tasks]
    else:
        lines.append("  _No open tasks matching this topic._")

    # Build context blob for LLM
    context_parts = []
    if hits:
        context_parts.append("Relevant files: " + ", ".join(h["file"] for h in hits[:8]))
    if commits:
        context_parts.append("Recent commits: " + "; ".join(commits[:5]))
    if tasks:
        context_parts.append("Open tasks: " + "; ".join(tasks[:5]))

    if llm.available():
        context_text = "\n".join(context_parts) if context_parts else "No specific code context found."
        talking_points = llm.ask(
            f"We're having a meeting about: '{topic}'\n\n"
            f"Project context:\n{context_text}\n\n"
            f"Generate 5 sharp talking points for this meeting, including: "
            f"current status, key decisions needed, risks, dependencies, and next steps.",
            system="You are an executive chief of staff preparing a meeting briefing.",
            max_tokens=600,
        )
        if talking_points:
            lines += ["", "## 🤖 AI-Generated Talking Points"]
            lines.append(talking_points)
        questions = llm.ask(
            f"For a meeting about '{topic}' in this CRE intelligence platform project, "
            f"what are the 3 most important questions that should be answered?",
            system="You are a sharp product manager.",
            max_tokens=300,
        )
        if questions:
            lines += ["", "## ❓ Key Questions to Answer"]
            lines.append(questions)
    else:
        lines += ["", "## 💡 Suggested Talking Points"]
        lines += [
            "  1. Current status and progress since last meeting",
            "  2. Blockers or unresolved decisions",
            "  3. Resource or dependency needs",
            "  4. What needs to ship this week",
            "  5. Risks and open questions",
        ]

    lines += ["", "---", f"_Generated by Chief of Staff for topic: '{topic}'_"]

    doc = "\n".join(lines)
    print(doc)

    out = STATE_DIR / f"prep_{re.sub(r'[^a-z0-9]', '_', topic.lower())[:40]}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(doc)
    print(f"\n[cos] Saved to {out.relative_to(REPO_DIR)}", flush=True)
