"""
Daily Briefing — cos briefing
Scans the repo and produces a structured markdown status report.
"""

import subprocess
import json
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

from . import llm

REPO_DIR   = Path(__file__).parent.parent.parent
CACHE_DIR  = REPO_DIR / "cache"
PIPELINE   = REPO_DIR / ".pipeline"


# ── Git helpers ────────────────────────────────────────────────────────────────

def _git(*args, cwd=REPO_DIR):
    r = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True, text=True)
    return r.stdout.strip()


def _recent_commits(hours=24):
    since = f"--since={hours} hours ago"
    log = _git("log", since, "--oneline", "--no-merges")
    return [l.strip() for l in log.splitlines() if l.strip()]


def _current_branch():
    return _git("branch", "--show-current")


def _stale_branches(days=14):
    raw = _git("branch", "-v", "--sort=-committerdate")
    stale = []
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    for line in raw.splitlines():
        parts = line.strip().lstrip("* ").split()
        if not parts:
            continue
        branch = parts[0]
        if branch in ("HEAD", _current_branch()):
            continue
        ts_raw = _git("log", "-1", "--format=%ct", branch)
        try:
            if float(ts_raw) < cutoff:
                stale.append(branch)
        except ValueError:
            pass
    return stale


def _scan_todos():
    results = []
    extensions = {".py", ".md", ".txt", ".js", ".ts"}
    skip_dirs  = {".git", "__pycache__", "node_modules", ".pipeline", "chief-of-staff"}
    for root, dirs, files in os.walk(REPO_DIR):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if Path(fname).suffix not in extensions:
                continue
            fpath = Path(root) / fname
            try:
                for i, line in enumerate(fpath.read_text(errors="ignore").splitlines(), 1):
                    upper = line.upper()
                    if "TODO" in upper or "FIXME" in upper or "HACK" in upper or "XXX" in upper:
                        rel = fpath.relative_to(REPO_DIR)
                        results.append(f"  `{rel}:{i}` — {line.strip()}")
            except Exception:
                pass
    return results


def _github_prs():
    """Fetch open PRs from GitHub API (public repo, no auth needed)."""
    remote = _git("remote", "get-url", "origin")
    # Parse owner/repo from https or ssh URL
    if "github.com" not in remote:
        return []
    remote = remote.rstrip(".git")
    if remote.startswith("git@"):
        path = remote.split("github.com:")[-1]
    else:
        path = remote.split("github.com/")[-1]
    owner, repo = path.split("/")[-2], path.split("/")[-1]
    url = f"https://api.github.com/repos/{owner}/{repo}/pulls?state=open&per_page=10"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cos-agent/1.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            prs = json.loads(r.read())
            return [f"  #{p['number']} — {p['title']} (@{p['user']['login']})" for p in prs]
    except Exception:
        return []


def _cache_health():
    items = []
    now = datetime.now(timezone.utc).timestamp()
    if not CACHE_DIR.exists():
        return ["  Cache directory not found."]
    for f in sorted(CACHE_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            ts   = data.get("timestamp") or data.get("last_updated") or data.get("cached_at")
            if ts:
                age_h = (now - datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()) / 3600
                flag  = " ⚠️  STALE" if age_h > 8 else ""
                items.append(f"  `{f.name}` — {age_h:.1f}h ago{flag}")
            else:
                items.append(f"  `{f.name}` — no timestamp")
        except Exception:
            items.append(f"  `{f.name}` — unreadable")
    return items or ["  No cache files found."]


def _sync_status():
    state_file = PIPELINE / "last_sync.json"
    if not state_file.exists():
        return "  Sync agent not yet run."
    try:
        s = json.loads(state_file.read_text())
        checked = s.get("last_checked", "unknown")
        synced  = s.get("last_synced",  "never")
        sha     = s.get("current_sha",  "unknown")[:12]
        status  = s.get("status",       "unknown")
        return f"  Last checked: {checked} | Last synced: {synced} | HEAD: {sha} | Status: {status}"
    except Exception:
        return "  Could not read sync state."


# ── Main ───────────────────────────────────────────────────────────────────────

def run():
    now  = datetime.now().strftime("%A, %B %-d %Y — %-I:%M %p")
    commits = _recent_commits(24)
    todos   = _scan_todos()
    prs     = _github_prs()
    stale   = _stale_branches()
    cache   = _cache_health()
    sync    = _sync_status()

    lines = [
        f"# 📋 Chief of Staff — Daily Briefing",
        f"_{now}_",
        "",
        "---",
        "",
        f"## 🔀 Recent Commits (last 24h) — {len(commits)} commit(s)",
    ]
    if commits:
        lines += [f"  - {c}" for c in commits]
    else:
        lines.append("  _No commits in the last 24 hours._")

    lines += ["", f"## 🔁 Open Pull Requests — {len(prs)} open"]
    if prs:
        lines += prs
    else:
        lines.append("  _No open PRs._")

    lines += ["", f"## 📝 TODO / FIXME / HACK — {len(todos)} found"]
    if todos:
        lines += todos[:20]
        if len(todos) > 20:
            lines.append(f"  _…and {len(todos)-20} more._")
    else:
        lines.append("  _None found._")

    lines += ["", f"## 🌿 Stale Branches (>14 days) — {len(stale)} found"]
    if stale:
        lines += [f"  - `{b}`" for b in stale]
    else:
        lines.append("  _No stale branches._")

    lines += ["", "## 🗄️  Agent Cache Health"]
    lines += cache

    lines += ["", "## 🔄 Sync Agent"]
    lines.append(sync)

    # Optional LLM summary
    if llm.available() and commits:
        commit_text = "\n".join(commits)
        summary = llm.ask(
            f"Summarize these git commits in 2-3 sentences for an executive briefing:\n{commit_text}",
            system="You are a concise engineering executive assistant.",
            max_tokens=200,
        )
        if summary:
            lines += ["", "## 🤖 AI Summary of Recent Work"]
            lines.append(summary)

    lines += ["", "---", "_Generated by Chief of Staff_"]

    report = "\n".join(lines)
    print(report)

    # Save to state/
    out = Path(__file__).parent.parent / "state" / "last_briefing.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
