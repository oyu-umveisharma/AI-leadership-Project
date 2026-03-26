#!/usr/bin/env python3
"""
Autonomous Sync Agent — AI Leadership Project
Polls GitHub for new commits, pulls updates, installs deps, and logs all activity.
Run on a schedule via launchd (see com.aayman.ai-leadership-sync.plist).
"""

import subprocess
import sys
import json
import logging
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
REPO_DIR    = Path(__file__).parent.parent          # AI-leadership-Project/
LOG_FILE    = Path(__file__).parent / "logs" / "sync.log"
STATE_FILE  = Path(__file__).parent / "last_sync.json"
REPO_OWNER  = "oyu-umveisharma"
REPO_NAME   = "AI-leadership-Project"
BRANCH      = "main"
GITHUB_API  = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits/{BRANCH}"

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


def run(cmd: list[str], cwd=None) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def get_remote_sha() -> str | None:
    """Fetch the latest commit SHA from GitHub API."""
    try:
        req = urllib.request.Request(
            GITHUB_API,
            headers={"Accept": "application/vnd.github.v3+json",
                     "User-Agent": "sync-agent/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data["sha"]
    except urllib.error.URLError as e:
        log.warning(f"GitHub API unreachable: {e}")
        return None
    except Exception as e:
        log.error(f"Failed to fetch remote SHA: {e}")
        return None


def get_local_sha() -> str:
    """Return the current local HEAD SHA."""
    _, sha, _ = run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR)
    return sha


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def pull_updates() -> bool:
    """Pull latest changes. Returns True if successful."""
    code, out, err = run(["git", "pull", "origin", BRANCH], cwd=REPO_DIR)
    if code == 0:
        log.info(f"git pull succeeded:\n{out}")
        return True
    else:
        log.error(f"git pull failed (exit {code}):\n{err}")
        return False


def install_deps() -> bool:
    """Re-run pip install if requirements.txt changed."""
    req_file = REPO_DIR / "requirements.txt"
    if not req_file.exists():
        return True
    code, out, err = run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_file), "--quiet"]
    )
    if code == 0:
        log.info("Dependencies installed/verified.")
        return True
    else:
        log.error(f"pip install failed:\n{err}")
        return False


def check_and_sync():
    log.info("── Sync agent tick ──────────────────────────────────")

    remote_sha = get_remote_sha()
    local_sha  = get_local_sha()
    state      = load_state()

    log.info(f"Local  SHA: {local_sha[:12] if local_sha else 'unknown'}")
    log.info(f"Remote SHA: {remote_sha[:12] if remote_sha else 'unknown (API unavailable)'}")

    if remote_sha is None:
        log.warning("Skipping sync — could not reach GitHub API.")
        return

    if remote_sha == local_sha:
        log.info("Repository is up to date. No action needed.")
        state["last_checked"] = datetime.now().isoformat()
        save_state(state)
        return

    log.info(f"New commits detected ({local_sha[:8]} → {remote_sha[:8]}). Pulling…")

    if pull_updates():
        new_sha = get_local_sha()
        deps_ok = install_deps()
        state.update({
            "last_synced":    datetime.now().isoformat(),
            "last_checked":   datetime.now().isoformat(),
            "previous_sha":   local_sha,
            "current_sha":    new_sha,
            "deps_installed": deps_ok,
            "status":         "ok" if deps_ok else "pull_ok_deps_failed",
        })
        log.info(f"Sync complete. Now at {new_sha[:12]}.")
    else:
        state.update({
            "last_checked": datetime.now().isoformat(),
            "status":       "pull_failed",
        })

    save_state(state)


if __name__ == "__main__":
    check_and_sync()
