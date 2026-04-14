"""
Audit Logger — records every agent run with timestamp, latency, and output summary.

Writes to cache/audit_log.csv for traceability and the System Monitor tab.
"""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / "cache"
AUDIT_LOG = CACHE_DIR / "audit_log.csv"
_log_lock = threading.Lock()

HEADERS = [
    "timestamp",
    "agent_name",
    "status",
    "latency_ms",
    "output_summary",
    "record_count",
    "error",
]


def _ensure_csv():
    """Create the CSV file with headers if it doesn't exist."""
    if not AUDIT_LOG.exists():
        CACHE_DIR.mkdir(exist_ok=True)
        with open(AUDIT_LOG, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)


def log_agent_run(
    agent_name: str,
    status: str,
    latency_ms: float,
    output_summary: str = "",
    record_count: int = 0,
    error: str = "",
):
    """
    Append a single audit row for an agent run.

    Parameters
    ----------
    agent_name : str
        Agent identifier (e.g., "migration", "pricing").
    status : str
        "ok", "error", or "running".
    latency_ms : float
        Wall-clock time for the agent run in milliseconds.
    output_summary : str
        Brief description of output (e.g., "51 states, top=TX").
    record_count : int
        Number of records produced.
    error : str
        Error message if status is "error".
    """
    with _log_lock:
        _ensure_csv()
        with open(AUDIT_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                agent_name,
                status,
                round(latency_ms, 1),
                output_summary[:200],
                record_count,
                error[:200] if error else "",
            ])


def read_audit_log(limit: int = 50) -> list:
    """
    Read the most recent `limit` audit log entries.

    Returns list of dicts with HEADERS as keys, newest first.
    """
    _ensure_csv()
    rows = []
    with open(AUDIT_LOG, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Return newest first, limited
    return rows[-limit:][::-1]


def get_agent_stats() -> dict:
    """
    Compute per-agent statistics from the audit log.

    Returns dict: {agent_name: {runs, avg_latency_ms, last_status, error_count}}
    """
    _ensure_csv()
    stats = {}
    with open(AUDIT_LOG, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("agent_name", "")
            if name not in stats:
                stats[name] = {
                    "runs": 0,
                    "total_latency_ms": 0.0,
                    "error_count": 0,
                    "last_status": "",
                    "last_run": "",
                }
            s = stats[name]
            s["runs"] += 1
            try:
                s["total_latency_ms"] += float(row.get("latency_ms", 0))
            except ValueError:
                pass
            if row.get("status") == "error":
                s["error_count"] += 1
            s["last_status"] = row.get("status", "")
            s["last_run"] = row.get("timestamp", "")

    # Compute averages
    for name, s in stats.items():
        s["avg_latency_ms"] = round(s["total_latency_ms"] / s["runs"], 1) if s["runs"] > 0 else 0
        del s["total_latency_ms"]

    return stats


class AgentTimer:
    """
    Context manager for timing and logging agent runs.

    Usage:
        with AgentTimer("migration") as timer:
            # ... do agent work ...
            timer.set_summary("51 states loaded", record_count=51)
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.summary = ""
        self.record_count = 0
        self._start = None
        self._error = None

    def set_summary(self, summary: str, record_count: int = 0):
        self.summary = summary
        self.record_count = record_count

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.time() - self._start) * 1000
        if exc_type:
            log_agent_run(
                self.agent_name,
                status="error",
                latency_ms=latency_ms,
                error=str(exc_val),
            )
        else:
            log_agent_run(
                self.agent_name,
                status="ok",
                latency_ms=latency_ms,
                output_summary=self.summary,
                record_count=self.record_count,
            )
        return False  # Don't suppress exceptions
