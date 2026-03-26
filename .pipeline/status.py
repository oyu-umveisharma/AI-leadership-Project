#!/usr/bin/env python3
"""Quick status check — prints last sync state and recent log lines."""

import json
from pathlib import Path
from datetime import datetime

STATE_FILE = Path(__file__).parent / "last_sync.json"
LOG_FILE   = Path(__file__).parent / "logs" / "sync.log"

print("=" * 55)
print(" AI Leadership Project — Sync Agent Status")
print("=" * 55)

if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text())
    for k, v in state.items():
        print(f"  {k:<20} {v}")
else:
    print("  No sync state found yet.")

print()
print("── Recent log (last 20 lines) " + "─" * 25)
if LOG_FILE.exists():
    lines = LOG_FILE.read_text().splitlines()
    for line in lines[-20:]:
        print(" ", line)
else:
    print("  No log file yet.")
print("=" * 55)
