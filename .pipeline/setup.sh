#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Sync Agent Setup — AI Leadership Project
# Installs the autonomous GitHub sync agent as a macOS launchd daemon.
# Run once per machine: bash .pipeline/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$PIPELINE_DIR")"
PYTHON_BIN="$(which python3)"
PLIST_LABEL="com.$(whoami).ai-leadership-sync"
PLIST_DEST="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
LOG_DIR="$PIPELINE_DIR/logs"

echo "── AI Leadership Project — Sync Agent Setup ─────────────"
echo "  Repo:    $REPO_DIR"
echo "  Python:  $PYTHON_BIN"
echo "  User:    $(whoami)"
echo "  Plist:   $PLIST_DEST"
echo "──────────────────────────────────────────────────────────"

# Create logs directory
mkdir -p "$LOG_DIR"

# Generate plist with this user's paths
cat > "$PLIST_DEST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_BIN}</string>
        <string>${PIPELINE_DIR}/sync_agent.py</string>
    </array>

    <!-- Run every 5 minutes -->
    <key>StartInterval</key>
    <integer>300</integer>

    <!-- Also run immediately when loaded -->
    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd_stdout.log</string>

    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd_stderr.log</string>

    <key>KeepAlive</key>
    <false/>

    <key>ThrottleInterval</key>
    <integer>60</integer>
</dict>
</plist>
EOF

echo "  Plist written."

# Unload existing instance if present, then load
launchctl unload "$PLIST_DEST" 2>/dev/null && echo "  Unloaded previous instance." || true
launchctl load "$PLIST_DEST" && echo "  Loaded into launchd."

echo ""
echo "✓ Sync agent is running. It will check for repo updates every 5 minutes."
echo ""
echo "Useful commands:"
echo "  Check status :  python3 .pipeline/status.py"
echo "  Manual sync  :  python3 .pipeline/sync_agent.py"
echo "  Stop agent   :  launchctl unload ~/Library/LaunchAgents/${PLIST_LABEL}.plist"
echo "  Restart agent:  launchctl load  ~/Library/LaunchAgents/${PLIST_LABEL}.plist"
