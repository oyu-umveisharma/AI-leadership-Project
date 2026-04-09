#!/bin/bash
# Run the CRE Intelligence Platform using the correct Python environment.
# Using "python3 -m streamlit" ensures the same Python that has all
# dependencies installed (yfinance, apscheduler, groq, etc.) is used,
# rather than any stale system-level streamlit binary.

cd "$(dirname "$0")"

# Clear stale bytecode cache before launch to prevent ImportErrors after git pulls
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Use python3.12 explicitly to avoid the Python 3.9 streamlit binary in PATH
# falling back to python3 if 3.12 is not found by that name
if command -v python3.12 &>/dev/null; then
    python3.12 -m streamlit run app/cre_app.py "$@"
else
    python3 -m streamlit run app/cre_app.py "$@"
fi
