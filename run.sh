#!/bin/bash
# Run the CRE Intelligence Platform using the correct Python environment.
# Using "python3 -m streamlit" ensures the same Python that has all
# dependencies installed (yfinance, apscheduler, groq, etc.) is used,
# rather than any stale system-level streamlit binary.

cd "$(dirname "$0")"

# Clear stale bytecode cache before launch to prevent ImportErrors after git pulls
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

python3 -m streamlit run app/cre_app.py "$@"
