#!/bin/bash
# Run the CRE Intelligence Platform using the correct Python environment.
# Using "python3 -m streamlit" ensures the same Python that has all
# dependencies installed (yfinance, apscheduler, groq, etc.) is used,
# rather than any stale system-level streamlit binary.

cd "$(dirname "$0")"
python3 -m streamlit run app/cre_app.py "$@"
