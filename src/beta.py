"""
Beta Regression — OLS regression of stock monthly returns vs S&P 500.
Mirrors the Excel 'Equity beta' sheet methodology.
"""

import numpy as np
import pandas as pd
from scipy import stats
from .data_engine import get_monthly_returns, get_sp500_returns


def compute_beta(ticker: str, years: int = 5) -> dict:
    stock_ret = get_monthly_returns(ticker, years)
    mkt_ret = get_sp500_returns(years)

    # Align on common dates
    combined = pd.DataFrame({"stock": stock_ret, "market": mkt_ret}).dropna()
    if len(combined) < 12:
        return {"beta": None, "alpha": None, "r_squared": None, "n_obs": len(combined)}

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        combined["market"], combined["stock"]
    )

    return {
        "beta": round(slope, 4),
        "alpha": round(intercept, 6),
        "r_squared": round(r_value ** 2, 4),
        "p_value": round(p_value, 4),
        "std_err": round(std_err, 4),
        "n_obs": len(combined),
        "stock_returns": combined["stock"],
        "market_returns": combined["market"],
    }
