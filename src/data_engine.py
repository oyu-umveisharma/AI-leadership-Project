"""
Data Engine — fetches financial statements and market data via yfinance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def get_company_info(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", None),
        "employees": info.get("fullTimeEmployees", None),
        "description": info.get("longBusinessSummary", ""),
        "website": info.get("website", ""),
        "currency": info.get("financialCurrency", "USD"),
        "price": info.get("currentPrice", info.get("regularMarketPrice", None)),
        "shares_outstanding": info.get("sharesOutstanding", None),
        "total_debt": info.get("totalDebt", None),
        "beta": info.get("beta", None),
    }


def get_income_statement(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        df = t.financials  # annual, columns = dates, rows = line items
        return df
    except Exception:
        return pd.DataFrame()


def get_balance_sheet(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        return t.balance_sheet
    except Exception:
        return pd.DataFrame()


def get_cash_flow(ticker: str) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    try:
        return t.cashflow
    except Exception:
        return pd.DataFrame()


def get_monthly_returns(ticker: str, years: int = 5) -> pd.Series:
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    df = yf.download(ticker, start=start, end=end, interval="1mo", progress=False, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    closes = df["Close"].squeeze()
    returns = closes.pct_change().dropna()
    return returns


def get_sp500_returns(years: int = 5) -> pd.Series:
    return get_monthly_returns("^GSPC", years=years)


def safe_get(df: pd.DataFrame, *keys) -> pd.Series:
    """Try multiple row-label variants; return first match."""
    for key in keys:
        if key in df.index:
            return df.loc[key].astype(float)
    return pd.Series(dtype=float)
