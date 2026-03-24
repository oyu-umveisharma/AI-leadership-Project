"""
NWC Analysis — Net Working Capital calculations following the M&A Excel model.
"""

import pandas as pd
import numpy as np
from .data_engine import safe_get

_EMPTY_COLS = ["Year", "Total Current Assets", "Cash",
               "Total Current Liabilities", "Short-Term Debt",
               "NWC", "Sales", "NWC/Sales", "Increase in NWC"]


def _scalar(val):
    """Force any pandas/numpy object to a plain Python float."""
    try:
        if isinstance(val, (pd.Series, pd.DataFrame)):
            val = val.values.flat[0]
        result = float(val)
        return result if not np.isnan(result) else np.nan
    except Exception:
        return np.nan


def compute_nwc(balance_sheet, income_statement):
    try:
        if balance_sheet is None or (hasattr(balance_sheet, 'empty') and balance_sheet.empty):
            return pd.DataFrame(columns=_EMPTY_COLS)
        if income_statement is None or (hasattr(income_statement, 'empty') and income_statement.empty):
            return pd.DataFrame(columns=_EMPTY_COLS)

        bs  = balance_sheet
        inc = income_statement

        total_ca = safe_get(bs,  "Current Assets", "Total Current Assets")
        cash     = safe_get(bs,  "Cash And Cash Equivalents", "Cash")
        total_cl = safe_get(bs,  "Current Liabilities", "Total Current Liabilities")
        std      = safe_get(bs,  "Current Debt", "Short Long Term Debt",
                            "Current Portion Of Long Term Debt",
                            "CurrentDebtAndCapitalLeaseObligation")
        sales    = safe_get(inc, "Total Revenue", "Revenue")

        years = sorted(set(bs.columns) & set(inc.columns), reverse=True)[:5]
        if not years:
            return pd.DataFrame(columns=_EMPTY_COLS)

        rows = []
        for yr in years:
            def _get(series, yr):
                try:
                    return _scalar(series[yr]) if yr in series.index else np.nan
                except Exception:
                    return np.nan

            ca  = _get(total_ca, yr)
            c   = _get(cash,     yr)
            cl  = _get(total_cl, yr)
            s   = _get(std,      yr)
            rev = _get(sales,    yr)

            if np.isnan(s):
                s = 0.0

            try:
                nwc_val = float(ca) - float(c) - (float(cl) - float(s))
            except Exception:
                nwc_val = np.nan

            try:
                nwc_sales = nwc_val / rev if (np.isfinite(nwc_val) and np.isfinite(rev) and rev != 0) else np.nan
            except Exception:
                nwc_sales = np.nan

            rows.append({
                "Year":                      str(yr.year) if hasattr(yr, "year") else str(yr),
                "Total Current Assets":      ca,
                "Cash":                      c,
                "Total Current Liabilities": cl,
                "Short-Term Debt":           s,
                "NWC":                       nwc_val,
                "Sales":                     rev,
                "NWC/Sales":                 nwc_sales,
                "Increase in NWC":           np.nan,
            })

        if not rows:
            return pd.DataFrame(columns=_EMPTY_COLS)

        df = pd.DataFrame(rows)
        # compute increase in NWC (positive = cash outflow)
        nwc_series = pd.to_numeric(df["NWC"], errors="coerce")
        df["Increase in NWC"] = nwc_series.diff(-1) * -1
        return df

    except Exception:
        return pd.DataFrame(columns=_EMPTY_COLS)


def nwc_average_ratio(nwc_df):
    try:
        if nwc_df is None or nwc_df.empty or "NWC/Sales" not in nwc_df.columns:
            return 0.18
        result = pd.to_numeric(nwc_df["NWC/Sales"], errors="coerce").dropna().mean()
        return float(result) if np.isfinite(result) else 0.18
    except Exception:
        return 0.18
