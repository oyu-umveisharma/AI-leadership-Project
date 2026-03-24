"""
Standalone DCF Valuation — mirrors the 'standalone for PCH' sheet.

FCF = NOPAT + D&A - Capex - ΔNWC
NOPAT = EBIT × (1 - tax_rate)
TV   = FCF_(n+1) / (WACC − g)     [Gordon Growth]
EV   = Σ PV(FCF_t)  +  PV(TV)
"""

import numpy as np
import pandas as pd
from .data_engine import safe_get


def extract_dcf_inputs(income_statement, cash_flow, nwc_df):
    """Pull historical EBIT, D&A, Capex, NWC from financial statements."""
    ebit  = safe_get(income_statement, "EBIT", "Operating Income", "Total Operating Income As Reported")
    da    = safe_get(cash_flow, "Depreciation And Amortization", "Depreciation Amortization Depletion",
                    "Depreciation", "DepreciationAmortization")
    capex = safe_get(cash_flow, "Capital Expenditure", "Capital Expenditures", "Capex",
                     "Purchase Of Property Plant And Equipment").abs()
    sales = safe_get(income_statement, "Total Revenue", "Revenue")
    return {"ebit": ebit, "da": da, "capex": capex, "sales": sales, "nwc_df": nwc_df}


def _positive_mean_ratio(num_s: pd.Series, den_s: pd.Series) -> float:
    """
    Compute mean ratio, skipping years where numerator is negative.
    This prevents goodwill-impairment or restructuring years from distorting
    the long-run operating margin (matches Excel exclusion of 2023 impairment year).
    """
    if num_s.empty or den_s.empty:
        return 0.0
    common = num_s.index.intersection(den_s.index)
    if len(common) == 0:
        return 0.0
    ratios = (num_s[common] / den_s[common]).replace([np.inf, -np.inf], np.nan).dropna()
    positive = ratios[ratios > 0]
    return float(positive.mean()) if not positive.empty else float(ratios.mean()) if not ratios.empty else 0.0


def mean_ratio(num_s: pd.Series, den_s: pd.Series) -> float:
    """Compute mean ratio (all years, no sign filter — for D&A, Capex)."""
    if num_s.empty or den_s.empty:
        return 0.0
    common = num_s.index.intersection(den_s.index)
    if len(common) == 0:
        return 0.0
    ratios = (num_s[common] / den_s[common]).replace([np.inf, -np.inf], np.nan).dropna()
    return float(ratios.mean()) if not ratios.empty else 0.0


def run_dcf(
    ebit_margin: float,
    base_revenue: float,
    da_pct_revenue: float,
    capex_pct_revenue: float,
    nwc_pct_revenue: float,
    wacc: float,
    tax_rate: float = 0.21,
    growth_rate: float = 0.015,
    projection_years: int = 9,
    synergy_fcfs: list = None,       # optional per-year after-tax synergy additions
) -> dict:
    """
    Projects FCF for projection_years, computes terminal value and Enterprise Value.

    synergy_fcfs: list of length >= projection_years with additional after-tax synergy FCF
                  per year (used for the with-synergies DCF version in Step 6).
    """
    years = list(range(1, projection_years + 1))
    syn = synergy_fcfs if (synergy_fcfs and len(synergy_fcfs) >= projection_years) else [0.0] * projection_years

    revenues, ebit_vals, nopat_vals, da_vals, capex_vals, delta_nwc_vals, fcf_vals = [], [], [], [], [], [], []
    prev_nwc = base_revenue * nwc_pct_revenue

    for i, yr in enumerate(years):
        rev      = base_revenue * ((1 + growth_rate) ** yr)
        ebit     = rev * ebit_margin
        # Only apply tax shield when EBIT > 0; don't add back tax on operating losses
        nopat    = ebit * (1 - tax_rate) if ebit > 0 else ebit
        da       = rev * da_pct_revenue
        capex    = rev * capex_pct_revenue
        nwc      = rev * nwc_pct_revenue
        d_nwc    = nwc - prev_nwc
        fcf      = nopat + da - capex - d_nwc + syn[i]
        prev_nwc = nwc

        revenues.append(rev);    ebit_vals.append(ebit);  nopat_vals.append(nopat)
        da_vals.append(da);      capex_vals.append(capex)
        delta_nwc_vals.append(d_nwc); fcf_vals.append(fcf)

    disc     = [1 / (1 + wacc) ** yr for yr in years]
    pv_fcfs  = [fcf * d for fcf, d in zip(fcf_vals, disc)]
    pv_total = sum(pv_fcfs)

    # Terminal Value — Gordon Growth, discounted to today at year-n factor
    tv_fcf = fcf_vals[-1] * (1 + growth_rate)
    tv     = tv_fcf / (wacc - growth_rate) if wacc > growth_rate else 0.0
    pv_tv  = tv * disc[-1]
    ev     = pv_total + pv_tv

    proj_df = pd.DataFrame({
        "Year":            [f"Year {y}" for y in years],
        "Revenue":         revenues,
        "EBIT":            ebit_vals,
        "NOPAT":           nopat_vals,
        "D&A":             da_vals,
        "Capex":           capex_vals,
        "ΔNWC":            delta_nwc_vals,
        "FCF":             fcf_vals,
        "Discount Factor": disc,
        "PV of FCF":       pv_fcfs,
    })

    return {
        "projection_df":      proj_df,
        "pv_fcf_total":       pv_total,
        "terminal_value":     tv,
        "pv_terminal_value":  pv_tv,
        "enterprise_value":   ev,
        "growth_rate":        growth_rate,
        "wacc":               wacc,
        "ebit_margin":        ebit_margin,
        "da_pct":             da_pct_revenue,      # stored for sensitivity re-use
        "capex_pct":          capex_pct_revenue,   # stored for sensitivity re-use
        "fcf_vals":           fcf_vals,
        "pv_fcfs":            pv_fcfs,
        "revenues":           revenues,
        "years":              years,
    }
