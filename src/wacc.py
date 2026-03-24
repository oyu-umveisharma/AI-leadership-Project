"""
WACC Calculation — mirrors the PCH WACC / WACC w synergies sheets.

WACC = wd × (1 - τ) × rd + we × re

re = rf + β × rp        (CAPM)
rd = rf + β_debt × rp   (or yield on corporate bonds)
"""

import numpy as np


# Market constants (user-adjustable via app)
RF_DEFAULT = 0.0411          # 10-year Treasury yield
MRP_DEFAULT = 0.0576         # Implied market risk premium (Damodaran)
TAX_RATE_DEFAULT = 0.21
DEBT_BETA_DEFAULT = 0.17     # Average BB-rated debt beta


def compute_wacc(
    market_equity: float,
    total_debt: float,
    excess_cash: float,
    equity_beta: float,
    rf: float = RF_DEFAULT,
    mrp: float = MRP_DEFAULT,
    tax_rate: float = TAX_RATE_DEFAULT,
    debt_beta: float = DEBT_BETA_DEFAULT,
    use_bond_yield: bool = True,
    bond_yield: float = 0.054,
) -> dict:
    net_debt = total_debt - excess_cash
    total_capital = net_debt + market_equity
    wd = net_debt / total_capital if total_capital != 0 else 0
    we = market_equity / total_capital if total_capital != 0 else 0

    re = rf + equity_beta * mrp
    if use_bond_yield:
        rd = bond_yield
    else:
        rd = rf + debt_beta * mrp

    wacc = wd * (1 - tax_rate) * rd + we * re

    return {
        "WACC": round(wacc, 6),
        "wd": round(wd, 6),
        "we": round(we, 6),
        "re": round(re, 6),
        "rd": round(rd, 6),
        "net_debt": net_debt,
        "market_equity": market_equity,
        "tax_rate": tax_rate,
        "rf": rf,
        "mrp": mrp,
        "equity_beta": equity_beta,
        "debt_beta": debt_beta,
    }


def compute_excess_cash(cash: float, revenue: float, excess_cash_pct: float = 0.02) -> float:
    """
    Excess cash = max(0, cash - operating_cash_needed)
    Operating cash needed ~= 2% of revenue (conservative assumption).
    """
    operating_cash_needed = revenue * excess_cash_pct
    return max(0.0, cash - operating_cash_needed)
