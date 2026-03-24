"""
Synergy Analysis — mirrors 'Synergy Assumptions' + 'Synergies final' Excel sheets.

Benchmarked against Bayer / Merck Consumer Care acquisition:
  • Integration costs : 2 % of enterprise deal value, phased 50 / 30 / 20 over 3 years
  • Revenue synergies : 18 % of target revenue at run-rate, phased in (20 % yr1 → 100 % yr2+)
  • EBIT margin on synergies: 24 % (blended operating margin of combined entity)
"""

import numpy as np
import pandas as pd


def compute_synergies(
    target_revenue: float,
    acquirer_revenue: float,
    deal_equity_value: float,
    target_net_debt: float,
    wacc: float,
    tax_rate: float = 0.21,
    growth_rate: float = 0.015,
    rev_synergy_pct: float = 0.18,          # % of target revenue (Excel: 18 %)
    ebit_margin_syn: float = 0.24,           # EBIT margin on incremental synergy revenue
    integration_cost_pct: float = 0.02,      # % of enterprise deal value (Excel: 2 %)
    op_loss_yr1: float = 0.0,               # one-time operating disruption in yr 1
    integration_timing: tuple = (0.50, 0.30, 0.20),   # cost phasing yr 1/2/3
    rev_syn_timing: tuple = (0.20, 1.0),               # rev synergy: 20 % yr1, 100 % yr2+
    projection_years: int = 9,
    # ── backward-compat kwargs (old callers used combined-rev approach) ──
    ramp_years: int = 3,
    rev_syn_pct: float = None,
    cost_syn_pct: float = None,
    combined_cogs_plus_sga: float = 0.0,
) -> dict:
    """
    Compute year-by-year after-tax synergy FCFs.

    Returns dict with:
      synergy_df          — year-by-year breakdown table
      synergy_fcfs        — list[float] of after-tax FCF additions (pass to run_dcf)
      total_pv_synergies  — PV(FCFs) + PV(TV) of synergy stream
      pv_synergies        — PV of FCF portion only
      pv_tv_synergies     — PV of terminal value
      rev_synergy_full    — full run-rate revenue synergy
      integration_cost_total
    """
    # ── Handle old-style calls (combined revenue × %) ──────────────────
    if rev_syn_pct is not None:
        rev_synergy_full = (target_revenue + acquirer_revenue) * rev_syn_pct
    else:
        rev_synergy_full = target_revenue * rev_synergy_pct

    enterprise_deal_value = deal_equity_value + target_net_debt
    integration_cost_total = enterprise_deal_value * integration_cost_pct

    rows = []
    synergy_fcfs = []

    for yr in range(1, projection_years + 1):
        # Revenue synergy ramp
        ramp = rev_syn_timing[0] if yr == 1 else rev_syn_timing[1]
        rev_syn_gross = rev_synergy_full * ramp * ((1 + growth_rate) ** (yr - 1))
        ebit_syn      = rev_syn_gross * ebit_margin_syn
        nopat_syn     = ebit_syn * (1 - tax_rate)

        # Integration costs (phased over 3 years, then 0)
        int_cost = integration_cost_total * integration_timing[yr - 1] if yr <= len(integration_timing) else 0.0

        # One-time operating disruption (yr 1 only)
        op_loss_net = op_loss_yr1 * (1 - tax_rate) if yr == 1 else 0.0

        # Backward-compat: old cost synergy
        cost_syn_bt = 0.0
        if cost_syn_pct:
            ramp_old = min(yr / max(ramp_years, 1), 1.0)
            cost_syn_bt = combined_cogs_plus_sga * cost_syn_pct * ramp_old * (1 - tax_rate)

        after_tax_fcf = nopat_syn - int_cost - op_loss_net + cost_syn_bt
        pv = after_tax_fcf / (1 + wacc) ** yr

        synergy_fcfs.append(after_tax_fcf)
        rows.append({
            "Year":                    f"Year {yr}",
            "Revenue Synergy (gross)": rev_syn_gross,
            "EBIT from Synergies":     ebit_syn,
            "Integration Costs":       int_cost,
            "After-Tax Synergy FCF":   after_tax_fcf,
            "PV of Synergy":           pv,
            # backward compat aliases
            "Revenue Synergy":         rev_syn_gross,
            "Cost Synergy":            cost_syn_bt / (1 - tax_rate) if cost_syn_pct else 0.0,
            "After-Tax Synergy":       after_tax_fcf,
        })

    df = pd.DataFrame(rows)
    pv_syn_fcf = df["PV of Synergy"].sum()

    # Terminal value of synergy stream
    last_fcf  = df["After-Tax Synergy FCF"].iloc[-1] * (1 + growth_rate)
    tv_syn    = last_fcf / (wacc - growth_rate) if wacc > growth_rate else 0.0
    pv_tv_syn = tv_syn / (1 + wacc) ** projection_years

    return {
        "synergy_df":               df,
        "synergy_fcfs":             synergy_fcfs,
        "pv_synergies":             pv_syn_fcf,
        "pv_tv_synergies":          pv_tv_syn,
        "total_pv_synergies":       pv_syn_fcf + pv_tv_syn,
        "rev_synergy_full":         rev_synergy_full,
        "cost_synergy_full":        0.0,
        "integration_cost_total":   integration_cost_total,
    }


def compute_deal_summary(
    standalone_ev_target: float,
    total_pv_synergies: float,
    target_net_debt: float,
    target_shares: float,
    acquirer_market_cap: float,
    offer_premium_pct: float = 0.25,
) -> dict:
    implied_eq   = standalone_ev_target - target_net_debt
    implied_pps  = implied_eq / target_shares if target_shares else 0.0

    synergy_ev   = standalone_ev_target + total_pv_synergies
    synergy_eq   = synergy_ev - target_net_debt
    synergy_pps  = synergy_eq / target_shares if target_shares else 0.0

    offer_price  = implied_pps * (1 + offer_premium_pct)
    deal_value   = offer_price * target_shares if target_shares else 0.0

    return {
        "standalone_ev":            standalone_ev_target,
        "synergy_ev":               synergy_ev,
        "implied_equity_value":     implied_eq,
        "implied_price_per_share":  implied_pps,
        "synergy_price_per_share":  synergy_pps,
        "synergy_equity":           synergy_eq,
        "offer_price":              offer_price,
        "total_deal_value":         deal_value,
        "offer_premium_pct":        offer_premium_pct,
        "pv_synergies":             total_pv_synergies,
    }
