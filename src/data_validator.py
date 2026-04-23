"""
Pandera Data Validator — pre-flight harness checks.

Addresses Dr. Zhang's Week 5 feedback:
    "Data validation should run BEFORE any AI step, not as post-hoc checks."

Each fetched frame is run through a schema before it's persisted to cache or
consumed by downstream AI-adjacent code. Validation failures are logged to
`cache/audit_log.csv` with status="warning" and error="pandera_validation: ...".
Data is returned unchanged — the validator surfaces problems without silently
filtering, so every bad row is auditable.

Schemas
-------
reit_prices       Price > 0 and < 10_000
cap_rates         2.0 ≤ rate ≤ 15.0
migration_scores  0 ≤ score ≤ 100
pop_growth        -10 ≤ pct ≤ +20
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema


# ── Schemas ──────────────────────────────────────────────────────────────────

REIT_PRICES_SCHEMA = DataFrameSchema({
    "Price": Column(
        float,
        checks=[Check.greater_than(0), Check.less_than(10_000)],
        nullable=False,
        coerce=True,
    ),
})

CAP_RATE_SCHEMA = DataFrameSchema({
    "cap_rate": Column(
        float,
        checks=[Check.in_range(2.0, 15.0)],
        nullable=False,
        coerce=True,
    ),
})

MIGRATION_SCORE_SCHEMA = DataFrameSchema({
    "composite_score": Column(
        float,
        checks=[Check.in_range(0.0, 100.0)],
        nullable=False,
        coerce=True,
    ),
})

POP_GROWTH_SCHEMA = DataFrameSchema({
    "pop_growth_pct": Column(
        float,
        checks=[Check.in_range(-10.0, 20.0)],
        nullable=False,
        coerce=True,
    ),
})


# ── Logging helper ────────────────────────────────────────────────────────────

def _log(agent_name: str, detail: str, record_count: int = 0) -> None:
    """Write a pandera validation failure line to the audit log."""
    try:
        from src.audit_logger import log_agent_run
        log_agent_run(
            agent_name=agent_name,
            status="warning",
            latency_ms=0,
            output_summary=f"pandera validation: {detail[:120]}",
            record_count=record_count,
            error=f"pandera_validation: {detail[:200]}",
        )
    except Exception:
        pass


# ── Core validator ───────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, schema: DataFrameSchema, agent_name: str) -> dict:
    """
    Run a DataFrame through a Pandera schema.

    Returns
    -------
    dict with keys:
        ok              — True if all rows passed
        failure_count   — number of failing row/check combos (from lazy=True)
        failure_cases   — Pandera failure_cases DataFrame (or empty DF)
        df              — the input DataFrame, UNCHANGED
    """
    if df is None or df.empty:
        return {"ok": True, "failure_count": 0, "failure_cases": pd.DataFrame(), "df": df}

    try:
        schema.validate(df, lazy=True)
        return {"ok": True, "failure_count": 0, "failure_cases": pd.DataFrame(), "df": df}
    except pa.errors.SchemaErrors as err:
        fc = getattr(err, "failure_cases", pd.DataFrame())
        n  = len(fc) if isinstance(fc, pd.DataFrame) else 0
        sample = ""
        if isinstance(fc, pd.DataFrame) and not fc.empty:
            # Compact "col=value" summary of the first few failures
            cols = [c for c in ("column", "check", "failure_case", "index") if c in fc.columns]
            sample = fc[cols].head(5).to_dict(orient="records") if cols else fc.head(5).to_dict(orient="records")
        _log(agent_name, f"{n} row/check failures — sample={sample}", record_count=n)
        return {"ok": False, "failure_count": n, "failure_cases": fc if isinstance(fc, pd.DataFrame) else pd.DataFrame(), "df": df}
    except Exception as e:
        _log(agent_name, f"unexpected validator error: {e}")
        return {"ok": False, "failure_count": -1, "failure_cases": pd.DataFrame(), "df": df}


# ── Convenience wrappers (one per data type) ─────────────────────────────────

def validate_reit_prices(df: pd.DataFrame) -> dict:
    """Validate a REIT pricing DataFrame. Expects a 'Price' column."""
    return validate(df, REIT_PRICES_SCHEMA, "pricing")


def validate_migration_frame(df: pd.DataFrame) -> dict:
    """
    Validate a migration-scores DataFrame. Runs BOTH the composite-score and
    population-growth schemas and aggregates results.
    """
    out = {"ok": True, "failure_count": 0, "failure_cases": pd.DataFrame(), "df": df}
    if df is None or df.empty:
        return out

    if "composite_score" in df.columns:
        r1 = validate(df, MIGRATION_SCORE_SCHEMA, "migration")
        out["ok"] = out["ok"] and r1["ok"]
        out["failure_count"] += max(r1["failure_count"], 0)
    if "pop_growth_pct" in df.columns:
        r2 = validate(df, POP_GROWTH_SCHEMA, "migration")
        out["ok"] = out["ok"] and r2["ok"]
        out["failure_count"] += max(r2["failure_count"], 0)
    return out


def validate_cap_rate_dict(market_cap_rates: dict[str, dict[str, Any]]) -> dict:
    """
    Validate the nested `{market: {property_type: cap_rate_pct}}` structure
    produced by the cap-rate agent. Flattens to long-format before checking.
    """
    if not market_cap_rates:
        return {"ok": True, "failure_count": 0, "failure_cases": pd.DataFrame(), "df": pd.DataFrame()}

    rows = []
    for mkt, pts in market_cap_rates.items():
        if not isinstance(pts, dict):
            continue
        for pt, val in pts.items():
            try:
                rows.append({"market": mkt, "property_type": pt, "cap_rate": float(val)})
            except (TypeError, ValueError):
                rows.append({"market": mkt, "property_type": pt, "cap_rate": float("nan")})
    df = pd.DataFrame(rows)
    return validate(df, CAP_RATE_SCHEMA, "cap_rate")


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Happy path
    good = pd.DataFrame({"Price": [50.1, 123.4, 987.6]})
    print("REIT good:", validate_reit_prices(good)["ok"])

    # Failure path — out of range
    bad = pd.DataFrame({"Price": [-5.0, 50_000.0, 42.0]})
    r = validate_reit_prices(bad)
    print("REIT bad:", r["ok"], "failures:", r["failure_count"])

    mig_bad = pd.DataFrame({"composite_score": [50, 120, -10], "pop_growth_pct": [1.0, 50.0, -20.0]})
    r = validate_migration_frame(mig_bad)
    print("Migration bad:", r["ok"], "failures:", r["failure_count"])

    cr_bad = {"Austin, TX": {"Office": 1.0, "Industrial": 18.0, "Retail": 7.1, "Multifamily": 5.8}}
    r = validate_cap_rate_dict(cr_bad)
    print("Cap bad:", r["ok"], "failures:", r["failure_count"])
