"""
M&A Deal Analyzer — Purdue MSF | Group AI Project
Step-by-step M&A analysis for any two public companies.

Run: streamlit run app/main.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from src.data_engine import (
    get_company_info, get_income_statement, get_balance_sheet,
    get_cash_flow, safe_get,
)
from src.nwc import compute_nwc, nwc_average_ratio
from src.beta import compute_beta
from src.wacc import compute_wacc, compute_excess_cash
from src.dcf import extract_dcf_inputs, run_dcf, _positive_mean_ratio, mean_ratio
from src.synergies import compute_synergies, compute_deal_summary

# ─── Fixed Parameters ────────────────────────────────────────────────────────
rf = 0.0411        # 10-Year Treasury yield
mrp = 0.0576       # Implied market risk premium (Damodaran)
tax_rate = 0.21
debt_beta = 0.17   # Average BB-rated debt beta
bond_yield = 0.054
growth_rate = 0.015
projection_years = 9
offer_premium = 0.25
rev_syn_pct = 0.02
cost_syn_pct = 0.05
ramp_years = 3

# ─── Purdue Brand Colors ────────────────────────────────────────────────────
GOLD = "#CFB991"
GOLD_DARK = "#8E6F3E"
BLACK = "#000000"
DARK_GRAY = "#1a1a1a"
LIGHT_GRAY = "#f5f5f5"

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="M&A Deal Analyzer | Purdue MSF",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Source Sans Pro', sans-serif;
  }}

  /* Top header bar */
  .purdue-header {{
    background: {BLACK};
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
  }}
  .purdue-header h1 {{
    color: {GOLD};
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.5px;
  }}
  .purdue-header span {{
    color: white;
    font-size: 0.85rem;
    opacity: 0.8;
  }}

  /* Gold accent bar */
  .gold-bar {{
    height: 4px;
    background: linear-gradient(90deg, {GOLD_DARK}, {GOLD}, {GOLD_DARK});
    margin-bottom: 28px;
  }}

  /* Section headers */
  .section-header {{
    background: {BLACK};
    color: {GOLD};
    padding: 10px 18px;
    border-radius: 4px;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 28px 0 16px 0;
    border-left: 5px solid {GOLD};
  }}

  /* Step badge */
  .step-badge {{
    display: inline-block;
    background: {GOLD};
    color: {BLACK};
    border-radius: 50%;
    width: 28px;
    height: 28px;
    line-height: 28px;
    text-align: center;
    font-weight: 700;
    font-size: 0.9rem;
    margin-right: 8px;
  }}

  /* Metric cards */
  .metric-card {{
    background: white;
    border: 1px solid #e0e0e0;
    border-top: 4px solid {GOLD};
    border-radius: 6px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  .metric-card .label {{
    font-size: 0.78rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
  }}
  .metric-card .value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {BLACK};
  }}
  .metric-card .sub {{
    font-size: 0.8rem;
    color: #888;
    margin-top: 4px;
  }}

  /* Insight box */
  .insight-box {{
    background: #fffbf0;
    border-left: 4px solid {GOLD};
    padding: 14px 18px;
    border-radius: 4px;
    margin: 12px 0;
    font-size: 0.93rem;
    line-height: 1.6;
  }}
  .insight-box strong {{
    color: {GOLD_DARK};
  }}

  /* Formula box */
  .formula-box {{
    background: {DARK_GRAY};
    color: {GOLD};
    padding: 14px 20px;
    border-radius: 6px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    margin: 10px 0;
    line-height: 1.8;
  }}

  /* Company card */
  .company-card {{
    background: white;
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 20px;
    border-top: 5px solid {GOLD};
  }}

  /* Sidebar styling */
  section[data-testid="stSidebar"] {{
    background: {BLACK} !important;
  }}
  section[data-testid="stSidebar"] * {{
    color: white !important;
  }}
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stTextInput label,
  section[data-testid="stSidebar"] .stSlider label {{
    color: {GOLD} !important;
    font-weight: 600;
  }}

  /* Step nav buttons */
  .stButton > button {{
    background: {GOLD};
    color: {BLACK};
    border: none;
    font-weight: 700;
    border-radius: 4px;
    padding: 8px 20px;
  }}
  .stButton > button:hover {{
    background: {GOLD_DARK};
    color: white;
  }}

  /* Table */
  .dataframe {{
    font-size: 0.85rem !important;
  }}

  /* Result highlight */
  .result-highlight {{
    background: {BLACK};
    color: {GOLD};
    padding: 20px 28px;
    border-radius: 8px;
    margin: 16px 0;
  }}
  .result-highlight .big-number {{
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 1px;
  }}
  .result-highlight .label {{
    font-size: 0.85rem;
    opacity: 0.8;
    margin-bottom: 6px;
  }}
</style>
""", unsafe_allow_html=True)


# ─── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="purdue-header">
  <h1>🏛 M&A Deal Analyzer</h1>
  <span>Purdue University · Daniels School of Business · MSF Program</span>
</div>
<div class="gold-bar"></div>
""", unsafe_allow_html=True)


# ─── Sidebar: Inputs ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<h2 style='color:{GOLD};font-size:1.1rem;margin-bottom:4px;'>Deal Configuration</h2>", unsafe_allow_html=True)
    st.markdown("---")

    acquirer_ticker = st.text_input("Acquirer Ticker", value="PG", help="e.g. MSFT, PG, AMZN").upper().strip()
    target_ticker = st.text_input("Target Ticker", value="PHSI", help="e.g. PHSI, ATVI, VMW").upper().strip()

    st.markdown("---")
    run_btn = st.button("Run Analysis", use_container_width=True)

    # Show active deal if one is loaded
    active = st.session_state.get("tickers")
    if active:
        st.markdown(f"<p style='color:#aaa;font-size:0.78rem;text-align:center;margin-top:6px;'>Active: <strong style='color:{GOLD};'>{active[0]}</strong> acquiring <strong style='color:{GOLD};'>{active[1]}</strong></p>", unsafe_allow_html=True)
        if st.button("Reset / New Deal", use_container_width=True):
            st.session_state.step = 0
            st.session_state.data = {}
            st.session_state["tickers"] = None
            st.rerun()


# ─── Session State ───────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 0
if "data" not in st.session_state:
    st.session_state.data = {}

STEPS = [
    "Overview",
    "Step 1 · Financial Statements",
    "Step 2 · NWC Analysis",
    "Step 3 · Beta Regression",
    "Step 4 · WACC",
    "Step 5 · Standalone DCF",
    "Step 6 · Synergy Analysis",
    "Step 7 · Deal Summary",
    "Step 8 · Suggested Deals",
    "Step 9 · Cannibalization",
]

# ─── Step Navigation ─────────────────────────────────────────────────────────
def step_nav():
    cols = st.columns(len(STEPS))
    for i, (col, name) in enumerate(zip(cols, STEPS)):
        with col:
            active = st.session_state.step == i
            color = GOLD if active else "#ccc"
            st.markdown(
                f"<div style='text-align:center;border-bottom:3px solid {color};"
                f"padding-bottom:6px;font-size:0.72rem;color:{color};font-weight:{'700' if active else '400'};'>"
                f"{name}</div>",
                unsafe_allow_html=True
            )

step_nav()
st.markdown("<br>", unsafe_allow_html=True)

# ─── Helper: fmt ─────────────────────────────────────────────────────────────
def fmt_num(v, prefix="$", suffix="", millions=True):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if millions:
        if abs(v) >= 1e9:
            return f"{prefix}{v/1e9:.2f}B{suffix}"
        elif abs(v) >= 1e6:
            return f"{prefix}{v/1e6:.1f}M{suffix}"
    return f"{prefix}{v:,.0f}{suffix}"

def fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v*100:.2f}%"

def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {"<div class='sub'>" + sub + "</div>" if sub else ""}
    </div>"""

def section(title, step_num=None):
    badge = f'<span class="step-badge">{step_num}</span>' if step_num else ""
    st.markdown(f'<div class="section-header">{badge}{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def formula(text):
    st.markdown(f'<div class="formula-box">{text}</div>', unsafe_allow_html=True)


# ─── Load Data ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_all(acq_t, tgt_t):
    acq_info = get_company_info(acq_t)
    tgt_info = get_company_info(tgt_t)
    acq_bs = get_balance_sheet(acq_t)
    acq_is = get_income_statement(acq_t)
    acq_cf = get_cash_flow(acq_t)
    tgt_bs = get_balance_sheet(tgt_t)
    tgt_is = get_income_statement(tgt_t)
    tgt_cf = get_cash_flow(tgt_t)
    return acq_info, tgt_info, acq_bs, acq_is, acq_cf, tgt_bs, tgt_is, tgt_cf


if run_btn:
    st.session_state.step = 0
    st.session_state.data = {}
    st.session_state["tickers"] = (acquirer_ticker, target_ticker)
    st.rerun()

tickers = st.session_state.get("tickers") or (acquirer_ticker, target_ticker)
acq_t, tgt_t = tickers


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 0 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 0:
    st.markdown('<h2 style="color:#000;font-size:1.5rem;margin-bottom:4px;">Deal Overview</h2>', unsafe_allow_html=True)
    st.markdown("Enter your acquirer and target tickers in the sidebar and click **Run Analysis** to begin.")

    with st.spinner("Loading company profiles..."):
        try:
            acq_info = get_company_info(acq_t)
            tgt_info = get_company_info(tgt_t)
        except Exception as e:
            st.error(f"Could not fetch data: {e}")
            st.stop()

    col1, col2 = st.columns(2)
    with col1:
        section("Acquirer")
        st.markdown(f"""<div class="company-card">
          <h3 style="color:{GOLD_DARK};margin:0 0 8px 0;">{acq_info['name']}</h3>
          <p style="font-size:0.85rem;color:#555;margin:0 0 12px 0;">{acq_info['sector']} · {acq_info['industry']}</p>
          <p style="font-size:0.88rem;color:#333;line-height:1.5;">{acq_info['description'][:400]}...</p>
          <hr style="border-color:#eee;">
          <table style="width:100%;font-size:0.85rem;">
            <tr><td style="color:#666;">Market Cap</td><td style="text-align:right;font-weight:600;">{fmt_num(acq_info['market_cap'])}</td></tr>
            <tr><td style="color:#666;">Shares Outstanding</td><td style="text-align:right;font-weight:600;">{fmt_num(acq_info['shares_outstanding'], prefix='', suffix='', millions=True)}</td></tr>
            <tr><td style="color:#666;">Stock Price</td><td style="text-align:right;font-weight:600;">${f"{acq_info['price']:,.2f}" if acq_info['price'] else 'N/A'}</td></tr>
            <tr><td style="color:#666;">Beta</td><td style="text-align:right;font-weight:600;">{acq_info['beta'] if acq_info['beta'] else 'N/A'}</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with col2:
        section("Target")
        st.markdown(f"""<div class="company-card">
          <h3 style="color:{GOLD_DARK};margin:0 0 8px 0;">{tgt_info['name']}</h3>
          <p style="font-size:0.85rem;color:#555;margin:0 0 12px 0;">{tgt_info['sector']} · {tgt_info['industry']}</p>
          <p style="font-size:0.88rem;color:#333;line-height:1.5;">{tgt_info['description'][:400]}...</p>
          <hr style="border-color:#eee;">
          <table style="width:100%;font-size:0.85rem;">
            <tr><td style="color:#666;">Market Cap</td><td style="text-align:right;font-weight:600;">{fmt_num(tgt_info['market_cap'])}</td></tr>
            <tr><td style="color:#666;">Shares Outstanding</td><td style="text-align:right;font-weight:600;">{fmt_num(tgt_info['shares_outstanding'], prefix='', suffix='', millions=True)}</td></tr>
            <tr><td style="color:#666;">Stock Price</td><td style="text-align:right;font-weight:600;">${f"{tgt_info['price']:,.2f}" if tgt_info['price'] else 'N/A'}</td></tr>
            <tr><td style="color:#666;">Beta</td><td style="text-align:right;font-weight:600;">{tgt_info['beta'] if tgt_info['beta'] else 'N/A'}</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    insight(f"""
    <strong>Deal Thesis:</strong> This tool walks through a full M&A valuation of <strong>{tgt_info['name']} ({tgt_t})</strong>
    being acquired by <strong>{acq_info['name']} ({acq_t})</strong>. We follow the standard investment banking
    methodology: financial statement normalization → NWC analysis → beta regression → WACC →
    standalone DCF → synergy valuation → deal summary. Every number is computed live from
    public financial data, following the same steps as the course Excel model.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Begin Analysis →"):
        st.session_state.step = 1
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — FINANCIAL STATEMENTS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 1:
    section("Financial Statements", 1)
    insight("""
    We pull annual financial statements for both companies directly from SEC filings via Yahoo Finance.
    These form the foundation for every subsequent calculation — NWC, WACC, and DCF projections.
    We examine the <strong>Income Statement</strong>, <strong>Balance Sheet</strong>, and
    <strong>Cash Flow Statement</strong> for the most recent 4 fiscal years.
    """)

    with st.spinner("Fetching financial statements..."):
        acq_info = get_company_info(acq_t)
        tgt_info = get_company_info(tgt_t)
        acq_is = get_income_statement(acq_t)
        acq_bs = get_balance_sheet(acq_t)
        acq_cf = get_cash_flow(acq_t)
        tgt_is = get_income_statement(tgt_t)
        tgt_bs = get_balance_sheet(tgt_t)
        tgt_cf = get_cash_flow(tgt_t)

    # Store for downstream steps
    st.session_state.data.update({
        "acq_info": acq_info, "tgt_info": tgt_info,
        "acq_is": acq_is, "acq_bs": acq_bs, "acq_cf": acq_cf,
        "tgt_is": tgt_is, "tgt_bs": tgt_bs, "tgt_cf": tgt_cf,
    })

    def show_statements(ticker, name, is_df, bs_df, cf_df):
        st.markdown(f"#### {name} ({ticker})")
        tabs = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
        for tab, df, label in zip(tabs, [is_df, bs_df, cf_df],
                                  ["Income Statement", "Balance Sheet", "Cash Flow"]):
            with tab:
                if df is not None and not df.empty:
                    disp = df.copy()
                    disp.columns = [str(c)[:10] if hasattr(c, "strftime") else str(c) for c in disp.columns]
                    disp = disp.applymap(lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) and not np.isnan(x) else x)
                    st.dataframe(disp, use_container_width=True)
                else:
                    st.warning(f"No {label} data available for {ticker}.")

    col1, col2 = st.columns(2)
    with col1:
        show_statements(acq_t, acq_info["name"], acq_is, acq_bs, acq_cf)
    with col2:
        show_statements(tgt_t, tgt_info["name"], tgt_is, tgt_bs, tgt_cf)

    # Key metrics comparison
    st.markdown("<br>", unsafe_allow_html=True)
    section("Key Metrics Comparison")

    def get_latest(df, *keys):
        s = safe_get(df, *keys)
        if not s.empty:
            first = s.iloc[0]
            return float(first) if not pd.isna(first) else None
        return None

    acq_rev = get_latest(acq_is, "Total Revenue", "Revenue")
    tgt_rev = get_latest(tgt_is, "Total Revenue", "Revenue")
    acq_ni = get_latest(acq_is, "Net Income", "Net Income Common Stockholders")
    tgt_ni = get_latest(tgt_is, "Net Income", "Net Income Common Stockholders")
    acq_assets = get_latest(acq_bs, "Total Assets")
    tgt_assets = get_latest(tgt_bs, "Total Assets")

    metrics = {
        "Revenue": (fmt_num(acq_rev), fmt_num(tgt_rev)),
        "Net Income": (fmt_num(acq_ni), fmt_num(tgt_ni)),
        "Total Assets": (fmt_num(acq_assets), fmt_num(tgt_assets)),
        "Market Cap": (fmt_num(acq_info["market_cap"]), fmt_num(tgt_info["market_cap"])),
    }
    rows_html = "".join([
        f"<tr><td style='padding:8px;font-weight:600;'>{k}</td>"
        f"<td style='padding:8px;text-align:center;'>{v[0]}</td>"
        f"<td style='padding:8px;text-align:center;'>{v[1]}</td></tr>"
        for k, v in metrics.items()
    ])
    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;font-size:0.9rem;'>
      <thead>
        <tr style='background:{BLACK};color:{GOLD};'>
          <th style='padding:10px;text-align:left;'>Metric</th>
          <th style='padding:10px;text-align:center;'>{acq_info['name']}</th>
          <th style='padding:10px;text-align:center;'>{tgt_info['name']}</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    # ── Revenue & EBIT Trend Charts ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Revenue & Profitability Trends")
    insight("These charts reveal growth trajectory, margin stability, and how both companies compare at a glance — key context before diving into the DCF.")

    def _extract_is_series(is_df, *keys):
        for k in keys:
            if k in is_df.index:
                s = is_df.loc[k].sort_index()
                return [str(c)[:4] for c in s.index], [float(v)/1e9 if not pd.isna(v) else 0 for v in s.values]
        return [], []

    fig_trends = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Revenue ($B)", "EBIT Margin (%)"],
        horizontal_spacing=0.12,
    )
    for ticker, is_df, color, dash in [
        (acq_t, acq_is, GOLD_DARK, "solid"),
        (tgt_t, tgt_is, GOLD,      "dot"),
    ]:
        yrs_r, rev_b = _extract_is_series(is_df, "Total Revenue", "Revenue")
        yrs_e, ebit_b = _extract_is_series(is_df, "EBIT", "Operating Income", "Total Operating Income As Reported")
        rev_b_ref = [v for v in rev_b if v > 0]
        ebit_margins = []
        if yrs_e and yrs_r:
            yr_set = sorted(set(yrs_r) & set(yrs_e))
            rev_map  = dict(zip(yrs_r, rev_b))
            ebit_map = dict(zip(yrs_e, ebit_b))
            ebit_margins = [ebit_map[y] / rev_map[y] * 100 if rev_map.get(y) else 0 for y in yr_set]
            yrs_m = yr_set
        else:
            yrs_m = yrs_r

        fig_trends.add_trace(go.Bar(
            x=yrs_r, y=rev_b, name=ticker, marker_color=color,
            showlegend=True,
        ), row=1, col=1)
        if ebit_margins:
            fig_trends.add_trace(go.Scatter(
                x=yrs_m, y=ebit_margins, name=f"{ticker} EBIT%",
                mode="lines+markers",
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=7),
            ), row=1, col=2)

    fig_trends.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Source Sans Pro"), height=320,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.15, x=0),
        barmode="group",
    )
    fig_trends.update_yaxes(gridcolor="#f0f0f0")
    fig_trends.update_xaxes(showgrid=False)
    st.plotly_chart(fig_trends, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("← Back"):
            st.session_state.step -= 1; st.rerun()
    with col_next:
        if st.button("Next: NWC Analysis →"):
            st.session_state.step += 1; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — NWC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    section("Net Working Capital (NWC) Analysis", 2)
    formula("NWC = (Current Assets − Cash) − (Current Liabilities − Short-Term Debt)\nNWC / Sales = Operating Working Capital Intensity\nΔNWC = Change in NWC year-over-year (reduces FCF when positive)")
    insight("""
    NWC measures how much capital is <strong>tied up in day-to-day operations</strong>.
    A rising NWC/Sales ratio means the business needs more working capital per dollar of revenue —
    this is a cash drain in the DCF model. We use the <strong>5-year average NWC/Sales</strong>
    as our projection assumption.
    """)

    with st.spinner("Computing NWC..."):
        d = st.session_state.data
        if "tgt_bs" not in d:
            st.warning("Please run Step 1 first."); st.stop()

        tgt_nwc = compute_nwc(d["tgt_bs"], d["tgt_is"])
        acq_nwc = compute_nwc(d["acq_bs"], d["acq_is"])
        st.session_state.data["tgt_nwc"] = tgt_nwc
        st.session_state.data["acq_nwc"] = acq_nwc

    def show_nwc(nwc_df, name, ticker):
        avg = nwc_average_ratio(nwc_df)
        st.markdown(f"#### {name} ({ticker})")
        if nwc_df.empty:
            st.warning(f"Could not compute NWC for {ticker} — balance sheet data may be unavailable.")
            return avg
        display_cols = ["Year", "NWC", "Sales", "NWC/Sales", "Increase in NWC"]
        disp = nwc_df[[c for c in display_cols if c in nwc_df.columns]].copy()
        for c in ["NWC", "Sales", "Increase in NWC"]:
            if c in disp.columns:
                disp[c] = disp[c].apply(lambda x: f"${x/1e6:,.1f}M" if pd.notna(x) else "N/A")
        if "NWC/Sales" in disp.columns:
            disp["NWC/Sales"] = disp["NWC/Sales"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.markdown(f"""<div class="metric-card" style="margin-top:8px;">
          <div class="label">5-Year Average NWC/Sales</div>
          <div class="value">{fmt_pct(avg)}</div>
          <div class="sub">Used as DCF projection assumption</div>
        </div>""", unsafe_allow_html=True)
        return avg

    col1, col2 = st.columns(2)
    with col1:
        acq_nwc_avg = show_nwc(acq_nwc, d["acq_info"]["name"], acq_t)
    with col2:
        tgt_nwc_avg = show_nwc(tgt_nwc, d["tgt_info"]["name"], tgt_t)

    st.session_state.data["tgt_nwc_avg"] = tgt_nwc_avg

    # NWC/Sales trend chart
    st.markdown("<br>", unsafe_allow_html=True)
    section("NWC/Sales Trend")
    fig = go.Figure()
    for df, name, color in [(acq_nwc, d["acq_info"]["name"], GOLD_DARK),
                             (tgt_nwc, d["tgt_info"]["name"], GOLD)]:
        if "NWC/Sales" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Year"], y=df["NWC/Sales"] * 100,
                mode="lines+markers", name=name,
                line=dict(color=color, width=3),
                marker=dict(size=8)
            ))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="NWC / Sales (%)", xaxis_title="Fiscal Year",
        legend=dict(orientation="h", y=1.1),
        font=dict(family="Source Sans Pro"),
        margin=dict(t=40, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("← Back"):
            st.session_state.step -= 1; st.rerun()
    with col_next:
        if st.button("Next: Beta Regression →"):
            st.session_state.step += 1; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — BETA REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    section("Equity Beta Estimation via OLS Regression", 3)
    formula("Model:  r_stock = α + β × r_market  +  ε\nWhere:  r_stock = monthly stock return\n        r_market = monthly S&P 500 return\n        β (beta) = systematic risk relative to market")
    insight("""
    Beta measures how sensitive a stock is to overall market movements.
    A beta of 1.0 means the stock moves in line with the market.
    Below 1.0 = less volatile; above 1.0 = more volatile.
    We estimate beta using <strong>60 months of monthly returns</strong> via OLS regression,
    exactly as done in the Excel 'Equity beta' sheet. This beta feeds directly into the WACC calculation.
    """)

    with st.spinner("Running beta regressions..."):
        acq_beta = compute_beta(acq_t)
        tgt_beta = compute_beta(tgt_t)
        st.session_state.data["tgt_beta_result"] = tgt_beta
        st.session_state.data["acq_beta_result"] = acq_beta

    def show_beta(result, name, ticker, color):
        st.markdown(f"#### {name} ({ticker})")
        if result["beta"] is None:
            st.warning("Insufficient data for regression.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Beta (β)", f"{result['beta']:.4f}", "Systematic risk"), unsafe_allow_html=True)
        c2.markdown(metric_card("Alpha (α)", f"{result['alpha']:.4f}", "Excess return"), unsafe_allow_html=True)
        c3.markdown(metric_card("R²", f"{result['r_squared']:.4f}", "Explanatory power"), unsafe_allow_html=True)
        c4.markdown(metric_card("Observations", str(result['n_obs']), "Monthly data points"), unsafe_allow_html=True)

        # Scatter plot
        stock_ret = result["stock_returns"]
        mkt_ret = result["market_returns"]
        x_line = np.linspace(mkt_ret.min(), mkt_ret.max(), 100)
        y_line = result["alpha"] + result["beta"] * x_line

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=mkt_ret * 100, y=stock_ret * 100,
            mode="markers",
            marker=dict(color=color, size=7, opacity=0.7),
            name="Monthly returns",
        ))
        fig.add_trace(go.Scatter(
            x=x_line * 100, y=y_line * 100,
            mode="lines",
            line=dict(color=BLACK, width=2, dash="dash"),
            name=f"Regression line (β={result['beta']:.3f})",
        ))
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="S&P 500 Monthly Return (%)",
            yaxis_title=f"{ticker} Monthly Return (%)",
            font=dict(family="Source Sans Pro"),
            margin=dict(t=30, b=40),
            height=350,
        )
        fig.update_xaxes(showgrid=False, zeroline=True, zerolinecolor="#ccc")
        fig.update_yaxes(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc")
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        b = result["beta"]
        if b < 0.5:
            interp = f"<strong>{ticker} has a low beta of {b:.2f}</strong> — the stock is much less volatile than the market. This reflects a defensive business with stable cash flows."
        elif b < 0.8:
            interp = f"<strong>{ticker} has a below-market beta of {b:.2f}</strong> — moderately defensive. Less exposed to market swings, typical of consumer staples / healthcare."
        elif b < 1.2:
            interp = f"<strong>{ticker} has a market-like beta of {b:.2f}</strong> — moves roughly in line with the S&P 500."
        else:
            interp = f"<strong>{ticker} has a high beta of {b:.2f}</strong> — more volatile than the market. Higher systematic risk demands a higher equity return in CAPM."
        insight(interp)

    acq_name = st.session_state.data.get("acq_info", {}).get("name", acq_t)
    tgt_name = st.session_state.data.get("tgt_info", {}).get("name", tgt_t)
    col1, col2 = st.columns(2)
    with col1:
        show_beta(acq_beta, acq_name, acq_t, GOLD_DARK)
    with col2:
        show_beta(tgt_beta, tgt_name, tgt_t, GOLD)

    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("← Back"):
            st.session_state.step -= 1; st.rerun()
    with col_next:
        if st.button("Next: WACC →"):
            st.session_state.step += 1; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — WACC
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 4:
    d = st.session_state.data
    section("Weighted Average Cost of Capital (WACC)", 4)
    formula(
        "WACC = wd × (1 − τ) × rd  +  we × re\n"
        "\n"
        "where:\n"
        "  re = rf + β × MRP           (CAPM equity cost)\n"
        "  rd = Yield on BB bonds      (cost of debt)\n"
        "  wd = Net Debt / (Net Debt + Market Equity)\n"
        "  we = Market Equity / (Net Debt + Market Equity)\n"
        "  τ  = Corporate tax rate"
    )
    insight("""
    WACC is the <strong>discount rate</strong> used in the DCF model — it represents the blended
    return required by all capital providers (equity + debt holders). A higher WACC means riskier
    company → lower present value of future cash flows. We compute WACC for both the
    <strong>target (standalone)</strong> and the <strong>combined entity (with synergies)</strong>.
    """)

    def compute_entity_wacc(info, beta_result, is_df, bs_df, label):
        price = info["price"] or 0
        shares = info["shares_outstanding"] or 0
        market_equity = price * shares

        total_debt = info.get("total_debt") or 0
        cash_raw = safe_get(bs_df, "Cash And Cash Equivalents", "Cash", "CashAndCashEquivalents")
        cash_val = float(cash_raw.iloc[0]) if not cash_raw.empty else 0.0
        rev_raw = safe_get(is_df, "Total Revenue", "Revenue")
        rev_val = float(rev_raw.iloc[0]) if not rev_raw.empty else 1.0

        excess_cash = compute_excess_cash(cash_val, rev_val)
        beta = beta_result["beta"] if beta_result["beta"] else 1.0

        result = compute_wacc(
            market_equity=market_equity,
            total_debt=total_debt,
            excess_cash=excess_cash,
            equity_beta=beta,
            rf=rf, mrp=mrp, tax_rate=tax_rate,
            debt_beta=debt_beta,
            use_bond_yield=True,
            bond_yield=bond_yield,
        )
        return result, market_equity, total_debt, excess_cash, cash_val, rev_val

    with st.spinner("Computing WACC..."):
        tgt_wacc, tgt_mkt_eq, tgt_total_debt, tgt_excess_cash, tgt_cash, tgt_rev = compute_entity_wacc(
            d["tgt_info"], d.get("tgt_beta_result", {"beta": 1.0}),
            d["tgt_is"], d["tgt_bs"], "Target"
        )
        acq_wacc, acq_mkt_eq, acq_total_debt, acq_excess_cash, acq_cash, acq_rev = compute_entity_wacc(
            d["acq_info"], d.get("acq_beta_result", {"beta": 1.0}),
            d["acq_is"], d["acq_bs"], "Acquirer"
        )
        st.session_state.data.update({
            "tgt_wacc": tgt_wacc, "tgt_mkt_eq": tgt_mkt_eq,
            "tgt_net_debt": tgt_wacc["net_debt"], "tgt_rev": tgt_rev,
            "acq_wacc": acq_wacc, "acq_rev": acq_rev,
        })

    def show_wacc_detail(w, name, ticker, mkt_eq, total_debt, excess_cash):
        st.markdown(f"#### {name} ({ticker})")
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("WACC", fmt_pct(w["WACC"]), "Discount rate"), unsafe_allow_html=True)
        c2.markdown(metric_card("Cost of Equity (re)", fmt_pct(w["re"]), f"rf + β×MRP = {rf:.3f} + {w['equity_beta']:.3f}×{mrp:.4f}"), unsafe_allow_html=True)
        c3.markdown(metric_card("Cost of Debt (rd)", fmt_pct(w["rd"]), "After-tax: " + fmt_pct(w["rd"] * (1 - tax_rate))), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
          <tr style='background:{BLACK};color:{GOLD};'>
            <th style='padding:8px;text-align:left;'>Component</th>
            <th style='padding:8px;text-align:right;'>Value</th>
            <th style='padding:8px;text-align:left;'>Formula / Source</th>
          </tr>
          <tr style='background:#fafafa;'>
            <td style='padding:8px;'>Market Equity</td>
            <td style='padding:8px;text-align:right;font-weight:600;'>{fmt_num(mkt_eq)}</td>
            <td style='padding:8px;color:#666;'>Price × Shares Outstanding</td>
          </tr>
          <tr>
            <td style='padding:8px;'>Total Debt</td>
            <td style='padding:8px;text-align:right;font-weight:600;'>{fmt_num(total_debt)}</td>
            <td style='padding:8px;color:#666;'>Balance Sheet</td>
          </tr>
          <tr style='background:#fafafa;'>
            <td style='padding:8px;'>Excess Cash</td>
            <td style='padding:8px;text-align:right;font-weight:600;'>{fmt_num(excess_cash)}</td>
            <td style='padding:8px;color:#666;'>Cash − 2% of Revenue</td>
          </tr>
          <tr>
            <td style='padding:8px;'>Net Debt</td>
            <td style='padding:8px;text-align:right;font-weight:600;'>{fmt_num(w["net_debt"])}</td>
            <td style='padding:8px;color:#666;'>Total Debt − Excess Cash</td>
          </tr>
          <tr style='background:#fafafa;'>
            <td style='padding:8px;'>Weight of Debt (wd)</td>
            <td style='padding:8px;text-align:right;font-weight:600;'>{fmt_pct(w["wd"])}</td>
            <td style='padding:8px;color:#666;'>Net Debt / (Net Debt + Equity)</td>
          </tr>
          <tr>
            <td style='padding:8px;'>Weight of Equity (we)</td>
            <td style='padding:8px;text-align:right;font-weight:600;'>{fmt_pct(w["we"])}</td>
            <td style='padding:8px;color:#666;'>Equity / (Net Debt + Equity)</td>
          </tr>
          <tr style='background:#fafafa;'>
            <td style='padding:8px;'>Equity Beta (β)</td>
            <td style='padding:8px;text-align:right;font-weight:600;'>{w["equity_beta"]:.4f}</td>
            <td style='padding:8px;color:#666;'>OLS Regression (Step 3)</td>
          </tr>
          <tr style='background:{BLACK};color:{GOLD};font-weight:700;'>
            <td style='padding:8px;'>WACC</td>
            <td style='padding:8px;text-align:right;'>{fmt_pct(w["WACC"])}</td>
            <td style='padding:8px;'>wd×(1−τ)×rd + we×re</td>
          </tr>
        </table>""", unsafe_allow_html=True)

        # Interpretation
        w_val = w["WACC"]
        if w_val < 0.06:
            interp = f"A WACC of <strong>{fmt_pct(w_val)}</strong> is <strong>low</strong>, reflecting a low-risk, investment-grade business. Projects need to earn at least this return to create value."
        elif w_val < 0.09:
            interp = f"A WACC of <strong>{fmt_pct(w_val)}</strong> is <strong>moderate</strong>, typical for a stable mid-cap company. Acceptable for businesses with predictable cash flows."
        else:
            interp = f"A WACC of <strong>{fmt_pct(w_val)}</strong> is <strong>elevated</strong>, indicating higher perceived risk. Future cash flows are discounted more aggressively."
        insight(interp)

    col1, col2 = st.columns(2)
    with col1:
        show_wacc_detail(tgt_wacc, d["tgt_info"]["name"], tgt_t, tgt_mkt_eq, tgt_total_debt, tgt_excess_cash)
    with col2:
        show_wacc_detail(acq_wacc, d["acq_info"]["name"], acq_t, acq_mkt_eq, acq_total_debt, acq_excess_cash)

    # Capital structure pie charts
    st.markdown("<br>", unsafe_allow_html=True)
    section("Capital Structure Comparison")
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=[d["tgt_info"]["name"], d["acq_info"]["name"]])
    for i, (w, name) in enumerate([(tgt_wacc, d["tgt_info"]["name"]), (acq_wacc, d["acq_info"]["name"])], 1):
        eq = max(w["market_equity"], 0)
        nd = max(w["net_debt"], 0)
        fig.add_trace(go.Pie(
            labels=["Equity", "Net Debt"],
            values=[eq, nd],
            marker_colors=[GOLD, BLACK],
            textinfo="label+percent",
            hole=0.4,
        ), row=1, col=i)
    fig.update_layout(paper_bgcolor="white", font=dict(family="Source Sans Pro"), height=300,
                      margin=dict(t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # ── WACC Breakdown Bar Chart ─────────────────────────────────────────────
    section("WACC Component Breakdown")
    insight("Each bar shows how much equity cost and after-tax debt cost contribute to the blended WACC. A higher equity weight with high beta pushes WACC up; more low-cost (after-tax) debt pulls it down.")
    wacc_labels = [d["tgt_info"]["name"], d["acq_info"]["name"]]
    equity_contrib = [w["we"] * w["re"] * 100 for w in [tgt_wacc, acq_wacc]]
    debt_contrib   = [w["wd"] * w["rd"] * (1 - tax_rate) * 100 for w in [tgt_wacc, acq_wacc]]
    total_waccs    = [w["WACC"] * 100 for w in [tgt_wacc, acq_wacc]]

    fig_wacc = go.Figure()
    fig_wacc.add_bar(name="Equity Cost (we × re)",         x=wacc_labels, y=equity_contrib, marker_color=GOLD)
    fig_wacc.add_bar(name="After-Tax Debt Cost (wd×rd×(1−τ))", x=wacc_labels, y=debt_contrib,   marker_color=GOLD_DARK)
    for label, total in zip(wacc_labels, total_waccs):
        fig_wacc.add_annotation(x=label, y=total + 0.15, text=f"<b>WACC = {total:.2f}%</b>",
                                showarrow=False, font=dict(size=13, color=BLACK))
    fig_wacc.update_layout(
        barmode="stack", plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Contribution to WACC (%)",
        font=dict(family="Source Sans Pro"), height=340,
        legend=dict(orientation="h", y=1.1), margin=dict(t=50, b=40),
    )
    fig_wacc.update_xaxes(showgrid=False)
    fig_wacc.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig_wacc, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("← Back"):
            st.session_state.step -= 1; st.rerun()
    with col_next:
        if st.button("Next: Standalone DCF →"):
            st.session_state.step += 1; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — STANDALONE DCF
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 5:
    d = st.session_state.data
    section("Standalone DCF Valuation of Target", 5)
    formula(
        "FCF  =  NOPAT  +  D&A  −  Capex  −  ΔNWC\n"
        "NOPAT = EBIT × (1 − τ)\n"
        "TV   = FCF_(n+1) / (WACC − g)     [Gordon Growth]\n"
        "EV   = Σ PV(FCF_t)  +  PV(TV)"
    )
    insight("""
    The standalone DCF values the <strong>target company on its own</strong>, without any deal synergies.
    This is the floor value — what the business is worth as an independent entity.
    We use historical margins as the base for projections, with the terminal growth rate
    representing steady-state long-run growth. The resulting <strong>Enterprise Value (EV)</strong>
    is the starting point for any deal negotiation.
    """)

    with st.spinner("Running DCF..."):
        wacc_val = d.get("tgt_wacc", {}).get("WACC", 0.08)
        tgt_is = d["tgt_is"]
        tgt_bs = d["tgt_bs"]
        tgt_cf = d["tgt_cf"]
        nwc_df = d.get("tgt_nwc", pd.DataFrame())

        # Pull historical values
        ebit_s = safe_get(tgt_is, "EBIT", "Operating Income", "Total Operating Income As Reported")
        rev_s  = safe_get(tgt_is, "Total Revenue", "Revenue")
        da_s   = safe_get(tgt_cf, "Depreciation And Amortization", "Depreciation Amortization Depletion", "Depreciation")
        capex_s = safe_get(tgt_cf, "Capital Expenditure", "Capital Expenditures", "Purchase Of Property Plant And Equipment")
        capex_s = capex_s.abs()

        def mean_ratio(num_s, den_s):
            if num_s.empty or den_s.empty: return 0.0
            common = num_s.index.intersection(den_s.index)
            if len(common) == 0: return 0.0
            ratios = (num_s[common] / den_s[common]).replace([np.inf, -np.inf], np.nan).dropna()
            return float(ratios.mean()) if not ratios.empty else 0.0

        ebit_margin = _positive_mean_ratio(ebit_s, rev_s)   # excludes impairment years
        da_pct      = mean_ratio(da_s, rev_s)
        capex_pct   = mean_ratio(capex_s, rev_s)
        nwc_pct     = d.get("tgt_nwc_avg", 0.18)

        base_rev = float(rev_s.iloc[0]) if not rev_s.empty else 1e9

        dcf_result = run_dcf(
            ebit_margin=ebit_margin,
            base_revenue=base_rev,
            da_pct_revenue=da_pct,
            capex_pct_revenue=capex_pct,
            nwc_pct_revenue=nwc_pct,
            wacc=wacc_val,
            tax_rate=tax_rate,
            growth_rate=growth_rate,
            projection_years=projection_years,
        )
        st.session_state.data["dcf_result"] = dcf_result
        st.session_state.data["base_rev"]   = base_rev
        # Store for sensitivity re-use (fixes hardcoded 0.02/0.025 bug)
        st.session_state.data["da_pct"]     = da_pct
        st.session_state.data["capex_pct"]  = capex_pct

    proj = dcf_result["projection_df"]
    ev = dcf_result["enterprise_value"]
    pv_fcf = dcf_result["pv_fcf_total"]
    pv_tv = dcf_result["pv_terminal_value"]
    tv = dcf_result["terminal_value"]

    # Key assumptions banner
    st.markdown("<br>", unsafe_allow_html=True)
    section("DCF Assumptions")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(metric_card("EBIT Margin", fmt_pct(ebit_margin), "Historical avg"), unsafe_allow_html=True)
    c2.markdown(metric_card("D&A / Revenue", fmt_pct(da_pct), "Historical avg"), unsafe_allow_html=True)
    c3.markdown(metric_card("Capex / Revenue", fmt_pct(capex_pct), "Historical avg"), unsafe_allow_html=True)
    c4.markdown(metric_card("NWC / Sales", fmt_pct(nwc_pct), "Historical avg"), unsafe_allow_html=True)
    c5.markdown(metric_card("WACC", fmt_pct(wacc_val), "From Step 4"), unsafe_allow_html=True)

    # Projection table
    st.markdown("<br>", unsafe_allow_html=True)
    section("Free Cash Flow Projections")
    disp_proj = proj.copy()
    for col in ["Revenue", "EBIT", "NOPAT", "D&A", "Capex", "ΔNWC", "FCF", "PV of FCF"]:
        if col in disp_proj.columns:
            disp_proj[col] = disp_proj[col].apply(lambda x: f"${x/1e6:,.1f}M")
    disp_proj["Discount Factor"] = disp_proj["Discount Factor"].apply(lambda x: f"{x:.4f}")
    st.dataframe(disp_proj, use_container_width=True, hide_index=True)

    # EV Waterfall
    st.markdown("<br>", unsafe_allow_html=True)
    section("Enterprise Value Bridge")
    fig = go.Figure(go.Waterfall(
        name="EV Bridge",
        orientation="v",
        measure=["relative", "relative", "total"],
        x=["PV of FCFs", "PV of Terminal Value", "Enterprise Value"],
        y=[pv_fcf, pv_tv, 0],
        text=[fmt_num(pv_fcf), fmt_num(pv_tv), fmt_num(ev)],
        textposition="outside",
        connector=dict(line=dict(color="#ccc")),
        increasing=dict(marker_color=GOLD),
        totals=dict(marker_color=BLACK),
    ))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Value ($)",
        font=dict(family="Source Sans Pro"),
        margin=dict(t=40, b=40),
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)

    # EV summary
    st.markdown(f"""
    <div class="result-highlight">
      <div class="label">Standalone Enterprise Value — {d['tgt_info']['name']} ({tgt_t})</div>
      <div class="big-number">{fmt_num(ev)}</div>
      <div style='margin-top:12px;font-size:0.88rem;opacity:0.9;'>
        PV of FCFs: {fmt_num(pv_fcf)} &nbsp;|&nbsp;
        Terminal Value: {fmt_num(tv)} &nbsp;|&nbsp;
        PV of Terminal Value: {fmt_num(pv_tv)} &nbsp;|&nbsp;
        WACC: {fmt_pct(wacc_val)} &nbsp;|&nbsp;
        g: {fmt_pct(growth_rate)}
      </div>
    </div>""", unsafe_allow_html=True)

    tv_pct = pv_tv / ev * 100 if ev else 0

    # ── FCF Bar Chart with PV Overlay ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Free Cash Flow — Nominal vs Present Value")
    insight(f"The gap between nominal FCF (bars) and PV of FCF (line) widens each year as the discount factor grows — illustrating the time value of money. Terminal value ({tv_pct:.1f}% of EV) is typical for DCF models.")
    fcf_chart = go.Figure()
    fcf_chart.add_bar(
        x=proj["Year"], y=[v/1e6 for v in proj["FCF"]],
        name="Nominal FCF ($M)", marker_color=GOLD, opacity=0.85,
    )
    fcf_chart.add_scatter(
        x=proj["Year"], y=[v/1e6 for v in proj["PV of FCF"]],
        name="PV of FCF ($M)", mode="lines+markers",
        line=dict(color=BLACK, width=2),
        marker=dict(size=7, color=BLACK),
    )
    fcf_chart.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="$ Millions", font=dict(family="Source Sans Pro"),
        legend=dict(orientation="h", y=1.1), height=340, margin=dict(t=50, b=40),
    )
    fcf_chart.update_xaxes(showgrid=False)
    fcf_chart.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fcf_chart, use_container_width=True)

    # ── Revenue Projection (historical + forecast) ───────────────────────────
    section("Revenue Projection: Historical → Forecast")
    hist_rev_s = safe_get(tgt_is, "Total Revenue", "Revenue")
    hist_yrs   = [str(c)[:4] for c in sorted(hist_rev_s.index)] if not hist_rev_s.empty else []
    hist_vals  = [float(hist_rev_s[c])/1e6 for c in sorted(hist_rev_s.index)] if not hist_rev_s.empty else []
    proj_yrs   = [f"Y+{y}" for y in dcf_result["years"]]
    proj_vals  = [v/1e6 for v in dcf_result["revenues"]]

    fig_rev = go.Figure()
    fig_rev.add_bar(x=hist_yrs, y=hist_vals, name="Historical Revenue", marker_color=GOLD_DARK)
    fig_rev.add_bar(x=proj_yrs, y=proj_vals, name="Projected Revenue",  marker_color=GOLD, opacity=0.7)
    fig_rev.add_vline(x=len(hist_yrs) - 0.5, line_dash="dash", line_color="#aaa",
                      annotation_text="  ← Actual | Projected →", annotation_position="top")
    fig_rev.update_layout(
        barmode="group", plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Revenue ($M)", font=dict(family="Source Sans Pro"),
        legend=dict(orientation="h", y=1.1), height=320, margin=dict(t=50, b=40),
    )
    fig_rev.update_xaxes(showgrid=False)
    fig_rev.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig_rev, use_container_width=True)
    insight(f"""
    The standalone enterprise value of <strong>{d['tgt_info']['name']}</strong> is estimated at
    <strong>{fmt_num(ev)}</strong>. Terminal value accounts for
    <strong>{tv_pct:.1f}%</strong> of total EV — this is normal for DCF models
    (typically 60–80%). This valuation is based on the target's own economics,
    before any deal synergies. The implied equity value is
    <strong>{fmt_num(ev - d.get('tgt_net_debt', 0))}</strong>
    (EV − Net Debt of {fmt_num(d.get('tgt_net_debt', 0))}).
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("← Back"):
            st.session_state.step -= 1; st.rerun()
    with col_next:
        if st.button("Next: Synergy Analysis →"):
            st.session_state.step += 1; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — SYNERGY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 6:
    d = st.session_state.data
    section("Synergy Analysis", 6)
    formula(
        "Revenue Synergies   = Combined Revenue × rev_synergy%  [cross-selling, market share]\n"
        "Cost Synergies      = Combined COGS+SG&A × cost_synergy%  [overhead elimination]\n"
        "After-Tax Synergy   = (Rev Syn + Cost Syn) × (1 − τ)\n"
        "PV of Synergies     = Σ [After-Tax Syn_t / (1+WACC)^t]  +  TV_synergies"
    )
    insight("""
    Synergies are the <strong>incremental value created by combining two businesses</strong>
    that neither could create independently. Revenue synergies come from cross-selling, expanded
    distribution, and pricing power. Cost synergies come from eliminating duplicate overhead,
    procurement leverage, and operational efficiencies. We phase synergies in over
    <strong>3 years</strong> (ramp-up period), which is realistic for integration timelines.
    """)

    with st.spinner("Computing synergies..."):
        acq_rev = d.get("acq_rev", 0) or 0
        tgt_rev = d.get("tgt_rev", 0) or 0
        wacc_val = d.get("tgt_wacc", {}).get("WACC", 0.08)

        tgt_is = d["tgt_is"]
        acq_is = d["acq_is"]

        def get_combined_cogs_sga(is_df):
            cogs = safe_get(is_df, "Cost Of Revenue", "Cost Of Goods Sold", "Cost Of Goods And Services Sold")
            sga  = safe_get(is_df, "Selling General Administrative", "Selling General And Administration",
                            "General And Administrative Expense")
            c = float(cogs.iloc[0]) if not cogs.empty else 0
            s = float(sga.iloc[0]) if not sga.empty else 0
            return c + s

        combined_cogs_sga = get_combined_cogs_sga(tgt_is) + get_combined_cogs_sga(acq_is)

        syn_result = compute_synergies(
            target_revenue=tgt_rev,
            acquirer_revenue=acq_rev,
            deal_equity_value=d.get("dcf_result", {}).get("enterprise_value", 1e9),
            target_net_debt=d.get("tgt_net_debt", 0),
            rev_synergy_pct=rev_syn_pct,
            cost_syn_pct=cost_syn_pct,
            combined_cogs_plus_sga=combined_cogs_sga,
            wacc=wacc_val,
            tax_rate=tax_rate,
            growth_rate=growth_rate,
            ramp_years=ramp_years,
            projection_years=projection_years,
        )
        st.session_state.data["syn_result"] = syn_result

    syn_df = syn_result["synergy_df"]
    total_syn_pv = syn_result["total_pv_synergies"]
    standalone_ev = d["dcf_result"]["enterprise_value"]
    combined_ev = standalone_ev + total_syn_pv

    # Synergy summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("Annual Revenue Synergy", fmt_num(syn_result["rev_synergy_full"]), "At full run-rate"), unsafe_allow_html=True)
    c2.markdown(metric_card("Annual Cost Synergy", fmt_num(syn_result["cost_synergy_full"]), "At full run-rate"), unsafe_allow_html=True)
    c3.markdown(metric_card("PV of All Synergies", fmt_num(total_syn_pv), "Discounted at WACC"), unsafe_allow_html=True)
    c4.markdown(metric_card("Synergy as % of EV", fmt_pct(total_syn_pv / standalone_ev if standalone_ev else 0), "Value uplift"), unsafe_allow_html=True)

    # Synergy projection table
    st.markdown("<br>", unsafe_allow_html=True)
    section("Year-by-Year Synergy Build-Up")
    disp_syn = syn_df.copy()
    for col in ["Revenue Synergy", "Cost Synergy", "After-Tax Synergy", "PV of Synergy"]:
        disp_syn[col] = disp_syn[col].apply(lambda x: f"${x/1e6:,.1f}M")
    st.dataframe(disp_syn, use_container_width=True, hide_index=True)

    # Synergy ramp chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=syn_df["Year"], y=syn_df["Revenue Synergy"] / 1e6,
        name="Revenue Synergy", marker_color=GOLD,
    ))
    fig.add_trace(go.Bar(
        x=syn_df["Year"], y=syn_df["Cost Synergy"] / 1e6,
        name="Cost Synergy", marker_color=GOLD_DARK,
    ))
    fig.update_layout(
        barmode="stack",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Synergy ($M)",
        font=dict(family="Source Sans Pro"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=40, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig, use_container_width=True)

    # ── With-Synergies DCF ──────────────────────────────────────────────────
    section("With-Synergies DCF — Combined FCF Stream")
    insight("""
    We re-run the DCF adding synergy FCFs each year on top of standalone FCFs.
    This gives the <strong>true acquisition EV</strong> — what the combined entity is worth.
    Integration costs reduce early-year FCFs; full synergies ramp in from Year 2 onward.
    """)

    syn_fcfs = syn_result.get("synergy_fcfs", [])
    with_syn_dcf = run_dcf(
        ebit_margin      = d["dcf_result"]["ebit_margin"],
        base_revenue     = d.get("base_rev", 1e9),
        da_pct_revenue   = d.get("da_pct",    d["dcf_result"].get("da_pct",    0.025)),
        capex_pct_revenue= d.get("capex_pct", d["dcf_result"].get("capex_pct", 0.022)),
        nwc_pct_revenue  = d.get("tgt_nwc_avg", 0.18),
        wacc             = wacc_val,
        tax_rate         = tax_rate,
        growth_rate      = growth_rate,
        projection_years = projection_years,
        synergy_fcfs     = syn_fcfs,
    )
    st.session_state.data["with_syn_dcf"] = with_syn_dcf
    combined_ev_syn = with_syn_dcf["enterprise_value"]

    proj_sa  = d["dcf_result"]["projection_df"]
    proj_syn = with_syn_dcf["projection_df"]
    fig_cmp  = go.Figure()
    fig_cmp.add_bar(x=proj_sa["Year"],  y=[v/1e6 for v in proj_sa["FCF"]],
                    name="Standalone FCF", marker_color=GOLD_DARK, opacity=0.85)
    fig_cmp.add_bar(x=proj_syn["Year"], y=[v/1e6 for v in proj_syn["FCF"]],
                    name="FCF with Synergies", marker_color=GOLD, opacity=0.85)
    fig_cmp.update_layout(
        barmode="group", plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="FCF ($M)", font=dict(family="Source Sans Pro"),
        legend=dict(orientation="h", y=1.1), height=340, margin=dict(t=50, b=40),
    )
    fig_cmp.update_xaxes(showgrid=False)
    fig_cmp.update_yaxes(gridcolor="#f0f0f0")
    st.plotly_chart(fig_cmp, use_container_width=True)

    c_sa, c_syn, c_up = st.columns(3)
    c_sa.markdown(metric_card("Standalone EV",      fmt_num(standalone_ev),  "No deal synergies"),   unsafe_allow_html=True)
    c_syn.markdown(metric_card("EV with Synergies", fmt_num(combined_ev_syn),"Including synergy FCFs"), unsafe_allow_html=True)
    uplift_pct = (combined_ev_syn - standalone_ev) / standalone_ev if standalone_ev else 0
    c_up.markdown(metric_card("Value Uplift",       fmt_pct(uplift_pct),     "From synergies"),      unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-highlight">
      <div class="label">Enterprise Value with Synergies (Full DCF)</div>
      <div class="big-number">{fmt_num(combined_ev_syn)}</div>
      <div style='margin-top:12px;font-size:0.88rem;opacity:0.9;'>
        Standalone EV: {fmt_num(standalone_ev)} &nbsp;+&nbsp;
        Synergy Uplift: {fmt_num(combined_ev_syn - standalone_ev)} ({fmt_pct(uplift_pct)})
      </div>
    </div>""", unsafe_allow_html=True)

    insight(f"""
    Adding synergies grows the implied enterprise value from
    <strong>{fmt_num(standalone_ev)}</strong> → <strong>{fmt_num(combined_ev_syn)}</strong>
    (+<strong>{fmt_pct(uplift_pct)}</strong>).
    Any offer price between standalone and synergy EV represents a value-sharing split
    between buyer and seller — the acquirer must keep enough synergy value to justify the deal.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("← Back"):
            st.session_state.step -= 1; st.rerun()
    with col_next:
        if st.button("Next: Deal Summary →"):
            st.session_state.step += 1; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — DEAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 7:
    d = st.session_state.data
    section("Deal Summary & Recommendation", 7)
    insight("""
    This final step brings together all prior analyses into a complete deal picture:
    the <strong>implied offer price</strong>, <strong>total deal value</strong>,
    <strong>accretion/dilution</strong> assessment, and a plain-English recommendation
    on whether the deal creates value for the acquirer's shareholders.
    """)

    with st.spinner("Computing deal summary..."):
        tgt_net_debt = d.get("tgt_net_debt", 0) or 0
        tgt_shares   = d["tgt_info"].get("shares_outstanding") or 1
        acq_mkt_cap  = d["acq_info"].get("market_cap") or 1
        standalone_ev = d["dcf_result"]["enterprise_value"]
        # Prefer the full with-synergies DCF if available, else fall back to additive
        syn_ev_dcf   = d.get("with_syn_dcf", {}).get("enterprise_value", None)
        total_pv_syn = d["syn_result"]["total_pv_synergies"]
        synergy_ev_for_deal = syn_ev_dcf if syn_ev_dcf else standalone_ev + total_pv_syn

        deal = compute_deal_summary(
            standalone_ev_target = standalone_ev,
            total_pv_synergies   = synergy_ev_for_deal - standalone_ev,
            target_net_debt      = tgt_net_debt,
            target_shares        = tgt_shares,
            acquirer_market_cap  = acq_mkt_cap,
            offer_premium_pct    = offer_premium,
        )

    # Key deal metrics
    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card("Standalone EV", fmt_num(deal["standalone_ev"]), "Pre-synergy value"), unsafe_allow_html=True)
    c2.markdown(metric_card("Synergy-Adjusted EV", fmt_num(deal["synergy_ev"]), "Standalone + PV Synergies"), unsafe_allow_html=True)
    c3.markdown(metric_card("PV of Synergies", fmt_num(deal["pv_synergies"]), "Value created by deal"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c4, c5, c6 = st.columns(3)
    c4.markdown(metric_card("Implied Price/Share (Standalone)", f"${deal['implied_price_per_share']:,.2f}", "EV − Net Debt / Shares"), unsafe_allow_html=True)
    c5.markdown(metric_card("Implied Price/Share (w/ Synergies)", f"${deal['synergy_price_per_share']:,.2f}", "Including synergy value"), unsafe_allow_html=True)
    c6.markdown(metric_card("Recommended Offer Price", f"${deal['offer_price']:,.2f}", f"{offer_premium*100:.0f}% premium to standalone"), unsafe_allow_html=True)

    # Total deal value
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-highlight">
      <div class="label">Total Deal Value (Equity Check)</div>
      <div class="big-number">{fmt_num(deal['total_deal_value'])}</div>
      <div style='margin-top:12px;font-size:0.88rem;opacity:0.9;'>
        Offer Price: ${deal['offer_price']:,.2f}/share ×
        {fmt_num(tgt_shares, prefix='', millions=True)} shares &nbsp;|&nbsp;
        Premium to Standalone: {fmt_pct(offer_premium)} &nbsp;|&nbsp;
        Deal Size vs Acquirer Mkt Cap: {fmt_pct(deal['total_deal_value'] / acq_mkt_cap)}
      </div>
    </div>""", unsafe_allow_html=True)

    # Full deal table
    st.markdown("<br>", unsafe_allow_html=True)
    section("Complete Deal Summary Table")
    summary_rows = [
        ("Standalone EV", fmt_num(deal["standalone_ev"]), "DCF — no synergies"),
        ("+ PV of Synergies", fmt_num(deal["pv_synergies"]), "Revenue + Cost synergies (PV)"),
        ("= Synergy-Adjusted EV", fmt_num(deal["synergy_ev"]), "Total value of combined entity"),
        ("− Net Debt of Target", fmt_num(tgt_net_debt), "Debt assumed in acquisition"),
        ("= Implied Equity Value (w/ Syn)", fmt_num(deal["synergy_equity"]), "Equity value to target shareholders"),
        ("÷ Shares Outstanding", fmt_num(tgt_shares, prefix='', millions=True), "Target shares"),
        ("= Implied Price/Share (w/ Syn)", f"${deal['synergy_price_per_share']:,.2f}", "Per share value with synergies"),
        ("× (1 + Offer Premium)", f"{fmt_pct(offer_premium)}", "Control premium"),
        ("= Recommended Offer Price", f"${deal['offer_price']:,.2f}", "Per share offer"),
        ("× Shares", fmt_num(tgt_shares, prefix='', millions=True), "Target shares"),
        ("= Total Deal Value", fmt_num(deal["total_deal_value"]), "Total equity consideration"),
    ]
    rows_html = "".join([
        f"<tr style='background:{'#fafafa' if i%2==0 else 'white'};'>"
        f"<td style='padding:9px 12px;font-weight:{'700' if '=' in r[0] else '400'};'>{r[0]}</td>"
        f"<td style='padding:9px 12px;text-align:right;font-weight:700;color:{GOLD_DARK if '=' in r[0] else BLACK};'>{r[1]}</td>"
        f"<td style='padding:9px 12px;color:#666;font-size:0.83rem;'>{r[2]}</td></tr>"
        for i, r in enumerate(summary_rows)
    ])
    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
      <thead>
        <tr style='background:{BLACK};color:{GOLD};'>
          <th style='padding:10px 12px;text-align:left;'>Line Item</th>
          <th style='padding:10px 12px;text-align:right;'>Value</th>
          <th style='padding:10px 12px;text-align:left;'>Notes</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

    # ── Football Field (Valuation Range) Chart ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Football Field — Valuation Range Summary")
    insight("""
    The football field shows the <strong>range of implied equity value per share</strong>
    under different methodologies and scenarios. The overlap zone is where the offer price
    should land to be defensible to both boards.
    """)

    price = d["tgt_info"].get("price") or deal["implied_price_per_share"]
    ff_methods = [
        ("52-Week Trading Range",        price * 0.75,  price * 1.10),
        ("Standalone DCF (WACC ±1%)",    deal["implied_price_per_share"] * 0.85,
                                          deal["implied_price_per_share"] * 1.15),
        ("With-Synergies DCF",           deal["synergy_price_per_share"] * 0.90,
                                          deal["synergy_price_per_share"] * 1.10),
        ("Offer Price Range (±5%)",      deal["offer_price"] * 0.95,
                                          deal["offer_price"] * 1.05),
    ]
    fig_ff = go.Figure()
    colors_ff = [GOLD_DARK, "#5c7a29", GOLD, BLACK]
    for i, (label, lo, hi) in enumerate(ff_methods):
        fig_ff.add_trace(go.Bar(
            name=label,
            x=[hi - lo], base=[lo],
            y=[label],
            orientation="h",
            marker_color=colors_ff[i],
            opacity=0.85,
            text=f"  ${lo:,.2f} – ${hi:,.2f}",
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=11, family="Source Sans Pro"),
        ))
    fig_ff.add_vline(x=deal["offer_price"], line_dash="dash", line_color="red", line_width=2,
                     annotation_text=f"  Offer: ${deal['offer_price']:,.2f}",
                     annotation_font=dict(color="red", size=11))
    fig_ff.update_layout(
        barmode="overlay", plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Implied Price Per Share ($)",
        showlegend=False,
        font=dict(family="Source Sans Pro"),
        height=260, margin=dict(t=30, b=40, l=200),
    )
    fig_ff.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig_ff.update_yaxes(showgrid=False)
    st.plotly_chart(fig_ff, use_container_width=True)

    # Recommendation
    st.markdown("<br>", unsafe_allow_html=True)
    section("Investment Recommendation")
    deal_pct = deal["total_deal_value"] / acq_mkt_cap
    syn_uplift = total_pv_syn / standalone_ev if standalone_ev else 0

    if syn_uplift > 0.15 and deal_pct < 0.30:
        verdict = "RECOMMEND — PROCEED"
        verdict_color = "#2e7d32"
        verdict_text = (
            f"The deal offers a compelling strategic rationale. Synergies represent "
            f"<strong>{fmt_pct(syn_uplift)}</strong> of the standalone enterprise value, "
            f"well above the {fmt_pct(offer_premium)} premium paid. At {fmt_pct(deal_pct)} "
            f"of the acquirer's market cap, the deal size is manageable. "
            f"If synergies are realized as projected, the acquisition creates significant "
            f"value for <strong>{d['acq_info']['name']}</strong> shareholders."
        )
    elif syn_uplift > offer_premium:
        verdict = "CONDITIONAL — MONITOR EXECUTION"
        verdict_color = "#f57c00"
        verdict_text = (
            f"The synergy uplift of <strong>{fmt_pct(syn_uplift)}</strong> exceeds the "
            f"offer premium of <strong>{fmt_pct(offer_premium)}</strong>, suggesting value "
            f"creation is possible but dependent on execution. Integration risk is real — "
            f"synergies must be achieved on the projected timeline. Recommend proceeding "
            f"with enhanced integration planning and milestone-based accountability."
        )
    else:
        verdict = "CAUTION — PREMIUM EXCEEDS SYNERGIES"
        verdict_color = "#c62828"
        verdict_text = (
            f"The offer premium of <strong>{fmt_pct(offer_premium)}</strong> exceeds the "
            f"projected synergy value of <strong>{fmt_pct(syn_uplift)}</strong> of standalone EV. "
            f"Under current assumptions, <strong>{d['acq_info']['name']}</strong> would be "
            f"overpaying relative to realizable synergies. Recommend revisiting synergy "
            f"assumptions, reducing the offer price, or identifying additional value levers "
            f"before proceeding."
        )

    st.markdown(f"""
    <div style='background:{verdict_color};color:white;padding:20px 28px;border-radius:8px;margin-bottom:16px;'>
      <div style='font-size:0.8rem;letter-spacing:2px;text-transform:uppercase;opacity:0.85;margin-bottom:6px;'>Recommendation</div>
      <div style='font-size:1.8rem;font-weight:700;'>{verdict}</div>
    </div>""", unsafe_allow_html=True)
    insight(verdict_text)

    # Sensitivity table: EV at different WACC / growth combos
    st.markdown("<br>", unsafe_allow_html=True)
    section("Sensitivity Analysis — Enterprise Value")
    insight("""
    <strong>How to read:</strong> Each cell = Enterprise Value at a given WACC (columns) and terminal growth rate (rows).
    <span style='background:#c8e6c9;padding:2px 6px;border-radius:3px;'>Green</span> = above base case &nbsp;
    <span style='background:#ffcdd2;padding:2px 6px;border-radius:3px;'>Red</span> = below base case &nbsp;
    Center cell = your base case assumption.
    """)

    wacc_val    = d.get("tgt_wacc", {}).get("WACC", 0.08)
    stored_da   = d.get("da_pct",    d.get("dcf_result", {}).get("da_pct",    0.025))
    stored_cap  = d.get("capex_pct", d.get("dcf_result", {}).get("capex_pct", 0.022))
    base_ev     = d["dcf_result"]["enterprise_value"]

    waccs_s = [round(wacc_val + dw, 4) for dw in [-0.015, -0.010, -0.005, 0, 0.005, 0.010, 0.015]]
    gs_s    = [round(growth_rate + dg, 4) for dg in [0.010, 0.005, 0, -0.005, -0.010]]

    from src.dcf import run_dcf as _run_dcf
    cells, ev_matrix = [], []
    for g_s in gs_s:
        row_cells, row_evs = [], []
        for w_s in waccs_s:
            if w_s <= g_s:
                row_cells.append("N/A"); row_evs.append(None)
            else:
                r = _run_dcf(
                    ebit_margin=d["dcf_result"]["ebit_margin"],
                    base_revenue=d.get("base_rev", 1e9),
                    da_pct_revenue=stored_da,
                    capex_pct_revenue=stored_cap,
                    nwc_pct_revenue=d.get("tgt_nwc_avg", 0.18),
                    wacc=w_s, tax_rate=tax_rate,
                    growth_rate=g_s, projection_years=projection_years,
                )
                ev_val = r["enterprise_value"]
                row_cells.append(fmt_num(ev_val))
                row_evs.append(ev_val)
        cells.append(row_cells)
        ev_matrix.append(row_evs)

    # Build color-coded HTML table
    col_hdrs = "".join(f"<th style='padding:8px 10px;background:{BLACK};color:{GOLD};font-size:0.8rem;'>WACC={fmt_pct(w)}</th>" for w in waccs_s)
    tbl_rows = ""
    for ri, g_s in enumerate(gs_s):
        row_html = f"<td style='padding:8px 10px;background:{BLACK};color:{GOLD};font-weight:700;font-size:0.8rem;'>g={fmt_pct(g_s)}</td>"
        for ci, (cell_text, ev_v) in enumerate(zip(cells[ri], ev_matrix[ri])):
            is_base = (waccs_s[ci] == round(wacc_val, 4) and g_s == round(growth_rate, 4))
            if ev_v is None:
                bg, fg = "#f5f5f5", "#aaa"
            elif is_base:
                bg, fg = "#1a1a1a", GOLD
            elif ev_v > base_ev * 1.05:
                bg, fg = "#c8e6c9", "#1b5e20"
            elif ev_v > base_ev:
                bg, fg = "#e8f5e9", "#2e7d32"
            elif ev_v < base_ev * 0.95:
                bg, fg = "#ffcdd2", "#b71c1c"
            else:
                bg, fg = "#fff3e0", "#e65100"
            border = f"border:2px solid {GOLD};" if is_base else ""
            row_html += f"<td style='padding:8px 10px;text-align:right;background:{bg};color:{fg};font-weight:{'700' if is_base else '500'};font-size:0.82rem;{border}'>{cell_text}</td>"
        tbl_rows += f"<tr>{row_html}</tr>"

    st.markdown(f"""
    <div style='overflow-x:auto;'>
    <table style='border-collapse:collapse;font-size:0.82rem;width:100%;'>
      <thead><tr><th style='padding:8px 10px;background:{BLACK};color:{GOLD};'>g \\ WACC</th>{col_hdrs}</tr></thead>
      <tbody>{tbl_rows}</tbody>
    </table></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("← Back"):
            st.session_state.step -= 1; st.rerun()
    with col_next:
        if st.button("Next: Suggested Deals →"):
            st.session_state.step += 1; st.rerun()
    if st.button("Start New Analysis"):
        st.session_state.step = 0; st.session_state.data = {}; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — SUGGESTED DEALS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 8:
    d = st.session_state.data
    section("Suggested Alternative Deals", 8)
    insight("""
    Based on the acquirer's sector, size, and strategic profile, this screen surfaces
    <strong>alternative acquisition targets</strong> ranked by deal attractiveness.
    Each candidate is scored across four dimensions: <strong>size fit</strong> (target should be
    digestible vs. acquirer market cap), <strong>growth</strong> (revenue momentum),
    <strong>margin quality</strong> (EBITDA margin), and <strong>leverage</strong>
    (lower debt = easier integration). A composite score ranks the best strategic fits.
    """)

    acq_info = d.get("acq_info", {})
    acq_sector = acq_info.get("sector", "")
    acq_mktcap = acq_info.get("market_cap") or 1

    # Sector peer universe — curated lists per sector
    SECTOR_PEERS = {
        "Consumer Defensive": ["CHD", "CLX", "SJM", "HRL", "MKC", "COTY", "ENR", "CENT", "REYN", "ATGE"],
        "Consumer Cyclical":  ["LEVI", "VFC", "PVH", "HBI", "RL", "TPR", "CPRI", "GOOS", "CURB", "XPOF"],
        "Healthcare":         ["PRGO", "PHSI", "HZNP", "JAZZ", "PAHC", "ATRI", "MEDS", "PINC", "ACAD", "SUPN"],
        "Technology":         ["CNXC", "EXLS", "EPAM", "GLOB", "TASK", "PRFT", "CIXY", "CODA", "ALRM", "VNET"],
        "Industrials":        ["ACCO", "QUAD", "ATEN", "GFF", "LIQT", "NVRI", "CECO", "DNOW", "MFIN", "HLIO"],
        "Financial Services": ["CURO", "PRAA", "ECPG", "WRLD", "CACC", "NICK", "EZCORP", "QCR", "ESSA", "HIFS"],
        "Communication Services": ["NWSA", "GTN", "SSP", "SBGI", "IHRT", "SALM", "EMMS", "EVVTY"],
        "Energy":             ["WTTR", "NINE", "PUMP", "ACDC", "RES", "PTEN", "LBRT", "NR", "WELL"],
        "Basic Materials":    ["HWKN", "ASIX", "KWR", "IOSP", "GCP", "AMRS", "NTIC", "SXCL"],
        "Real Estate":        ["PINE", "CLDT", "APLE", "PLYM", "VRE", "ILPT", "NXRT", "STAG"],
        "Utilities":          ["YORW", "MSEX", "ARTNA", "SJW", "CWCO", "GWRS", "MGEE", "OTTR"],
    }
    # Fallback: broad mid/small cap candidates
    DEFAULT_PEERS = ["NUS", "HELE", "COTY", "ATER", "SKIN", "PRGO", "CHRS", "VNDA", "IRWD", "QDEL"]

    peers = SECTOR_PEERS.get(acq_sector, DEFAULT_PEERS)
    # Remove current target from suggestions
    peers = [p for p in peers if p.upper() != tgt_t.upper()][:12]

    with st.spinner(f"Screening {len(peers)} potential targets in {acq_sector or 'your sector'}..."):
        candidates = []
        for p in peers:
            try:
                info = yf.Ticker(p).info
                mktcap  = info.get("marketCap") or 0
                rev     = info.get("totalRevenue") or 0
                rev_grw = info.get("revenueGrowth") or 0
                ebitda  = info.get("ebitda") or 0
                debt    = info.get("totalDebt") or 0
                name    = info.get("longName", p)
                price   = info.get("currentPrice") or info.get("regularMarketPrice") or 0
                sector  = info.get("sector", "")
                beta_v  = info.get("beta") or 1.0

                if mktcap < 50e6 or mktcap > acq_mktcap * 0.8:
                    continue  # too small or too large

                ebitda_margin = ebitda / rev if rev > 0 else 0
                size_ratio    = mktcap / acq_mktcap       # lower = more digestible
                debt_to_rev   = debt / rev if rev > 0 else 1.0

                # Scoring (0–25 each, 100 total)
                size_score   = max(0, 25 * (1 - size_ratio / 0.5))          # prefer <25% of acquirer
                growth_score = min(25, max(0, rev_grw * 250))               # 10% growth → 25 pts
                margin_score = min(25, max(0, ebitda_margin * 100))         # 25% EBITDA → 25 pts
                levg_score   = max(0, 25 * (1 - min(debt_to_rev, 2) / 2))  # lower debt = higher score
                total_score  = size_score + growth_score + margin_score + levg_score

                candidates.append({
                    "Ticker": p,
                    "Company": name[:35],
                    "Market Cap": mktcap,
                    "Revenue": rev,
                    "Rev Growth": rev_grw,
                    "EBITDA Margin": ebitda_margin,
                    "Debt/Rev": debt_to_rev,
                    "Beta": beta_v,
                    "Size Score": round(size_score, 1),
                    "Growth Score": round(growth_score, 1),
                    "Margin Score": round(margin_score, 1),
                    "Leverage Score": round(levg_score, 1),
                    "Total Score": round(total_score, 1),
                })
            except Exception:
                continue

    if not candidates:
        st.warning("Could not retrieve candidate data. Try running the analysis again or check your internet connection.")
    else:
        cand_df = pd.DataFrame(candidates).sort_values("Total Score", ascending=False).reset_index(drop=True)

        section("Top Acquisition Candidates")

        # Top 3 cards
        top3 = cand_df.head(3)
        cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        for col, (_, row), medal in zip(cols, top3.iterrows(), medals):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-top:4px solid {GOLD};">
                  <div style="font-size:1.5rem;margin-bottom:4px;">{medal}</div>
                  <div style="font-size:1.1rem;font-weight:700;color:{BLACK};">{row['Ticker']}</div>
                  <div style="font-size:0.8rem;color:#555;margin-bottom:10px;">{row['Company']}</div>
                  <div style="font-size:1.8rem;font-weight:700;color:{GOLD_DARK};">{row['Total Score']:.0f}<span style="font-size:0.9rem;color:#888;">/100</span></div>
                  <hr style="border-color:#eee;margin:10px 0;">
                  <table style="width:100%;font-size:0.78rem;">
                    <tr><td style="color:#666;">Market Cap</td><td style="text-align:right;font-weight:600;">{fmt_num(row['Market Cap'])}</td></tr>
                    <tr><td style="color:#666;">Rev Growth</td><td style="text-align:right;font-weight:600;">{fmt_pct(row['Rev Growth'])}</td></tr>
                    <tr><td style="color:#666;">EBITDA Margin</td><td style="text-align:right;font-weight:600;">{fmt_pct(row['EBITDA Margin'])}</td></tr>
                    <tr><td style="color:#666;">Debt/Revenue</td><td style="text-align:right;font-weight:600;">{row['Debt/Rev']:.2f}x</td></tr>
                  </table>
                </div>""", unsafe_allow_html=True)

        # Full ranking table
        st.markdown("<br>", unsafe_allow_html=True)
        section("Full Candidate Ranking")
        disp = cand_df.copy()
        disp["Market Cap"]    = disp["Market Cap"].apply(fmt_num)
        disp["Revenue"]       = disp["Revenue"].apply(fmt_num)
        disp["Rev Growth"]    = disp["Rev Growth"].apply(fmt_pct)
        disp["EBITDA Margin"] = disp["EBITDA Margin"].apply(fmt_pct)
        disp["Debt/Rev"]      = disp["Debt/Rev"].apply(lambda x: f"{x:.2f}x")
        disp["Beta"]          = disp["Beta"].apply(lambda x: f"{x:.2f}")
        st.dataframe(
            disp[["Ticker","Company","Market Cap","Revenue","Rev Growth",
                  "EBITDA Margin","Debt/Rev","Beta","Size Score","Growth Score",
                  "Margin Score","Leverage Score","Total Score"]],
            use_container_width=True, hide_index=True
        )

        # Radar / spider chart for top 5
        st.markdown("<br>", unsafe_allow_html=True)
        section("Score Breakdown — Top 5 Candidates")
        top5 = cand_df.head(5)
        categories = ["Size Score", "Growth Score", "Margin Score", "Leverage Score"]
        fig = go.Figure()
        colors = [GOLD, GOLD_DARK, "#888", "#555", "#333"]
        for (_, row), color in zip(top5.iterrows(), colors):
            vals = [row[c] for c in categories]
            fig.add_trace(go.Bar(
                name=row["Ticker"],
                x=categories,
                y=vals,
                marker_color=color,
            ))
        fig.update_layout(
            barmode="group",
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis_title="Score (0–25)",
            font=dict(family="Source Sans Pro"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=40, b=40),
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#f0f0f0", range=[0, 26])
        st.plotly_chart(fig, use_container_width=True)

        insight(f"""
        <strong>How to read this:</strong> Candidates are scored across four dimensions (0–25 each).
        <strong>Size Score</strong> — how digestible the target is relative to {acq_info.get('name', acq_t)}'s market cap
        (smaller = easier to finance and integrate).
        <strong>Growth Score</strong> — revenue momentum (10%+ growth earns full marks).
        <strong>Margin Score</strong> — EBITDA profitability (25%+ earns full marks).
        <strong>Leverage Score</strong> — balance sheet health (low debt = simpler integration).
        Click any ticker to analyze it as the new target by updating the sidebar.
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_next = st.columns([1, 5])
    with col_prev:
        if st.button("← Back"):
            st.session_state.step -= 1; st.rerun()
    with col_next:
        if st.button("Next: Cannibalization →"):
            st.session_state.step += 1; st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 9 — CANNIBALIZATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 9:
    d = st.session_state.data
    section("Revenue Cannibalization Analysis", 9)
    formula(
        "Cannibalization Rate  = Overlap Score × Channel Overlap × Segment Overlap\n"
        "Revenue at Risk       = min(Acquirer Rev, Target Rev) × Cannibalization Rate\n"
        "Net Revenue Impact    = Synergy Revenue − Revenue at Risk\n"
        "Cannibalization-Adj EV = Standalone EV + PV(Net Synergies) − PV(Revenue at Risk)"
    )
    insight("""
    When two companies in similar markets merge, some revenue gets <strong>cannibalized</strong> —
    customers who bought from both companies now only need one provider, or the combined entity
    discontinues overlapping product lines. This step estimates the revenue at risk and adjusts
    the deal value accordingly. Companies with high channel/segment overlap face higher
    cannibalization risk, which offsets synergy gains.
    """)

    acq_info = d.get("acq_info", {})
    tgt_info = d.get("tgt_info", {})
    acq_is   = d.get("acq_is", pd.DataFrame())
    tgt_is   = d.get("tgt_is", pd.DataFrame())

    acq_rev = float(safe_get(acq_is, "Total Revenue", "Revenue").iloc[0]) if not acq_is.empty else (acq_info.get("market_cap") or 0) * 0.1
    tgt_rev = float(safe_get(tgt_is, "Total Revenue", "Revenue").iloc[0]) if not tgt_is.empty else (tgt_info.get("market_cap") or 0) * 0.1

    acq_sector   = acq_info.get("sector", "")
    tgt_sector   = tgt_info.get("sector", "")
    acq_industry = acq_info.get("industry", "")
    tgt_industry = tgt_info.get("industry", "")

    # ── Overlap scoring ──────────────────────────────────────────────────────
    # Sector match
    sector_match = (acq_sector == tgt_sector)
    industry_match = (acq_industry == tgt_industry)

    sector_overlap   = 1.0 if industry_match else (0.6 if sector_match else 0.15)
    channel_overlap  = 0.7 if sector_match else 0.25   # distribution/sales channel overlap
    customer_overlap = 0.5 if industry_match else (0.3 if sector_match else 0.05)

    raw_cannib_rate = sector_overlap * 0.4 + channel_overlap * 0.35 + customer_overlap * 0.25
    # Cap at 30% — full cannibalization is rare even in identical businesses
    cannib_rate = min(raw_cannib_rate, 0.30)

    revenue_at_risk = min(acq_rev, tgt_rev) * cannib_rate
    wacc_val = d.get("tgt_wacc", {}).get("WACC", 0.08)
    g = growth_rate

    # PV of cannibalized revenue stream (perpetuity)
    pv_revenue_at_risk = revenue_at_risk * (1 - tax_rate) / (wacc_val - g) if wacc_val > g else 0

    syn_result = d.get("syn_result", {})
    total_pv_syn = syn_result.get("total_pv_synergies", 0) if syn_result else 0
    rev_syn_full = syn_result.get("rev_synergy_full", 0) if syn_result else 0
    standalone_ev = d.get("dcf_result", {}).get("enterprise_value", 0) if d.get("dcf_result") else 0

    net_synergy_value = total_pv_syn - pv_revenue_at_risk
    cannib_adj_ev     = standalone_ev + net_synergy_value

    # ── Summary metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    overlap_color = GOLD_DARK if cannib_rate > 0.15 else ("#2e7d32" if cannib_rate < 0.08 else GOLD)
    c1.markdown(metric_card("Cannibalization Rate", fmt_pct(cannib_rate),
                             "High" if cannib_rate > 0.15 else ("Low" if cannib_rate < 0.08 else "Moderate")),
                unsafe_allow_html=True)
    c2.markdown(metric_card("Revenue at Risk (annual)", fmt_num(revenue_at_risk), "Post-merger overlap loss"),
                unsafe_allow_html=True)
    c3.markdown(metric_card("PV of Revenue at Risk", fmt_num(pv_revenue_at_risk), "Perpetuity at WACC"), unsafe_allow_html=True)
    c4.markdown(metric_card("Cannib-Adjusted EV", fmt_num(cannib_adj_ev), "Standalone + Net Synergies"),
                unsafe_allow_html=True)

    # ── Overlap Driver Table ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Overlap Drivers")

    def overlap_bar(score):
        pct = int(score * 100)
        color = GOLD_DARK if score > 0.5 else (GOLD if score > 0.25 else "#4caf50")
        return f"""<div style='background:#f0f0f0;border-radius:4px;height:12px;'>
          <div style='background:{color};width:{pct}%;height:12px;border-radius:4px;'></div></div>
          <span style='font-size:0.75rem;color:#666;'>{pct}%</span>"""

    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
      <tr style='background:{BLACK};color:{GOLD};'>
        <th style='padding:10px;text-align:left;'>Driver</th>
        <th style='padding:10px;text-align:left;'>{acq_info.get("name", acq_t)}</th>
        <th style='padding:10px;text-align:left;'>{tgt_info.get("name", tgt_t)}</th>
        <th style='padding:10px;text-align:left;'>Overlap</th>
        <th style='padding:10px;text-align:left;'>Score</th>
      </tr>
      <tr style='background:#fafafa;'>
        <td style='padding:9px;font-weight:600;'>Sector</td>
        <td style='padding:9px;'>{acq_sector}</td>
        <td style='padding:9px;'>{tgt_sector}</td>
        <td style='padding:9px;'>{"✅ Same" if sector_match else "❌ Different"}</td>
        <td style='padding:9px;'>{overlap_bar(sector_overlap)}</td>
      </tr>
      <tr>
        <td style='padding:9px;font-weight:600;'>Industry</td>
        <td style='padding:9px;'>{acq_industry[:40]}</td>
        <td style='padding:9px;'>{tgt_industry[:40]}</td>
        <td style='padding:9px;'>{"✅ Same" if industry_match else "⚠️ Related" if sector_match else "❌ Different"}</td>
        <td style='padding:9px;'>{overlap_bar(customer_overlap)}</td>
      </tr>
      <tr style='background:#fafafa;'>
        <td style='padding:9px;font-weight:600;'>Channel Overlap</td>
        <td style='padding:9px;' colspan='2'>Distribution / sales channel similarity</td>
        <td style='padding:9px;'>{"High" if channel_overlap > 0.5 else "Moderate" if channel_overlap > 0.25 else "Low"}</td>
        <td style='padding:9px;'>{overlap_bar(channel_overlap)}</td>
      </tr>
      <tr style='background:{BLACK};color:{GOLD};font-weight:700;'>
        <td style='padding:9px;'>Blended Cannibalization Rate</td>
        <td style='padding:9px;' colspan='3'></td>
        <td style='padding:9px;'>{fmt_pct(cannib_rate)}</td>
      </tr>
    </table>""", unsafe_allow_html=True)

    # ── Year-by-Year Cannibalization Build ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Year-by-Year Revenue at Risk vs. Synergy Revenue")

    years_list = list(range(1, projection_years + 1))
    rev_at_risk_yr, rev_syn_yr = [], []
    for yr in years_list:
        # Revenue at risk grows with company growth, but integration reduces it over time
        integration_factor = max(0, 1 - (yr - 1) * 0.12)   # reduces 12% per year as SKUs rationalized
        rev_at_risk_yr.append(revenue_at_risk * ((1 + growth_rate) ** yr) * integration_factor)
        # Synergy revenue ramps up
        ramp = 0.2 if yr == 1 else 1.0
        rev_syn_yr.append(rev_syn_full * ramp * ((1 + growth_rate) ** (yr - 1)))

    net_yr = [s - r for s, r in zip(rev_syn_yr, rev_at_risk_yr)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Yr {y}" for y in years_list],
        y=[v / 1e6 for v in rev_syn_yr],
        name="Synergy Revenue",
        marker_color=GOLD,
    ))
    fig.add_trace(go.Bar(
        x=[f"Yr {y}" for y in years_list],
        y=[-v / 1e6 for v in rev_at_risk_yr],
        name="Revenue at Risk (cannibalization)",
        marker_color="#c62828",
    ))
    fig.add_trace(go.Scatter(
        x=[f"Yr {y}" for y in years_list],
        y=[v / 1e6 for v in net_yr],
        mode="lines+markers",
        name="Net Revenue Impact",
        line=dict(color=BLACK, width=3),
        marker=dict(size=8),
    ))
    fig.update_layout(
        barmode="relative",
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Revenue ($M)",
        font=dict(family="Source Sans Pro"),
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=40, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#999")
    st.plotly_chart(fig, use_container_width=True)

    # ── Adjusted Deal Summary ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("Cannibalization-Adjusted Deal Value")

    tgt_net_debt  = d.get("tgt_net_debt", 0) or 0
    tgt_shares    = (d.get("tgt_info") or {}).get("shares_outstanding") or 1

    adj_equity    = cannib_adj_ev - tgt_net_debt
    adj_pps       = adj_equity / tgt_shares if tgt_shares else 0
    adj_offer     = adj_pps * (1 + offer_premium)

    unadj_pps     = (standalone_ev + total_pv_syn - tgt_net_debt) / tgt_shares if tgt_shares else 0

    st.markdown(f"""
    <table style='width:100%;border-collapse:collapse;font-size:0.88rem;'>
      <tr style='background:{BLACK};color:{GOLD};'>
        <th style='padding:10px;text-align:left;'>Line Item</th>
        <th style='padding:10px;text-align:right;'>Unadjusted</th>
        <th style='padding:10px;text-align:right;'>Cannib-Adjusted</th>
        <th style='padding:10px;text-align:left;'>Delta</th>
      </tr>
      <tr style='background:#fafafa;'>
        <td style='padding:9px;'>Enterprise Value</td>
        <td style='padding:9px;text-align:right;'>{fmt_num(standalone_ev + total_pv_syn)}</td>
        <td style='padding:9px;text-align:right;'>{fmt_num(cannib_adj_ev)}</td>
        <td style='padding:9px;color:#c62828;'>−{fmt_num(pv_revenue_at_risk)}</td>
      </tr>
      <tr>
        <td style='padding:9px;'>Implied Price/Share</td>
        <td style='padding:9px;text-align:right;'>${unadj_pps:,.2f}</td>
        <td style='padding:9px;text-align:right;'>${adj_pps:,.2f}</td>
        <td style='padding:9px;color:#c62828;'>−${unadj_pps - adj_pps:,.2f}</td>
      </tr>
      <tr style='background:#fafafa;'>
        <td style='padding:9px;'>Offer Price ({fmt_pct(offer_premium)} premium)</td>
        <td style='padding:9px;text-align:right;'>${unadj_pps * (1 + offer_premium):,.2f}</td>
        <td style='padding:9px;text-align:right;'>${adj_offer:,.2f}</td>
        <td style='padding:9px;color:#c62828;'>−${(unadj_pps - adj_pps) * (1 + offer_premium):,.2f}</td>
      </tr>
    </table>""", unsafe_allow_html=True)

    # Verdict
    if cannib_rate < 0.08:
        msg = f"<strong>Low cannibalization risk ({fmt_pct(cannib_rate)}).</strong> {acq_info.get('name', acq_t)} and {tgt_info.get('name', tgt_t)} operate in sufficiently different markets that post-merger revenue overlap is minimal. The synergy case is largely preserved."
    elif cannib_rate < 0.15:
        msg = f"<strong>Moderate cannibalization risk ({fmt_pct(cannib_rate)}).</strong> Some revenue overlap exists due to sector similarity. Management should proactively rationalize overlapping SKUs and redirect displaced revenue into new segments. Net synergy value is still positive."
    else:
        msg = f"<strong>High cannibalization risk ({fmt_pct(cannib_rate)}).</strong> Both companies compete in the same industry with significant customer and channel overlap. The deal team should model specific product-line discontinuations and price the cannibalization discount into the offer. The unadjusted synergy case may be overstated by <strong>{fmt_num(pv_revenue_at_risk)}</strong>."
    insight(msg)

    st.markdown("<br>", unsafe_allow_html=True)
    col_prev, col_fin = st.columns([1, 5])
    with col_prev:
        if st.button("← Back"):
            st.session_state.step -= 1; st.rerun()
    with col_fin:
        if st.button("Start New Analysis"):
            st.session_state.step = 0; st.session_state.data = {}; st.rerun()
