"""
Commercial Real Estate Intelligence Platform
Purdue MSF | Group AI Project

Eight background agents update independently on schedules:
  Agent 1 — Population & Migration    (every 6h)
  Agent 2 — CRE Pricing & Profit      (every 1h)
  Agent 3 — Company Predictions       (every 24h, LLM)
  Agent 4 — Debugger / Monitor        (every 30min)
  Agent 5 — News & Announcements      (every 4h)
  Agent 6 — Interest Rate & Debt      (every 1h, requires FRED_API_KEY)
  Agent 7 — Energy & Construction     (every 6h)
  Agent 8 — Sustainability & ESG      (every 6h)

Run: streamlit run app/cre_app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── Page config MUST be first Streamlit command ──────────────────────────────
st.set_page_config(
    page_title="CRE Intelligence | Purdue MSF",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Start background agents ──────────────────────────────────────────────────
from src.cre_agents import (
    start_scheduler, read_cache, cache_age_label, get_status,
)

@st.cache_resource
def _init_scheduler():
    """Called once per server process — keeps one scheduler alive regardless of reruns or new sessions."""
    start_scheduler()
    return True

_init_scheduler()

# ── Brand ────────────────────────────────────────────────────────────────────
GOLD      = "#CFB991"
GOLD_DARK = "#8E6F3E"
BLACK     = "#000000"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] {{ font-family: 'Source Sans Pro', sans-serif; }}

  .cre-header {{
    background: {BLACK};
    padding: 18px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
  }}
  .cre-header h1 {{ color: {GOLD}; font-size: 1.6rem; font-weight: 700; margin: 0; }}
  .cre-header span {{ color: white; font-size: 0.85rem; opacity: 0.8; }}
  .gold-bar {{ height: 4px; background: linear-gradient(90deg, {GOLD_DARK}, {GOLD}, {GOLD_DARK}); margin-bottom: 24px; }}

  .agent-card {{
    background: {BLACK};
    border-radius: 8px;
    padding: 20px 24px;
    margin: 12px 0;
    border-left: 5px solid {GOLD};
  }}
  .agent-card .agent-label {{
    color: {GOLD};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 8px;
  }}
  .agent-card .agent-text {{
    color: #eee;
    font-size: 0.92rem;
    line-height: 1.7;
    white-space: pre-wrap;
  }}

  .metric-card {{
    background: white;
    border: 1px solid #e0e0e0;
    border-top: 4px solid {GOLD};
    border-radius: 6px;
    padding: 16px;
    text-align: center;
  }}
  .metric-card .label {{ font-size: 0.75rem; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric-card .value {{ font-size: 1.6rem; font-weight: 700; color: {BLACK}; margin: 4px 0; }}
  .metric-card .sub   {{ font-size: 0.78rem; color: #888; }}

  .section-header {{
    background: {BLACK};
    color: {GOLD};
    padding: 9px 16px;
    border-radius: 4px;
    font-size: 1rem;
    font-weight: 700;
    margin: 24px 0 14px 0;
    border-left: 5px solid {GOLD};
  }}

  .listing-card {{
    background: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-left: 4px solid {GOLD};
    border-radius: 6px;
    padding: 14px 18px;
    margin: 8px 0;
  }}
  .listing-card .l-price {{ font-size: 1.4rem; font-weight: 700; color: {BLACK}; }}
  .listing-card .l-address {{ font-size: 0.9rem; color: #333; margin: 2px 0; }}
  .listing-card .l-detail {{ font-size: 0.8rem; color: #666; }}
  .listing-card .l-tag {{
    display: inline-block;
    background: {GOLD};
    color: {BLACK};
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-right: 4px;
    margin-top: 4px;
  }}

  .status-ok    {{ color: #2e7d32; font-weight: 700; }}
  .status-error {{ color: #b71c1c; font-weight: 700; }}
  .status-run   {{ color: #e65100; font-weight: 700; }}
  .status-idle  {{ color: #757575; }}

  div[data-testid="stTabs"] button {{
    font-weight: 600 !important;
    font-size: 0.92rem !important;
  }}
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="cre-header">
  <h1> CRE Intelligence Platform</h1>
  <span>Purdue University · Daniels School of Business · MSF Program &nbsp;|&nbsp; {datetime.today().strftime('%B %d, %Y')}</span>
</div>
<div class="gold-bar"></div>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
      {"<div class='sub'>" + sub + "</div>" if sub else ""}
    </div>"""

def stale_banner(cache_key: str):
    c = read_cache(cache_key)
    if c["data"] is None:
        st.warning(f"⏳ Agent is fetching {cache_key} data for the first time — please wait ~30 seconds and refresh.")
        return False
    age = cache_age_label(cache_key)
    if c.get("stale"):
        st.warning(f"⚠ Data is stale (last updated {age}). Agent may be restarting.")
    else:
        st.caption(f" Last updated: {age} · Auto-refreshes in background")
    return True

def agent_last_updated(agent_name: str):
    st.caption(f"Last updated: {cache_age_label(agent_name)}")


# ── Meet the Team (fixed footer) ──────────────────────────────────────────────
st.markdown(f"""
<style>
  .meet-team-footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    background: {BLACK};
    padding: 14px 32px;
    border-top: 3px solid {GOLD};
    text-align: center;
  }}
  .meet-team-footer .label {{
    color: {GOLD};
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 6px;
  }}
  .meet-team-footer .names {{
    display: flex;
    justify-content: center;
    gap: 36px;
    flex-wrap: wrap;
  }}
  .meet-team-footer a {{
    color: {GOLD} !important;
    text-decoration: underline !important;
    font-size: 0.9rem;
    font-weight: 600;
  }}
  .meet-team-footer .course {{
    color: #888;
    font-size: 0.7rem;
    margin-top: 6px;
  }}
  /* Push page content up so footer doesn't overlap last element */
  .main .block-container {{ padding-bottom: 90px; }}
</style>
<div class="meet-team-footer">
  <div class="label">Meet the Team</div>
  <div class="names">
    <a href="https://www.linkedin.com/in/aayman-afzal/" target="_blank">Aayman Afzal</a>
    <a href="https://www.linkedin.com/in/ajinkyakodnikar/" target="_blank">Ajinkya Kodnikar</a>
    <a href="https://www.linkedin.com/in/oyu-amar/" target="_blank">Oyu Amar</a>
    <a href="https://www.linkedin.com/in/ricardo-ruiz1/" target="_blank">Ricardo Ruiz</a>
  </div>
  <div class="course">MGMT 690 · AI Leadership · Purdue Daniels School of Business</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════
main_tab_re, main_tab_energy = st.tabs(["Real Estate", "Energy"])

with main_tab_re:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Migration Intelligence",
        "Pricing & Profit",
        "Company Predictions",
        "Cheapest Buildings",
        "Industry Announcements",
        "System Monitor",
    ])


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 1 — POPULATION & MIGRATION
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("#### Where are people and companies moving in the US?")
        st.markdown(
            "Agent 1 tracks **domestic population migration** and **corporate headquarters relocations** "
            "to surface the highest-demand markets for CRE investment. Updates every 6 hours."
        )
        agent_last_updated("migration")

        cache = read_cache("migration")
        if not stale_banner("migration") or cache["data"] is None:
            st.stop()

        data     = cache["data"]
        mig_df   = pd.DataFrame(data["migration"])
        metros_df = pd.DataFrame(data["metros"])

        # ── KPI strip ──────────────────────────────────────────────────────────
        top1      = mig_df.iloc[0]
        top_gain  = mig_df[mig_df["pop_growth_pct"] > 0].shape[0]
        top_loss  = mig_df[mig_df["pop_growth_pct"] < 0].shape[0]
        avg_grow  = mig_df["pop_growth_pct"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("#1 Destination", top1["state_abbr"], top1["state_name"]), unsafe_allow_html=True)
        c2.markdown(metric_card("States Growing", str(top_gain), "Positive net migration"), unsafe_allow_html=True)
        c3.markdown(metric_card("States Shrinking", str(top_loss), "Negative net migration"), unsafe_allow_html=True)
        c4.markdown(metric_card("Avg Pop Growth", f"{avg_grow:.2f}%", "All states YoY"), unsafe_allow_html=True)

        # ── US CHOROPLETH MAP ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" US Population Growth Map — Where America is Moving")

        map_col, legend_col = st.columns([3, 1])
        with map_col:
            fig_map = go.Figure(go.Choropleth(
                locations=mig_df["state_abbr"],
                z=mig_df["composite_score"],
                locationmode="USA-states",
                colorscale=[
                    [0.0,  "#b71c1c"],
                    [0.25, "#ef5350"],
                    [0.45, "#ffcdd2"],
                    [0.55, "#fff9c4"],
                    [0.70, "#a5d6a7"],
                    [0.85, "#388e3c"],
                    [1.0,  "#1b5e20"],
                ],
                zmin=0, zmax=100,
                colorbar=dict(
                    title=dict(text="Migration<br>Score", font=dict(size=11)),
                    tickfont=dict(size=10), thickness=14, len=0.7,
                ),
                text=mig_df.apply(
                    lambda r: f"<b>{r['state_name']}</b><br>"
                              f"Pop Growth: {r['pop_growth_pct']:+.2f}%<br>"
                              f"Business Score: {r['biz_score']}<br>"
                              f"Composite: {r['composite_score']}<br>"
                              f"Key Companies: {r['key_companies']}<br>"
                              f"<i>{r['growth_drivers']}</i>",
                    axis=1
                ),
                hovertemplate="%{text}<extra></extra>",
            ))
            fig_map.update_layout(
                geo=dict(scope="usa", showlakes=True, lakecolor="lightblue",
                         bgcolor="white", showland=True, landcolor="#f5f5f5",
                         projection_scale=1, center=dict(lat=38, lon=-96)),
                paper_bgcolor="white",
                margin=dict(t=10, b=10, l=0, r=0),
                height=460,
                font=dict(family="Source Sans Pro", color="#1a1a1a"),
                dragmode=False,
            )
            st.plotly_chart(fig_map, use_container_width=True, config={"scrollZoom": False, "displayModeBar": False})
            st.caption(
                "Darker green indicates states with the strongest combined population inflow and business migration. "
                "These markets historically see the earliest and sharpest increases in CRE demand — "
                "particularly for multifamily, industrial, and mixed-use properties."
            )

        with legend_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for color, label in [
                ("#1b5e20", " High Growth (70–100)"),
                ("#388e3c", " Growing (55–70)"),
                ("#fff9c4", " Stable (45–55)"),
                ("#ef5350", " Declining (25–45)"),
                ("#b71c1c", "⛔ High Outflow (<25)"),
            ]:
                st.markdown(
                    f"<div style='display:flex;align-items:center;margin:6px 0;font-size:0.82rem;'>"
                    f"<div style='width:16px;height:16px;background:{color};border-radius:3px;margin-right:8px;flex-shrink:0;'></div>"
                    f"{label}</div>",
                    unsafe_allow_html=True
                )
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Score = 60% Pop Growth + 40% Business Migration Index")

        # ── Top 10 States ──────────────────────────────────────────────────────
        section(" Top 10 States for CRE Investment (Migration Score)")
        top10 = mig_df.head(10)
        fig_bar = go.Figure(go.Bar(
            x=top10["composite_score"], y=top10["state_abbr"],
            orientation="h",
            marker=dict(color=top10["composite_score"],
                        colorscale=[[0, GOLD_DARK], [1, GOLD]], showscale=False),
            text=top10["composite_score"].apply(lambda x: f"{x:.0f}"),
            textposition="outside",
            textfont=dict(color="#1a1a1a", size=12),
            customdata=top10[["state_name", "pop_growth_pct", "key_companies"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Pop Growth: %{customdata[1]:+.2f}%<br>"
                "Key Companies: %{customdata[2]}<extra></extra>"
            ),
        ))
        fig_bar.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", range=[0, 110],
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            yaxis=dict(autorange="reversed", tickfont=dict(color="#1a1a1a", size=12)),
            margin=dict(t=20, b=20, l=60, r=60),
            height=320, font=dict(family="Source Sans Pro", color="#1a1a1a"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption(
            "Rankings combine population growth rate (60% weight) and business migration index (40% weight). "
            "States at the top represent the most favorable macro conditions for new CRE investment — "
            "higher scores correlate with rising rental demand, tighter vacancy, and upward rent pressure."
        )

        # ── Metro Table ────────────────────────────────────────────────────────
        section(" Top Metro Areas — Population, Jobs & CRE Demand")
        metros_disp = metros_df.copy()
        metros_disp["Pop Growth %"] = metros_disp["Pop Growth %"].apply(lambda x: f"{x:+.1f}%")
        metros_disp["Job Growth %"] = metros_disp["Job Growth %"].apply(lambda x: f"{x:+.1f}%")
        metros_disp["Corp HQ Moves"] = metros_disp["Corp HQ Moves"].apply(lambda x: f"{x:+d}")

        def color_demand(val):
            colors = {
                "Very High": "background-color:#c8e6c9;color:#1b5e20;font-weight:700",
                "High":      "background-color:#dcedc8;color:#33691e;font-weight:600",
                "Moderate":  "background-color:#fff9c4;color:#f57f17",
                "Weak":      "background-color:#ffccbc;color:#bf360c",
                "Declining": "background-color:#ffcdd2;color:#b71c1c;font-weight:700",
            }
            return colors.get(val, "")

        st.dataframe(
            metros_disp.style.applymap(color_demand, subset=["CRE Demand"]),
            use_container_width=True, hide_index=True
        )

        # ── Bubble chart ───────────────────────────────────────────────────────
        section(" Business Migration vs. Population Growth")
        biz_data = mig_df[mig_df["biz_score"] != 50].copy()
        fig_bubble = px.scatter(
            biz_data, x="pop_growth_pct", y="biz_score",
            size="population", color="composite_score",
            text="state_abbr",
            color_continuous_scale=[[0, "#b71c1c"], [0.5, GOLD], [1, "#1b5e20"]],
            labels={"pop_growth_pct": "Population Growth % (YoY)",
                    "biz_score": "Business Migration Score",
                    "composite_score": "Composite"},
            hover_data={"state_name": True, "key_companies": True, "growth_drivers": True},
        )
        fig_bubble.update_traces(textposition="middle center", textfont=dict(size=9, color="#1a1a1a"))
        fig_bubble.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc",
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            coloraxis_showscale=False, margin=dict(t=20, b=40),
            height=380, font=dict(family="Source Sans Pro", color="#1a1a1a"),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.caption(
            "Each bubble is a state. Bubble size reflects the composite migration score. "
            "The ideal CRE investment target sits in the upper-right — high population growth AND strong business migration. "
            "States in the lower-left are losing both residents and corporate presence."
        )


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 2 — CRE PRICING & PROFIT
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### Where are the highest profit margins in commercial real estate today?")
        st.markdown(
            "Agent 2 pulls **live REIT pricing**, estimates **cap rates** and **NOI margins** by property type, "
            "and ranks market × property type combinations. Updates every hour."
        )
        agent_last_updated("pricing")

        from src.cre_pricing import compute_profit_matrix, get_top_opportunities, get_property_type_summary, CAP_RATE_BENCHMARKS

        cache_p = read_cache("pricing")
        if not stale_banner("pricing") or cache_p["data"] is None:
            st.stop()

        pdata    = cache_p["data"]
        reit_df  = pd.DataFrame(pdata["reits"])
        top_opps = pd.DataFrame(pdata["top_opps"])
        pt_summary = pd.DataFrame(pdata["pt_summary"])

        # ── KPI strip ──────────────────────────────────────────────────────────
        best_type  = pt_summary.iloc[0]
        worst_type = pt_summary.iloc[-1]
        best_opp   = top_opps.iloc[0]
        avg_cap    = pt_summary["Cap Rate"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Best Property Type", best_type["Property Type"].split("/")[0].strip(),
                                 f"Margin: {best_type['Eff Profit Margin']*100:.1f}%"), unsafe_allow_html=True)
        c2.markdown(metric_card("Weakest Type", worst_type["Property Type"].split("/")[0].strip(),
                                 "High vacancy / rent decline"), unsafe_allow_html=True)
        c3.markdown(metric_card("Avg Market Cap Rate", f"{avg_cap*100:.2f}%",
                                 "Blended all property types"), unsafe_allow_html=True)
        c4.markdown(metric_card("Top Market", best_opp["Market"].split(",")[0],
                                 f"{best_opp['Property Type'].split('/')[0].strip()} · Score {best_opp['Profit Score']}"),
                    unsafe_allow_html=True)

        # ── Live REIT Prices ────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(f" Live REIT Prices — {datetime.today().strftime('%B %d, %Y')}")

        prop_types = reit_df["Property Type"].unique()
        tabs_reit  = st.tabs(list(prop_types))

        for tab_r, pt in zip(tabs_reit, prop_types):
            with tab_r:
                sub   = reit_df[reit_df["Property Type"] == pt].copy()
                col1, col2 = st.columns([2, 1])
                with col1:
                    valid = sub[sub["Price"].notna()]
                    if not valid.empty:
                        colors = [GOLD if r >= 0 else "#c62828" for r in valid["Daily Return"].fillna(0)]
                        fig_p = go.Figure(go.Bar(
                            x=valid["Ticker"], y=valid["Price"],
                            marker_color=colors,
                            text=valid["Price"].apply(lambda x: f"${x:.2f}" if x else "N/A"),
                            textposition="outside",
                            customdata=valid[["Company","Daily Return","Div Yield","Market Cap"]].values,
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>Price: $%{y:.2f}<br>"
                                "Daily Return: %{customdata[1]:+.2%}<br>"
                                "Div Yield: %{customdata[2]:.2%}<br>"
                                "Market Cap: $%{customdata[3]:,.0f}<extra></extra>"
                            ),
                        ))
                        fig_p.update_layout(
                            plot_bgcolor="white", paper_bgcolor="white",
                            yaxis_title="Price ($)", margin=dict(t=30, b=30),
                            height=260, font=dict(family="Source Sans Pro", color="#1a1a1a"),
                        )
                        fig_p.update_xaxes(showgrid=False, tickfont=dict(color="#1a1a1a"))
                        fig_p.update_yaxes(gridcolor="#f0f0f0", tickfont=dict(color="#1a1a1a"),
                                           title_font=dict(color="#1a1a1a"))
                        st.plotly_chart(fig_p, use_container_width=True)
                        st.caption(
                            "Live REIT share prices sourced from Yahoo Finance. "
                            "Higher dividend yield indicates more income relative to price — "
                            "useful for comparing income-generating potential across property types."
                        )
                with col2:
                    bench = CAP_RATE_BENCHMARKS.get(pt, {})
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(metric_card("Cap Rate", f"{bench.get('cap_rate',0)*100:.2f}%", "Market benchmark"), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(metric_card("NOI Margin", f"{bench.get('noi_margin',0)*100:.1f}%", "After opex, before CapEx"), unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(metric_card("Rent Growth YoY", f"{bench.get('rent_growth',0)*100:+.1f}%", "Market consensus"), unsafe_allow_html=True)

                disp = sub[["Ticker","Company","Market Focus","Price","Daily Return","Div Yield","Cap Rate","NOI Margin","Vacancy Rate","Rent Growth"]].copy()
                disp["Price"]        = disp["Price"].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                disp["Daily Return"] = disp["Daily Return"].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A")
                disp["Div Yield"]    = disp["Div Yield"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
                disp["Cap Rate"]     = disp["Cap Rate"].apply(lambda x: f"{x*100:.2f}%")
                disp["NOI Margin"]   = disp["NOI Margin"].apply(lambda x: f"{x*100:.1f}%")
                disp["Vacancy Rate"] = disp["Vacancy Rate"].apply(lambda x: f"{x*100:.1f}%")
                disp["Rent Growth"]  = disp["Rent Growth"].apply(lambda x: f"{x*100:+.1f}%")
                st.dataframe(disp, use_container_width=True, hide_index=True)

        # ── Profit Margin Heatmap ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Profit Margin Heatmap — Market × Property Type")

        profit_df = compute_profit_matrix()
        pivot = profit_df.pivot_table(
            index="Property Type", columns="Market",
            values="Eff Profit Margin", aggfunc="mean"
        )
        pivot.columns = [c.split(",")[0] for c in pivot.columns]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values * 100, x=list(pivot.columns), y=list(pivot.index),
            colorscale=[[0.0, "#b71c1c"], [0.3, "#ef9a9a"], [0.5, "#fff9c4"], [0.7, "#a5d6a7"], [1.0, "#1b5e20"]],
            text=[[f"{v:.1f}%" for v in row] for row in pivot.values * 100],
            texttemplate="%{text}", textfont=dict(size=9, color="#1a1a1a"),
            hovertemplate="<b>%{y}</b><br>%{x}<br>Margin: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="Eff Margin %", thickness=14, len=0.8,
                          tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
        ))
        fig_heat.update_layout(
            paper_bgcolor="white",
            xaxis=dict(tickangle=-35, tickfont=dict(size=9, color="#1a1a1a")),
            yaxis=dict(tickfont=dict(size=9, color="#1a1a1a")),
            margin=dict(t=20, b=100, l=180, r=20),
            height=420, font=dict(family="Source Sans Pro", color="#1a1a1a"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption(
            "Effective Profit Margin = NOI Margin × (1 − Vacancy) × (1 + Rent Growth). "
            "Green cells are the most profitable combinations of market and property type. "
            "Sunbelt markets (Dallas, Austin, Miami, Nashville) consistently outperform gateway cities "
            "due to lower cap rate compression and stronger rent growth trajectories."
        )

        # ── Top 10 Opportunities ───────────────────────────────────────────────
        section(" Top 10 Highest Profit Margin Opportunities Right Now")
        st.dataframe(top_opps, use_container_width=True, hide_index=True)
        st.caption(
            "Ranked by Effective Profit Margin across all tracked markets and property types. "
            "Industrial and Data Center assets in Sunbelt markets dominate the top rankings due to "
            "low vacancy, high NOI margins, and strong rent growth — key indicators of durable cash flow."
        )

        # ── Property Type Comparison ───────────────────────────────────────────
        section(" Property Type Performance Comparison")
        colors_pt = [GOLD if i == 0 else (GOLD_DARK if i < 3 else "#aaa") for i in range(len(pt_summary))]
        fig_pt = go.Figure()
        fig_pt.add_trace(go.Bar(
            x=pt_summary["Property Type"].str.split("/").str[0].str.strip(),
            y=pt_summary["Eff Profit Margin"] * 100,
            marker_color=colors_pt,
            text=pt_summary["Eff Profit Margin"].apply(lambda x: f"{x*100:.1f}%"),
            textposition="outside", name="Effective Profit Margin",
        ))
        fig_pt.add_trace(go.Scatter(
            x=pt_summary["Property Type"].str.split("/").str[0].str.strip(),
            y=pt_summary["Cap Rate"] * 100,
            mode="lines+markers", name="Cap Rate",
            line=dict(color=BLACK, width=2, dash="dash"),
            marker=dict(size=7), yaxis="y2",
        ))
        fig_pt.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Effective Profit Margin (%)", gridcolor="#f0f0f0",
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            yaxis2=dict(title="Cap Rate (%)", overlaying="y", side="right", showgrid=False,
                        tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            legend=dict(orientation="h", y=1.1, font=dict(color="#1a1a1a")),
            margin=dict(t=40, b=60),
            height=360, font=dict(family="Source Sans Pro", color="#1a1a1a"),
        )
        fig_pt.update_xaxes(showgrid=False, tickangle=-15, tickfont=dict(color="#1a1a1a"))
        st.plotly_chart(fig_pt, use_container_width=True)
        st.caption(
            "Bars show effective profit margin (left axis); the line shows cap rate (right axis). "
            "Industrial leads on margin due to near-zero vacancy and rapid rent growth driven by e-commerce and reshoring. "
            "Office carries the highest cap rate but the lowest margin — reflecting elevated vacancy and negative rent growth in most markets. "
            "Source: CBRE, JLL, and Green Street 2024–2025 reports. Live REIT prices from Yahoo Finance."
        )

        # ── Rate-Adjusted View ─────────────────────────────────────────────────
        cache_r2 = read_cache("rates")
        rdata2   = cache_r2.get("data") or {}
        cap_adj2 = rdata2.get("cap_rate_adjustments", [])
        cur_10y2 = rdata2.get("current_10y")

        if cap_adj2 and cur_10y2:
            st.markdown("<br>", unsafe_allow_html=True)
            section(f" Rate-Adjusted Cap Rates — 10Y at {cur_10y2:.2f}%")
            show_adj = st.toggle("Show rate-adjusted profit matrix", value=True, key="rate_adj_toggle")
            if show_adj:
                adj_df2 = pd.DataFrame(cap_adj2)
                baseline2 = rdata2.get("baseline_10y", 4.0)
                delta2    = cur_10y2 - baseline2
                direction2 = "above" if delta2 > 0 else "below"

                st.caption(
                    f"10Y Treasury ({cur_10y2:.2f}%) is {abs(delta2):.2f}% {direction2} the "
                    f"{baseline2:.1f}% baseline. Cap rates are adjusted using sector-specific betas."
                )

                colors_adj = [
                    "#b71c1c" if r > 0 else "#1b5e20"
                    for r in adj_df2["Rate Adjustment bps"]
                ]
                fig_adj = go.Figure()
                pt_short = adj_df2["Property Type"].str.split("/").str[0].str.strip()
                fig_adj.add_trace(go.Bar(
                    name="Static Cap Rate (%)",
                    x=pt_short,
                    y=adj_df2["Baseline Cap Rate"],
                    marker_color="#CFB991",
                    text=adj_df2["Baseline Cap Rate"].apply(lambda v: f"{v:.2f}%"),
                    textposition="inside",
                ))
                fig_adj.add_trace(go.Bar(
                    name="Rate-Adjusted Cap Rate (%)",
                    x=pt_short,
                    y=adj_df2["Adjusted Cap Rate"],
                    marker_color=colors_adj,
                    opacity=0.85,
                    text=adj_df2["Rate Adjustment bps"].apply(
                        lambda d: f"{'+' if d > 0 else ''}{d:.0f}bps"
                    ),
                    textposition="outside",
                ))
                fig_adj.update_layout(
                    barmode="group",
                    paper_bgcolor="white", plot_bgcolor="white",
                    yaxis=dict(title="Cap Rate (%)", ticksuffix="%", gridcolor="#f0f0f0",
                               tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                    xaxis=dict(tickangle=-15, tickfont=dict(color="#1a1a1a", size=10)),
                    legend=dict(orientation="h", y=1.1, font=dict(color="#1a1a1a")),
                    margin=dict(t=40, b=60), height=340,
                    font=dict(family="Source Sans Pro", color="#1a1a1a"),
                )
                st.plotly_chart(fig_adj, use_container_width=True)
                st.caption(
                    "Shows the static (benchmark) cap rate alongside the rate-adjusted cap rate for each property type. "
                    "When the 10-year Treasury rises above the baseline, cap rates expand — meaning asset values fall "
                    "for the same NOI. Office and retail are most sensitive; industrial and multifamily are more resilient."
                )


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 3 — CONFIRMED FACILITY ANNOUNCEMENTS
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### Which companies have announced new plants, factories, and facilities?")
        st.markdown(
            "Agent 3 scans live news feeds and uses AI to extract **confirmed** corporate facility announcements — "
            "new manufacturing plants, data centers, warehouses, training centers, and headquarters. Updates every 24 hours."
        )
        agent_last_updated("predictions")

        cache3 = read_cache("predictions")
        if not stale_banner("predictions") or cache3["data"] is None:
            st.stop()

        pdata3 = cache3["data"]

        # ── Confirmed Announcements ────────────────────────────────────────────
        section("Confirmed Plant & Facility Announcements")
        confirmed = pdata3.get("confirmed_announcements", [])

        # Filter out error placeholders
        confirmed = [a for a in confirmed if a.get("company") and a.get("company") != "Error"]

        if not confirmed:
            st.info(
                "No confirmed facility announcements found in the current news cycle. "
                "Agent 3 refreshes every 24 hours — check back after the next run or ensure GROQ_API_KEY is set in .env."
            )
        else:
            # ── Summary metrics ────────────────────────────────────────────────
            type_counts = {}
            for a in confirmed:
                t = a.get("type", "Other")
                type_counts[t] = type_counts.get(t, 0) + 1

            m_cols = st.columns(min(4, len(type_counts) + 1))
            m_cols[0].metric("Total Announcements", len(confirmed))
            for i, (t, cnt) in enumerate(sorted(type_counts.items(), key=lambda x: -x[1])[:3], 1):
                m_cols[i].metric(t, cnt)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Type badge colors ──────────────────────────────────────────────
            TYPE_COLORS = {
                "Manufacturing Plant":    ("#1b5e20", "#e8f5e9"),
                "Warehouse / Distribution": ("#0d47a1", "#e3f2fd"),
                "Data Center":            ("#4a148c", "#f3e5f5"),
                "Semiconductor Fab":      ("#b71c1c", "#ffebee"),
                "Battery Plant":          ("#e65100", "#fff3e0"),
                "Headquarters":           ("#263238", "#eceff1"),
                "Training Center":        ("#004d40", "#e0f2f1"),
                "Research & Development": ("#880e4f", "#fce4ec"),
                "Other":                  ("#5d4037", "#efebe9"),
            }

            # ── Announcement cards ─────────────────────────────────────────────
            for ann in confirmed:
                co      = ann.get("company", "Unknown")
                ticker  = ann.get("ticker", "")
                atype   = ann.get("type", "Other")
                loc     = ann.get("location", "")
                invest  = ann.get("investment", "")
                jobs    = ann.get("jobs", "")
                detail  = ann.get("detail", "")
                impact  = ann.get("cre_impact", "")
                source  = ann.get("source", "")

                badge_fg, badge_bg = TYPE_COLORS.get(atype, ("#5d4037", "#efebe9"))
                ticker_str = f" &nbsp;·&nbsp; <span style='color:#888;font-size:0.8rem;'>{ticker}</span>" if ticker else ""
                invest_str = f"<b>Investment:</b> {invest}" if invest else ""
                jobs_str   = f"<b>Jobs:</b> {jobs}" if jobs else ""
                meta_parts = [x for x in [invest_str, jobs_str] if x]
                meta_line  = "&nbsp;&nbsp;|&nbsp;&nbsp;".join(meta_parts) if meta_parts else ""

                st.markdown(f"""
                <div style="background:#fff;border:1px solid #e0e0e0;border-radius:8px;
                            padding:16px 20px;margin-bottom:12px;border-left:4px solid {badge_fg};">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                    <span style="background:{badge_bg};color:{badge_fg};font-size:0.72rem;
                                 font-weight:700;padding:2px 8px;border-radius:4px;
                                 text-transform:uppercase;letter-spacing:0.5px;">{atype}</span>
                    <span style="font-size:1rem;font-weight:700;color:#1a1a1a;">{co}{ticker_str}</span>
                    <span style="font-size:0.85rem;color:#555;margin-left:auto;">{loc}</span>
                  </div>
                  <div style="font-size:0.9rem;color:#333;margin-bottom:6px;">{detail}</div>
                  {"<div style='font-size:0.82rem;color:#555;margin-bottom:6px;'>" + meta_line + "</div>" if meta_line else ""}
                  {"<div style='font-size:0.82rem;color:#1b5e20;'><b>CRE Opportunity:</b> " + impact + "</div>" if impact else ""}
                  <div style="font-size:0.72rem;color:#aaa;margin-top:6px;">Source: {source}</div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                "Announcements extracted from live RSS feeds: Reuters, Manufacturing.net, IndustryWeek, "
                "PR Newswire, Business Wire, Dept. of Energy, Commerce Dept., EDA, and industry publications. "
                "Only confirmed/announced projects are shown — not speculation."
            )

        # ── Top 5 States context ───────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section("Top 5 States Attracting Corporate Investment")
        top5 = pdata3.get("top5_states", [])
        if top5:
            top5_df = pd.DataFrame(top5)
            cols_show = [c for c in ["state_name","state_abbr","pop_growth_pct","biz_score","key_companies","growth_drivers"] if c in top5_df.columns]
            top5_df = top5_df[cols_show].copy()
            if "pop_growth_pct" in top5_df.columns:
                top5_df["pop_growth_pct"] = top5_df["pop_growth_pct"].apply(lambda x: f"{x:+.2f}%")
            rename = {"state_name": "State", "state_abbr": "Abbr", "pop_growth_pct": "Pop Growth",
                      "biz_score": "Business Score", "key_companies": "Recent Corporate Moves",
                      "growth_drivers": "Growth Drivers"}
            top5_df.rename(columns=rename, inplace=True)
            st.dataframe(top5_df, use_container_width=True, hide_index=True)
            st.caption(
                "These states rank highest on combined population growth and business migration scores. "
                "Cross-reference with announcements above to identify where CRE demand is building fastest."
            )


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 4 — CHEAPEST BUILDINGS
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("#### Cheapest commercial buildings to purchase in the top 3 migration destination states")
        st.markdown(
            "Agent 3 sources the lowest-price commercial listings in the states with the highest "
            "migration and business growth scores — identifying acquisition opportunities before demand peaks. "
            "Updates every 24 hours alongside company predictions."
        )
        agent_last_updated("predictions")

        cache4 = read_cache("predictions")
        if not stale_banner("predictions") or cache4["data"] is None:
            st.stop()

        pdata4   = cache4["data"]
        listings = pdata4.get("listings", {})
        top3_abbr = pdata4.get("top3_abbr", [])

        if not listings:
            st.info("Listings will appear after the first scheduled agent run (every 24 hours).")
        else:
            from src.cre_listings import format_listing_card

            for abbr in top3_abbr:
                state_listings = listings.get(abbr, [])
                if not state_listings:
                    continue

                # Get migration cache for state name
                mig_cache = read_cache("migration")
                state_name = abbr
                if mig_cache["data"]:
                    mig_df2 = pd.DataFrame(mig_cache["data"]["migration"])
                    row = mig_df2[mig_df2["state_abbr"] == abbr]
                    if not row.empty:
                        state_name = row.iloc[0]["state_name"]

                section(f" {state_name} ({abbr}) — Cheapest Commercial Properties")
                st.caption(f"Showing {len(state_listings)} lowest-price listings sorted by asking price")

                for listing in state_listings:
                    if isinstance(listing, dict):
                        price_fmt  = f"${listing.get('price', 0):,}"
                        sqft_fmt   = f"{listing.get('sqft', 0):,} sqft"
                        ppsf_fmt   = f"${listing.get('price_per_sqft', 0):.0f}/sqft"
                        cap_fmt    = f"{listing.get('cap_rate', 0):.2f}% cap rate"
                        noi_fmt    = f"${listing.get('noi_annual', 0):,}/yr NOI"
                        dom_fmt    = f"{listing.get('days_on_market', 0)}d on market"
                        built_fmt  = f"Built {listing.get('year_built', 'N/A')}"
                        pt_fmt     = listing.get("property_type", "")
                        addr_fmt   = f"{listing.get('address', '')}, {listing.get('city', '')}, {listing.get('state', '')}"
                        highlights = listing.get("highlights", "")

                        st.markdown(f"""
                        <div class="listing-card">
                          <div class="l-price">{price_fmt}</div>
                          <div class="l-address">{addr_fmt}</div>
                          <div style="margin:4px 0;">
                            <span class="l-tag">{pt_fmt}</span>
                            <span class="l-tag">{cap_fmt}</span>
                            <span class="l-tag">{dom_fmt}</span>
                          </div>
                          <div class="l-detail">{sqft_fmt} · {ppsf_fmt} · {noi_fmt} · {built_fmt}</div>
                          {"<div class='l-detail' style='color:#555;margin-top:4px;font-style:italic;'> " + highlights + "</div>" if highlights else ""}
                        </div>
                        """, unsafe_allow_html=True)

            # ── Why these markets? ────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            section(" Why These Markets? Investment Thesis")

            mig_cache2 = read_cache("migration")
            if mig_cache2["data"]:
                mig_df3 = pd.DataFrame(mig_cache2["data"]["migration"])
                top3_rows = mig_df3[mig_df3["state_abbr"].isin(top3_abbr)].head(3)
                for _, row in top3_rows.iterrows():
                    with st.expander(f" {row['state_name']} ({row['state_abbr']}) — Why Invest Here?"):
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Population Growth", f"{row['pop_growth_pct']:+.2f}%", "YoY")
                        col_b.metric("Business Score", str(row['biz_score']), "Migration index")
                        col_c.metric("Composite Score", str(row['composite_score']), "0–100 scale")
                        st.markdown(f"**Key Companies:** {row['key_companies']}")
                        st.markdown(f"**Growth Drivers:** {row['growth_drivers']}")


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 5 — SYSTEM MONITOR / DEBUGGER
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("#### Where are companies building? Live facility & investment announcements across the US")
        st.markdown(
            "Agent 5 monitors **news wires, government press releases, and industry publications** "
            "every 4 hours — surfacing companies that have announced new manufacturing plants, "
            "training centers, data centers, warehouses, and other large facilities. "
            "Sources include Reuters, IndustryWeek, PR Newswire, DOE, Commerce Dept, and EDA."
        )
        agent_last_updated("news")

        cache_news = read_cache("news")
        if not stale_banner("news") or cache_news["data"] is None:
            st.stop()

        ndata = cache_news["data"]

        # ── KPI strip ─────────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric_card("Articles Found", str(ndata.get("article_count", 0)), "Facility announcements"), unsafe_allow_html=True)
        c2.markdown(metric_card("Sources Checked", str(ndata.get("sources_checked", 0)), "News + gov feeds"), unsafe_allow_html=True)
        fetched = ndata.get("fetched_at", "")
        fetched_label = datetime.fromisoformat(fetched).strftime("%b %d, %Y %I:%M %p") if fetched else "N/A"
        c3.markdown(metric_card("Last Scan", fetched_label, "Updates every 4 hours"), unsafe_allow_html=True)

        # ── AI Summary ────────────────────────────────────────────────────────
        section(" Agent 5 — AI Investment Brief: Facility Announcements")
        summary = ndata.get("summary", "")
        if summary:
            st.markdown(f"""
            <div class="agent-card">
              <div class="agent-label"> Agent 5 · Industry Announcements · {datetime.today().strftime('%b %d, %Y')}</div>
              <div class="agent-text">{summary}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("News summary is generated every 4 hours. Ensure GROQ_API_KEY is set in .env.")

        # ── Raw Article Feed ─────────────────────────────────────────────────
        raw = ndata.get("raw_articles", [])
        if raw:
            section(f" Raw Announcement Feed ({len(raw)} articles)")

            feed_type_filter = st.selectbox(
                "Filter by source type",
                options=["All", "news", "industry", "press", "government"],
                key="news_filter",
            )

            source_colors = {
                "government": "#1565c0",
                "industry":   "#2e7d32",
                "press":      "#6a1b9a",
                "news":       "#bf360c",
            }

            for art in raw:
                if feed_type_filter != "All" and art.get("feed_type") != feed_type_filter:
                    continue
                ft    = art.get("feed_type", "news")
                color = source_colors.get(ft, "#555")
                link  = art.get("link", "#")
                title = art.get("title", "No title")
                desc  = art.get("description", "")[:280]
                src   = art.get("source", "")
                date  = art.get("pub_date", "")[:22]

                href = f'<a href="{link}" target="_blank" style="color:{color};font-weight:700;text-decoration:none;">{title}</a>' if link and link != "#" else f'<span style="font-weight:700;">{title}</span>'

                st.markdown(f"""
                <div style="border-left:3px solid {color};padding:10px 14px;margin:6px 0;background:#fafafa;border-radius:4px;">
                  <div style="font-size:0.7rem;color:{color};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
                    {src} &nbsp;·&nbsp; {ft.upper()} &nbsp;·&nbsp; {date}
                  </div>
                  <div style="font-size:0.9rem;margin-bottom:4px;">{href}</div>
                  <div style="font-size:0.8rem;color:#555;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        st.caption(
            "Sources: Reuters, Manufacturing.net, IndustryWeek, PR Newswire, Business Wire, "
            "US Dept of Energy, US Dept of Commerce, EDA, Expansion Solutions, Site Selection Magazine."
        )


    with tab6:
        st.markdown("#### Background Agent Monitor — Agent 4 runs every 30 minutes")
        st.markdown(
            "Agent 4 continuously verifies that all data sources are live, caches are fresh, "
            "and APIs are reachable. This tab shows the live health dashboard."
        )

        # ── Agent Status ────────────────────────────────────────────────────────
        section(" Agent Status")
        status = get_status()

        agent_labels = {
            "migration":      ("Agent 1", "Population & Migration", "Every 6h"),
            "pricing":        ("Agent 2", "REIT Pricing",           "Every 1h"),
            "predictions":    ("Agent 3", "Company Predictions",    "Every 24h"),
            "debugger":       ("Agent 4", "Debugger / Monitor",     "Every 30min"),
            "news":           ("Agent 5", "Industry Announcements", "Every 4h"),
            "rates":          ("Agent 6", "Interest Rate & Debt",   "Every 1h"),
            "energy":         ("Agent 7", "Energy & Construction",  "Every 6h"),
            "sustainability": ("Agent 8", "Sustainability & ESG",   "Every 6h"),
        }

        cols = st.columns(len(agent_labels))
        for col, (agent_key, (num, name, freq)) in zip(cols, agent_labels.items()):
            s = status.get(agent_key, {})
            st_val = s.get("status", "idle")
            runs   = s.get("runs", 0)
            err    = s.get("last_error", None)
            icon   = {"ok": "", "running": "⏳", "error": "", "idle": ""}.get(st_val, "")
            color  = {"ok": "status-ok", "running": "status-run", "error": "status-error", "idle": "status-idle"}.get(st_val, "")
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{num} · {freq}</div>
              <div class="value">{icon}</div>
              <div class="sub">{name}</div>
              <div class="{color}" style="font-size:0.75rem;margin-top:4px;">{st_val.upper()} · {runs} runs</div>
              {"<div style='color:#b71c1c;font-size:0.7rem;margin-top:4px;'>⚠ " + str(err)[:60] + "</div>" if err else ""}
            </div>
            """, unsafe_allow_html=True)

        # ── Cache Ages ─────────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Cache Status")
        cache_keys = [
            ("migration",           "Every 6h",    7),
            ("pricing",             "Every 1h",    2),
            ("predictions",         "Every 24h",   25),
            ("debugger",            "Every 30min", 1),
            ("news",                "Every 4h",    5),
            ("rates",               "Every 1h",    2),
            ("energy_data",         "Every 6h",    7),
            ("sustainability_data", "Every 6h",    7),
        ]
        c_cols = st.columns(len(cache_keys))
        for col, (key, freq, max_h) in zip(c_cols, cache_keys):
            c = read_cache(key)
            age_label = cache_age_label(key)
            stale = c.get("stale", True)
            has_data = c["data"] is not None
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{key.title()} Cache</div>
              <div class="value">{"" if has_data and not stale else ("⚠" if has_data else "")}</div>
              <div class="sub">{age_label}</div>
              <div style="font-size:0.72rem;color:#888;margin-top:4px;">Refresh: {freq} · Max age: {max_h}h</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Health Report from Debugger Agent ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Health Report — Agent 4 Output")

        dbg_cache = read_cache("debugger")
        if dbg_cache["data"]:
            dbg = dbg_cache["data"]
            checked = dbg.get("checked_at", "unknown")
            st.caption(f"Last health check: {checked}")

            issues  = dbg.get("issues", [])
            healthy = dbg.get("healthy", [])

            if healthy:
                st.markdown("**Healthy systems:**")
                for h in healthy:
                    st.markdown(f"- {h}")

            if issues:
                st.markdown("**Issues detected:**")
                for i in issues:
                    st.markdown(f"- {i}")
            elif healthy:
                st.success("All systems healthy — no issues detected.")

            # ── Agent sub-status from debugger ────────────────────────────────
            agent_status_in_dbg = dbg.get("agent_status", {})
            if agent_status_in_dbg:
                st.markdown("<br>", unsafe_allow_html=True)
                section(" Last Known Agent States (from Debugger)")
                dbg_df = pd.DataFrame([
                    {
                        "Agent":    k,
                        "Status":   v.get("status", "?"),
                        "Last Run": v.get("last_run", "Never"),
                        "Runs":     v.get("runs", 0),
                        "Last Error": (v.get("last_error") or "")[:80],
                    }
                    for k, v in agent_status_in_dbg.items()
                ])
                st.dataframe(dbg_df, use_container_width=True, hide_index=True)
        else:
            st.info("Debugger agent has not completed its first run yet. Refresh in 30 seconds.")

        st.caption(
            "All agents run independently in background threads managed by APScheduler. "
            "Data is stored in JSON cache files and survives Streamlit reruns. "
            "Requires GROQ_API_KEY in .env for Agent 3 predictions."
        )



with main_tab_energy:
    tab_rates, tab_energy, tab_esg = st.tabs([
        "Rate Environment",
        "Energy & Construction Costs",
        "Sustainability",
    ])


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — RATE ENVIRONMENT
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_rates:
        st.markdown("#### How do current interest rates affect CRE cap rates, valuations, and REIT debt risk?")
        st.markdown(
            "Agent 6 pulls **live rate data from FRED**, classifies the rate environment, "
            "computes dynamic cap rate adjustments by property type, and scores REIT refinancing risk. "
            "Updates every hour. Requires `FRED_API_KEY` in `.env`."
        )
        agent_last_updated("rates")

        cache_r = read_cache("rates")
        rdata   = cache_r.get("data") or {}

        if not rdata or rdata.get("error"):
            err = rdata.get("error") if rdata else None
            if err:
                st.warning(f"⚠ {err}")
            else:
                st.info("Rate data is fetched every hour. Check that `FRED_API_KEY` is set in `.env`.")
            st.stop()

        rates       = rdata.get("rates", {})
        env         = rdata.get("environment", {})
        cap_adj     = rdata.get("cap_rate_adjustments", [])
        debt_risk   = rdata.get("reit_debt_risk", [])
        yc          = rdata.get("yield_curve", {})
        current_10y = rdata.get("current_10y")
        baseline    = rdata.get("baseline_10y", 4.0)
        cached_at   = rdata.get("cached_at", "")

        # ── Signal Banner ───────────────────────────────────────────────────────
        signal  = env.get("signal", "CAUTIOUS")
        sig_clr = {"BULLISH": "#1b5e20", "CAUTIOUS": "#e65100", "BEARISH": "#b71c1c"}.get(signal, "#333")
        bg_clr  = {"BULLISH": "#e8f5e9", "CAUTIOUS": "#fff3e0", "BEARISH": "#ffebee"}.get(signal, "#f5f5f5")
        st.markdown(f"""
        <div style="background:{bg_clr};border-left:6px solid {sig_clr};
                    padding:18px 24px;border-radius:6px;margin-bottom:20px;">
          <div style="font-size:1.4rem;font-weight:700;color:{sig_clr};">
            {env.get('icon','⚪')} Rate Environment: {signal}
          </div>
          <div style="color:#333;margin-top:6px;font-size:0.95rem;">{env.get('summary','')}</div>
          <ul style="margin-top:10px;color:#444;font-size:0.88rem;">
            {"".join(f"<li>{b}</li>" for b in env.get('bullets', []))}
          </ul>
          <div style="font-size:0.75rem;color:#888;margin-top:8px;">Last updated: {cached_at[:19].replace('T',' ')}</div>
        </div>""", unsafe_allow_html=True)

        # ── Key Rate Cards ──────────────────────────────────────────────────────
        section(" Current Rates")
        display_rates = [
            "10Y Treasury", "2Y Treasury", "Fed Funds Rate",
            "SOFR", "30Y Mortgage", "Prime Rate", "IG Corp Spread",
        ]
        card_cols = st.columns(len(display_rates))
        for col, name in zip(card_cols, display_rates):
            r = rates.get(name)
            if not r:
                col.markdown(metric_card(name, "N/A", "No data"), unsafe_allow_html=True)
                continue
            curr  = r["current"]
            d1w   = r.get("delta_1w")
            unit  = r["unit"]
            val_s = f"{curr:.2f}{unit}" if unit == "%" else f"{curr:.0f}{unit}"
            delta_s = ""
            if d1w is not None:
                arrow = "▲" if d1w > 0 else ("▼" if d1w < 0 else "→")
                color = "#b71c1c" if d1w > 0 else ("#1b5e20" if d1w < 0 else "#555")
                delta_s = f"<span style='color:{color};font-size:0.78rem;'>{arrow} {abs(d1w):.2f}{unit} 1W</span>"
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{name}</div>
              <div class="value">{val_s}</div>
              <div class="sub">{delta_s}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Yield Curve + Rate Comparison Table ─────────────────────────────────
        col_yc, col_tbl = st.columns([3, 2])
        with col_yc:
            section(" Yield Curve (Current Shape)")
            if yc:
                tenor_order = ["3M", "2Y", "5Y", "10Y", "30Y"]
                tenors  = [t for t in tenor_order if t in yc]
                values  = [yc[t] for t in tenors]
                inverted = yc.get("2Y", 0) > yc.get("10Y", 0)
                line_clr = "#b71c1c" if inverted else "#1b5e20"
                fig_yc = go.Figure()
                fig_yc.add_trace(go.Scatter(
                    x=tenors, y=values,
                    mode="lines+markers+text",
                    line=dict(color=line_clr, width=3),
                    marker=dict(size=10, color=line_clr),
                    text=[f"{v:.2f}%" for v in values],
                    textposition="top center",
                    textfont=dict(size=11, color="#1a1a1a"),
                    hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
                ))
                fig_yc.add_hline(y=values[0] if values else 0, line_dash="dot",
                                  line_color="#aaa", line_width=1)
                fig_yc.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    yaxis=dict(title="Yield (%)", ticksuffix="%",
                               gridcolor="#f0f0f0", tickfont=dict(color="#1a1a1a"),
                               title_font=dict(color="#1a1a1a")),
                    xaxis=dict(tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                    margin=dict(t=30, b=30, l=60, r=20), height=300,
                    font=dict(family="Source Sans Pro", color="#1a1a1a"),
                    annotations=[dict(
                        text="⚠ INVERTED" if inverted else "Normal slope",
                        x=0.5, y=1.08, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=12,
                        color="#b71c1c" if inverted else "#1b5e20", family="Source Sans Pro"),
                    )],
                )
                st.plotly_chart(fig_yc, use_container_width=True)
                st.caption(
                    "A normal (upward-sloping) yield curve signals healthy economic expectations. "
                    "An inverted curve — where short-term rates exceed long-term — historically precedes recessions "
                    "and tightens CRE lending conditions as banks compress their net interest margins."
                )
            else:
                st.info("Yield curve data unavailable.")

        with col_tbl:
            section(" Rate Change Summary")
            tbl_rows = []
            for name in display_rates:
                r = rates.get(name)
                if not r:
                    continue
                unit = r["unit"]
                def _fmt(v):
                    if v is None: return "—"
                    return f"{v:+.2f}{unit}" if unit == "%" else f"{v:+.0f}{unit}"
                tbl_rows.append({
                    "Rate":   name,
                    "Now":    f"{r['current']:.2f}{unit}" if unit == "%" else f"{r['current']:.0f}{unit}",
                    "1W Δ":   _fmt(r.get("delta_1w")),
                    "1M Δ":   _fmt(r.get("delta_1m")),
                    "1Y Δ":   _fmt(r.get("delta_1y")),
                })
            if tbl_rows:
                tbl_df = pd.DataFrame(tbl_rows)
                st.dataframe(tbl_df, use_container_width=True, hide_index=True, height=280)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Rate Trends (12 months) ──────────────────────────────────────────────
        section(" Rate Trends — Past 12 Months")
        trend_series = ["10Y Treasury", "2Y Treasury", "Fed Funds Rate", "SOFR"]
        trend_colors = ["#1565c0", "#e65100", "#1b5e20", "#6a1b9a"]
        fig_tr = go.Figure()
        for sname, clr in zip(trend_series, trend_colors):
            r = rates.get(sname)
            if not r or not r.get("series"):
                continue
            series = r["series"]
            # Filter to last 365 calendar days
            cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            pts = [o for o in series if o["date"] >= cutoff]
            if not pts:
                continue
            dates  = [o["date"] for o in pts]
            values = [o["value"] for o in pts]
            fig_tr.add_trace(go.Scatter(
                x=dates, y=values, name=sname,
                mode="lines", line=dict(color=clr, width=2),
                hovertemplate=f"{sname}: %{{y:.3f}}%<br>%{{x}}<extra></extra>",
            ))
        # Shade inverted periods (2Y > 10Y)
        t10_s = rates.get("10Y Treasury", {}).get("series", [])
        t2_s  = rates.get("2Y Treasury",  {}).get("series", [])
        if t10_s and t2_s:
            t10_d = {o["date"]: o["value"] for o in t10_s}
            t2_d  = {o["date"]: o["value"] for o in t2_s}
            cutoff = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            inv_dates = sorted(d for d in t10_d if d >= cutoff and d in t2_d and t2_d[d] > t10_d[d])
            if inv_dates:
                fig_tr.add_vrect(
                    x0=inv_dates[0], x1=inv_dates[-1],
                    fillcolor="rgba(183,28,28,0.07)", line_width=0,
                    annotation_text="Inverted", annotation_position="top left",
                    annotation_font=dict(color="#b71c1c", size=10),
                )
        fig_tr.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            yaxis=dict(title="Rate (%)", ticksuffix="%", gridcolor="#f0f0f0",
                       tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            xaxis=dict(tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            legend=dict(orientation="h", y=1.08, font=dict(color="#1a1a1a", size=11)),
            margin=dict(t=40, b=40), height=380,
            font=dict(family="Source Sans Pro", color="#1a1a1a"),
        )
        st.plotly_chart(fig_tr, use_container_width=True)
        st.caption(
            "Tracks the 10-year Treasury, 2-year Treasury, and Fed Funds rate over the past 12 months. "
            "The spread between the 10Y and 2Y (the yield curve slope) is a key leading indicator — "
            "a narrowing or inverted spread signals reduced appetite for long-duration CRE debt."
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Cap Rate Adjustment Impact ───────────────────────────────────────────
        section(f" Cap Rate Adjustment — Current 10Y ({current_10y:.2f}%) vs. {baseline:.1f}% Baseline")
        if cap_adj:
            adj_df = pd.DataFrame(cap_adj)
            col_bars, col_impact = st.columns([3, 2])
            with col_bars:
                pt_labels = [p.split("/")[0].strip() for p in adj_df["Property Type"]]
                base_caps = adj_df["Baseline Cap Rate"].tolist()
                adj_caps  = adj_df["Adjusted Cap Rate"].tolist()
                adj_bps   = adj_df["Rate Adjustment bps"].tolist()
                bar_colors = ["#b71c1c" if v > 0 else "#1b5e20" for v in adj_bps]

                fig_cap = go.Figure()
                fig_cap.add_trace(go.Bar(
                    name="Baseline Cap Rate",
                    x=pt_labels, y=base_caps,
                    marker_color="#CFB991",
                    text=[f"{v:.2f}%" for v in base_caps],
                    textposition="inside", textfont=dict(color="#1a1a1a", size=10),
                ))
                fig_cap.add_trace(go.Bar(
                    name="Rate-Adjusted Cap Rate",
                    x=pt_labels, y=adj_caps,
                    marker_color=bar_colors,
                    opacity=0.85,
                    text=[f"{v:.2f}%\n({'+' if d>0 else ''}{d:.0f}bps)" for v, d in zip(adj_caps, adj_bps)],
                    textposition="outside", textfont=dict(color="#1a1a1a", size=9),
                ))
                fig_cap.update_layout(
                    barmode="group",
                    paper_bgcolor="white", plot_bgcolor="white",
                    yaxis=dict(title="Cap Rate (%)", ticksuffix="%", gridcolor="#f0f0f0",
                               tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                    xaxis=dict(tickangle=-20, tickfont=dict(color="#1a1a1a", size=9)),
                    legend=dict(orientation="h", y=1.1, font=dict(color="#1a1a1a")),
                    margin=dict(t=40, b=60), height=360,
                    font=dict(family="Source Sans Pro", color="#1a1a1a"),
                )
                st.plotly_chart(fig_cap, use_container_width=True)
                st.caption(
                    f"Adjustment = (10Y Treasury − {baseline:.1f}% baseline) × property-type interest rate beta. "
                    "Red bars mean cap rates have expanded from the baseline — asset values have declined for the same NOI. "
                    "Green bars mean cap rate compression — favorable for existing owners but tougher for new buyers on yield."
                )

            with col_impact:
                st.markdown("**Profit Margin Impact by Property Type**")
                disp_adj = adj_df[["Property Type", "Baseline Cap Rate", "Adjusted Cap Rate",
                                    "Rate Adjustment bps", "Static Margin %", "Adj Margin %", "Margin Delta bps"]].copy()
                disp_adj["Property Type"] = disp_adj["Property Type"].str.split("/").str[0].str.strip()

                def _colour_delta(val):
                    try:
                        v = float(val)
                        if v < 0: return "color: #b71c1c"
                        if v > 0: return "color: #1b5e20"
                    except: pass
                    return ""

                styled = disp_adj.style.applymap(_colour_delta, subset=["Margin Delta bps"])
                st.dataframe(styled, use_container_width=True, hide_index=True, height=290)
                delta_10y = (current_10y or 0) - baseline
                direction = "above" if delta_10y > 0 else "below"
                st.caption(
                    f"10Y is {abs(delta_10y):.2f}% {direction} the {baseline:.1f}% baseline. "
                    "Higher cap rates → lower property multiples → compressed effective margins."
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── REIT Refinancing Risk ────────────────────────────────────────────────
        section(" REIT Near-Term Refinancing Risk")
        if debt_risk:
            debt_df = pd.DataFrame(debt_risk)
            risk_colors = {"High": "#b71c1c", "Medium": "#e65100", "Low": "#1b5e20"}

            col_chart, col_tbl2 = st.columns([2, 3])
            with col_chart:
                sorted_df = debt_df.sort_values("Risk %", ascending=True).tail(20)
                bar_clrs  = [risk_colors.get(r, "#888") for r in sorted_df["Risk Level"]]
                fig_debt  = go.Figure(go.Bar(
                    x=sorted_df["Risk %"],
                    y=sorted_df["Ticker"],
                    orientation="h",
                    marker_color=bar_clrs,
                    text=sorted_df["Risk %"].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                    customdata=sorted_df[["Name", "Near-Term Debt $B", "Total Debt $B"]].values,
                    hovertemplate=(
                        "<b>%{y} — %{customdata[0]}</b><br>"
                        "Near-term: $%{customdata[1]:.2f}B<br>"
                        "Total debt: $%{customdata[2]:.2f}B<br>"
                        "Risk: %{x:.1f}%<extra></extra>"
                    ),
                ))
                fig_debt.update_layout(
                    paper_bgcolor="white", plot_bgcolor="white",
                    xaxis=dict(title="Near-Term Debt / Total Debt (%)", ticksuffix="%",
                               gridcolor="#f0f0f0", tickfont=dict(color="#1a1a1a"),
                               title_font=dict(color="#1a1a1a")),
                    yaxis=dict(tickfont=dict(color="#1a1a1a", size=9)),
                    margin=dict(t=20, b=40, l=60, r=60), height=460,
                    font=dict(family="Source Sans Pro", color="#1a1a1a"),
                )
                fig_debt.add_vline(x=25, line_dash="dash", line_color="#b71c1c",
                                    annotation_text="High risk threshold",
                                    annotation_font=dict(color="#b71c1c", size=9))
                fig_debt.add_vline(x=10, line_dash="dot", line_color="#e65100",
                                    annotation_text="Med threshold",
                                    annotation_font=dict(color="#e65100", size=9))
                st.plotly_chart(fig_debt, use_container_width=True)
                st.caption(
                    "Estimates the share of each REIT's debt maturing within 12 months relative to its market cap. "
                    "REITs above the medium threshold face refinancing pressure — in a high-rate environment, "
                    "rolling debt at elevated rates compresses FFO and can force asset sales or equity dilution."
                )

            with col_tbl2:
                def _risk_style(val):
                    return {"High": "color:#b71c1c;font-weight:700",
                            "Medium": "color:#e65100;font-weight:600",
                            "Low": "color:#1b5e20"}.get(val, "")
                styled_debt = debt_df.style.applymap(_risk_style, subset=["Risk Level"])
                st.dataframe(styled_debt, use_container_width=True, hide_index=True, height=460)

            high_risk = debt_df[debt_df["Risk Level"] == "High"]
            if not high_risk.empty:
                names = ", ".join(high_risk["Ticker"].tolist())
                st.warning(
                    f"⚠ **High refinancing risk:** {names} — these REITs have ≥25% of debt maturing "
                    f"within 12 months. Elevated 10Y rates ({current_10y:.2f}%) increase rollover costs."
                )
            else:
                st.success(" No REITs with high near-term refinancing risk detected.")

            st.caption(
                "Near-term debt = current portion of long-term debt from latest quarterly balance sheet (yfinance). "
                "Risk % = near-term / total debt. High ≥ 25%, Medium 10–25%, Low < 10%."
            )
        else:
            st.info("REIT debt data not yet available. The agent will populate this on its next run.")

        # ── Rate-Adjusted Pricing Note ───────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if cap_adj and current_10y:
            delta = current_10y - baseline
            direction_word = "above" if delta > 0 else "below"
            impact_word    = "expanding cap rates" if delta > 0 else "compressing cap rates"
            st.info(
                f" **Impact on Pricing & Profit tab:** With 10Y at **{current_10y:.2f}%** "
                f"({abs(delta):.2f}% {direction_word} the {baseline:.1f}% static benchmark), "
                f"rates are currently **{impact_word}** across all property types. "
                f"The adjustments above are applied in the rate-adjusted view on the Pricing & Profit tab."
            )

        st.caption(
            "Data: Federal Reserve Bank of St. Louis (FRED). "
            "Cap rate adjustment model: adjusted = benchmark + (10Y − baseline) × sector beta. "
            "This is research, not financial advice."
        )


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — ENERGY & CONSTRUCTION COSTS
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_energy:
        st.markdown("#### How are energy and material costs affecting CRE construction?")
        st.markdown(
            "Agent 6 tracks **oil, natural gas, copper, and steel** prices to derive a "
            "**Construction Cost Signal** that indicates whether building costs are rising or easing. Updates every 6 hours."
        )
        agent_last_updated("energy")

        cache_e = read_cache("energy_data")
        if cache_e["data"] is None:
            st.warning("⏳ Energy agent is fetching data for the first time — please wait ~30 seconds and refresh.")
            st.stop()
        age_e = cache_age_label("energy_data")
        st.caption(f" Last updated: {age_e} · Auto-refreshes in background")

        edata = cache_e["data"]
        commodities = edata.get("commodities", [])
        cost_signal = edata.get("construction_cost_signal", "UNKNOWN")
        avg_momentum = edata.get("avg_momentum_pct", 0)

        # ── KPI strip ──────────────────────────────────────────────────────────
        signal_color = {"HIGH": "", "MODERATE": "", "LOW": ""}.get(cost_signal, "⚪")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Construction Cost Signal", f"{signal_color} {cost_signal}",
                                 "Based on commodity momentum"), unsafe_allow_html=True)
        c2.markdown(metric_card("Avg Momentum", f"{avg_momentum:+.1f}%",
                                 "vs 60-day SMA"), unsafe_allow_html=True)
        c3.markdown(metric_card("Commodities Tracked", str(len(commodities)),
                                 "Oil, Gas, Copper, Steel, Energy"), unsafe_allow_html=True)
        c4.markdown(metric_card("Trading Days", str(edata.get("trading_days_analysed", 0)),
                                 "6-month lookback"), unsafe_allow_html=True)

        # ── Commodity Price Table ──────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section(" Commodity Prices vs 60-Day Moving Average")

        if commodities:
            comm_df = pd.DataFrame(commodities)

            # Bar chart — % above/below SMA
            colors_comm = [GOLD if p >= 0 else "#c62828" for p in comm_df["pct_above_sma"]]
            fig_comm = go.Figure(go.Bar(
                x=comm_df["label"], y=comm_df["pct_above_sma"],
                marker_color=colors_comm,
                text=comm_df["pct_above_sma"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
                customdata=comm_df[["latest_price", "sma_60"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>Price: $%{customdata[0]:.2f}<br>"
                    "SMA-60: $%{customdata[1]:.2f}<br>"
                    "vs SMA: %{y:+.1f}%<extra></extra>"
                ),
            ))
            fig_comm.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="% Above/Below SMA-60",
                yaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                xaxis=dict(tickfont=dict(color="#1a1a1a")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#1a1a1a"),
            )
            st.plotly_chart(fig_comm, use_container_width=True)
            st.caption(
                "Each bar shows the commodity's latest price versus its 60-day moving average. "
                "Bars above the baseline indicate costs are elevated — signaling higher construction costs for developers. "
                "Copper and steel are the most direct inputs for building; oil and natural gas drive operating expenses and transport."
            )

            # Detail table
            section(" Commodity Detail")
            disp_comm = comm_df[["label", "latest_price", "sma_60", "pct_above_sma"]].copy()
            disp_comm.columns = ["Commodity", "Latest Price", "SMA-60", "% vs SMA"]
            disp_comm["Latest Price"] = disp_comm["Latest Price"].apply(lambda x: f"${x:.2f}")
            disp_comm["SMA-60"] = disp_comm["SMA-60"].apply(lambda x: f"${x:.2f}")
            disp_comm["% vs SMA"] = disp_comm["% vs SMA"].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(disp_comm, use_container_width=True, hide_index=True)

        st.caption(
            "Data sourced from Yahoo Finance (USO, UNG, XLE, CPER, SLX). "
            "Construction Cost Signal: HIGH (>+5%), MODERATE (±5%), LOW (<−5%) based on avg momentum vs 60-day SMA."
        )


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB — SUSTAINABILITY & ESG
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab_esg:
        st.markdown("#### Is green capital flowing into real estate?")
        st.markdown(
            "Agent 7 monitors **clean-energy ETFs** and **green REITs** to gauge ESG momentum "
            "relative to the broad market (SPY). Updates every 6 hours."
        )
        agent_last_updated("sustainability")

        cache_s = read_cache("sustainability_data")
        if cache_s["data"] is None:
            st.warning("⏳ Sustainability agent is fetching data for the first time — please wait ~30 seconds and refresh.")
            st.stop()
        age_s = cache_age_label("sustainability_data")
        st.caption(f" Last updated: {age_s} · Auto-refreshes in background")

        sdata = cache_s["data"]
        clean_energy = sdata.get("clean_energy", [])
        green_reits = sdata.get("green_reits", [])
        esg_signal = sdata.get("esg_momentum_signal", "UNKNOWN")
        bench_ret = sdata.get("benchmark_return_pct", 0)
        avg_clean_ret = sdata.get("avg_clean_energy_return_pct", 0)

        # ── KPI strip ──────────────────────────────────────────────────────────
        esg_icon = {"STRONG": "", "NEUTRAL": "", "WEAK": ""}.get(esg_signal, "⚪")
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("ESG Momentum Signal", f"{esg_icon} {esg_signal}",
                                 "Clean energy vs SPY"), unsafe_allow_html=True)
        c2.markdown(metric_card("Clean Energy Avg Return", f"{avg_clean_ret:+.1f}%",
                                 "6-month trailing"), unsafe_allow_html=True)
        c3.markdown(metric_card("SPY Benchmark Return", f"{bench_ret:+.1f}%",
                                 "6-month trailing"), unsafe_allow_html=True)
        spread = (avg_clean_ret or 0) - (bench_ret or 0)
        c4.markdown(metric_card("Spread vs Market", f"{spread:+.1f} pp",
                                 "Positive = outperforming"), unsafe_allow_html=True)

        # ── Clean Energy ETFs ──────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section("⚡ Clean Energy ETF Performance (ICLN, TAN, QCLN)")

        if clean_energy:
            ce_df = pd.DataFrame(clean_energy)
            fig_ce = go.Figure()
            colors_ce = [GOLD if r >= 0 else "#c62828" for r in ce_df["period_return_pct"]]
            fig_ce.add_trace(go.Bar(
                x=ce_df["label"], y=ce_df["period_return_pct"],
                marker_color=colors_ce,
                text=ce_df["period_return_pct"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
                name="6mo Return",
                customdata=ce_df[["latest_price", "sma_60", "pct_above_sma"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>Price: $%{customdata[0]:.2f}<br>"
                    "6mo Return: %{y:+.1f}%<br>"
                    "vs SMA-60: %{customdata[2]:+.1f}%<extra></extra>"
                ),
            ))
            # Add SPY benchmark line
            fig_ce.add_hline(y=bench_ret, line_dash="dash", line_color=BLACK,
                             annotation_text=f"SPY: {bench_ret:+.1f}%",
                             annotation_position="top right")
            fig_ce.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="6-Month Return (%)",
                yaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                xaxis=dict(tickfont=dict(color="#1a1a1a")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#1a1a1a"),
            )
            st.plotly_chart(fig_ce, use_container_width=True)
            st.caption(
                "Tracks ICLN (global clean energy), TAN (solar), and QCLN (clean tech) ETFs. "
                "Sustained outperformance signals growing institutional capital flows into green energy — "
                "a leading indicator of demand for solar-ready industrial space, EV charging infrastructure, and LEED-certified buildings."
            )

        # ── Green REITs ────────────────────────────────────────────────────────
        section(" Green REIT Performance (PLD, EQIX, ARE)")

        if green_reits:
            gr_df = pd.DataFrame(green_reits)
            colors_gr = [GOLD if r >= 0 else "#c62828" for r in gr_df["period_return_pct"]]
            fig_gr = go.Figure(go.Bar(
                x=gr_df["label"], y=gr_df["period_return_pct"],
                marker_color=colors_gr,
                text=gr_df["period_return_pct"].apply(lambda x: f"{x:+.1f}%"),
                textposition="outside",
                customdata=gr_df[["latest_price", "sma_60", "pct_above_sma"]].values,
                hovertemplate=(
                    "<b>%{x}</b><br>Price: $%{customdata[0]:.2f}<br>"
                    "6mo Return: %{y:+.1f}%<br>"
                    "vs SMA-60: %{customdata[2]:+.1f}%<extra></extra>"
                ),
            ))
            fig_gr.add_hline(y=bench_ret, line_dash="dash", line_color=BLACK,
                             annotation_text=f"SPY: {bench_ret:+.1f}%",
                             annotation_position="top right")
            fig_gr.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_title="6-Month Return (%)",
                yaxis=dict(gridcolor="#f0f0f0", zeroline=True, zerolinecolor="#ccc",
                           tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
                xaxis=dict(tickfont=dict(color="#1a1a1a")),
                margin=dict(t=30, b=30), height=320,
                font=dict(family="Source Sans Pro", color="#1a1a1a"),
            )
            st.plotly_chart(fig_gr, use_container_width=True)
            st.caption(
                "Prologis (PLD) is the world's largest industrial REIT with a strong LEED portfolio. "
                "Equinix (EQIX) powers its data centers with 90%+ renewable energy. "
                "Alexandria (ARE) focuses on carbon-neutral life science campuses. "
                "Outperformance vs. SPY indicates ESG-focused capital allocators are actively rotating into these names."
            )

        # ── Combined Detail Table ──────────────────────────────────────────────
        section(" Full Detail — Clean Energy & Green REITs")
        all_esg = clean_energy + green_reits
        if all_esg:
            esg_df = pd.DataFrame(all_esg)
            disp_esg = esg_df[["label", "latest_price", "period_return_pct", "sma_60", "pct_above_sma"]].copy()
            disp_esg.columns = ["Security", "Price", "6mo Return", "SMA-60", "% vs SMA"]
            disp_esg["Price"] = disp_esg["Price"].apply(lambda x: f"${x:.2f}")
            disp_esg["6mo Return"] = disp_esg["6mo Return"].apply(lambda x: f"{x:+.1f}%")
            disp_esg["SMA-60"] = disp_esg["SMA-60"].apply(lambda x: f"${x:.2f}")
            disp_esg["% vs SMA"] = disp_esg["% vs SMA"].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(disp_esg, use_container_width=True, hide_index=True)

        st.caption(
            "Data sourced from Yahoo Finance. ESG Momentum Signal: STRONG (clean energy outperforms SPY by >2pp), "
            "NEUTRAL (±2pp), WEAK (underperforms by >2pp). Green REITs: Prologis (LEED), Equinix (renewables), Alexandria (carbon-neutral)."
        )


