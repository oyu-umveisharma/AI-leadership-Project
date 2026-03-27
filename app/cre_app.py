"""
Commercial Real Estate Intelligence Platform
Purdue MSF | Group AI Project

Four background agents update independently on schedules:
  Agent 1 — Population & Migration    (every 6h)
  Agent 2 — CRE Pricing & Profit      (every 1h)
  Agent 3 — Company Predictions       (every 24h, LLM)
  Agent 4 — Debugger / Monitor        (every 30min)

Run: streamlit run app/cre_app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Start background agents ──────────────────────────────────────────────────
from src.cre_agents import (
    start_scheduler, force_run, read_cache, cache_age_label, get_status,
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

st.set_page_config(
    page_title="CRE Intelligence | Purdue MSF",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
  <h1>🏢 CRE Intelligence Platform</h1>
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
        st.warning(f"⚠️ Data is stale (last updated {age}). Agent may be restarting.")
    else:
        st.caption(f"🔄 Last updated: {age} · Auto-refreshes in background")
    return True

def agent_force_button(agent_name: str, label: str, key_suffix: str = ""):
    key = f"force_{agent_name}{key_suffix}"
    col_btn, col_age = st.columns([2, 3])
    with col_btn:
        if st.button(f"⚡ Force Refresh {label}", key=key):
            force_run(agent_name)
            st.toast(f"{label} triggered — data will update in ~15s", icon="⚡")
    with col_age:
        st.caption(f"Last run: {cache_age_label(agent_name)}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN TAB
# ═══════════════════════════════════════════════════════════════════════════════
(main_tab,) = st.tabs(["🏢  Real Estate"])

with main_tab:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🗺️  Migration Intelligence",
        "💰  Pricing & Profit",
        "🔮  Company Predictions",
        "🏗️  Cheapest Buildings",
        "📰  Industry Announcements",
        "🛠️  System Monitor",
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
        agent_force_button("migration", "Migration Agent")

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
        section("🗺️ US Population Growth Map — Where America is Moving")

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
                         bgcolor="white", showland=True, landcolor="#f5f5f5"),
                paper_bgcolor="white",
                margin=dict(t=10, b=10, l=0, r=0),
                height=460,
                font=dict(family="Source Sans Pro", color="#1a1a1a"),
            )
            st.plotly_chart(fig_map, use_container_width=True)

        with legend_col:
            st.markdown("<br><br>", unsafe_allow_html=True)
            for color, label in [
                ("#1b5e20", "🟢 High Growth (70–100)"),
                ("#388e3c", "🟩 Growing (55–70)"),
                ("#fff9c4", "🟡 Stable (45–55)"),
                ("#ef5350", "🔴 Declining (25–45)"),
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
        section("🏆 Top 10 States for CRE Investment (Migration Score)")
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

        # ── Metro Table ────────────────────────────────────────────────────────
        section("🏙️ Top Metro Areas — Population, Jobs & CRE Demand")
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
        section("🏭 Business Migration vs. Population Growth")
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


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 2 — CRE PRICING & PROFIT
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### Where are the highest profit margins in commercial real estate today?")
        st.markdown(
            "Agent 2 pulls **live REIT pricing**, estimates **cap rates** and **NOI margins** by property type, "
            "and ranks market × property type combinations. Updates every hour."
        )
        agent_force_button("pricing", "Pricing Agent")

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
        section(f"📈 Live REIT Prices — {datetime.today().strftime('%B %d, %Y')}")

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
        section("🔥 Profit Margin Heatmap — Market × Property Type")

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
        st.caption("Effective Profit Margin = NOI Margin × (1 − Vacancy) × (1 + Rent Growth).")

        # ── Top 10 Opportunities ───────────────────────────────────────────────
        section("🏆 Top 10 Highest Profit Margin Opportunities Right Now")
        st.dataframe(top_opps, use_container_width=True, hide_index=True)

        # ── Property Type Comparison ───────────────────────────────────────────
        section("📊 Property Type Performance Comparison")
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
            "Cap rates and NOI margins sourced from CBRE, JLL, and Green Street 2024–2025 market reports. "
            "REIT prices are live from Yahoo Finance. This is research, not financial advice."
        )


    # ═══════════════════════════════════════════════════════════════════════════════
    #  TAB 3 — COMPANY PREDICTIONS
    # ═══════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### Which companies are most likely to relocate or expand in the next 12 months?")
        st.markdown(
            "Agent 3 uses an LLM (Llama 3.3-70B via Groq) to analyze migration signals, tax policy, "
            "labor markets, and corporate announcements — then predicts HQ moves. Updates every 24 hours."
        )
        agent_force_button("predictions", "Predictions Agent")

        cache3 = read_cache("predictions")
        if not stale_banner("predictions") or cache3["data"] is None:
            st.stop()

        pdata3 = cache3["data"]

        # ── AI Predictions ────────────────────────────────────────────────────
        section("🔮 AI-Predicted Corporate Relocations & Expansions (Next 12 Months)")
        pred_text = pdata3.get("predictions_text", "")
        if pred_text:
            st.markdown(f"""
            <div class="agent-card">
              <div class="agent-label">🤖 Agent 3 · Corporate Relocation Intelligence · {datetime.today().strftime('%b %d, %Y')}</div>
              <div class="agent-text">{pred_text}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Predictions not yet available. Click Force Refresh or check that GROQ_API_KEY is set in .env")

        # ── Top 5 States breakdown ─────────────────────────────────────────────
        section("📊 Top 5 States Driving Predictions")
        top5 = pdata3.get("top5_states", [])
        if top5:
            top5_df = pd.DataFrame(top5)
            cols_show = [c for c in ["state_name","state_abbr","pop_growth_pct","biz_score","key_companies","growth_drivers"] if c in top5_df.columns]
            top5_df = top5_df[cols_show].copy()
            if "pop_growth_pct" in top5_df.columns:
                top5_df["pop_growth_pct"] = top5_df["pop_growth_pct"].apply(lambda x: f"{x:+.2f}%")
            st.dataframe(top5_df, use_container_width=True, hide_index=True)


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
        agent_force_button("predictions", "Listings Agent", key_suffix="_listings")

        cache4 = read_cache("predictions")
        if not stale_banner("predictions") or cache4["data"] is None:
            st.stop()

        pdata4   = cache4["data"]
        listings = pdata4.get("listings", {})
        top3_abbr = pdata4.get("top3_abbr", [])

        if not listings:
            st.info("No listings available yet. Click Force Refresh above or wait for the agent to complete its first run.")
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

                section(f"🏗️ {state_name} ({abbr}) — Cheapest Commercial Properties")
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
                          {"<div class='l-detail' style='color:#555;margin-top:4px;font-style:italic;'>✓ " + highlights + "</div>" if highlights else ""}
                        </div>
                        """, unsafe_allow_html=True)

            # ── Why these markets? ────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            section("💡 Why These Markets? Investment Thesis")

            mig_cache2 = read_cache("migration")
            if mig_cache2["data"]:
                mig_df3 = pd.DataFrame(mig_cache2["data"]["migration"])
                top3_rows = mig_df3[mig_df3["state_abbr"].isin(top3_abbr)].head(3)
                for _, row in top3_rows.iterrows():
                    with st.expander(f"📍 {row['state_name']} ({row['state_abbr']}) — Why Invest Here?"):
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
        agent_force_button("news", "News Agent", key_suffix="_news")

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
        section("🤖 Agent 5 — AI Investment Brief: Facility Announcements")
        summary = ndata.get("summary", "")
        if summary:
            st.markdown(f"""
            <div class="agent-card">
              <div class="agent-label">🤖 Agent 5 · Industry Announcements · {datetime.today().strftime('%b %d, %Y')}</div>
              <div class="agent-text">{summary}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Summary not yet available. Click Force Refresh or ensure GROQ_API_KEY is set in .env")

        # ── Raw Article Feed ─────────────────────────────────────────────────
        raw = ndata.get("raw_articles", [])
        if raw:
            section(f"📋 Raw Announcement Feed ({len(raw)} articles)")

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

        col_refresh, _ = st.columns([2, 5])
        with col_refresh:
            if st.button("⚡ Force Debugger Run"):
                force_run("debugger")
                st.toast("Debugger triggered — refreshing in ~10s", icon="🛠️")

        # ── Agent Status ────────────────────────────────────────────────────────
        section("🤖 Agent Status")
        status = get_status()

        agent_labels = {
            "migration":   ("Agent 1", "Population & Migration", "Every 6h"),
            "pricing":     ("Agent 2", "REIT Pricing",           "Every 1h"),
            "predictions": ("Agent 3", "Company Predictions",    "Every 24h"),
            "debugger":    ("Agent 4", "Debugger / Monitor",     "Every 30min"),
            "news":        ("Agent 5", "Industry Announcements", "Every 4h"),
        }

        cols = st.columns(5)
        for col, (agent_key, (num, name, freq)) in zip(cols, agent_labels.items()):
            s = status.get(agent_key, {})
            st_val = s.get("status", "idle")
            runs   = s.get("runs", 0)
            err    = s.get("last_error", None)
            icon   = {"ok": "✅", "running": "⏳", "error": "❌", "idle": "💤"}.get(st_val, "❓")
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
        section("📦 Cache Status")
        cache_keys = [
            ("migration",   "Every 6h",   7),
            ("pricing",     "Every 1h",   2),
            ("predictions", "Every 24h",  25),
            ("debugger",    "Every 30min", 1),
            ("news",        "Every 4h",   5),
        ]
        c_cols = st.columns(5)
        for col, (key, freq, max_h) in zip(c_cols, cache_keys):
            c = read_cache(key)
            age_label = cache_age_label(key)
            stale = c.get("stale", True)
            has_data = c["data"] is not None
            col.markdown(f"""
            <div class="metric-card">
              <div class="label">{key.title()} Cache</div>
              <div class="value">{"✅" if has_data and not stale else ("⚠️" if has_data else "❌")}</div>
              <div class="sub">{age_label}</div>
              <div style="font-size:0.72rem;color:#888;margin-top:4px;">Refresh: {freq} · Max age: {max_h}h</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Health Report from Debugger Agent ──────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section("🔍 Health Report — Agent 4 Output")

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
                section("📋 Last Known Agent States (from Debugger)")
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

        # ── Manual force run buttons ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        section("⚡ Manual Agent Triggers")
        st.markdown("Force any agent to run immediately (runs in background — data appears after ~15-30s refresh):")
        b1, b2, b3, b4, b5 = st.columns(5)
        with b1:
            if st.button("🗺️ Run Migration Agent"):
                force_run("migration")
                st.toast("Migration agent triggered", icon="🗺️")
        with b2:
            if st.button("💰 Run Pricing Agent"):
                force_run("pricing")
                st.toast("Pricing agent triggered", icon="💰")
        with b3:
            if st.button("🔮 Run Predictions Agent"):
                force_run("predictions")
                st.toast("Predictions agent triggered (takes ~30s)", icon="🔮")
        with b4:
            if st.button("🛠️ Run Debugger Agent"):
                force_run("debugger")
                st.toast("Debugger agent triggered", icon="🛠️")
        with b5:
            if st.button("📰 Run News Agent"):
                force_run("news")
                st.toast("News agent triggered (takes ~20s)", icon="📰")

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(
            "All agents run independently in background threads managed by APScheduler. "
            "Data is stored in JSON cache files and survives Streamlit reruns. "
            "Requires GROQ_API_KEY in .env for Agent 3 predictions."
        )


# ── Meet the Team ─────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background:{BLACK};padding:28px 32px;border-top:4px solid {GOLD};text-align:center;">
  <div style="color:{GOLD};font-size:0.72rem;text-transform:uppercase;letter-spacing:3px;margin-bottom:16px;">Meet the Team</div>
  <div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;">
    <a href="https://www.linkedin.com/in/aayman-afzal/" target="_blank"
       style="color:{GOLD} !important;text-decoration:underline !important;font-size:0.95rem;font-weight:600;padding-bottom:2px;">
      Aayman Afzal
    </a>
    <a href="https://www.linkedin.com/in/ajinkyakodnikar/" target="_blank"
       style="color:{GOLD} !important;text-decoration:underline !important;font-size:0.95rem;font-weight:600;padding-bottom:2px;">
      Ajinkya Kodnikar
    </a>
    <a href="https://www.linkedin.com/in/oyu-amar/" target="_blank"
       style="color:{GOLD} !important;text-decoration:underline !important;font-size:0.95rem;font-weight:600;padding-bottom:2px;">
      Oyu Amar
    </a>
    <a href="https://www.linkedin.com/in/ricardo-ruiz1/" target="_blank"
       style="color:{GOLD} !important;text-decoration:underline !important;font-size:0.95rem;font-weight:600;padding-bottom:2px;">
      Ricardo Ruiz
    </a>
  </div>
  <div style="color:#666;font-size:0.75rem;margin-top:16px;">MGMT 690 · AI Leadership · Purdue Daniels School of Business</div>
</div>
""", unsafe_allow_html=True)
