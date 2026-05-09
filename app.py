"""
app.py — Quant Trading Strategy Dashboard
Run: streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.fetcher      import fetch_ohlcv, get_live_quote, annualization_factor, PRESET_SYMBOLS, SYMBOL_NAMES
from strategies.indicators import add_all
from strategies.signals    import STRATEGY_FN, DEFAULT_PARAMS
from backtest.engine       import run as backtest_run
from backtest.optimizer    import grid_search, walk_forward, PARAM_GRIDS

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quant Strategy Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2e3450;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; }
    .metric-label { font-size: 0.75rem; color: #888; margin-top: 4px; }
    .positive { color: #00d4aa; }
    .negative { color: #ff4d6d; }
    .neutral  { color: #a0aec0; }
    .signal-long  { background:#0d3321; color:#00d4aa; border-radius:6px; padding:4px 12px; font-weight:700; }
    .signal-short { background:#3d0d1a; color:#ff4d6d; border-radius:6px; padding:4px 12px; font-weight:700; }
    .signal-flat  { background:#1a1d27; color:#888;    border-radius:6px; padding:4px 12px; }
    div[data-testid="stSidebar"] { background-color: #0e1117; border-right: 1px solid #1e2130; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2130; border-radius: 8px 8px 0 0;
        padding: 8px 20px; color: #888;
    }
    .stTabs [aria-selected="true"] { background: #252a3a; color: #fff; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
COLORS = {
    "Trend Following":   "#00d4aa",
    "Mean Reversion":    "#7c83fd",
    "Donchian Breakout": "#f9a825",
    "Buy & Hold":        "#607d8b",
    "price":             "#e0e0e0",
}

@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker, timeframe):
    df = fetch_ohlcv(ticker, timeframe)
    return add_all(df)

def fmt_pct(v, suffix="%", decimals=2):
    if v is None: return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.{decimals}f}{suffix}"

def color_val(v, positive_good=True):
    if v is None or v == 0: return "neutral"
    return ("positive" if v > 0 else "negative") if positive_good else \
           ("negative" if v > 0 else "positive")

def metric_card(label, value, color_class="neutral", sub=None):
    sub_html = f"<div class='metric-label'>{sub}</div>" if sub else ""
    return f"""
    <div class='metric-card'>
        <div class='metric-value {color_class}'>{value}</div>
        <div class='metric-label'>{label}</div>
        {sub_html}
    </div>"""

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Symbol input
    st.markdown("**Asset**")
    preset_group = st.selectbox("Preset group", list(PRESET_SYMBOLS.keys()), index=0)
    preset_list  = PRESET_SYMBOLS[preset_group]
    ticker_input = st.selectbox("Symbol", preset_list, index=0)
    custom_tick  = st.text_input("Or type any ticker (e.g. NVDA, SPY)", "")
    ticker = custom_tick.strip().upper() if custom_tick.strip() else ticker_input

    st.markdown("---")

    # Timeframe
    st.markdown("**Timeframe**")
    timeframe = st.radio("", ["1d", "1h", "30m"], horizontal=True, index=0)

    st.markdown("---")

    # Strategy
    st.markdown("**Strategy**")
    strategy_name = st.selectbox("", list(STRATEGY_FN.keys()))

    st.markdown("---")

    # Strategy parameters (dynamic)
    st.markdown("**Parameters**")
    params = {}
    defs = DEFAULT_PARAMS[strategy_name]

    if strategy_name == "Trend Following":
        params["rsi_long"]       = st.slider("RSI Long threshold",  40, 65, defs["rsi_long"])
        params["rsi_short"]      = st.slider("RSI Short threshold", 35, 60, defs["rsi_short"])
        params["max_vol_regime"] = st.slider("Max Vol Regime",    1.0, 3.0, defs["max_vol_regime"], 0.1)
        params["min_hold"]       = st.slider("Min Hold (bars)",      1, 10,  defs["min_hold"])

    elif strategy_name == "Mean Reversion":
        params["rsi_oversold"]   = st.slider("RSI Oversold",    15, 40, defs["rsi_oversold"])
        params["rsi_overbought"] = st.slider("RSI Overbought",  60, 85, defs["rsi_overbought"])
        params["bb_oversold"]    = st.slider("BB Lower thresh", 0.0, 0.2, defs["bb_oversold"], 0.01)
        params["bb_overbought"]  = st.slider("BB Upper thresh", 0.8, 1.0, defs["bb_overbought"], 0.01)
        params["max_vol_regime"] = st.slider("Max Vol Regime",  0.5, 1.5, defs["max_vol_regime"], 0.1)
        params["min_hold"]       = st.slider("Min Hold (bars)",   1, 10,  defs["min_hold"])

    else:  # Donchian Breakout
        params["min_hold"]       = st.slider("Min Hold (bars)", 1, 10, defs["min_hold"])

    st.markdown("---")

    # Backtest settings
    st.markdown("**Backtest Settings**")
    fee_rate  = st.slider("Fee rate (%)",      0.0, 0.20, 0.05, 0.01) / 100
    slippage  = st.slider("Slippage (%)",      0.0, 0.10, 0.02, 0.01) / 100
    long_only = st.toggle("Long only", value=True)

    run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

# ─── Main area ────────────────────────────────────────────────────────────────
st.markdown(f"## 📈 {ticker}  ·  {strategy_name}")

# Live quote bar
quote = get_live_quote(ticker)
q1, q2, q3, q4, q5 = st.columns(5)
with q1:
    price_str = f"${quote['price']:,.4f}" if quote["price"] else "—"
    c = color_val(quote.get("change_pct"), True)
    st.markdown(metric_card("Last Price", price_str, c), unsafe_allow_html=True)
with q2:
    chg = quote.get("change_pct")
    st.markdown(metric_card("Day Change", fmt_pct(chg), color_val(chg)), unsafe_allow_html=True)
with q3:
    h52 = f"${quote['high_52w']:,.2f}" if quote["high_52w"] else "—"
    st.markdown(metric_card("52W High", h52, "neutral"), unsafe_allow_html=True)
with q4:
    l52 = f"${quote['low_52w']:,.2f}" if quote["low_52w"] else "—"
    st.markdown(metric_card("52W Low", l52, "neutral"), unsafe_allow_html=True)
with q5:
    vol_str = f"{quote['volume']:,}" if quote["volume"] else "—"
    st.markdown(metric_card("Avg 3M Vol", vol_str, "neutral"), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Load data
with st.spinner(f"Fetching {ticker} data…"):
    try:
        df = load_data(ticker, timeframe)
    except Exception as e:
        st.error(f"❌ Could not fetch data for **{ticker}**: {e}")
        st.stop()

# Apply strategy signals
df = STRATEGY_FN[strategy_name](df, params)
current_signal = int(df["signal"].iloc[-1])
sig_label = {1: "🟢 LONG", -1: "🔴 SHORT", 0: "⚪ FLAT"}[current_signal]
sig_class  = {1: "signal-long", -1: "signal-short", 0: "signal-flat"}[current_signal]

st.markdown(f"**Current Signal:** <span class='{sig_class}'>{sig_label}</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Chart", "📉 Backtest", "🔄 Trade Log", "🏆 Compare", "⚙️ Optimize", "📐 Walk-Forward"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Chart with indicators
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    plot_len = st.slider("Bars to display", 60, len(df), min(252, len(df)), key="chart_bars")
    dfc = df.tail(plot_len).copy()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.50, 0.18, 0.16, 0.16],
        subplot_titles=("Price & Signals", "Volume", "RSI", "MACD"),
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=dfc.index, open=dfc["open"], high=dfc["high"],
        low=dfc["low"],  close=dfc["close"],
        name="OHLC",
        increasing_line_color="#00d4aa", decreasing_line_color="#ff4d6d",
        showlegend=False,
    ), row=1, col=1)

    # EMAs
    for col, name, color in [("ema8","EMA 8","#7c83fd"),("ema21","EMA 21","#f9a825"),("ema50","EMA 50","#ff6b6b")]:
        fig.add_trace(go.Scatter(x=dfc.index, y=dfc[col], name=name,
                                 line=dict(color=color, width=1.2), opacity=0.85), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=dfc.index, y=dfc["bb_upper"], name="BB Upper",
                             line=dict(color="#607d8b", width=1, dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfc.index, y=dfc["bb_lower"], name="BB Lower",
                             line=dict(color="#607d8b", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(96,125,139,0.08)", showlegend=False), row=1, col=1)

    # Signal markers
    longs  = dfc[dfc["signal"] == 1]
    shorts = dfc[dfc["signal"] == -1]

    if len(longs):
        fig.add_trace(go.Scatter(
            x=longs.index, y=longs["low"] * 0.995,
            mode="markers", name="Long Signal",
            marker=dict(symbol="triangle-up", size=9, color="#00d4aa"),
        ), row=1, col=1)

    if len(shorts):
        fig.add_trace(go.Scatter(
            x=shorts.index, y=shorts["high"] * 1.005,
            mode="markers", name="Short Signal",
            marker=dict(symbol="triangle-down", size=9, color="#ff4d6d"),
        ), row=1, col=1)

    # ── Volume ───────────────────────────────────────────────────────────────
    vol_colors = ["#00d4aa" if c >= o else "#ff4d6d"
                  for c, o in zip(dfc["close"], dfc["open"])]
    fig.add_trace(go.Bar(x=dfc.index, y=dfc["volume"], name="Volume",
                         marker_color=vol_colors, showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=dfc.index, y=dfc["vol_ma20"], name="Vol MA20",
                             line=dict(color="#f9a825", width=1), showlegend=False), row=2, col=1)

    # ── RSI ──────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(x=dfc.index, y=dfc["rsi"], name="RSI",
                             line=dict(color="#7c83fd", width=1.5)), row=3, col=1)
    for level, color in [(70, "rgba(255,77,109,0.4)"), (30, "rgba(0,212,170,0.4)"), (50, "rgba(255,255,255,0.15)")]:
        fig.add_hline(y=level, line_dash="dot", line_color=color, row=3, col=1)

    # ── MACD ─────────────────────────────────────────────────────────────────
    hist_colors = ["#00d4aa" if v >= 0 else "#ff4d6d" for v in dfc["macd_hist"]]
    fig.add_trace(go.Bar(x=dfc.index, y=dfc["macd_hist"], name="MACD Hist",
                         marker_color=hist_colors, showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=dfc.index, y=dfc["macd"], name="MACD",
                             line=dict(color="#00d4aa", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=dfc.index, y=dfc["macd_signal"], name="Signal",
                             line=dict(color="#f9a825", width=1.2)), row=4, col=1)

    fig.update_layout(
        height=750, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=11)),
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color="#a0aec0"),
    )
    for i in range(1, 5):
        fig.update_xaxes(showgrid=True, gridcolor="#1e2130", row=i, col=1)
        fig.update_yaxes(showgrid=True, gridcolor="#1e2130", row=i, col=1)

    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Backtest results
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not run_bt and "bt_result" not in st.session_state:
        st.info("👈 Configure parameters in the sidebar and click **▶ Run Backtest**.")
    else:
        if run_bt:
            ann = annualization_factor(timeframe)
            with st.spinner("Running backtest…"):
                result = backtest_run(df, ann, fee_rate, slippage, long_only=long_only)
            st.session_state["bt_result"] = result
            st.session_state["bt_ticker"] = ticker

        result = st.session_state["bt_result"]
        m = result["metrics"]

        # ── Metric cards ─────────────────────────────────────────────────────
        st.markdown("### Performance Summary")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        cards = [
            (c1, "Total Return",   fmt_pct(m["total_return_pct"]),  color_val(m["total_return_pct"])),
            (c2, "Ann. Return",    fmt_pct(m["ann_return_pct"]),     color_val(m["ann_return_pct"])),
            (c3, "Sharpe Ratio",   f"{m['sharpe']:.2f}",             "positive" if m["sharpe"] > 1 else "negative"),
            (c4, "Sortino Ratio",  f"{m['sortino']:.2f}",            "positive" if m["sortino"] > 1 else "negative"),
            (c5, "Max Drawdown",   fmt_pct(m["max_drawdown_pct"]),   "negative"),
            (c6, "vs Buy & Hold",  fmt_pct(m["total_return_pct"] - m["bh_return_pct"]), color_val(m["total_return_pct"] - m["bh_return_pct"])),
        ]
        for col, label, val, cls in cards:
            with col:
                st.markdown(metric_card(label, val, cls), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        cards2 = [
            (c1, "# Trades",      str(m["num_trades"]),           "neutral"),
            (c2, "Win Rate",      fmt_pct(m["win_rate_pct"]),      color_val(m["win_rate_pct"] - 50)),
            (c3, "Profit Factor", f"{m['profit_factor']:.2f}",     "positive" if m["profit_factor"] > 1 else "negative"),
            (c4, "Avg Win",       fmt_pct(m["avg_win_pct"], "%", 3), "positive"),
            (c5, "Avg Loss",      fmt_pct(m["avg_loss_pct"], "%", 3), "negative"),
            (c6, "Ann. Vol",      fmt_pct(m["ann_vol_pct"]),       "neutral"),
        ]
        for col, label, val, cls in cards2:
            with col:
                st.markdown(metric_card(label, val, cls), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Equity curve ─────────────────────────────────────────────────────
        st.markdown("### Equity Curve vs Buy & Hold")
        eq  = result["equity"]
        bh  = result["bh_equity"]
        dd  = m["dd_series"]

        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             vertical_spacing=0.04, row_heights=[0.70, 0.30],
                             subplot_titles=("Cumulative Return (normalized)", "Drawdown"))

        fig2.add_trace(go.Scatter(x=eq.index, y=eq, name=f"Strategy ({strategy_name})",
                                  line=dict(color=COLORS[strategy_name], width=2)), row=1, col=1)
        fig2.add_trace(go.Scatter(x=bh.index, y=bh, name="Buy & Hold",
                                  line=dict(color=COLORS["Buy & Hold"], width=1.5, dash="dash")), row=1, col=1)
        fig2.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=1, col=1)

        fig2.add_trace(go.Scatter(x=dd.index, y=dd * 100, name="Drawdown %",
                                  line=dict(color="#ff4d6d", width=1),
                                  fill="tozeroy", fillcolor="rgba(255,77,109,0.15)"), row=2, col=1)

        fig2.update_layout(
            height=500, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)),
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(color="#a0aec0"),
        )
        for i in range(1, 3):
            fig2.update_xaxes(showgrid=True, gridcolor="#1e2130", row=i, col=1)
            fig2.update_yaxes(showgrid=True, gridcolor="#1e2130", row=i, col=1)

        st.plotly_chart(fig2, use_container_width=True)

        # Return distribution
        st.markdown("### Return Distribution")
        strat_ret = result["strat_ret"].dropna()
        strat_ret_pct = strat_ret * 100

        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=strat_ret_pct, nbinsx=60, name="Strategy Returns",
            marker_color=COLORS[strategy_name], opacity=0.75,
        ))
        fig3.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)
        fig3.add_vline(x=strat_ret_pct.mean(), line_dash="dot",
                       line_color="#f9a825", annotation_text="Mean")
        fig3.update_layout(
            height=300, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            xaxis_title="Return per Bar (%)", yaxis_title="Frequency",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Trade Log
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if "bt_result" not in st.session_state:
        st.info("Run backtest first to see the trade log.")
    else:
        trades = st.session_state["bt_result"]["trades"]
        if trades.empty:
            st.warning("No completed trades found. Try adjusting strategy parameters.")
        else:
            st.markdown(f"### Trade Log — {len(trades)} trades")

            def color_pnl(val):
                color = "#00d4aa" if val > 0 else "#ff4d6d"
                return f"color: {color}"

            disp = trades.copy()
            disp["entry_date"] = disp["entry_date"].dt.strftime("%Y-%m-%d")
            disp["exit_date"]  = disp["exit_date"].dt.strftime("%Y-%m-%d")
            disp = disp.rename(columns={
                "entry_date": "Entry", "exit_date": "Exit",
                "side": "Side", "entry_price": "Entry $",
                "exit_price": "Exit $", "pnl_pct": "P&L %",
                "bars_held": "Bars",
            })

            st.dataframe(
                disp.style.applymap(color_pnl, subset=["P&L %"]),
                use_container_width=True, height=420,
            )

            # PnL waterfall / cumulative
            cum_pnl = trades["pnl_pct"].cumsum()
            fig4 = go.Figure()
            bar_colors = ["#00d4aa" if v > 0 else "#ff4d6d" for v in trades["pnl_pct"]]
            fig4.add_trace(go.Bar(
                x=list(range(len(trades))), y=trades["pnl_pct"],
                name="Trade P&L %", marker_color=bar_colors,
            ))
            fig4.add_trace(go.Scatter(
                x=list(range(len(trades))), y=cum_pnl,
                name="Cumulative P&L %",
                line=dict(color="#f9a825", width=2), yaxis="y2",
            ))
            fig4.update_layout(
                height=320, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                yaxis=dict(title="Per-trade P&L %", showgrid=True, gridcolor="#1e2130"),
                yaxis2=dict(title="Cumulative P&L %", overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.1, x=0),
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Multi-symbol comparison
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Multi-Symbol Strategy Comparison")

    col_a, col_b = st.columns([3, 1])
    with col_a:
        compare_tickers = st.multiselect(
            "Select tickers to compare",
            options=PRESET_SYMBOLS["US Large Cap"] + PRESET_SYMBOLS["Major ETFs"],
            default=["AAPL", "MSFT", "NVDA", "SPY"],
            max_selections=8,
        )
    with col_b:
        compare_strategy = st.selectbox("Strategy", list(STRATEGY_FN.keys()), key="cmp_strat")

    if st.button("▶ Run Comparison", type="primary"):
        ann = annualization_factor(timeframe)
        rows = []
        prog = st.progress(0, text="Loading…")

        for i, t in enumerate(compare_tickers):
            prog.progress((i + 1) / len(compare_tickers), text=f"Processing {t}…")
            try:
                d = load_data(t, timeframe)
                d = STRATEGY_FN[compare_strategy](d, DEFAULT_PARAMS[compare_strategy])
                res = backtest_run(d, ann, fee_rate, slippage, long_only=long_only)
                mm  = res["metrics"]
                rows.append({
                    "Ticker":        t,
                    "Total Ret %":   mm["total_return_pct"],
                    "Ann Ret %":     mm["ann_return_pct"],
                    "Sharpe":        mm["sharpe"],
                    "Sortino":       mm["sortino"],
                    "Max DD %":      mm["max_drawdown_pct"],
                    "Win Rate %":    mm["win_rate_pct"],
                    "# Trades":      mm["num_trades"],
                    "Profit Factor": mm["profit_factor"],
                    "B&H Ret %":     mm["bh_return_pct"],
                })
            except Exception as e:
                rows.append({"Ticker": t, "Total Ret %": None, "Error": str(e)})

        prog.empty()
        cmp_df = pd.DataFrame(rows).set_index("Ticker")
        st.session_state["cmp_result"] = cmp_df

    if "cmp_result" in st.session_state:
        cmp_df = st.session_state["cmp_result"]

        def style_comparison(df):
            styled = df.style
            for col in ["Total Ret %", "Ann Ret %", "Sharpe", "Sortino", "Win Rate %", "Profit Factor"]:
                if col in df.columns:
                    styled = styled.background_gradient(cmap="RdYlGn", subset=[col])
            for col in ["Max DD %"]:
                if col in df.columns:
                    styled = styled.background_gradient(cmap="RdYlGn_r", subset=[col])
            return styled

        st.dataframe(style_comparison(cmp_df), use_container_width=True)

        # Sharpe bar chart
        if "Sharpe" in cmp_df.columns and cmp_df["Sharpe"].notna().any():
            fig5 = go.Figure()
            colors = ["#00d4aa" if v > 1 else "#f9a825" if v > 0 else "#ff4d6d"
                      for v in cmp_df["Sharpe"]]
            fig5.add_trace(go.Bar(
                x=cmp_df.index, y=cmp_df["Sharpe"],
                marker_color=colors, name="Sharpe Ratio",
            ))
            fig5.add_hline(y=1.0, line_dash="dash", line_color="white",
                           annotation_text="Sharpe = 1", opacity=0.6)
            fig5.update_layout(
                height=300, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                title="Sharpe Ratio by Ticker",
                xaxis_title="Ticker", yaxis_title="Sharpe",
                margin=dict(l=0, r=0, t=40, b=0),
                showlegend=False,
            )
            st.plotly_chart(fig5, use_container_width=True)

            # Return scatter: strategy vs B&H
            if "B&H Ret %" in cmp_df.columns:
                fig6 = px.scatter(
                    cmp_df.reset_index(), x="B&H Ret %", y="Total Ret %",
                    text="Ticker", color="Sharpe",
                    color_continuous_scale="RdYlGn",
                    title="Strategy Return vs Buy & Hold Return",
                    template="plotly_dark",
                    size_max=15,
                )
                fig6.add_shape(type="line",
                               x0=cmp_df["B&H Ret %"].min(), y0=cmp_df["B&H Ret %"].min(),
                               x1=cmp_df["B&H Ret %"].max(), y1=cmp_df["B&H Ret %"].max(),
                               line=dict(dash="dot", color="white", width=1))
                fig6.update_traces(textposition="top center", marker=dict(size=14))
                fig6.update_layout(
                    height=400, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    margin=dict(l=0, r=0, t=50, b=0),
                )
                st.plotly_chart(fig6, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#444; font-size:0.75rem;'>"
    "Quant Strategy Dashboard · Data via yfinance · For research purposes only · Not financial advice"
    "</div>",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Parameter Optimisation (Grid Search)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### ⚙️ Parameter Grid Search")
    st.markdown(
        "Exhaustively tests all parameter combinations and ranks them by your chosen objective. "
        "Uses the **currently loaded asset & timeframe**."
    )

    col_o1, col_o2, col_o3 = st.columns(3)
    with col_o1:
        opt_strategy = st.selectbox("Strategy to optimise", list(STRATEGY_FN.keys()), key="opt_strat")
    with col_o2:
        opt_objective = st.selectbox(
            "Optimisation objective",
            ["sharpe", "sortino", "calmar", "total_return_pct"],
            format_func=lambda x: {
                "sharpe": "Sharpe Ratio",
                "sortino": "Sortino Ratio",
                "calmar": "Calmar Ratio",
                "total_return_pct": "Total Return %",
            }[x],
        )
    with col_o3:
        top_n = st.slider("Show top N results", 5, 30, 10)

    # Show grid size preview
    grid_size = 1
    for v in PARAM_GRIDS[opt_strategy].values():
        grid_size *= len(v)
    st.info(f"Grid size: **{grid_size} combinations** for {opt_strategy}")

    run_opt = st.button("▶ Run Grid Search", type="primary", key="run_opt")

    if run_opt:
        ann = annualization_factor(timeframe)
        prog_bar  = st.progress(0, text="Running grid search…")
        prog_text = st.empty()

        def opt_progress(i, total):
            prog_bar.progress(i / total, text=f"Testing combination {i}/{total}…")

        with st.spinner("Optimising…"):
            opt_result = grid_search(
                df=df,
                strategy_name=opt_strategy,
                strategy_fn=STRATEGY_FN[opt_strategy],
                ann=ann,
                fee=fee_rate,
                slip=slippage,
                long_only=long_only,
                objective=opt_objective,
                top_n=top_n,
                progress_cb=opt_progress,
            )

        prog_bar.empty()
        st.session_state["opt_result"]    = opt_result
        st.session_state["opt_strategy"]  = opt_strategy
        st.session_state["opt_objective"] = opt_objective

    if "opt_result" in st.session_state:
        opt_result   = st.session_state["opt_result"]
        opt_strategy = st.session_state["opt_strategy"]
        obj_col      = st.session_state["opt_objective"]

        st.markdown(f"#### Top {len(opt_result)} Results — ranked by `{obj_col}`")

        # Identify param columns vs metric columns
        param_keys   = list(PARAM_GRIDS[opt_strategy].keys())
        metric_cols  = ["sharpe", "sortino", "total_return_pct", "max_drawdown_pct",
                        "win_rate_pct", "num_trades", "profit_factor", "calmar"]
        display_cols = param_keys + [c for c in metric_cols if c in opt_result.columns]
        display_df   = opt_result[display_cols].copy()

        # Rename for readability
        rename = {
            "sharpe": "Sharpe", "sortino": "Sortino", "calmar": "Calmar",
            "total_return_pct": "Return %", "max_drawdown_pct": "Max DD %",
            "win_rate_pct": "Win Rate %", "num_trades": "# Trades",
            "profit_factor": "Profit Factor",
        }
        display_df = display_df.rename(columns=rename)

        def style_opt(df):
            s = df.style
            for c in ["Sharpe", "Sortino", "Return %", "Win Rate %", "Profit Factor", "Calmar"]:
                if c in df.columns:
                    s = s.background_gradient(cmap="RdYlGn", subset=[c])
            if "Max DD %" in df.columns:
                s = s.background_gradient(cmap="RdYlGn_r", subset=["Max DD %"])
            return s

        st.dataframe(style_opt(display_df), use_container_width=True, height=420)

        # Highlight best params
        best = opt_result.iloc[0]
        best_params_display = {k: best[k] for k in param_keys}
        st.success(f"**Best params:** {best_params_display}")
        st.caption("💡 Click '▶ Run Backtest' in the Backtest tab after manually updating the sidebar sliders to these values.")

        # Scatter: two most important params vs objective
        if len(param_keys) >= 2:
            st.markdown("#### Parameter Sensitivity")
            px_col1, px_col2 = st.columns(2)
            with px_col1:
                xp = st.selectbox("X-axis param", param_keys, index=0, key="opt_x")
            with px_col2:
                yp = st.selectbox("Y-axis param", param_keys, index=min(1, len(param_keys)-1), key="opt_y")

            scatter_df = opt_result[param_keys + [obj_col, "num_trades"]].copy()
            fig_s = px.scatter(
                scatter_df, x=xp, y=yp, color=obj_col,
                size=scatter_df["num_trades"].clip(lower=1),
                color_continuous_scale="RdYlGn",
                title=f"{obj_col} across {xp} vs {yp}",
                template="plotly_dark",
            )
            fig_s.update_layout(
                height=380, paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_s, use_container_width=True)

        # Objective distribution
        fig_hist = go.Figure(go.Histogram(
            x=opt_result[obj_col], nbinsx=20,
            marker_color="#7c83fd", opacity=0.8,
        ))
        fig_hist.add_vline(
            x=float(opt_result[obj_col].iloc[0]),
            line_dash="dash", line_color="#00d4aa",
            annotation_text="Best",
        )
        fig_hist.update_layout(
            height=260, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            title=f"Distribution of {obj_col} across all combos",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 📐 Walk-Forward Validation")
    st.markdown("""
Walk-forward tests whether your strategy's optimised parameters **generalise** to unseen data.
Each fold: optimise on in-sample → test on out-of-sample → compare results.

| Efficiency Ratio | Interpretation |
|---|---|
| ≥ 0.70 | 🟢 Robust — OOS performance close to IS |
| 0.40 – 0.70 | 🟡 Moderate — some decay, acceptable |
| < 0.40 | 🔴 Over-fitted — strategy likely curve-fitted |
    """)

    wf_c1, wf_c2, wf_c3, wf_c4 = st.columns(4)
    with wf_c1:
        wf_strategy = st.selectbox("Strategy", list(STRATEGY_FN.keys()), key="wf_strat")
    with wf_c2:
        wf_splits = st.slider("Number of folds", 3, 8, 5, key="wf_splits")
    with wf_c3:
        wf_is_pct = st.slider("IS fraction", 0.5, 0.8, 0.7, 0.05, key="wf_is",
                               help="Fraction of each fold used for in-sample optimisation")
    with wf_c4:
        wf_anchored = st.toggle("Anchored IS window", value=False,
                                 help="ON = expanding IS; OFF = rolling IS")

    wf_objective = st.selectbox(
        "IS optimisation objective", ["sharpe", "sortino", "calmar", "total_return_pct"],
        format_func=lambda x: {"sharpe":"Sharpe","sortino":"Sortino",
                                "calmar":"Calmar","total_return_pct":"Total Return %"}[x],
        key="wf_obj",
    )

    grid_size_wf = 1
    for v in PARAM_GRIDS[wf_strategy].values():
        grid_size_wf *= len(v)
    total_evals = grid_size_wf * wf_splits
    st.info(f"Total evaluations: **{total_evals}** ({grid_size_wf} combos × {wf_splits} folds)")

    run_wf = st.button("▶ Run Walk-Forward", type="primary", key="run_wf")

    if run_wf:
        ann       = annualization_factor(timeframe)
        wf_bar    = st.progress(0, text="Running walk-forward…")

        def wf_progress(i, total):
            wf_bar.progress(i / total, text=f"Fold evaluation {i}/{total}…")

        try:
            with st.spinner("Running walk-forward validation…"):
                wf_result = walk_forward(
                    df=df,
                    strategy_name=wf_strategy,
                    strategy_fn=STRATEGY_FN[wf_strategy],
                    ann=ann,
                    fee=fee_rate,
                    slip=slippage,
                    long_only=long_only,
                    n_splits=wf_splits,
                    is_pct=wf_is_pct,
                    anchored=wf_anchored,
                    objective=wf_objective,
                    progress_cb=wf_progress,
                )
            wf_bar.empty()
            st.session_state["wf_result"] = wf_result
        except ValueError as e:
            wf_bar.empty()
            st.error(str(e))

    if "wf_result" in st.session_state:
        wfr = st.session_state["wf_result"]
        sm  = wfr["summary"]
        folds_df = wfr["folds"]

        # ── Summary cards ─────────────────────────────────────────────────────
        st.markdown("#### Summary")
        sw1, sw2, sw3, sw4, sw5 = st.columns(5)
        summary_cards = [
            (sw1, "Avg IS Sharpe",  f"{sm['avg_is_sharpe']:.3f}",  "neutral"),
            (sw2, "Avg OOS Sharpe", f"{sm['avg_oos_sharpe']:.3f}", color_val(sm["avg_oos_sharpe"])),
            (sw3, "Efficiency Ratio", f"{sm['efficiency_ratio']:.2f}",
             "positive" if sm["efficiency_ratio"] >= 0.7 else
             "neutral"  if sm["efficiency_ratio"] >= 0.4 else "negative"),
            (sw4, "Rating", sm["rating"], "neutral"),
            (sw5, "Valid Folds", str(sm["n_valid_folds"]), "neutral"),
        ]
        for col, label, val, cls in summary_cards:
            with col:
                st.markdown(metric_card(label, val, cls), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── IS vs OOS equity curves ────────────────────────────────────────────
        st.markdown("#### IS vs OOS Equity Curves (per fold, normalised to 1.0)")
        oos_eq = wfr["oos_equity"]
        is_eq  = wfr["is_equity"]

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Scatter(
            x=is_eq.index, y=is_eq,
            name="In-Sample (best params)",
            line=dict(color="#7c83fd", width=1.5, dash="dash"),
            opacity=0.8,
        ))
        fig_wf.add_trace(go.Scatter(
            x=oos_eq.index, y=oos_eq,
            name="Out-of-Sample",
            line=dict(color="#00d4aa", width=2),
        ))
        fig_wf.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.2)")

        # Shade OOS windows
        for _, row in folds_df.iterrows():
            fig_wf.add_vrect(
                x0=row["oos_start"], x1=row["oos_end"],
                fillcolor="rgba(0,212,170,0.05)",
                line_width=0, layer="below",
            )

        fig_wf.update_layout(
            height=380, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=True, gridcolor="#1e2130"),
            yaxis=dict(showgrid=True, gridcolor="#1e2130", title="Normalised Equity"),
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # ── IS vs OOS Sharpe per fold bar chart ────────────────────────────────
        st.markdown("#### IS vs OOS Sharpe by Fold")
        fig_bar = go.Figure()
        fold_labels = [f"Fold {i}" for i in folds_df.index]
        fig_bar.add_trace(go.Bar(
            name="IS Sharpe", x=fold_labels, y=folds_df["is_sharpe"],
            marker_color="#7c83fd", opacity=0.8,
        ))
        fig_bar.add_trace(go.Bar(
            name="OOS Sharpe", x=fold_labels, y=folds_df["oos_sharpe"],
            marker_color="#00d4aa", opacity=0.9,
        ))
        fig_bar.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
        fig_bar.add_hline(y=1, line_dash="dot", line_color="#f9a825",
                          annotation_text="Sharpe = 1", annotation_font_color="#f9a825")
        fig_bar.update_layout(
            barmode="group", height=320, template="plotly_dark",
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            legend=dict(orientation="h", y=1.08, x=0),
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(showgrid=True, gridcolor="#1e2130"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Fold detail table ──────────────────────────────────────────────────
        st.markdown("#### Fold Detail")
        display_folds = folds_df.copy()
        display_folds["is_start"]  = display_folds["is_start"].dt.strftime("%Y-%m-%d")
        display_folds["is_end"]    = display_folds["is_end"].dt.strftime("%Y-%m-%d")
        display_folds["oos_start"] = display_folds["oos_start"].dt.strftime("%Y-%m-%d")
        display_folds["oos_end"]   = display_folds["oos_end"].dt.strftime("%Y-%m-%d")
        display_folds["best_params"] = display_folds["best_params"].apply(str)

        display_folds = display_folds.rename(columns={
            "is_start": "IS Start", "is_end": "IS End",
            "oos_start": "OOS Start", "oos_end": "OOS End",
            "best_params": "Best Params",
            "is_sharpe": "IS Sharpe", "oos_sharpe": "OOS Sharpe",
            "is_return_pct": "IS Ret %", "oos_return_pct": "OOS Ret %",
            "is_trades": "IS Trades", "oos_trades": "OOS Trades",
            "is_win_rate": "IS WR %", "oos_win_rate": "OOS WR %",
        })

        def style_wf(df):
            s = df.style
            for c in ["IS Sharpe", "OOS Sharpe", "IS Ret %", "OOS Ret %"]:
                if c in df.columns:
                    s = s.background_gradient(cmap="RdYlGn", subset=[c])
            return s

        st.dataframe(style_wf(display_folds), use_container_width=True, height=260)

        st.caption(
            "💡 **How to read this:** Green OOS Sharpe bars close to IS bars = strategy is robust. "
            "Large IS/OOS gap = over-fitting. Efficiency Ratio > 0.7 is a good threshold for real deployment."
        )
