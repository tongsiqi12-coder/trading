"""
utils/charts.py
All Plotly figures used by the Streamlit app.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

COLORS = {
    "strategy": "#00C49F",
    "bh":       "#8884d8",
    "long":     "#26a69a",
    "short":    "#ef5350",
    "neutral":  "#90a4ae",
    "dd":       "#ef5350",
    "macd":     "#2196F3",
    "signal":   "#FF9800",
}


def candlestick_with_signals(df: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Candlestick + EMAs + Bollinger Bands + Buy/Sell arrows.
    Bottom sub-panel: RSI.
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.25, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} — Price & Signals", "Volume", "RSI"),
    )

    # ── Candlesticks ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="Price", increasing_line_color=COLORS["long"],
        decreasing_line_color=COLORS["short"],
    ), row=1, col=1)

    # EMAs
    for col, label, color in [
        ("ema_fast", "EMA Fast", "#FFA726"),
        ("ema_mid",  "EMA Mid",  "#42A5F5"),
        ("ema_slow", "EMA Slow", "#AB47BC"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=label,
                line=dict(color=color, width=1.2),
            ), row=1, col=1)

    # Bollinger Bands (shaded)
    if "bb_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_upper"], name="BB Upper",
            line=dict(color="rgba(150,150,150,0.4)", dash="dot", width=1),
            showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["bb_lower"], name="BB Lower",
            fill="tonexty",
            fillcolor="rgba(150,150,150,0.07)",
            line=dict(color="rgba(150,150,150,0.4)", dash="dot", width=1),
            showlegend=False,
        ), row=1, col=1)

    # ── Buy / Sell markers ────────────────────────────────────────────────────
    if "signal" in df.columns:
        sig_change = df["signal"] != df["signal"].shift(1)
        entries    = df[sig_change & (df["signal"] != 0)]

        buys  = entries[entries["signal"] ==  1]
        sells = entries[entries["signal"] == -1]

        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["low"] * 0.995,
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=10, color=COLORS["long"]),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["high"] * 1.005,
            mode="markers", name="Sell / Short",
            marker=dict(symbol="triangle-down", size=10, color=COLORS["short"]),
        ), row=1, col=1)

    # ── Volume ────────────────────────────────────────────────────────────────
    colors = [COLORS["long"] if c >= o else COLORS["short"]
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=colors, showlegend=False,
    ), row=2, col=1)

    # ── RSI ───────────────────────────────────────────────────────────────────
    if "rsi" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi"], name="RSI",
            line=dict(color="#CE93D8", width=1.5),
        ), row=3, col=1)
        for level, color in [(70, "rgba(239,83,80,0.3)"), (30, "rgba(38,166,154,0.3)")]:
            fig.add_hline(y=level, line_dash="dash",
                          line_color=color, row=3, col=1)

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def equity_curve(equity: pd.Series, bh_equity: pd.Series,
                 strat_ret: pd.Series) -> go.Figure:
    """Equity curve + drawdown sub-panel."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        subplot_titles=("Portfolio Value", "Drawdown (%)"),
    )

    fig.add_trace(go.Scatter(
        x=equity.index, y=equity,
        name="Strategy", line=dict(color=COLORS["strategy"], width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=bh_equity.index, y=bh_equity,
        name="Buy & Hold", line=dict(color=COLORS["bh"], width=1.5, dash="dot"),
    ), row=1, col=1)

    # Drawdown
    roll_max = equity.expanding().max()
    dd = (equity - roll_max) / roll_max * 100
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, name="Drawdown",
        fill="tozeroy",
        line=dict(color=COLORS["dd"], width=1),
        fillcolor="rgba(239,83,80,0.25)",
    ), row=2, col=1)

    fig.update_layout(
        height=500, template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_yaxes(title_text="$", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "macd" not in df.columns:
        return fig

    colors = [COLORS["long"] if v >= 0 else COLORS["short"]
              for v in df["macd_hist"]]
    fig.add_trace(go.Bar(
        x=df.index, y=df["macd_hist"],
        name="MACD Hist", marker_color=colors,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd"],
        name="MACD", line=dict(color=COLORS["macd"], width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["macd_sig"],
        name="Signal", line=dict(color=COLORS["signal"], width=1.5),
    ))
    fig.update_layout(
        height=250, template="plotly_dark",
        title="MACD",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h"),
    )
    return fig


def multi_ticker_comparison(results: dict) -> go.Figure:
    """
    Overlay normalised equity curves for multiple tickers.
    *results* = {ticker: backtest_result_dict}
    """
    fig = go.Figure()
    palette = [
        "#00C49F", "#8884d8", "#FFA726", "#ef5350",
        "#42A5F5", "#AB47BC", "#66BB6A", "#FF7043",
    ]
    for i, (ticker, res) in enumerate(results.items()):
        eq = res["equity"] / res["equity"].iloc[0] * 100   # normalise to 100
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq, name=ticker,
            line=dict(color=palette[i % len(palette)], width=2),
        ))

    fig.update_layout(
        height=420, template="plotly_dark",
        title="Strategy Performance — Normalised to 100",
        yaxis_title="Indexed Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def trade_pnl_histogram(trades: pd.DataFrame) -> go.Figure:
    if trades.empty:
        return go.Figure()
    fig = go.Figure(go.Histogram(
        x=trades["pnl_pct"],
        nbinsx=30,
        marker_color=[
            COLORS["long"] if v > 0 else COLORS["short"]
            for v in trades["pnl_pct"]
        ],
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="white")
    fig.update_layout(
        height=300, template="plotly_dark",
        title="Trade P&L Distribution (%)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig
