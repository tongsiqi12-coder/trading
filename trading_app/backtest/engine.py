"""
backtest/engine.py — Vectorized backtest with full cost accounting.
Returns equity curve + trade log + performance metrics dict.
"""

import pandas as pd
import numpy as np


def run(
    df: pd.DataFrame,
    ann_factor: float,
    fee_rate: float   = 0.0005,   # 0.05% per side (realistic for retail)
    slippage: float   = 0.0002,   # 0.02% per side
    atr_sl_mult: float = 1.5,
    atr_tp_mult: float = 2.5,
    long_only: bool   = True,
) -> dict:
    """
    Vectorized backtest on a df that already has a 'signal' column.

    Signal is position for NEXT bar (shift by 1 for execution realism).
    Costs applied on every position change.

    Returns:
        metrics : dict of performance stats
        equity  : pd.Series (cumulative returns)
        trades  : pd.DataFrame (trade log)
    """
    df = df.copy()

    if long_only:
        df["signal"] = df["signal"].clip(lower=0)   # suppress shorts

    # ── Position and returns ──────────────────────────────────────────────────
    pos    = df["signal"].shift(1).fillna(0)          # execute next bar open
    ret    = df["close"].pct_change()

    # Transaction cost on position change
    trades_mask = (pos != pos.shift(1)).astype(float)
    cost        = trades_mask * (fee_rate + slippage)

    strat_ret = pos * ret - cost
    equity    = (1 + strat_ret).cumprod()
    bh_equity = (1 + ret).cumprod()                   # Buy & Hold benchmark

    # ── Trade log ─────────────────────────────────────────────────────────────
    entries = df[(pos != 0) & (pos.shift(1) == 0)].copy()
    exits   = df[(pos == 0) & (pos.shift(1) != 0)].copy()

    trade_records = []
    entry_iter = iter(entries.iterrows())
    exit_iter  = iter(exits.iterrows())

    try:
        e_idx, e_row = next(entry_iter)
        x_idx, x_row = next(exit_iter)
        while True:
            while x_idx <= e_idx:
                x_idx, x_row = next(exit_iter)
            entry_price = e_row["close"]
            exit_price  = x_row["close"]
            direction   = e_row["signal"]
            pnl_pct     = direction * (exit_price - entry_price) / entry_price
            net_pnl     = pnl_pct - 2 * (fee_rate + slippage)
            trade_records.append({
                "entry_date":  e_idx,
                "exit_date":   x_idx,
                "side":        "Long" if direction == 1 else "Short",
                "entry_price": round(entry_price, 4),
                "exit_price":  round(exit_price, 4),
                "pnl_pct":     round(net_pnl * 100, 3),
                "bars_held":   (df.index.get_loc(x_idx) - df.index.get_loc(e_idx)),
            })
            e_idx, e_row = next(entry_iter)
            x_idx, x_row = next(exit_iter)
    except StopIteration:
        pass

    trades_df = pd.DataFrame(trade_records)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = _compute_metrics(strat_ret, equity, bh_equity, trades_df, ann_factor)
    return {
        "metrics": metrics,
        "equity":  equity,
        "bh_equity": bh_equity,
        "strat_ret": strat_ret,
        "trades":  trades_df,
    }


def _compute_metrics(
    strat_ret: pd.Series,
    equity: pd.Series,
    bh_equity: pd.Series,
    trades_df: pd.DataFrame,
    ann_factor: float,
) -> dict:

    ret = strat_ret.dropna()
    n   = len(ret)

    # Total return
    total_ret = equity.iloc[-1] - 1 if len(equity) > 0 else 0.0

    # Annualized return & vol
    ann_ret = (1 + total_ret) ** (ann_factor / max(n, 1)) - 1
    ann_vol = ret.std() * np.sqrt(ann_factor) if n > 1 else 0.0

    # Sharpe (risk-free ≈ 0 for simplicity; can parameterize)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino
    downside = ret[ret < 0].std() * np.sqrt(ann_factor)
    sortino  = ann_ret / downside if downside > 0 else 0.0

    # Max Drawdown
    roll_max = equity.expanding().max()
    dd_series = (equity - roll_max) / roll_max
    max_dd = dd_series.min()

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0

    # Buy & hold comparison
    bh_ret = bh_equity.iloc[-1] - 1 if len(bh_equity) > 0 else 0.0

    # Trade stats
    if len(trades_df) > 0:
        wins       = trades_df[trades_df["pnl_pct"] > 0]
        losses     = trades_df[trades_df["pnl_pct"] <= 0]
        win_rate   = len(wins) / len(trades_df)
        avg_win    = wins["pnl_pct"].mean()   if len(wins)   > 0 else 0.0
        avg_loss   = losses["pnl_pct"].mean() if len(losses) > 0 else 0.0
        pf_denom   = abs(losses["pnl_pct"].sum())
        profit_fac = wins["pnl_pct"].sum() / pf_denom if pf_denom > 0 else np.inf
        avg_bars   = trades_df["bars_held"].mean()
    else:
        win_rate = avg_win = avg_loss = profit_fac = avg_bars = 0.0

    return {
        "total_return_pct":  round(total_ret  * 100, 2),
        "ann_return_pct":    round(ann_ret    * 100, 2),
        "ann_vol_pct":       round(ann_vol    * 100, 2),
        "sharpe":            round(sharpe,  3),
        "sortino":           round(sortino, 3),
        "max_drawdown_pct":  round(max_dd   * 100, 2),
        "calmar":            round(calmar,  3),
        "bh_return_pct":     round(bh_ret   * 100, 2),
        "num_trades":        len(trades_df),
        "win_rate_pct":      round(win_rate  * 100, 1),
        "avg_win_pct":       round(avg_win,  3),
        "avg_loss_pct":      round(avg_loss, 3),
        "profit_factor":     round(profit_fac, 2) if profit_fac != np.inf else 999.0,
        "avg_bars_held":     round(avg_bars, 1),
        "dd_series":         dd_series,
    }
