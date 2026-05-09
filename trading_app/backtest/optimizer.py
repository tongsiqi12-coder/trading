"""
backtest/optimizer.py
─────────────────────
1. Grid Search         — exhaustive search over param grid, ranked by Sharpe
2. Walk-Forward        — rolling in-sample optimize → out-of-sample test
                         Produces IS vs OOS equity curves + efficiency ratio

Design notes
────────────
- All computation is pure Python/NumPy/Pandas; no scipy/sklearn dependency.
- Each grid evaluation re-uses the pre-computed indicator df (expensive part)
  and only re-runs the cheap signal + backtest step.
- Walk-forward uses anchored or rolling windows (user's choice).
"""

import itertools
import warnings
import pandas as pd
import numpy as np
from typing import Callable

from backtest.engine import run as _bt_run

warnings.filterwarnings("ignore")


# ─── Parameter grid definitions per strategy ─────────────────────────────────
PARAM_GRIDS = {
    "Trend Following": {
        "rsi_long":        [48, 52, 55],
        "rsi_short":       [42, 45, 48],
        "max_vol_regime":  [1.5, 1.8, 2.2],
        "min_hold":        [2, 3, 5],
    },
    "Mean Reversion": {
        "rsi_oversold":    [25, 30, 35],
        "rsi_overbought":  [65, 70, 75],
        "bb_oversold":     [0.05, 0.10, 0.15],
        "bb_overbought":   [0.85, 0.90, 0.95],
        "max_vol_regime":  [0.8, 0.9, 1.1],
        "min_hold":        [2, 3],
    },
    "Donchian Breakout": {
        "min_hold":        [2, 3, 5],
        "max_vol_regime":  [1.5, 2.0, 2.5],
    },
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _expand_grid(grid: dict) -> list[dict]:
    """Cartesian product of a param grid → list of param dicts."""
    keys   = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _eval_one(df: pd.DataFrame, strategy_fn: Callable, params: dict,
              ann: float, fee: float, slip: float, long_only: bool) -> dict:
    """Run one param combination. Returns metrics dict + params."""
    try:
        d = strategy_fn(df, params)
        r = _bt_run(d, ann, fee_rate=fee, slippage=slip, long_only=long_only)
        m = r["metrics"]
        # Penalise results with too few trades (over-fitted)
        if m["num_trades"] < 5:
            m["sharpe"]  = -99.0
            m["sortino"] = -99.0
        return {**params, **{k: v for k, v in m.items() if k != "dd_series"}}
    except Exception:
        return {**params, "sharpe": -99.0, "sortino": -99.0,
                "total_return_pct": 0, "num_trades": 0}


# ─── 1. Grid Search ───────────────────────────────────────────────────────────

def grid_search(
    df: pd.DataFrame,
    strategy_name: str,
    strategy_fn: Callable,
    ann: float,
    fee: float  = 0.0005,
    slip: float = 0.0002,
    long_only: bool = True,
    objective: str  = "sharpe",   # "sharpe" | "sortino" | "calmar" | "total_return_pct"
    top_n: int = 10,
    progress_cb: Callable | None = None,   # optional: called with (i, total)
) -> pd.DataFrame:
    """
    Exhaustive grid search over PARAM_GRIDS[strategy_name].
    Returns a DataFrame of top_n results sorted by objective metric.
    """
    grid    = PARAM_GRIDS[strategy_name]
    combos  = _expand_grid(grid)
    total   = len(combos)
    results = []

    for i, params in enumerate(combos):
        row = _eval_one(df, strategy_fn, params, ann, fee, slip, long_only)
        results.append(row)
        if progress_cb:
            progress_cb(i + 1, total)

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(objective, ascending=False).head(top_n)
    result_df = result_df.reset_index(drop=True)
    result_df.index += 1   # rank from 1
    return result_df


# ─── 2. Walk-Forward Validation ───────────────────────────────────────────────

def walk_forward(
    df: pd.DataFrame,
    strategy_name: str,
    strategy_fn: Callable,
    ann: float,
    fee: float       = 0.0005,
    slip: float      = 0.0002,
    long_only: bool  = True,
    n_splits: int    = 5,
    is_pct: float    = 0.70,      # fraction of each fold used for in-sample
    anchored: bool   = False,     # True = expanding IS window; False = rolling
    objective: str   = "sharpe",
    progress_cb: Callable | None = None,
) -> dict:
    """
    Walk-Forward Analysis.

    Splits df into n_splits folds. For each fold:
      - In-sample (IS)  : run grid search, pick best params
      - Out-of-sample (OOS): run those params on OOS window

    Returns:
      folds        : list of fold detail dicts
      oos_equity   : concatenated OOS equity curve (pd.Series)
      is_equity    : concatenated IS equity curve  (pd.Series, best params)
      summary      : aggregate IS vs OOS metrics
      efficiency   : OOS Sharpe / IS Sharpe (robustness ratio, >0.5 is good)
    """
    n    = len(df)
    fold_size = n // n_splits

    folds      = []
    oos_pieces = []
    is_pieces  = []

    for fold_i in range(n_splits):
        # ── Window boundaries ────────────────────────────────────────────────
        if anchored:
            is_start = 0
        else:
            is_start = fold_i * fold_size

        oos_end   = min((fold_i + 1) * fold_size + int(fold_size * (1 - is_pct)), n)
        is_end    = is_start + int((oos_end - is_start) * is_pct)
        oos_start = is_end

        if oos_start >= oos_end or is_end - is_start < 60:
            continue   # too short, skip

        is_df  = df.iloc[is_start:is_end].copy()
        oos_df = df.iloc[oos_start:oos_end].copy()

        # ── IS optimisation ──────────────────────────────────────────────────
        grid   = PARAM_GRIDS[strategy_name]
        combos = _expand_grid(grid)
        is_results = []

        for j, params in enumerate(combos):
            row = _eval_one(is_df, strategy_fn, params, ann, fee, slip, long_only)
            is_results.append(row)
            if progress_cb:
                # Nested progress: fold_i * total_combos + j
                progress_cb(fold_i * len(combos) + j + 1, n_splits * len(combos))

        is_df_res = pd.DataFrame(is_results).sort_values(objective, ascending=False)
        best_params = {k: is_df_res.iloc[0][k]
                       for k in PARAM_GRIDS[strategy_name].keys()}
        # Re-cast int params
        for k in ["min_hold"]:
            if k in best_params:
                best_params[k] = int(best_params[k])

        # ── IS equity (best params) ───────────────────────────────────────────
        is_strat = strategy_fn(is_df, best_params)
        is_res   = _bt_run(is_strat, ann, fee_rate=fee, slippage=slip, long_only=long_only)

        # ── OOS equity ────────────────────────────────────────────────────────
        oos_strat = strategy_fn(oos_df, best_params)
        oos_res   = _bt_run(oos_strat, ann, fee_rate=fee, slippage=slip, long_only=long_only)

        is_m  = is_res["metrics"]
        oos_m = oos_res["metrics"]

        folds.append({
            "fold":            fold_i + 1,
            "is_start":        df.index[is_start],
            "is_end":          df.index[is_end - 1],
            "oos_start":       df.index[oos_start],
            "oos_end":         df.index[oos_end - 1],
            "best_params":     best_params,
            "is_sharpe":       is_m["sharpe"],
            "oos_sharpe":      oos_m["sharpe"],
            "is_return_pct":   is_m["total_return_pct"],
            "oos_return_pct":  oos_m["total_return_pct"],
            "is_trades":       is_m["num_trades"],
            "oos_trades":      oos_m["num_trades"],
            "is_win_rate":     is_m["win_rate_pct"],
            "oos_win_rate":    oos_m["win_rate_pct"],
        })

        # Normalise equity curves to start at 1.0
        is_eq  = is_res["equity"]  / is_res["equity"].iloc[0]
        oos_eq = oos_res["equity"] / oos_res["equity"].iloc[0]
        is_pieces.append(is_eq)
        oos_pieces.append(oos_eq)

    if not folds:
        raise ValueError("Not enough data to run walk-forward. Try fewer splits or daily timeframe.")

    folds_df  = pd.DataFrame(folds).set_index("fold")
    oos_equity = pd.concat(oos_pieces).sort_index() if oos_pieces else pd.Series(dtype=float)
    is_equity  = pd.concat(is_pieces).sort_index()  if is_pieces  else pd.Series(dtype=float)

    # ── Aggregate summary ────────────────────────────────────────────────────
    valid = folds_df[folds_df["oos_trades"] >= 3]   # only folds with ≥3 OOS trades
    if len(valid) == 0:
        valid = folds_df

    avg_is_sharpe  = valid["is_sharpe"].mean()
    avg_oos_sharpe = valid["oos_sharpe"].mean()
    efficiency     = avg_oos_sharpe / avg_is_sharpe if avg_is_sharpe > 0 else 0.0

    # Overfitting rating
    if efficiency >= 0.7:
        rating = "🟢 Robust"
    elif efficiency >= 0.4:
        rating = "🟡 Moderate"
    else:
        rating = "🔴 Over-fitted"

    summary = {
        "avg_is_sharpe":    round(avg_is_sharpe,  3),
        "avg_oos_sharpe":   round(avg_oos_sharpe, 3),
        "efficiency_ratio": round(efficiency,     3),
        "rating":           rating,
        "avg_is_return":    round(valid["is_return_pct"].mean(),  2),
        "avg_oos_return":   round(valid["oos_return_pct"].mean(), 2),
        "avg_is_win_rate":  round(valid["is_win_rate"].mean(),    1),
        "avg_oos_win_rate": round(valid["oos_win_rate"].mean(),   1),
        "n_valid_folds":    len(valid),
    }

    return {
        "folds":       folds_df,
        "oos_equity":  oos_equity,
        "is_equity":   is_equity,
        "summary":     summary,
    }
