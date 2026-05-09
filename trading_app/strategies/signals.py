"""
strategies/signals.py — Three fully vectorized strategies.
Each returns the df with a 'signal' column: +1 long, -1 short, 0 flat.
"""

import pandas as pd
import numpy as np


def _apply_min_hold(signal: pd.Series, min_hold: int) -> pd.Series:
    """Suppress signal flips within min_hold bars (vectorized)."""
    s = signal.copy().astype(float)
    change     = s != s.shift(1)
    block      = change.cumsum()
    block_size = block.groupby(block).transform("count")
    mask = (block_size < min_hold) & change & (s != 0)
    s[mask] = np.nan
    return s.ffill().fillna(0).astype(int)


def trend_following(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """EMA trend + RSI filter + volume confirmation."""
    df = df.copy()
    p  = params

    long_cond = (
        (df["ema8"]  > df["ema21"]) &
        (df["ema21"] > df["ema50"]) &
        (df["close"] > df["sma200"]) &
        (df["rsi"]   > p.get("rsi_long", 52)) &
        df["vol_surge"] &
        (df["vol_regime"] < p.get("max_vol_regime", 1.8)) &
        (~df["high_risk"])
    )
    short_cond = (
        (df["ema8"]  < df["ema21"]) &
        (df["ema21"] < df["ema50"]) &
        (df["close"] < df["sma200"]) &
        (df["rsi"]   < p.get("rsi_short", 48)) &
        df["vol_surge"] &
        (df["vol_regime"] < p.get("max_vol_regime", 1.8)) &
        (~df["high_risk"])
    )

    signal = pd.Series(0, index=df.index)
    signal[long_cond]  =  1
    signal[short_cond] = -1
    df["signal"]   = _apply_min_hold(signal, p.get("min_hold", 3))
    df["strategy"] = "Trend Following"
    return df


def mean_reversion(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Bollinger Band extremes + RSI mean reversion."""
    df = df.copy()
    p  = params

    long_cond = (
        (df["bb_pct"] < p.get("bb_oversold",   0.05)) &
        (df["rsi"]    < p.get("rsi_oversold",   30))   &
        (df["vol_regime"] < p.get("max_vol_regime", 0.9)) &
        (~df["high_risk"])
    )
    short_cond = (
        (df["bb_pct"] > p.get("bb_overbought", 0.95)) &
        (df["rsi"]    > p.get("rsi_overbought", 70))   &
        (df["vol_regime"] < p.get("max_vol_regime", 0.9)) &
        (~df["high_risk"])
    )

    signal = pd.Series(0, index=df.index)
    signal[long_cond]  =  1
    signal[short_cond] = -1
    df["signal"]   = _apply_min_hold(signal, p.get("min_hold", 2))
    df["strategy"] = "Mean Reversion"
    return df


def donchian_breakout(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Donchian channel breakout + MACD confirmation + volume surge."""
    df = df.copy()
    p  = params

    long_cond = (
        (df["close"] > df["dc_high"].shift(1)) &
        (df["ema21"] > df["ema50"]) &
        (df["macd"]  > df["macd_signal"]) &
        df["vol_surge"] &
        (~df["high_risk"])
    )
    short_cond = (
        (df["close"] < df["dc_low"].shift(1)) &
        (df["ema21"] < df["ema50"]) &
        (df["macd"]  < df["macd_signal"]) &
        df["vol_surge"] &
        (~df["high_risk"])
    )

    signal = pd.Series(0, index=df.index)
    signal[long_cond]  =  1
    signal[short_cond] = -1
    df["signal"]   = _apply_min_hold(signal, p.get("min_hold", 2))
    df["strategy"] = "Donchian Breakout"
    return df


DEFAULT_PARAMS = {
    "Trend Following": {
        "rsi_long": 52, "rsi_short": 48,
        "max_vol_regime": 1.8, "min_hold": 3,
    },
    "Mean Reversion": {
        "bb_oversold": 0.05, "bb_overbought": 0.95,
        "rsi_oversold": 30, "rsi_overbought": 70,
        "max_vol_regime": 0.9, "min_hold": 2,
    },
    "Donchian Breakout": {
        "max_vol_regime": 2.0, "min_hold": 2,
    },
}

STRATEGY_FN = {
    "Trend Following":   trend_following,
    "Mean Reversion":    mean_reversion,
    "Donchian Breakout": donchian_breakout,
}
