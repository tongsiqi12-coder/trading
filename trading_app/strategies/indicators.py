"""
indicators.py — all vectorized, no Python for-loops.
Input: OHLCV DataFrame (lowercase cols). Returns enriched copy.
"""

import pandas as pd
import numpy as np


def add_all(df: pd.DataFrame, atr_len: int = 14, vol_len: int = 20) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── Trend EMAs / SMA ─────────────────────────────────────────────────────
    df["ema8"]   = close.ewm(span=8,   adjust=False).mean()
    df["ema21"]  = close.ewm(span=21,  adjust=False).mean()
    df["ema50"]  = close.ewm(span=50,  adjust=False).mean()
    df["sma200"] = close.rolling(200).mean()

    # ── ATR (Wilder smoothing) ────────────────────────────────────────────────
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"]     = tr.ewm(span=atr_len, adjust=False).mean()
    df["atr_pct"] = df["atr"] / close

    # ── Volatility regime ─────────────────────────────────────────────────────
    ret = close.pct_change()
    df["daily_ret"] = ret
    df["vol"]        = ret.rolling(vol_len, min_periods=5).std()
    df["vol_ma"]     = df["vol"].rolling(50).mean()
    df["vol_regime"] = df["vol"] / df["vol_ma"].replace(0, np.nan)

    # ── RSI (Wilder) ──────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    df["bb_mid"]   = close.rolling(vol_len).mean()
    bb_std         = close.rolling(vol_len).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    band_w         = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_pct"]   = (close - df["bb_lower"]) / band_w   # 0=lower band, 1=upper band

    # ── Donchian channels (20-period) ─────────────────────────────────────────
    df["dc_high"] = high.rolling(vol_len).max()
    df["dc_low"]  = low.rolling(vol_len).min()

    # ── Volume ────────────────────────────────────────────────────────────────
    df["vol_ma20"]  = vol.rolling(vol_len).mean()
    df["vol_surge"] = vol > df["vol_ma20"] * 1.5

    # ── Risk filter ───────────────────────────────────────────────────────────
    roll_max        = close.rolling(20, min_periods=1).max()
    df["dd_20"]     = (close / roll_max) - 1
    df["high_risk"] = (df["dd_20"] < -0.15) | (df["vol_regime"] > 2.5)

    return df.dropna(subset=["ema50", "rsi", "bb_pct"])
