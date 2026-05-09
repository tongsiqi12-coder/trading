"""
Data Fetcher — yfinance
Covers US stocks, ETFs, FX (via =X suffix), commodities futures (=F suffix)
No API key required.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# ── Preset symbol groups ──────────────────────────────────────────────────────
PRESET_SYMBOLS = {
    "US Large Cap": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "UNH"],
    "Major ETFs":   ["SPY", "QQQ", "IWM", "DIA", "VTI", "GLD", "TLT", "XLF", "XLK"],
    "FX Pairs":     ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
                     "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"],
    "Commodities":  ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F",
                     "ZW=F", "ZC=F", "ZS=F", "PL=F", "PA=F"],
}

# Human-readable display names
SYMBOL_NAMES = {
    # FX
    "EURUSD=X": "EUR/USD", "GBPUSD=X": "GBP/USD", "USDJPY=X": "USD/JPY",
    "AUDUSD=X": "AUD/USD", "USDCAD=X": "USD/CAD", "USDCHF=X": "USD/CHF",
    "NZDUSD=X": "NZD/USD", "EURGBP=X": "EUR/GBP", "EURJPY=X": "EUR/JPY",
    "GBPJPY=X": "GBP/JPY",
    # Commodities
    "GC=F": "Gold", "SI=F": "Silver", "CL=F": "Crude Oil (WTI)",
    "NG=F": "Natural Gas", "HG=F": "Copper",
    "ZW=F": "Wheat", "ZC=F": "Corn", "ZS=F": "Soybeans",
    "PL=F": "Platinum", "PA=F": "Palladium",
}

# Asset type: "equity" has real volume; "fx" and "commodity" do not
def asset_type(ticker: str) -> str:
    if ticker.endswith("=X"):
        return "fx"
    if ticker.endswith("=F"):
        return "commodity"
    return "equity"

# Timeframe config: interval string → annualization factor
TIMEFRAME_CONFIG = {
    "1d":  {"interval": "1d",  "period": "2y",  "ann": 252,        "label": "Daily"},
    "1h":  {"interval": "1h",  "period": "60d", "ann": 252 * 6.5,  "label": "Hourly"},
    "30m": {"interval": "30m", "period": "30d", "ann": 252 * 13,   "label": "30-Min"},
}


def fetch_ohlcv(ticker: str, timeframe: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker.
    FX and commodities: volume is synthetic (ATR-scaled proxy) since yfinance
    doesn't provide real volume for those instruments.
    Returns clean DataFrame with lowercase columns, DatetimeIndex.
    Also attaches 'asset_type' as a df attribute.
    """
    cfg  = TIMEFRAME_CONFIG[timeframe]
    atyp = asset_type(ticker)

    raw = yf.download(
        ticker,
        period=cfg["period"],
        interval=cfg["interval"],
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the symbol.")

    df = raw[["Open", "High", "Low", "Close"]].copy()
    df.columns = ["open", "high", "low", "close"]
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna()

    # Volume handling
    if "Volume" in raw.columns and atyp == "equity":
        df["volume"] = raw["Volume"].reindex(df.index).fillna(0)
    else:
        # FX / commodities: proxy volume via ATR momentum (relative range expansion)
        # This lets vol_surge fire when price range is large relative to recent avg
        rng = df["high"] - df["low"]
        df["volume"] = (rng / rng.rolling(20, min_periods=5).mean() * 1_000_000).fillna(1_000_000)

    # Store asset type as metadata attribute
    df.attrs["asset_type"] = atyp
    df.attrs["ticker"]     = ticker
    return df


def fetch_multi(tickers: list, timeframe: str = "1d") -> dict:
    """Fetch multiple tickers. Returns {ticker: DataFrame}."""
    result = {}
    for t in tickers:
        try:
            result[t] = fetch_ohlcv(t.strip().upper(), timeframe)
        except Exception:
            result[t] = None
    return result


def get_live_quote(ticker: str) -> dict:
    """Get latest price + day change via yfinance fast_info."""
    try:
        info = yf.Ticker(ticker).fast_info
        price    = info.last_price
        prev     = info.previous_close
        change_p = (price - prev) / prev * 100 if prev else 0.0
        return {
            "price":        round(price, 5),
            "change_pct":   round(change_p, 2),
            "high_52w":     round(info.year_high, 5),
            "low_52w":      round(info.year_low,  5),
            "volume":       int(info.three_month_average_volume or 0),
            "display_name": SYMBOL_NAMES.get(ticker, ticker),
            "asset_type":   asset_type(ticker),
        }
    except Exception:
        return {"price": None, "change_pct": None, "high_52w": None,
                "low_52w": None, "volume": None,
                "display_name": SYMBOL_NAMES.get(ticker, ticker),
                "asset_type": asset_type(ticker)}


def annualization_factor(timeframe: str) -> float:
    return TIMEFRAME_CONFIG[timeframe]["ann"]
