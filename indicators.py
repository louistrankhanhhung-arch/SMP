from __future__ import annotations

"""
indicators.py — Core, TradingView‑compatible indicators and enrich helpers
for Vietnam equities OHLCV DataFrames.

Expected input schema for enrichers: columns ["open","high","low","close","volume"],
DatetimeIndex (tz‑aware is fine). Returns the same DataFrame with new columns added.

Design goals
- Correct per textbook/TradingView behavior where applicable
- Vectorized Pandas/Numpy for speed; no loops
- Safe math: avoid divide‑by‑zero, robust to NaNs

Public API
- ema(series, span)
- sma(series, window)
- rsi(series, period=14)              # Wilder smoothing (alpha = 1/period)
- atr(df, period=14)                  # TR = max(H-L, |H-Cprev|, |L-Cprev|); Wilder RMA
- bollinger(series, window=20, num_std=2.0) -> (upper, mid, lower)
- rolling_zscore(series, window=20)
- enrich_indicators(df)
- enrich_more(df)
"""

import numpy as np
import pandas as pd
from typing import Tuple

# ---------- Core building blocks ----------

def _to_float_series(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    return s.astype(float)


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average (TV‑compatible)."""
    s = _to_float_series(series)
    return s.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average using Pandas ewm with adjust=False (TradingView‑style).
    Note: TV initializes EMA from the first value; using adjust=False yields same recursive form.
    """
    s = _to_float_series(series)
    # No min_periods to follow TV behavior of early warm‑up
    return s.ewm(span=span, adjust=False).mean()


def _rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA (a.k.a. SMMA) using EWM with alpha=1/period, adjust=False."""
    s = _to_float_series(series)
    alpha = 1.0 / float(period)
    return s.ewm(alpha=alpha, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder's smoothing (alpha = 1/period).

    Steps:
      1) delta = close.diff()
      2) gain = max(delta, 0), loss = max(-delta, 0)
      3) avg_gain = RMA(gain, period); avg_loss = RMA(loss, period)
      4) RS = avg_gain / avg_loss; RSI = 100 - 100/(1+RS)
    """
    s = _to_float_series(series)
    delta = s.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _rma(gain, period)
    avg_loss = _rma(loss, period)

    # Avoid division by zero: where avg_loss==0 and avg_gain==0 => RSI=50; if loss==0 but gain>0 => RSI=100
    rs = pd.Series(np.where(avg_loss <= 1e-12, np.inf, avg_gain / avg_loss), index=s.index)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))

    # If both avg_gain and avg_loss ~ 0 (flat series), set RSI=50
    both_zero = (avg_gain <= 1e-12) & (avg_loss <= 1e-12)
    rsi_val = rsi_val.where(~both_zero, 50.0)
    return rsi_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder).

    TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = RMA(TR, period)
    """
    high = _to_float_series(df["high"]) if "high" in df else None
    low  = _to_float_series(df["low"]) if "low" in df else None
    close = _to_float_series(df["close"]) if "close" in df else None
    if high is None or low is None or close is None:
        raise ValueError("atr requires columns: high, low, close")

    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return _rma(tr, period)


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. std computed with ddof=0 to match many trading platforms.
    Returns (upper, mid, lower).
    """
    s = _to_float_series(series)
    mid = sma(s, window)
    # Using ddof=0 via .rolling(...).std(ddof=0)
    std = s.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score with safe denominator (avoid divide-by-zero)."""
    s = _to_float_series(series)
    mean = s.rolling(window=window, min_periods=window).mean()
    std = s.rolling(window=window, min_periods=window).std(ddof=0)
    safe_std = std.where(std > 1e-12)
    z = (s - mean) / safe_std
    return z.replace([np.inf, -np.inf], np.nan)

# ---------- Enrichers ----------

def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add core indicators commonly used by strategies.

    Adds: ema20, ema50, rsi14, bb_upper/mid/lower, bb_width_pct, atr14,
          vol_sma20, vol_ratio
    """
    if not {"open","high","low","close","volume"}.issubset(df.columns):
        raise ValueError("enrich_indicators expects columns: open, high, low, close, volume")

    out = df.copy()
    close = _to_float_series(out["close"])

    # Moving averages & RSI
    out["ema20"] = ema(close, 20)
    out["ema50"] = ema(close, 50)
    out["rsi14"] = rsi(close, 14)

    # Bollinger
    bb_u, bb_m, bb_l = bollinger(close, window=20, num_std=2.0)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bb_u, bb_m, bb_l

    # Band width percentage based on mid as primary base, fallback to close if mid==0
    base = bb_m.copy()
    # Where |base| is ~0, fallback to close to avoid division by zero
    base = base.where(base.abs() > 1e-12, close)
    out["bb_width_pct"] = (bb_u - bb_l) / base * 100.0

    # ATR
    out["atr14"] = atr(out, 14)

    # Volume features
    vol = _to_float_series(out["volume"]) if "volume" in out else pd.Series(index=out.index, dtype=float)
    out["vol_sma20"] = sma(vol, 20)
    out["vol_ratio"] = vol / out["vol_sma20"]

    return out


def enrich_more(df: pd.DataFrame) -> pd.DataFrame:
    """Add supplemental features for strategy logic & risk filters.

    Adds: vol_z20, vol_up, vol_dn, body_pct, upper_wick_pct, lower_wick_pct,
          sma20, sma50

    Notes:
      - Candle body/wick % are relative to full range (H-L). Values are clipped to [0, 100].
      - If H==L (doji/flat), allocate 0 body / 0 wicks to avoid divide-by-zero.
    """
    if not {"open","high","low","close","volume"}.issubset(df.columns):
        raise ValueError("enrich_more expects columns: open, high, low, close, volume")

    out = df.copy()

    # Volume z-score & flags
    vol = _to_float_series(out["volume"]) if "volume" in out else pd.Series(index=out.index, dtype=float)
    out["vol_z20"] = rolling_zscore(vol, 20)

    # If vol_sma20 not already present, compute
    if "vol_sma20" not in out:
        out["vol_sma20"] = sma(vol, 20)

    out["vol_up"] = (vol > out["vol_sma20"]).astype(int)
    out["vol_dn"] = (vol < out["vol_sma20"]).astype(int)

    # Candle structure in percent
    o = _to_float_series(out["open"]) ; h = _to_float_series(out["high"]) ; l = _to_float_series(out["low"]) ; c = _to_float_series(out["close"]) 
    full = (h - l).abs()
    safe_full = full.where(full > 1e-12)

    body = (c - o).abs()
    upper_wick = (h - c.where(c >= o, o))
    lower_wick = ((o.where(c >= o, c)) - l)

    body_pct = (body / safe_full) * 100.0
    upper_wick_pct = (upper_wick / safe_full) * 100.0
    lower_wick_pct = (lower_wick / safe_full) * 100.0

    # Where full range is ~0, set all to 0
    zeros = safe_full.isna()
    body_pct = body_pct.where(~zeros, 0.0)
    upper_wick_pct = upper_wick_pct.where(~zeros, 0.0)
    lower_wick_pct = lower_wick_pct.where(~zeros, 0.0)

    # Clip to reasonable 0..100
    out["body_pct"] = body_pct.clip(lower=0.0, upper=100.0)
    out["upper_wick_pct"] = upper_wick_pct.clip(lower=0.0, upper=100.0)
    out["lower_wick_pct"] = lower_wick_pct.clip(lower=0.0, upper=100.0)

    # Soft SR MAs
    close = _to_float_series(out["close"]) 
    out["sma20"] = sma(close, 20)
    out["sma50"] = sma(close, 50)

    return out


# ---------- Quick self-check ----------
if __name__ == "__main__":
    # Synthetic example to verify shapes; replace with real OHLCV to test
    idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    data = pd.DataFrame({
        "open": np.linspace(10, 20, 120) + np.random.randn(120)*0.5,
        "high": np.linspace(10.5, 20.5, 120) + np.random.randn(120)*0.5,
        "low":  np.linspace(9.5, 19.5, 120) + np.random.randn(120)*0.5,
        "close":np.linspace(10.2, 20.2, 120) + np.random.randn(120)*0.5,
        "volume":np.random.randint(1000, 5000, 120).astype(float),
    }, index=idx)

    e1 = enrich_indicators(data)
    e2 = enrich_more(e1)
    print(e2.tail(3))
