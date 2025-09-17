from __future__ import annotations

"""
indicators.py — Core technical indicators and strategy-enrichment utilities
for Vietnam equities OHLCV DataFrames.

Input schema (expected):
    index  : DatetimeIndex (tz-aware preferred)
    columns: ["open","high","low","close","volume"] as float-like

Design goals:
- TradingView-compatible EMA/RSI (Wilder smoothing via EWMA alpha=1/period)
- Robust to NaNs / divide-by-zero
- Easy to reuse in your signal engine

Public API:
- sma(series, window)
- ema(series, span)
- rma(series, period)  # Wilder's RMA via EWM(alpha=1/period)
- rsi(series, period=14)
- atr(df, period=14)
- bollinger(series, window=20, num_std=2.0) -> (upper, mid, lower)
- rolling_zscore(series, window=20)
- enrich_indicators(df)
- enrich_more(df)
- calc_vp(df, window_bars=120, bins=24, top_k=5)  # volume profile zones

All functions return new Series/DataFrames aligned to the input index, leaving
original inputs unmodified.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List

# -----------------------------
# Helpers
# -----------------------------

def _to_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame or a Series")
        return x.iloc[:, 0]
    return x


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    out = numer.astype(float) / denom.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan)


# -----------------------------
# Core Indicators (correct & reusable)
# -----------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    s = _to_series(series).astype(float)
    return s.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average (TradingView-compatible default).

    Notes: adjust=False to use recursive formula; no bias correction.
    """
    s = _to_series(series).astype(float)
    return s.ewm(span=span, adjust=False).mean()


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's RMA via EWM with alpha=1/period (TV-compatible)."""
    s = _to_series(series).astype(float)
    alpha = 1.0 / float(period)
    return s.ewm(alpha=alpha, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (RMA) for better TV parity."""
    s = _to_series(series).astype(float)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    rma_up = rma(up, period)
    rma_dn = rma(down, period)
    rs = _safe_div(rma_up, rma_dn)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's RMA.

    TR = max(
        high - low,
        abs(high - prev_close),
        abs(low  - prev_close)
    )
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return rma(tr, period)


def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands with population std (ddof=0). Returns (upper, mid, lower)."""
    s = _to_series(series).astype(float)
    mid = s.rolling(window, min_periods=window).mean()
    # population std: ddof=0
    std = s.rolling(window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """(x - mean)/std with ddof=0, avoiding division by zero (returns 0 where std==0)."""
    s = _to_series(series).astype(float)
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0)
    z = _safe_div(s - mean, std)
    return z.fillna(0.0)


# -----------------------------
# Strategy Enrichment
# -----------------------------

def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add core indicators used broadly by strategies.

    Adds columns:
        - ema20, ema50, rsi14
        - bb_upper, bb_mid, bb_lower
        - bb_width_pct = (upper - lower) / base * 100, base = bb_mid fallback close
        - atr14
        - vol_sma20, vol_ratio = volume / vol_sma20
    """
    out = df.copy()

    # Core MAs & RSI
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["rsi14"] = rsi(out["close"], 14)

    # Bollinger
    bb_u, bb_m, bb_l = bollinger(out["close"], window=20, num_std=2.0)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bb_u, bb_m, bb_l

    # Width % with safe base
    base = bb_m.copy()
    base = base.replace(0, np.nan)
    base = base.fillna(out["close"].replace(0, np.nan))
    width = (bb_u - bb_l)
    out["bb_width_pct"] = _safe_div(width, base) * 100.0

    # ATR14
    out["atr14"] = atr(out, 14)

    # Volume features
    out["vol_sma20"] = sma(out["volume"], 20)
    out["vol_ratio"] = _safe_div(out["volume"], out["vol_sma20"]).fillna(0.0)

    return out


def enrich_more(df: pd.DataFrame) -> pd.DataFrame:
    """Additional features for strategies.

    Adds columns:
        - vol_z20, vol_up, vol_dn
        - body_pct, upper_wick_pct, lower_wick_pct (percent of candle range, clipped 0..100)
        - sma20, sma50 (soft S/R references)
    """
    out = df.copy()

    # Volume statistics
    out["vol_z20"] = rolling_zscore(out["volume"], 20)
    vol_sma20 = sma(out["volume"], 20)
    # Volume flags: spike + direction
    is_spike = out["volume"] > vol_sma20
    out["vol_up"] = (is_spike & (out["close"] >= out["open"]))
    out["vol_dn"] = (is_spike & (out["close"] < out["open"]))

    # Candle anatomy
    o = out["open"].astype(float)
    h = out["high"].astype(float)
    l = out["low"].astype(float)
    c = out["close"].astype(float)

    candle_range = (h - l).clip(lower=0.0)
    # avoid zero range -> use small epsilon to prevent div0; we keep pct=0 when range tiny
    eps = 1e-12
    denom = candle_range.where(candle_range > eps, np.nan)

    body = (c - o).abs()
    upper_wick = (h - np.maximum(o, c))
    lower_wick = (np.minimum(o, c) - l)

    body_pct = _safe_div(body, denom) * 100.0
    upper_wick_pct = _safe_div(upper_wick, denom) * 100.0
    lower_wick_pct = _safe_div(lower_wick, denom) * 100.0

    out["body_pct"] = body_pct.clip(lower=0.0, upper=100.0).fillna(0.0)
    out["upper_wick_pct"] = upper_wick_pct.clip(lower=0.0, upper=100.0).fillna(0.0)
    out["lower_wick_pct"] = lower_wick_pct.clip(lower=0.0, upper=100.0).fillna(0.0)

    # Soft S/R SMAs
    out["sma20"] = sma(out["close"], 20)
    out["sma50"] = sma(out["close"], 50)

    return out


# -----------------------------
# Liquidity zones / Volume Profile
# -----------------------------

def calc_vp(df: pd.DataFrame, window_bars: int = 120, bins: int = 24, top_k: int = 5) -> pd.DataFrame:
    """Compute a simple Volume Profile over the trailing window.

    Steps:
        1) Use HLC3 = (H+L+C)/3 as representative price per bar
        2) Bucket prices into `bins` between min..max
        3) Sum volumes by bucket
        4) Return top-k zones with columns: [low, high, mid, volume]
    """
    if bins < 1 or top_k < 1 or len(df) == 0:
        return pd.DataFrame(columns=["low","high","mid","volume"])  # empty

    tail = df.tail(int(window_bars)).copy()
    h = tail["high"].astype(float)
    l = tail["low"].astype(float)
    c = tail["close"].astype(float)
    v = tail["volume"].astype(float).fillna(0.0)

    hlc3 = (h + l + c) / 3.0
    pmin = float(hlc3.min())
    pmax = float(hlc3.max())
    if not np.isfinite(pmin) or not np.isfinite(pmax) or pmax <= pmin:
        return pd.DataFrame(columns=["low","high","mid","volume"])  # degenerate

    # Bin edges and labels
    edges = np.linspace(pmin, pmax, num=bins + 1)
    cats = pd.cut(hlc3, bins=edges, right=False, include_lowest=True)

    vol_by_bin = v.groupby(cats).sum(min_count=1).sort_values(ascending=False)
    vol_by_bin = vol_by_bin.dropna()
    if vol_by_bin.empty:
        return pd.DataFrame(columns=["low","high","mid","volume"])  # empty

    # Build result rows
    lows: List[float] = []
    highs: List[float] = []
    mids: List[float] = []
    vols: List[float] = []

    for interval, volsum in vol_by_bin.head(top_k).items():
        low_edge = float(interval.left)
        high_edge = float(interval.right)
        mid = (low_edge + high_edge) / 2.0
        lows.append(low_edge)
        highs.append(high_edge)
        mids.append(mid)
        vols.append(float(volsum))

    res = pd.DataFrame({
        "low": lows,
        "high": highs,
        "mid": mids,
        "volume": vols,
    })
    # Sort by price ascending for convenient display
    return res.sort_values("low").reset_index(drop=True)


# -----------------------------
# Minimal self-check (optional)
# -----------------------------
if __name__ == "__main__":
    # Quick synthetic check
    idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
    rng = np.random.default_rng(42)
    price = pd.Series(np.cumsum(rng.normal(0, 1, size=len(idx))) + 50, index=idx)
    high = price + rng.uniform(0.5, 1.5, size=len(idx))
    low = price - rng.uniform(0.5, 1.5, size=len(idx))
    open_ = price + rng.uniform(-0.5, 0.5, size=len(idx))
    close = price
    volume = pd.Series(rng.integers(1e5, 5e5, size=len(idx)), index=idx)
    df = pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":volume})

    enriched = enrich_more(enrich_indicators(df))
    print(enriched.tail(3))
    print(calc_vp(enriched).head())
