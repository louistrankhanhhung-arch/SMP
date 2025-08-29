
"""
feature_primitives.py — derive first-layer features from OHLCV
----------------------------------------------------------------
Input df columns: ["ts","open","high","low","close","volume"]
This module:
  1) Enriches with technical indicators (via indicators.enrich_indicators)
  2) Derives compact trend/momentum/volatility/position/SR features
  3) Provides a `compute_features_by_tf` entrypoint for {'1D': df1d, '1W': df1w}

Outputs
-------
- enrich_and_features(df, timeframe): 
    {
      'timeframe': '1D'|'1W',
      'df': <enriched DataFrame>,
      'features': <dict snapshot for last bar>
    }
- compute_features_by_tf({'1D': df1d, '1W': df1w}) -> dict[str, dict]

Notes
-----
- Designed to be lightweight and deterministic.
- Uses simple swing detection for SR pivots.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from indicators import enrich_indicators

# --- robust coerce helper ---
def _ensure_series(x):
    """Return a pandas Series regardless of input being Series/ndarray/list/scalar.
    Keeps length for array-like; for scalars, returns length-1 Series.
    """
    if isinstance(x, pd.Series):
        return x
    try:
        if isinstance(x, (list, tuple, np.ndarray)):
            return pd.Series(pd.to_numeric(x, errors="coerce"))
        return pd.Series([pd.to_numeric(x, errors="coerce")])
    except Exception:
        return pd.Series([np.nan])

# =========================
# Helpers
# =========================
def _pct(a, b):
    # (a/b - 1)*100 with safe division
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    with np.errstate(divide='ignore', invalid='ignore'):
        out = (a / b - 1.0) * 100.0
    return out

def _slope(series: pd.Series, window: int = 5) -> pd.Series:
    # Simple slope: price change over 'window' bars
    s = _ensure_series(pd.to_numeric(series, errors="coerce"))
    return s.diff(window)

def _rolling_extrema(series: pd.Series, window: int = 5, mode: str = "high") -> pd.Series:
    s = _ensure_series(pd.to_numeric(series, errors="coerce"))
    if mode == "high":
        return s.rolling(window, min_periods=1).max()
    else:
        return s.rolling(window, min_periods=1).min()

def _pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # Classic floor trader pivots (daily/weekly)
    h, l, c = pd.to_numeric(high, errors="coerce"), pd.to_numeric(low, errors="coerce"), pd.to_numeric(close, errors="coerce")
    P = (h.shift(1) + l.shift(1) + c.shift(1)) / 3.0
    R1 = 2*P - l.shift(1)
    S1 = 2*P - h.shift(1)
    return P, R1, S1

def _zscore(series: pd.Series, window: int = 20) -> pd.Series:
    s = _ensure_series(pd.to_numeric(series, errors="coerce"))
    mean = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=0)
    return (s - mean) / std

# =========================
# Core feature derivation
# =========================
def _derive_features(enriched: pd.DataFrame, timeframe: str) -> dict:
    last = enriched.iloc[-1]
    # Trend structure
    close = last["close"]
    ema20, ema50, ema200 = last["ema20"], last["ema50"], last["ema200"]
    bb_mid, bb_up, bb_lo = last["bb_mid"], last["bb_upper"], last["bb_lower"]
    rsi14 = last["rsi14"]
    atr14, atr_pct = last["atr14"], last["atr_pct"]
    macd_hist = last["macd_hist"]
    st_k, st_d = last["stoch_k"], last["stoch_d"]
    vol = last.get("volume", np.nan)

    # Distance metrics (%)
    dist_ema20 = float(_pct(close, ema20).iloc[-1]) if hasattr(ema20, 'iloc') else float(_pct(close, ema20))
    dist_ema50 = float(_pct(close, ema50).iloc[-1]) if hasattr(ema50, 'iloc') else float(_pct(close, ema50))
    dist_ema200 = float(_pct(close, ema200).iloc[-1]) if hasattr(ema200, 'iloc') else float(_pct(close, ema200))
    dist_bb_mid = float(_pct(close, bb_mid).iloc[-1]) if hasattr(bb_mid, 'iloc') else float(_pct(close, bb_mid))

    # Position within Bollinger (%B already exists 0..1 typical)
    bb_pctb = float(last.get("bb_pctb", np.nan))
    bb_width = float(last.get("bb_width", np.nan))

    # Volatility regimes
    vol_regime = "low" if bb_width < 0.05 else ("mid" if bb_width < 0.12 else "high")  # heuristic for VN 1D

    # MA trend flags (stacking + slope)
    # Slope over last 5 bars
    ema20_slope5 = float(_slope(enriched["ema20"], 5).iloc[-1])
    ema50_slope5 = float(_slope(enriched["ema50"], 5).iloc[-1])
    ema200_slope5 = float(_slope(enriched["ema200"], 5).iloc[-1])

    stacked_bull = (close > ema20) and (ema20 > ema50) and (ema50 > ema200)
    stacked_bear = (close < ema20) and (ema20 < ema50) and (ema50 < ema200)

    # Momentum flags
    rsi_zone = "overbought" if rsi14 >= 70 else ("oversold" if rsi14 <= 30 else "neutral")
    macd_bias = "bull" if macd_hist > 0 else ("bear" if macd_hist < 0 else "flat")
    stoch_cross = "k>dx" if st_k > st_d else ("k<dx" if st_k < st_d else "flat")

    # Volume features: volume vs 20-day average, zscore
    vol_ma20 = float(enriched["volume"].rolling(20).mean().iloc[-1]) if "volume" in enriched.columns else np.nan
    vol_ratio = float(vol / vol_ma20) if vol_ma20 and vol_ma20 > 0 else np.nan
    vol_z = float(_zscore(enriched["volume"], 20).iloc[-1]) if "volume" in enriched.columns else np.nan

    # Support/Resistance using rolling pivots & recent extrema
    hi20 = float(_rolling_extrema(enriched["high"], 20, "high").iloc[-1])
    lo20 = float(_rolling_extrema(enriched["low"], 20, "low").iloc[-1])
    hi50 = float(_rolling_extrema(enriched["high"], 50, "high").iloc[-1])
    lo50 = float(_rolling_extrema(enriched["low"], 50, "low").iloc[-1])
    P, R1, S1 = _pivot_points(enriched["high"], enriched["low"], enriched["close"])
    pivot = float(P.iloc[-1]) if len(P) else np.nan
    R1v = float(R1.iloc[-1]) if len(R1) else np.nan
    S1v = float(S1.iloc[-1]) if len(S1) else np.nan

    # Distances to SR (%)
    dist_hi20 = float(_pct(close, hi20))
    dist_lo20 = float(_pct(close, lo20))
    dist_hi50 = float(_pct(close, hi50))
    dist_lo50 = float(_pct(close, lo50))
    dist_pivot = float(_pct(close, pivot))

    # Return compact feature dict
    return {
        "timeframe": timeframe,
        "close": float(close),
        "ema20": float(ema20), "ema50": float(ema50), "ema200": float(ema200),
        "dist_ema20_pct": dist_ema20, "dist_ema50_pct": dist_ema50, "dist_ema200_pct": dist_ema200,
        "bb_mid": float(bb_mid), "bb_width": bb_width, "bb_pctb": bb_pctb, "dist_bb_mid_pct": dist_bb_mid,
        "rsi14": float(rsi14), "rsi_zone": rsi_zone,
        "macd_hist": float(macd_hist), "macd_bias": macd_bias,
        "stoch_k": float(st_k), "stoch_d": float(st_d), "stoch_cross": stoch_cross,
        "atr14": float(atr14), "atr_pct": float(atr_pct),
        "ema20_slope5": ema20_slope5, "ema50_slope5": ema50_slope5, "ema200_slope5": ema200_slope5,
        "stacked_bull": bool(stacked_bull), "stacked_bear": bool(stacked_bear),
        "vol": float(vol) if not np.isnan(vol) else None, "vol_ma20": vol_ma20, "vol_ratio": vol_ratio, "vol_z": vol_z,
        "hi20": hi20, "lo20": lo20, "hi50": hi50, "lo50": lo50,
        "pivot": pivot, "R1": R1v, "S1": S1v,
        "dist_hi20_pct": dist_hi20, "dist_lo20_pct": dist_lo20,
        "dist_hi50_pct": dist_hi50, "dist_lo50_pct": dist_lo50,
        "dist_pivot_pct": dist_pivot,
        "vol_regime": vol_regime,
    }

# =========================
# Public API
# =========================
def enrich_and_features(df: pd.DataFrame, timeframe: str) -> dict:
    """Enrich df with indicators and compute compact feature snapshot (last row).

    Returns dict with keys: 'timeframe', 'df', 'features'
    """
    if df is None or len(df) == 0:
        return {'timeframe': timeframe, 'df': df, 'features': {}}
    e = enrich_indicators(df)
    feats = _derive_features(e, timeframe=timeframe)
    out = {'timeframe': timeframe, 'df': e, 'features': feats}
    try:
        if ("volume" not in e.columns) or (getattr(e["volume"], "isna", lambda: None)().all()):
            out["warnings"] = ["no_volume"]
    except Exception:
        pass
    return out

def compute_features_by_tf(dfs_by_tf: Dict[str, pd.DataFrame]) -> Dict[str, dict]:
    """Compute features for each timeframe dict entry (e.g., {'1D': df1d, '1W': df1w})."""
    out: Dict[str, dict] = {}
    for tf, d in dfs_by_tf.items():
        try:
            out[tf] = enrich_and_features(d, timeframe=tf)
        except Exception as e:
            out[tf] = {'timeframe': tf, 'df': d, 'features': {}, 'error': str(e)}
    return out
