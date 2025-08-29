
"""
vnstock_api.py — Data access layer for Vietnam equities (VN30/HOSE/HNX/UPCoM)
--------------------------------------------------------------------------------
Purpose
-------
- Provide a thin, resilient interface to fetch OHLCV for 1D and 1W timeframes.
- Accept the *running* (in-progress) candle at scan time (include_partial=True).
- Normalize output columns to: ["ts","open","high","low","close","volume"].
- Keep a single place to swap/extend data providers (vnstock, vnstock3, custom).

Dependencies
------------
- Tries to import `vnstock3` first, then `vnstock` as a fallback.
- Add one of these to requirements.txt (recommended: vnstock3).

Usage
-----
from vnstock_api import fetch_ohlcv, fetch_ohlcv_batch

df = fetch_ohlcv("VCB", timeframe="1D", limit=600, include_partial=True)
data = fetch_ohlcv_batch(["VCB","CTG","STB"], timeframe="1W", limit=260)
"""

from __future__ import annotations
import os
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------
_PROVIDER = None
_PROVIDER_NAME = None

try:
    # Preferred provider
    from vnstock3 import Vnstock as _VN3
    _PROVIDER = "vnstock3"
    _PROVIDER_NAME = "vnstock3.Vnstock"
except Exception:
    try:
        # Fallback provider
        import vnstock as _VN1  # type: ignore
        _PROVIDER = "vnstock"
        _PROVIDER_NAME = "vnstock"
    except Exception:
        _PROVIDER = None
        _PROVIDER_NAME = None

# Asia/Ho_Chi_Minh timezone (UTC+7)
_TZ = "Asia/Ho_Chi_Minh"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _now_vn() -> pd.Timestamp:
    return pd.Timestamp.now(tz=_TZ)

def _infer_start_date(limit: int, timeframe: str) -> str:
    """Return YYYY-MM-DD start date based on limit bars and timeframe."""
    now = _now_vn()
    if timeframe.upper() == "1D":
        start = now - pd.Timedelta(days=int(limit * 1.6))  # buffer for holidays
    elif timeframe.upper() == "1W":
        start = now - pd.Timedelta(weeks=int(limit * 1.6))
    else:
        raise ValueError("Only 1D and 1W supported")
    return start.strftime("%Y-%m-%d")

def _date_str(ts: pd.Timestamp) -> str:
    return ts.tz_convert(_TZ).strftime("%Y-%m-%d")

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and shape columns to ts/open/high/low/close/volume, sort by ts ascending."""
    rename_map = {
        "time": "ts", "date": "ts", "TradingDate": "ts", "tradingDate": "ts",
        "open": "open", "Open": "open", "o": "open",
        "high": "high", "High": "high", "h": "high",
        "low": "low", "Low": "low", "l": "low",
        "close": "close", "Close": "close", "c": "close",
        "volume": "volume", "Volume": "volume", "v": "volume", "value": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    # Keep only expected columns if present
    keep = [c for c in ["ts","open","high","low","close","volume"] if c in df.columns]
    df = df[keep].copy()

    # Parse timestamps
    if "ts" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["ts"]):
            pass
        else:
            # Try parse common formats
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        # Localize/convert to Asia/Ho_Chi_Minh
        df["ts"] = df["ts"].dt.tz_convert(_TZ)

    # Ensure numeric
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

# ---------------------------------------------------------------------------
# Core fetchers
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, timeframe: str = "1D", limit: int = 600, include_partial: bool = True) -> pd.DataFrame:
    """Fetch OHLCV for a single symbol.

    Parameters
    ----------
    symbol : str
        VN ticker (e.g., 'VCB', 'CTG'). Exchange suffix not required.
    timeframe : {'1D','1W'}
        Resolution of bars. Only daily and weekly supported.
    limit : int
        Max number of bars to return (approximate; upstream API may return more/less).
    include_partial : bool
        If True, keep the in-progress candle (today/this week). If False, drop the last bar
        if its period has not closed yet.
    """
    if _PROVIDER is None:
        raise RuntimeError("No provider found. Please install 'vnstock3' (preferred) or 'vnstock'.")

    tf = timeframe.upper()
    if tf not in {"1D","1W"}:
        raise ValueError("Only 1D and 1W are supported by vnstock_api.fetch_ohlcv")

    start = _infer_start_date(limit=limit, timeframe=tf)
    end = _now_vn().strftime("%Y-%m-%d")

    if _PROVIDER == "vnstock3":
        # vnstock3 usage
        vs = _VN3()
        stock = vs.stock(symbol)
        # vnstock3 expects: symbol, resolution in {'1D','1W'}, start 'YYYY-mm-dd', end 'YYYY-mm-dd'
        raw = stock.history(period="custom", start=start, end=end, interval=tf)  # returns pd.DataFrame
        df = _normalize_df(raw)
    elif _PROVIDER == "vnstock":
        # vnstock (v1) usage
        from vnstock import stock_historical_data  # type: ignore
        raw = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution=tf)
        df = _normalize_df(raw)
    else:
        raise RuntimeError("Unsupported provider dispatch")

    # Optionally drop the last incomplete bar
    if not include_partial and len(df) > 0:
        last = df.iloc[-1]["ts"]
        if tf == "1D":
            is_incomplete = last.date() == _now_vn().date()
        else:  # '1W' — consider week incomplete if within the same ISO week
            now = _now_vn()
            is_incomplete = (last.isocalendar().week == now.isocalendar().week) and (last.year == now.year)
        if is_incomplete:
            df = df.iloc[:-1].copy()

    # Trim to limit (from the end)
    if limit and len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)
    return df

def fetch_ohlcv_batch(symbols: List[str], timeframe: str = "1D", limit: int = 600, include_partial: bool = True) -> Dict[str, pd.DataFrame]:
    """Batch fetch OHLCV for multiple symbols. Returns dict[symbol] -> DataFrame."""
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = fetch_ohlcv(sym, timeframe=timeframe, limit=limit, include_partial=include_partial)
            out[sym] = df
        except Exception as e:
            out[sym] = pd.DataFrame(columns=["ts","open","high","low","close","volume"])
            out[sym].attrs["error"] = str(e)
    return out

# ---------------------------------------------------------------------------
# Convenience: latest bar & trading calendar awareness
# ---------------------------------------------------------------------------
def latest_bar(symbol: str, timeframe: str = "1D", include_partial: bool = True):
    """Return the last bar as a dict or None if unavailable."""
    df = fetch_ohlcv(symbol, timeframe=timeframe, limit=2, include_partial=include_partial)
    if len(df) == 0:
        return None
    row = df.iloc[-1]
    return {
        "symbol": symbol,
        "timeframe": timeframe.upper(),
        "ts": row["ts"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]) if not pd.isna(row["volume"]) else None,
    }

if __name__ == "__main__":
    # Quick self-test (will only run if a provider is installed)
    try:
        test_df = fetch_ohlcv("VCB", timeframe="1D", limit=100, include_partial=True)
        print("Fetched rows:", len(test_df))
        print(test_df.tail())
    except Exception as e:
        print("Self-test failed:", e)
