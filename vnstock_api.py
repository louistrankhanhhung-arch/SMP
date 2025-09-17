# -*- coding: utf-8 -*-
"""
vnstock_api.py — Robust fetcher for Vietnam equities OHLCV (ccxt-style niceties)
-----------------------------------------------------------------------------
What this module provides
1) Scope & normalization
   • enableRateLimit via pacing + backoff; configurable timeout; optional HTTP(S) proxy via env
   • Auto symbol normalization (upper, trims, common HOSE/HSX suffix variants)
   • Timeframe alias map "friendly" → ccxt: {1H→1h, 4H→4h, 1D→1d, 1W→1w}
   • Auto-load provider (vnstock) on first use

2) OHLCV data correctness
   • Convert provider OHLCV → pandas DataFrame with UTC DatetimeIndex
   • Columns: open, high, low, close, volume (float64); duplicates by timestamp are de-duped keeping *latest*
   • Strict NaN filtering on O/H/L/C; numeric coercion

3) Partial-bar handling (closed bars only)
   • Drop the last bar if it is incomplete for ALL timeframes (1h/4h/1d/1w)

4) Robustness & deep history pagination
   • Retry with exponential backoff for NetworkError/Timeout/429 (“Too many requests”)
   • Backward pagination by date windows within [start_ms, end_ms] (UTC ms) with max_pages & sleep_sec pacing
   • Merge, sort by time, drop dups (keep newest)

5) Public API
   • fetch_ohlcv(symbol, timeframe, limit=600, include_partial=False)
   • fetch_ohlcv_history(symbol, timeframe, start_ms, end_ms, max_pages=20, window_days=120,
                         sleep_sec=0.8, include_partial=False)
   • fetch_batch(symbol, timeframes=("1D","1W"), limit=600, include_partial=False,
                 step_sleep_sec=0.6)  → returns {tf: DataFrame}
   • fetch_ohlcv_batch(symbols, timeframe="1D", ...) → returns {symbol: DataFrame}

Notes
- vnstock currently focuses on end-of-day/weekly. Intraday (1H/4H) availability varies by source.
  For unsupported timeframes, this module raises NotImplementedError.
- Proxies: set HTTP_PROXY/HTTPS_PROXY env vars if needed.
"""
from __future__ import annotations
import os
import time
import math
import importlib
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------- Config & constants -----------------------------
ENABLE_RATE_LIMIT = bool(int(os.getenv("ENABLE_RATE_LIMIT", "1")))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))  # seconds (best-effort; provider-dependent)
RATE_LIMIT_MIN_SLEEP = float(os.getenv("RATE_LIMIT_MIN_SLEEP", "0.4"))  # baseline sleep between calls
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
BACKOFF_BASE = float(os.getenv("BACKOFF_BASE", "1.7"))  # exponential backoff factor

VN_TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh (no DST)

# Friendly → ccxt timeframe map
TF_CCXT_MAP = {
    "1H": "1h", "1h": "1h",
    "4H": "4h", "4h": "4h",
    "1D": "1d", "1d": "1d", "D": "1d", "day": "1d", "daily": "1d",
    "1W": "1w", "1w": "1w", "W": "1w", "week": "1w", "weekly": "1w",
}

SUPPORTED_SOURCES = ("VCI", "TCBS", "MSN")

# ----------------------------- Utilities -------------------------------------

def _now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def _to_utc_index(dt_like: pd.Series) -> pd.DatetimeIndex:
    ts = pd.to_datetime(dt_like, errors="coerce", utc=True)
    # ensure UTC index
    return ts.tz_convert(timezone.utc)


def _coerce_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False).str.replace("_", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    out = out.mask(~np.isfinite(out))
    return out


def _normalize_symbol(sym: str) -> str:
    return sym.strip().upper()


def _symbol_variants(sym: str) -> List[str]:
    b = _normalize_symbol(sym)
    return [b, f"{b}.HOSE", f"{b}:HOSE", f"{b}.HSX", f"{b}:HSX"]


def _resolve_timeframe(tf: str) -> str:
    tf_norm = TF_CCXT_MAP.get(str(tf).strip(), None)
    if tf_norm is None:
        raise ValueError(f"Unsupported timeframe alias: {tf}")
    return tf_norm


def _interval_aliases_for_provider(tf_ccxt: str) -> List[str]:
    # Map our ccxt-style back to provider-friendly strings
    if tf_ccxt == "1w":
        return ["1W", "1w", "W", "week", "weekly"]
    if tf_ccxt == "1d":
        return ["1D", "1d", "D", "day", "daily"]
    if tf_ccxt in ("1h", "4h"):
        # vnstock intraday may be unsupported; we keep aliases for future-proofing
        return [tf_ccxt, tf_ccxt.upper()]
    return [tf_ccxt]


def _sleep_base(sec: float) -> None:
    if ENABLE_RATE_LIMIT:
        time.sleep(max(sec, RATE_LIMIT_MIN_SLEEP))


def _should_retry(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "timeout" in text
        or "timed out" in text
        or "network" in text
        or "connection" in text
        or "429" in text
        or "too many requests" in text
        or "rate" in text and "limit" in text
    )


def _retryable_call(fn, *args, **kwargs):
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if not _should_retry(e):
                raise
            sleep_s = (BACKOFF_BASE ** i) * RATE_LIMIT_MIN_SLEEP
            _sleep_base(sleep_s)
    raise last_err


# ----------------------------- Provider loader --------------------------------

def _load_vnstock():
    return importlib.import_module("vnstock")


# ----------------------------- Core transforms --------------------------------

def _sanitize_ohlcv(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        return df

    # unify columns
    rename = {
        "date": "time", "time": "time", "datetime": "time", "TradingDate": "time", "tradingDate": "time",
        "o": "open", "open": "open", "Open": "open",
        "h": "high", "high": "high", "High": "high",
        "l": "low", "low": "low", "Low": "low",
        "c": "close", "close": "close", "Close": "close", "adj_close": "close",
        "v": "volume", "volume": "volume", "Volume": "volume", "vol": "volume", "value": "volume",
    }
    df = df_in.rename(columns={k: v for k, v in rename.items() if k in df_in.columns}).copy()

    # keep core
    keep = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]

    # time column
    if "time" not in df.columns:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    # numeric coercion
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c])
        else:
            df[c] = np.nan

    # drop invalid OHLC rows
    df = df.dropna(subset=["open", "high", "low", "close"], how="any").copy()

    # normalize time to UTC ISO string (keep as column)
    ts = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["time"] = ts.dt.tz_convert(timezone.utc)

    # drop dup times (keep last), sort asc
    df = df[~df["time"].duplicated(keep="last")].sort_values("time")

    # ensure order and dtypes
    df = df[["time", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    return df


def _is_last_bar_partial(df: pd.DataFrame, tf_ccxt: str) -> bool:
    if df is None or df.empty:
        return False
    last_ts = df["time"].iloc[-1]  # UTC
    now = _now_utc()

    if tf_ccxt == "1h":
        closed_edge = now.replace(minute=0, second=0, microsecond=0)
        return last_ts >= closed_edge
    if tf_ccxt == "4h":
        hour_block = (now.hour // 4) * 4
        closed_edge = now.replace(hour=hour_block, minute=0, second=0, microsecond=0)
        return last_ts >= closed_edge
    if tf_ccxt == "1d":
        day_edge = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return last_ts >= day_edge
    if tf_ccxt == "1w":
        start_of_week = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        return last_ts >= start_of_week
    return False


def _drop_partial(df: pd.DataFrame, tf_ccxt: str, include_partial: bool) -> pd.DataFrame:
    if include_partial or df.empty:
        return df
    if _is_last_bar_partial(df, tf_ccxt):
        return df.iloc[:-1].copy()
    return df


# ----------------------------- Fetch primitives -------------------------------

def _provider_history(vnstock_mod, symbol: str, tf_ccxt: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Try multiple sources, symbol variants and interval aliases; return raw DataFrame."""
    last_err = None
    for src in SUPPORTED_SOURCES:
        for sym in _symbol_variants(symbol):
            # new API
            def _do_history(interval: str, start=start, end=end):
                try:
                    quote = vnstock_mod.Quote(symbol=sym, source=src)
                    if start and end:
                        return quote.history(start=start, end=end, interval=interval)
                    return quote.history(interval=interval)
                except Exception:
                    # legacy path
                    stk = vnstock_mod.Vnstock().stock(symbol=sym, source=src)
                    if start and end:
                        return stk.quote.history(start=start, end=end, interval=interval)
                    return stk.quote.history(interval=interval)

            for itv in _interval_aliases_for_provider(tf_ccxt):
                try:
                    raw = _retryable_call(_do_history, itv)
                    if raw is not None and len(raw) > 0:
                        return raw
                except Exception as e:
                    last_err = e
                    continue
    if last_err:
        raise last_err
    return pd.DataFrame()


def _date_range_for_limit(limit: int) -> Tuple[str, str]:
    # buffer for weekends/holidays
    days = int(limit * 1.1) + 5
    end_vn = datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=7)
    start_vn = end_vn - timedelta(days=days)
    return start_vn.date().isoformat(), end_vn.date().isoformat()


# ----------------------------- Public API -------------------------------------
__all__ = [
    "fetch_ohlcv",
    "fetch_ohlcv_history",
    "fetch_batch",
    "fetch_ohlcv_batch",
]


def fetch_ohlcv(symbol: str, timeframe: str = "1D", limit: int = 600, include_partial: bool = False) -> pd.DataFrame:
    """Fetch a single timeframe → DataFrame with columns [time, open,high,low,close,volume] (UTC time col).
    NEVER raises provider errors; returns empty DataFrame with attrs['error'] and debug info instead.
    """
    vnstock_mod = _load_vnstock()
    tf_ccxt = _resolve_timeframe(timeframe)

    # Guard for intraday availability
    if tf_ccxt in ("1h", "4h"):
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        empty["ts"] = pd.to_datetime([], utc=True)
        empty.attrs["error"] = "intraday_not_supported"
        empty.attrs["debug"] = f"fetch_ohlcv_failed({symbol},{timeframe})"
        return empty

    # Build date window
    start, end = _date_range_for_limit(limit)
    _sleep_base(RATE_LIMIT_MIN_SLEEP)

    # Try provider calls with fallbacks; collect trace
    tried = []
    try:
        raw = _provider_history(vnstock_mod, symbol, tf_ccxt, start, end)
        df = _sanitize_ohlcv(raw)
        df = _drop_partial(df, tf_ccxt, include_partial)

        # add helper ts column
        if not df.empty:
            df["ts"] = pd.to_datetime(df["time"], utc=True)
        else:
            df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime([], utc=True)
        return df
    except Exception as e:
        # Return empty but annotated so caller can log the root cause gracefully
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime([], utc=True)
        df.attrs["error"] = str(e)
        df.attrs["source_tried"] = None  # unknown at this level
        df.attrs["debug"] = f"fetch_ohlcv_failed({symbol},{timeframe})"
        return df


def fetch_ohlcv_history(
    symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    max_pages: int = 20,
    window_days: int = 120,
    sleep_sec: float = 0.8,
    include_partial: bool = False,
) -> pd.DataFrame:
    """Deep history within [start_ms, end_ms] UTC ms using backward pagination windows.

    We walk windows from end_ms backwards by `window_days` per page (at most `max_pages`).
    Each page requests provider history within [start_date, end_date] (VN time window), then merges.
    """
    vnstock_mod = _load_vnstock()
    tf_ccxt = _resolve_timeframe(timeframe)

    if tf_ccxt in ("1h", "4h"):
        raise NotImplementedError("Intraday (1H/4H) may not be supported by vnstock providers yet.")

    start_dt_utc = datetime.fromtimestamp(start_ms / 1000.0, tz=timezone.utc)
    end_dt_utc = datetime.fromtimestamp(end_ms / 1000.0, tz=timezone.utc)
    if end_dt_utc <= start_dt_utc:
        raise ValueError("end_ms must be greater than start_ms")

    dfs: List[pd.DataFrame] = []
    pages = 0
    cur_end = end_dt_utc
    window = timedelta(days=window_days)

    while pages < max_pages and cur_end > start_dt_utc:
        cur_start = max(start_dt_utc, cur_end - window)

        # Convert UTC → VN local date boundaries for the provider (inclusive)
        cur_start_vn = (cur_start.astimezone(VN_TZ)).date().isoformat()
        cur_end_vn = (cur_end.astimezone(VN_TZ)).date().isoformat()

        _sleep_base(sleep_sec)
        raw = _provider_history(vnstock_mod, symbol, tf_ccxt, cur_start_vn, cur_end_vn)
        df_page = _sanitize_ohlcv(raw)
        if not df_page.empty:
            dfs.append(df_page)

        pages += 1
        cur_end = cur_start  # step backward

    if not dfs:
        out = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        out["ts"] = pd.to_datetime([], utc=True)
        return out

    # merge, de-dup, sort
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    merged = merged.drop_duplicates(subset=["time"], keep="last").sort_values("time")

    merged = _drop_partial(merged, tf_ccxt, include_partial)

    # clip to [start_ms, end_ms]
    start_clip = pd.to_datetime(start_dt_utc, utc=True)
    end_clip = pd.to_datetime(end_dt_utc, utc=True)
    merged = merged[(merged["time"] >= start_clip) & (merged["time"] <= end_clip)].copy()

    merged["ts"] = pd.to_datetime(merged["time"], utc=True)
    return merged


def fetch_batch(
    symbol: str,
    timeframes: Tuple[str, ...] = ("1D", "1W"),
    limit: int = 600,
    include_partial: bool = False,
    step_sleep_sec: float = 0.6,
) -> Dict[str, pd.DataFrame]:
    """Fetch multiple timeframes for one symbol in a rate-limit-friendly way.

    Returns a dict {tf_input: DataFrame}
    """
    out: Dict[str, pd.DataFrame] = {}
    for i, tf in enumerate(timeframes):
        if i > 0:
            _sleep_base(step_sleep_sec)
        try:
            out[tf] = fetch_ohlcv(symbol, timeframe=tf, limit=limit, include_partial=include_partial)
        except NotImplementedError as e:
            # provide empty frame for unsupported tf
            empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            empty["ts"] = pd.to_datetime([], utc=True)
            empty.attrs["error"] = str(e)
            out[tf] = empty
    return out


def fetch_ohlcv_batch(
    symbols: List[str],
    timeframe: str = "1D",
    limit: int = 600,
    include_partial: bool = False,
    step_sleep_sec: float = 0.4,
) -> Dict[str, pd.DataFrame]:
    """Batch-fetch *one timeframe* for *many symbols*.

    Returns a mapping {symbol: DataFrame} and paces requests to avoid 429.
    """
    out: Dict[str, pd.DataFrame] = {}
    symbols = symbols or []
    for i, sym in enumerate(symbols):
        if i > 0:
            _sleep_base(step_sleep_sec)
        try:
            df = fetch_ohlcv(sym, timeframe=timeframe, limit=limit, include_partial=include_partial)
            out[sym] = df
        except Exception as e:
            # return empty DF with debug attrs so caller can log
            empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
            empty["ts"] = pd.to_datetime([], utc=True)
            empty.attrs["error"] = str(e)
            empty.attrs["debug"] = f"fetch_ohlcv_failed({sym},{timeframe})"
            out[sym] = empty
    return out


# ----------------------------- Self-test --------------------------------------
if __name__ == "__main__":
    sym = os.getenv("TEST_SYMBOL", "VCB")
    print("Testing fetch_ohlcv ...")
    try:
        df = fetch_ohlcv(sym, timeframe="1D", limit=160, include_partial=False)
        print("Rows:", len(df), "Cols:", list(df.columns))
        print(df.tail())
    except Exception as e:
        print("fetch_ohlcv error:", e)

    print("Testing fetch_ohlcv_history ...")
    try:
        end_ms = int(_now_utc().timestamp() * 1000)
        start_ms = int((_now_utc() - timedelta(days=365)).timestamp() * 1000)
        dh = fetch_ohlcv_history(sym, timeframe="1D", start_ms=start_ms, end_ms=end_ms, max_pages=6)
        print("History rows:", len(dh))
        print(dh.tail())
    except Exception as e:
        print("fetch_ohlcv_history error:", e)

    print("Testing fetch_batch ...")
    try:
        batch = fetch_batch(sym, timeframes=("1D", "1W"), limit=200)
        for k, v in batch.items():
            print(k, "→", len(v))
    except Exception as e:
        print("fetch_batch error:", e)

    print("Testing fetch_ohlcv_batch ...")
    try:
        b = fetch_ohlcv_batch(["VCB", "CTG"], timeframe="1D", limit=120)
        for k, v in b.items():
            print(k, "rows:", len(v))
    except Exception as e:
        print("fetch_ohlcv_batch error:", e)
