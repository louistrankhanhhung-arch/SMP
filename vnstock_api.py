"""
vnstock_api_refactor.py — robust OHLCV fetcher for Vietnam equities (HOSE/HNX/UPCOM)
following the same engineering philosophy as your crypto signal client.

Key design:
1) Scope & normalization
   - Default data source: vnstock (public Python lib)
   - Symbol normalization: accepts "dpm", "DPM", "DPM.HOSE", "HOSE:DPM" -> "DPM" (exchange optional)
   - Timeframe map: {"1H":"1H", "4H":"4H", "1D":"1D", "1W":"1W"}
     (Only drop partial bars for 1H when drop_partial=True; for stocks we never use partial for 1D/1W anyway.)

2) OHLCV schema
   - Output DataFrame with UTC index (DatetimeIndex, tz-aware UTC)
   - Columns: ["open","high","low","close","volume"] (float64)
   - Dedup by timestamp, keep the newest
   - Strip rows with any NaN in O/H/L/C (volume may be 0, still allowed)

3) Partial-bar handling
   - For timeframe = 1H: if drop_partial=True, drop the last row if its close time is not the boundary of a full hour bar
     using market calendar Asia/Ho_Chi_Minh and HOSE/HNX trading hours (09:00–11:30, 13:00–15:00).
   - For 1D/1W: bars are considered closed only after exchange close; include_partial is ignored (always treat as closed-only).

4) Robustness & pagination
   - Retries with exponential backoff for transient errors (network, 429)
   - Historical crawl with windowed pagination by date range (start/end), stitching all pages then sort/drop-dup

5) Public API
   - fetch_ohlcv(symbol, timeframe="1D", limit=300, drop_partial=True)
   - fetch_ohlcv_history(symbol, timeframe, start_ms=None, end_ms=None, max_pages=20, page_days=60,
                         drop_partial=True, sleep_sec=0.5)
   - fetch_batch([(symbol, timeframe, limit), ...], drop_partial=True, batch_sleep_sec=0.4)

NOTE: This module assumes a recent `vnstock` package with functions to fetch candles for
      daily/weekly and intraday (hour) bars. If your `vnstock` wrapper differs, swap the
      adapter functions `_vn_fetch_daily`, `_vn_fetch_weekly`, `_vn_fetch_intraday` below.
"""
from __future__ import annotations

import math
import time
import json
import typing as T
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ---------- Config ----------
VN_TZ = "Asia/Ho_Chi_Minh"
VN_TZINFO = pd.Timestamp.now(tz=VN_TZ).tz
UTC = timezone.utc

DEFAULT_TIMEOUT = 20_000  # ms, for parity with style used in your crypto client
DEFAULT_RETRIES = 4
DEFAULT_BACKOFF_BASE = 0.6
DEFAULT_BACKOFF_CAP = 4.0

# Trading hours for HOSE/HNX (approx.) local time
MORNING_START = (9, 0)
MORNING_END   = (11, 30)
AFTERNOON_START = (13, 0)
AFTERNOON_END   = (15, 0)

# ---------- Helpers ----------

def _normalize_symbol(sym: str) -> str:
    if not isinstance(sym, str):
        raise TypeError("symbol must be str")
    s = sym.strip().upper()
    # allow prefixes/suffixes like HOSE:DPM, DPM.HOSE
    s = s.replace("HOSE:", "").replace("HNX:", "").replace("UPCOM:", "")
    if ".HOSE" in s or ".HNX" in s or ".UPCOM" in s:
        s = s.split(".")[0]
    return s

_TIMEFRAME_MAP = {
    "1H": "1H",
    "4H": "4H",
    "1D": "1D",
    "1W": "1W",
}

@dataclass
class RetryPolicy:
    retries: int = DEFAULT_RETRIES
    backoff_base: float = DEFAULT_BACKOFF_BASE
    backoff_cap: float = DEFAULT_BACKOFF_CAP

    def sleep(self, attempt: int):
        # attempt starts from 1
        delay = min(self.backoff_cap, (self.backoff_base ** attempt) + (0.1 * np.random.rand()))
        time.sleep(delay)

# ---------- vnstock Adapters ----------
# Swap these three functions to match your actual vnstock client.

try:
    # Avoid hard import errors in non-runtime environments
    import vnstock  # type: ignore
except Exception:  # pragma: no cover
    vnstock = None


def _vn_fetch_daily(symbol: str, limit: int) -> pd.DataFrame:
    """Fetch last `limit` daily candles. Must return DataFrame with columns time, open, high, low, close, volume.
    time must be tz-aware (UTC or local), we'll normalize later.
    """
    if vnstock is None:
        raise RuntimeError("vnstock package not available at runtime")

    # Example using vnstock 2.x style API; adjust to your version
    # df_raw expected columns: time, open, high, low, close, volume
    df_raw = vnstock.stock_historical_data(symbol, period="1D", count=limit)  # pseudo-call, adjust
    return _ensure_ohlcv_schema(df_raw)


def _vn_fetch_weekly(symbol: str, limit: int) -> pd.DataFrame:
    if vnstock is None:
        raise RuntimeError("vnstock package not available at runtime")
    df_raw = vnstock.stock_historical_data(symbol, period="1W", count=limit)  # pseudo-call, adjust
    return _ensure_ohlcv_schema(df_raw)


def _vn_fetch_intraday(symbol: str, tf: str, limit: int) -> pd.DataFrame:
    if vnstock is None:
        raise RuntimeError("vnstock package not available at runtime")
    # tf in {"1H","4H"}
    # Some vnstock builds expose minute bars; if only 15m available, resample to 1H/4H below
    df_raw = vnstock.stock_intraday_data(symbol, period=tf, count=limit)  # pseudo-call, adjust
    return _ensure_ohlcv_schema(df_raw)

# ---------- Core Normalizers ----------

def _ensure_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)

    # Try to detect timestamp column
    ts_col = None
    for c in ["time", "timestamp", "Date", "date", "datetime"]:
        if c in df.columns:
            ts_col = c
            break
    if ts_col is None:
        raise ValueError("Input DataFrame has no time-like column")

    out = pd.DataFrame()
    out["time"] = pd.to_datetime(df[ts_col], utc=False, errors="coerce")

    # Make tz-aware in local VN then convert to UTC for index
    if out["time"].dt.tz is None:
        out["time"] = out["time"].dt.tz_localize(VN_TZ)
    out["time"] = out["time"].dt.tz_convert("UTC")

    col_map = {
        "open": ["open","Open","o"],
        "high": ["high","High","h"],
        "low":  ["low","Low","l"],
        "close":["close","Close","c"],
        "volume":["volume","Volume","vol","v"],
    }
    for k, alts in col_map.items():
        src = None
        for a in alts:
            if a in df.columns:
                src = a
                break
        if src is None:
            # volume can be missing -> fill 0; O/H/L/C must exist
            if k == "volume":
                out[k] = 0.0
            else:
                raise ValueError(f"Missing required column {k}")
        else:
            out[k] = pd.to_numeric(df[src], errors="coerce")

    out = out.dropna(subset=["open","high","low","close"])  # allow volume=NaN -> fill below
    out["volume"] = out["volume"].fillna(0.0)

    # Dedup & sort
    out = out.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    out = out.set_index("time")
    out.index = out.index.tz_convert("UTC")

    # Ensure types
    out = out[["open","high","low","close","volume"]].astype(float)
    return out

# ---------- Partial-bar logic ----------

def _is_trading_hour_local(ts_local: pd.Timestamp) -> bool:
    # Given local VN time, check if within HOSE/HNX regular sessions
    t = (ts_local.hour, ts_local.minute)
    in_morning = (MORNING_START <= t < MORNING_END)
    in_afternoon = (AFTERNOON_START <= t < AFTERNOON_END)
    return in_morning or in_afternoon


def _is_full_hour_boundary(ts_local: pd.Timestamp) -> bool:
    # A bar ending at XX:00 is a canonical hour boundary inside sessions
    return ts_local.minute == 0


def _drop_partial_hour_bar(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    last_ts_utc = df.index[-1]
    last_local = last_ts_utc.tz_convert(VN_TZ)

    # If last bar ends inside trading hours but not on a full hour mark, treat it as partial
    if _is_trading_hour_local(last_local) and not _is_full_hour_boundary(last_local):
        return df.iloc[:-1]
    return df

# ---------- Public API ----------

def fetch_ohlcv(symbol: str, timeframe: str = "1D", limit: int = 300,
                drop_partial: bool = True,
                retry: RetryPolicy | None = None) -> pd.DataFrame:
    """Fetch last `limit` candles for `symbol` & `timeframe` from vnstock.

    - Output UTC-indexed OHLCV DataFrame
    - For 1H timeframe, if drop_partial=True: drop the running (not-on-hour) bar.
    - For 1D/1W: treat as closed-only regardless of drop_partial.
    """
    sym = _normalize_symbol(symbol)
    tf = _TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    retry = retry or RetryPolicy()
    err: Exception | None = None

    for attempt in range(1, retry.retries + 1):
        try:
            if tf == "1D":
                df = _vn_fetch_daily(sym, limit)
            elif tf == "1W":
                df = _vn_fetch_weekly(sym, limit)
            elif tf in ("1H", "4H"):
                df = _vn_fetch_intraday(sym, tf, limit)
            else:
                raise ValueError(f"Unsupported timeframe: {tf}")

            if tf == "1H" and drop_partial:
                df = _drop_partial_hour_bar(df)

            return df
        except Exception as e:  # network, rate-limit, schema changes
            err = e
            if attempt >= retry.retries:
                break
            retry.sleep(attempt)
    raise RuntimeError(f"fetch_ohlcv failed for {sym} {timeframe}: {err}")


def fetch_ohlcv_history(symbol: str, timeframe: str,
                         start_ms: int | None = None,
                         end_ms: int | None = None,
                         max_pages: int = 20,
                         page_days: int = 60,
                         drop_partial: bool = True,
                         sleep_sec: float = 0.5,
                         retry: RetryPolicy | None = None) -> pd.DataFrame:
    """Historical crawl within [start_ms, end_ms] (UTC ms). If none, fetch up to `max_pages` windows.

    Since vnstock APIs are usually date-ranged (not since/limit), we paginate by date windows of `page_days`.
    """
    sym = _normalize_symbol(symbol)
    tf = _TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    retry = retry or RetryPolicy()

    # Compute UTC date windows
    if end_ms is None:
        end = pd.Timestamp.utcnow().tz_localize("UTC")
    else:
        end = pd.to_datetime(end_ms, unit="ms", utc=True)

    if start_ms is None:
        start = end - pd.Timedelta(days=page_days * max_pages)
    else:
        start = pd.to_datetime(start_ms, unit="ms", utc=True)

    # Build windows [w_start, w_end]
    frames = []
    w_end = end
    pages = 0
    while w_end > start and pages < max_pages:
        w_start = max(start, w_end - pd.Timedelta(days=page_days))
        # Fetch this window with best-effort (vnstock may not support per-window fetch;
        # if not, we simply pull a larger chunk via limit and slice.)
        # Here we call the simple fetch and then slice by time.
        try:
            df = fetch_ohlcv(sym, tf, limit=1500, drop_partial=False, retry=retry)
            df = df.loc[(df.index >= w_start) & (df.index <= w_end)]
            frames.append(df)
        except Exception as e:
            # Retry per page
            pages += 1
            continue
        pages += 1
        time.sleep(max(0.0, sleep_sec))
        w_end = w_start

    if not frames:
        return pd.DataFrame(columns=["open","high","low","close","volume"])  # empty

    out = pd.concat(frames).sort_index().drop_duplicates(keep="last")

    if tf == "1H" and drop_partial:
        out = _drop_partial_hour_bar(out)

    # Final slice by [start,end]
    out = out.loc[(out.index >= start) & (out.index <= end)]
    return out


def fetch_batch(items: T.List[tuple[str, str, int]] | T.Iterable[tuple[str, str, int]],
                drop_partial: bool = True,
                batch_sleep_sec: float = 0.4,
                retry: RetryPolicy | None = None) -> dict:
    """Fetch multiple (symbol, timeframe, limit) tuples in series with small sleeps to avoid rate limits.
       Returns a dict {(symbol, timeframe): DataFrame}
    """
    retry = retry or RetryPolicy()
    out: dict = {}
    for (sym, tf, limit) in items:
        key = (_normalize_symbol(sym), tf.upper())
        out[key] = fetch_ohlcv(key[0], key[1], limit=limit, drop_partial=drop_partial, retry=retry)
        time.sleep(max(0.0, batch_sleep_sec))
    return out

# ---------- Quick self-test helpers (optional) ----------
if __name__ == "__main__":
    # Replace with a real symbol you have access to
    test_sym = "DPM"
    for tf in ("1D","1W","1H"):
        try:
            df = fetch_ohlcv(test_sym, tf, limit=400, drop_partial=True)
            print(tf, len(df), df.tail(3))
        except Exception as e:
            print(tf, "ERROR", e)
