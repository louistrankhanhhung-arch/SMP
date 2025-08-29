# -*- coding: utf-8 -*-
"""
vnstock_api.py — Data access layer for Vietnam equities (VN30/HOSE/HNX/UPCoM)
----------------------------------------------------------------------------
- Uses unified `vnstock` package (>=3.2.x).
- Fetch OHLCV for 1D & 1W; includes the *running* candle when include_partial=True.
- Normalized columns: ["ts","open","high","low","close","volume"]
- Tries multiple sources if a provider returns empty/error: ["VCI","TCBS","MSN"]
- Tries symbol variants: "VCB", "VCB.HOSE", "VCB:HOSE"
- Tries interval aliases: 1D→["1D","1d","D","day","daily"], 1W→["1W","1w","W","week","weekly"]
"""
from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd
import importlib
from datetime import datetime, timedelta, timezone

_TZ = "Asia/Ho_Chi_Minh"
_SUPPORTED_SOURCES = ["VCI", "TCBS", "MSN"]
_INTERVAL_ALIASES = {
    "1D": ["1D", "1d", "D", "day", "daily"],
    "1W": ["1W", "1w", "W", "week", "weekly"],
}

def _now_vn() -> datetime:
    # naive local time for VN; we avoid pytz dependency
    # Asia/Ho_Chi_Minh is UTC+7 without DST
    return datetime.utcnow().replace(tzinfo=timezone.utc) + timedelta(hours=7)

def _check_provider() -> str:
    try:
        vnstock = importlib.import_module("vnstock")
        return getattr(vnstock, "__version__", "unknown")
    except Exception as e:
        raise RuntimeError(f"vnstock import failed: {e}")

def _to_date(limit: int) -> Tuple[str, str]:
    now = _now_vn().date()
    # buffer 10% to be safe for weekends/holidays
    days = int(limit * 1.1) + 5
    start = now - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d")

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["ts","open","high","low","close","volume"])
    rename_map = {
        "time": "ts", "date": "ts", "TradingDate": "ts", "tradingDate": "ts",
        "open": "open", "Open": "open", "o": "open",
        "high": "high", "High": "high", "h": "high",
        "low": "low", "Low": "low", "l": "low",
        "close": "close", "Close": "close", "c": "close",
        "volume": "volume", "Volume": "volume", "v": "volume", "value": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    keep = [c for c in ["ts","open","high","low","close","volume"] if c in df.columns]
    df = df[keep].copy()
    # Normalize timestamp to pandas datetime (naive, VN local)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    # Coerce numerics
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows without ts
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    return df

def _symbol_variants(sym: str) -> List[str]:
    # Some providers accept exchange suffix; we try a few common variants
    base = sym.strip().upper()
    return [base, f"{base}.HOSE", f"{base}:HOSE", f"{base}.HSX", f"{base}:HSX"]

def _history_with_fallbacks(obj, start: str, end: str, tf: str):
    """
    Try multiple interval aliases; if still empty, try again without date range.
    Returns (df, meta) where meta describes provider/interval used.
    """
    tried = []
    for itv in _INTERVAL_ALIASES.get(tf, [tf]):
        try:
            raw = obj.history(start=start, end=end, interval=itv)
            tried.append(itv)
            df_tmp = _normalize_df(raw)
            if len(df_tmp) > 0:
                return df_tmp, {"interval_used": itv, "range": "start-end"}
        except Exception as e:
            tried.append(f"{itv}:err={e}")
            continue
    # try without dates (provider default range)
    for itv in _INTERVAL_ALIASES.get(tf, [tf]):
        try:
            raw = obj.history(interval=itv)
            tried.append(f"{itv}(nodates)")
            df_tmp = _normalize_df(raw)
            if len(df_tmp) > 0:
                return df_tmp, {"interval_used": itv, "range": "nodates"}
        except Exception as e:
            tried.append(f"{itv}(nodates):err={e}")
            continue
    return pd.DataFrame(columns=["ts","open","high","low","close","volume"]), {"tried": tried}

def _drop_running_bar_if_needed(df: pd.DataFrame, timeframe: str, include_partial: bool) -> pd.DataFrame:
    if include_partial or len(df) == 0:
        return df
    # For 1D: drop bar if last ts is "today" VN time (market not closed/committed yet)
    last_ts = pd.to_datetime(df["ts"].iloc[-1])
    today_vn = _now_vn().date()
    if timeframe.upper().startswith("1D") and last_ts.date() >= today_vn:
        return df.iloc[:-1].copy()
    return df

def fetch_ohlcv(symbol: str, timeframe: str = "1D", limit: int = 600, include_partial: bool = True) -> pd.DataFrame:
    """
    Fetch a single symbol with fallbacks across sources, symbol variants and interval aliases.
    Returns normalized DataFrame with attrs:
        - source_used
        - interval_used
        - range  ("start-end" | "nodates")
        - error / source_tried / debug when empty
    """
    vnstock = importlib.import_module("vnstock")
    start, end = _to_date(limit)
    tf = timeframe
    last_err = None
    df = None
    used_src = None
    meta_used = {}

    for src_name in _SUPPORTED_SOURCES:
        for sym_variant in _symbol_variants(symbol):
            try:
                # New-style accessor
                try:
                    quote = vnstock.Quote(symbol=sym_variant, source=src_name)
                    df_tmp, meta = _history_with_fallbacks(quote, start, end, tf)
                except Exception:
                    # Legacy path
                    stk = vnstock.Vnstock().stock(symbol=sym_variant, source=src_name)
                    df_tmp, meta = _history_with_fallbacks(stk.quote, start, end, tf)
                if len(df_tmp) > 0:
                    df = df_tmp
                    used_src = src_name
                    meta_used = meta
                    break
            except Exception as e:
                last_err = str(e)
                continue
        if df is not None and len(df) > 0:
            break

    if df is None:
        df = pd.DataFrame(columns=["ts","open","high","low","close","volume"])
        if last_err:
            df.attrs["error"] = last_err
        df.attrs["source_tried"] = ",".join(_SUPPORTED_SOURCES)
        df.attrs["debug"] = "no_rows_after_all_fallbacks"
        return df

    # Optionally drop the running bar
    df = _drop_running_bar_if_needed(df, timeframe, include_partial)

    # Attach metadata
    if used_src:
        df.attrs["source_used"] = used_src
    if meta_used:
        for k, v in meta_used.items():
            df.attrs[k] = v
    return df

def fetch_ohlcv_batch(symbols: List[str], timeframe: str = "1D", limit: int = 600, include_partial: bool = True) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        out[sym] = fetch_ohlcv(sym, timeframe=timeframe, limit=limit, include_partial=include_partial)
    return out

# Self-test
if __name__ == "__main__":
    try:
        print("vnstock version:", _check_provider())
        test_df = fetch_ohlcv("VCB", timeframe="1D", limit=160, include_partial=True)
        print("Fetched rows:", len(test_df), "| source_used:", test_df.attrs.get("source_used"),
              "| interval:", test_df.attrs.get("interval_used"), "| range:", test_df.attrs.get("range"))
        print(test_df.tail())
    except Exception as e:
        print("Self-test failed:", e)
