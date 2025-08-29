"""
vnstock_api.py — Data access layer for Vietnam equities (VN30/HOSE/HNX/UPCoM)
------------------------------------------------------------------------------
- Uses unified `vnstock` package (>=3.2.x).
- Fetch OHLCV for 1D & 1W; includes the *running* candle when include_partial=True.
- Normalized columns: ["ts","open","high","low","close","volume"]
- Tries multiple sources if a provider returns empty/error: ["VCI","SSI","MBS","TVSI","HSC"]
"""
from __future__ import annotations
from typing import List, Dict
import pandas as pd
import importlib

_TZ = "Asia/Ho_Chi_Minh"
_INTERVAL_ALIASES = {
    "1D": ["1D", "1d", "D", "day", "daily"],
    "1W": ["1W", "1w", "W", "week", "weekly"],
}

def _check_provider() -> str:
    try:
        mod = importlib.import_module("vnstock")
        ver = getattr(mod, "__version__", "0")
        return str(ver)
    except Exception as e:
        raise RuntimeError("Missing dependency 'vnstock'. Please install: pip install -U vnstock") from e

def _now_vn() -> pd.Timestamp:
    return pd.Timestamp.now(tz=_TZ)

def _infer_start_date(limit: int, timeframe: str) -> str:
    now = _now_vn()
    if timeframe.upper() == "1D":
        start = now - pd.Timedelta(days=int(limit * 1.6))
    elif timeframe.upper() == "1W":
        start = now - pd.Timedelta(weeks=int(limit * 1.6))
    else:
        raise ValueError("Only 1D and 1W supported")
    return start.strftime("%Y-%m-%d")

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
    if "ts" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
            df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df["ts"] = df["ts"].dt.tz_convert(_TZ)
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

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

def fetch_ohlcv(symbol: str, timeframe: str = "1D", limit: int = 600, include_partial: bool = True) -> pd.DataFrame:
    """Fetch OHLCV for a single symbol using `vnstock` Quote.history.
    timeframe in {"1D","1W"}; tries multiple sources if needed.
    """
    ver = _check_provider()
    from vnstock import Vnstock, Quote  # type: ignore

    tf = timeframe.upper()
    if tf not in {"1D","1W"}:
        raise ValueError("Only 1D and 1W are supported by vnstock_api.fetch_ohlcv")

    start = _infer_start_date(limit=limit, timeframe=tf)
    end = _now_vn().strftime("%Y-%m-%d")

    sources = ["VCI", "SSI", "MBS", "TVSI", "HSC"]
    last_err = None
    df = None
    used = None
    meta_used = {}
    for src_name in sources:
        try:
            try:
                quote = Quote(symbol=symbol, source=src_name)
                df_tmp, meta = _history_with_fallbacks(quote, start, end, tf)
            except Exception:
                stock = Vnstock().stock(symbol=symbol, source=src_name)
                df_tmp, meta = _history_with_fallbacks(stock.quote, start, end, tf)
            if len(df_tmp) > 0:
                df = df_tmp
                used = src_name
                meta_used = meta
                break
        except Exception as e:
            last_err = str(e)
            continue

    if df is None:
        df = pd.DataFrame(columns=["ts","open","high","low","close","volume"])
        if last_err:
            df.attrs["error"] = last_err
        df.attrs["source_tried"] = ",".join(sources)
        df.attrs["debug"] = "no_rows_after_all_fallbacks"
        return df

    if not include_partial and len(df) > 0:
        last = df.iloc[-1]["ts"]
        if tf == "1D":
            is_incomplete = last.date() == _now_vn().date()
        else:
            now = _now_vn()
            is_incomplete = (last.isocalendar().week == now.isocalendar().week) and (last.year == now.year)
        if is_incomplete:
            df = df.iloc[:-1].copy()

    if limit and len(df) > limit:
        df = df.iloc[-limit:].reset_index(drop=True)

    if used:
        df.attrs["source_used"] = used
    if meta_used:
        for k, v in meta_used.items():
            df.attrs[k] = v
    return df

def fetch_ohlcv_batch(symbols: List[str], timeframe: str = "1D", limit: int = 600, include_partial: bool = True) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            out[sym] = fetch_ohlcv(sym, timeframe=timeframe, limit=limit, include_partial=include_partial)
        except Exception as e:
            df = pd.DataFrame(columns=["ts","open","high","low","close","volume"])
            df.attrs["error"] = str(e)
            out[sym] = df
    return out

def latest_bar(symbol: str, timeframe: str = "1D", include_partial: bool = True):
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
    try:
        print("vnstock version:", _check_provider())
        test_df = fetch_ohlcv("VCB", timeframe="1D", limit=100, include_partial=True)
        print("Fetched rows:", len(test_df), "| source_used:", test_df.attrs.get("source_used"))
        print(test_df.tail())
    except Exception as e:
        print("Self-test failed:", e)
