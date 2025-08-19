
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Literal

# =============================
# VNStock OHLCV fetcher (1H-1D-1W)
# - Index: UTC timestamps
# - Columns: ["open","high","low","close","volume"]
# - Weekly rule: W-FRI (đóng tuần vào thứ Sáu)
# =============================

_DEFAULT_LOCAL_TZ = "Asia/Ho_Chi_Minh"

def _to_utc_index(df: pd.DataFrame, ts_col: str, assume_local_tz: str = _DEFAULT_LOCAL_TZ) -> pd.DataFrame:
    """Ensure df index is UTC datetime and keep only OHLCV columns in correct order."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    # Parse time column
    ts = pd.to_datetime(df[ts_col], errors="coerce")
    # If tz-naive, assume local exchange time then convert to UTC
    try:
        is_tz_aware = ts.dt.tz is not None
    except Exception:
        is_tz_aware = False

    if not is_tz_aware:
        ts = ts.dt.tz_localize(assume_local_tz, nonexistent="NaT", ambiguous="NaT").dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    df = df.copy()
    df.index = ts
    if ts_col in df.columns:
        df.drop(columns=[ts_col], inplace=True, errors="ignore")

    # Keep only expected columns, preserve order
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    out = df[cols].copy()
    # Standardize types
    for c in ["open","high","low","close","volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(how="any")
    out = out.sort_index()
    return out

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV with standard aggregations. df index must be tz-aware UTC."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    need = ["open","high","low","close","volume"]
    base = df.copy()
    for c in need:
        if c not in base.columns:
            base[c] = np.nan

    o = base["open"].resample(rule, label="right", closed="right").first()
    h = base["high"].resample(rule, label="right", closed="right").max()
    l = base["low"].resample(rule, label="right", closed="right").min()
    c = base["close"].resample(rule, label="right", closed="right").last()
    v = base["volume"].resample(rule, label="right", closed="right").sum()

    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ["open","high","low","close","volume"]
    out = out.dropna(how="any").sort_index()
    return out

def _fetch_daily_gen2(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Daily via vnstock gen2."""
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        df = stock.quote.history(start=start, end=end, interval="1D")
        # Expected columns: time/open/high/low/close/volume  (sometimes 'date' or 'datetime')
        if "time" in df.columns:
            df = df.rename(columns={"time": "datetime"})
        if "datetime" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        if "datetime" not in df.columns:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        return _to_utc_index(df, "datetime")
    except Exception:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

def _fetch_daily_legacy(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Daily via vnstock legacy."""
    try:
        from vnstock import stock_historical_data
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution="1D")
        if "time" in df.columns:
            df = df.rename(columns={"time": "datetime"})
        if "datetime" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        if "datetime" not in df.columns:
            # Some versions put time in index
            if df.index.name and df.index.name.lower() in ("time","date","datetime"):
                df = df.reset_index().rename(columns={df.index.name:"datetime"})
            else:
                df["datetime"] = pd.to_datetime(df.index)
        return _to_utc_index(df, "datetime")
    except Exception:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

def _fetch_intraday_gen2(symbol: str) -> pd.DataFrame:
    """Intraday/tick via vnstock gen2, then normalize to minute-level if needed."""
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        intraday = None
        if hasattr(stock.quote, "intraday"):
            intraday = stock.quote.intraday(symbol=symbol, page_size=200000, show_log=False)
        if intraday is None or len(intraday) == 0:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        df = intraday.copy()
        if "time" in df.columns:
            df = df.rename(columns={"time": "datetime"})
        if "datetime" not in df.columns:
            if "tradingDate" in df.columns and "tradingTime" in df.columns:
                df["datetime"] = pd.to_datetime(df["tradingDate"] + " " + df["tradingTime"], errors="coerce")
            else:
                return pd.DataFrame(columns=["open","high","low","close","volume"])
        return _to_utc_index(df, "datetime")
    except Exception:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

def _fetch_intraday_legacy(symbol: str) -> pd.DataFrame:
    """Intraday via vnstock legacy API."""
    try:
        from vnstock import stock_intraday_data
        df = stock_intraday_data(symbol=symbol, page_num=0, page_size=100000)
        if "time" in df.columns:
            df = df.rename(columns={"time":"datetime"})
        return _to_utc_index(df, "datetime")
    except Exception:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

def _approx_intraday_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Create coarse intraday approximation (3 pseudo-hour bars) from daily OHLCV.
    Note: This is a last resort for when intraday is unavailable.
    """
    if daily is None or len(daily) == 0:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

    reps = []
    for t, row in daily.iterrows():
        # Three pseudo-hours: 09:30, 10:30, 14:00 (local, but we keep relative spacing in UTC)
        # Split volume roughly equally
        for k in range(3):
            reps.append({
                "datetime": pd.Timestamp(t) + pd.Timedelta(hours=k),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) / 3.0,
            })
    tmp = pd.DataFrame(reps)
    return _to_utc_index(tmp, "datetime")

def fetch_ohlcv(symbol: str, timeframe: Literal["1H","1D","1W"] = "1D", limit: int = 300) -> pd.DataFrame:
    """Public entry: fetch OHLCV for VN stocks with 1H / 1D / 1W support (W-FRI weekly)."""
    timeframe = timeframe.upper()
    if timeframe not in ("1H","1D","1W"):
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of: 1H, 1D, 1W.")

    today = datetime.now(timezone.utc).date()
    start = (today - timedelta(days=3*365)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    # ----- Daily path -----
    if timeframe == "1D":
        df = _fetch_daily_gen2(symbol, start, end)
        if len(df) == 0:
            df = _fetch_daily_legacy(symbol, start, end)
        return df.tail(limit)

    # ----- Weekly path (resample from daily with W-FRI) -----
    if timeframe == "1W":
        daily = _fetch_daily_gen2(symbol, start, end)
        if len(daily) == 0:
            daily = _fetch_daily_legacy(symbol, start, end)
        if len(daily) == 0:
            return daily
        o = daily["open"].resample("W-FRI", label="right", closed="right").first()
        h = daily["high"].resample("W-FRI", label="right", closed="right").max()
        l = daily["low"].resample("W-FRI", label="right", closed="right").min()
        c = daily["close"].resample("W-FRI", label="right", closed="right").last()
        v = daily["volume"].resample("W-FRI", label="right", closed="right").sum()
        out = pd.concat([o,h,l,c,v], axis=1)
        out.columns = ["open","high","low","close","volume"]
        out = out.dropna(how="any").sort_index()
        return out.tail(limit)

    # ----- 1H path: intraday → resample 1H → fallback from daily -----
    intraday = _fetch_intraday_gen2(symbol)
    if len(intraday) == 0:
        intraday = _fetch_intraday_legacy(symbol)
    if len(intraday) == 0:
        # fallback approximate from daily
        daily = _fetch_daily_gen2(symbol, start, end)
        if len(daily) == 0:
            daily = _fetch_daily_legacy(symbol, start, end)
        if len(daily) == 0:
            return daily
        intraday = _approx_intraday_from_daily(daily)

    out = _resample_ohlcv(intraday, rule="1H").tail(limit)
    return out
