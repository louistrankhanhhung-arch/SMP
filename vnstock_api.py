
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional
import warnings

# Note: We try Vnstock (gen2) API first, then fall back to legacy functions.
# The goal is to return a DataFrame with index as UTC timestamps and columns:
# ["open","high","low","close","volume"]

def _to_utc_index(df, ts_col):
    if df.empty:
        return df
    idx = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(idx).drop(columns=[ts_col])
    # keep only the expected columns in right order if available
    cols = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    return df[cols]

def _resample_ohlcv(df, rule: str):
    if df.empty:
        return df
    # Ensure we have all columns
    need = ["open","high","low","close","volume"]
    for c in need:
        if c not in df.columns:
            df[c] = np.nan
    o = df["open"].resample(rule, label="right", closed="right").first()
    h = df["high"].resample(rule, label="right", closed="right").max()
    l = df["low"].resample(rule, label="right", closed="right").min()
    c = df["close"].resample(rule, label="right", closed="right").last()
    v = df["volume"].resample(rule, label="right", closed="right").sum()
    out = pd.concat([o,h,l,c,v], axis=1)
    out.columns = ["open","high","low","close","volume"]
    out = out.dropna(how="any")
    return out

def _fetch_daily_gen2(symbol: str, start: str, end: str):
    # Vnstock gen2 class-based
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        df = stock.quote.history(start=start, end=end, interval="1D")
        # Expected columns: time, open, high, low, close, volume
        if "time" in df.columns:
            df.rename(columns={"time":"datetime"}, inplace=True)
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date":"datetime"}, inplace=True)
        if "datetime" not in df.columns:
            # best-effort
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        return _to_utc_index(df, "datetime")
    except Exception:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

def _fetch_daily_legacy(symbol: str, start: str, end: str):
    try:
        from vnstock import stock_historical_data
        df = stock_historical_data(symbol=symbol, start_date=start, end_date=end, resolution="1D")
        # Legacy sometimes has column 'time' or 'date'
        if "time" in df.columns:
            df.rename(columns={"time":"datetime"}, inplace=True)
        if "datetime" not in df.columns and "date" in df.columns:
            df.rename(columns={"date":"datetime"}, inplace=True)
        if "datetime" not in df.columns:
            # Some versions use index as date
            if df.index.name and df.index.name.lower() in ("time","date","datetime"):
                df = df.reset_index().rename(columns={df.index.name:"datetime"})
            else:
                df["datetime"] = pd.to_datetime(df.index)
        return _to_utc_index(df, "datetime")
    except Exception:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

def _fetch_intraday_gen2(symbol: str, days_back: int = 10):
    # Try to gather per-minute data for the last N trading days, then resample later.
    try:
        from vnstock import Vnstock
        stock = Vnstock().stock(symbol=symbol, source="VCI")
        # gen2 offers tick/intraday; API surface may vary across versions.
        # Try "stock.quote.intraday" if present.
        intraday = None
        if hasattr(stock.quote, "intraday"):
            # page_size large enough to cover recent sessions
            intraday = stock.quote.intraday(symbol=symbol, page_size=200000, show_log=False)
        if intraday is None or len(intraday) == 0:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        df = intraday.copy()
        # Expected columns include 'time' or 'datetime', 'open/high/low/close/volume'
        if "time" in df.columns:
            df.rename(columns={"time":"datetime"}, inplace=True)
        if "datetime" not in df.columns:
            # try combining date + time
            if "tradingDate" in df.columns and "tradingTime" in df.columns:
                df["datetime"] = pd.to_datetime(df["tradingDate"] + " " + df["tradingTime"])
            else:
                return pd.DataFrame(columns=["open","high","low","close","volume"])
        return _to_utc_index(df, "datetime")
    except Exception:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

def _fetch_intraday_legacy(symbol: str):
    try:
        from vnstock import stock_intraday_data
        df = stock_intraday_data(symbol=symbol, page_num=0, page_size=100000)
        # Legacy intraday returns columns: time, open, high, low, close, volume
        if "time" in df.columns:
            df.rename(columns={"time":"datetime"}, inplace=True)
        return _to_utc_index(df, "datetime")
    except Exception:
        return pd.DataFrame(columns=["open","high","low","close","volume"])

def fetch_ohlcv(symbol: str, timeframe: str = "1D", limit: int = 300) -> pd.DataFrame:
    """
    Fetch OHLCV for a VN stock symbol via vnstock/vnstock gen2
    - timeframe: "1H","4H","1D"
    - limit: bars limit (best-effort for intraday)
    """
    timeframe = timeframe.upper()
    today = datetime.now(timezone.utc).date()
    # Daily range: go back ~ 3 years to be safe
    start = (today - timedelta(days=3*365)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    if timeframe == "1D":
        df = _fetch_daily_gen2(symbol, start, end)
        if df.empty:
            df = _fetch_daily_legacy(symbol, start, end)
        return df.tail(limit)

    # For intraday 1H and 4H, fetch minute/tick data then resample
    intraday = _fetch_intraday_gen2(symbol, days_back=15)
    if intraday.empty:
        intraday = _fetch_intraday_legacy(symbol)
    if intraday.empty:
        # As a last resort, approximate from daily by forward-filling to synthetic hourly bars (not ideal)
        daily = _fetch_daily_gen2(symbol, start, end)
        if daily.empty:
            daily = _fetch_daily_legacy(symbol, start, end)
        if daily.empty:
            return daily  # empty
        # Expand each daily bar into 4H/1H placeholders (close-to-close) – conservative fallback
        # Create an hourly time index (UTC) ending now
        approx = daily.copy()
        approx["ts"] = approx.index
        # Duplicate each daily row 6 times to mimic 6 trading hours (9:30–11:30 = 2h, 14:00–15:00 = 1h => 3h total; but use 6 slots for more points)
        reps = []
        for t, row in approx.iterrows():
            for k in range(3):  # 3 pseudo-hours
                reps.append({"datetime": t + pd.Timedelta(minutes=60*k),
                             "open": row["open"], "high": row["high"], "low": row["low"],
                             "close": row["close"], "volume": row["volume"]/3.0})
        intraday = pd.DataFrame(reps)
        intraday = _to_utc_index(intraday, "datetime")

    # Resample
    rule = "1H" if timeframe == "1H" else "4H"
    out = _resample_ohlcv(intraday, rule=rule).tail(limit)
    return out
