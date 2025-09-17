
# indicators.py — technical indicators for OHLCV DataFrames
# Expected columns: ["ts","open","high","low","close","volume"]
# Output: original cols + common indicators with consistent names.
from __future__ import annotations
import pandas as pd
import numpy as np

# =========================
# Utilities
# =========================
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _safe_div(a, b):
    """Elementwise safe division that preserves pandas Series type when possible."""
    a = pd.to_numeric(a, errors="coerce") if not isinstance(a, pd.Series) else pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce") if not isinstance(b, pd.Series) else pd.to_numeric(b, errors="coerce")
    try:
        res = a.divide(b)
        # Replace inf/-inf where b == 0
        if isinstance(res, pd.Series):
            res = res.replace([np.inf, -np.inf], np.nan)
        return res
    except Exception:
        with np.errstate(divide='ignore', invalid='ignore'):
            res = np.divide(a, b)
        res = np.where(np.isfinite(res), res, np.nan)
        return res

def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    # Ưu tiên 'ts', fallback sang 'time'
    col = "ts" if "ts" in df.columns else ("time" if "time" in df.columns else None)
    if col:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        df = df.sort_values(col).reset_index(drop=True)
    return df

# =========================
# Moving Averages & Oscillators
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    series = _to_num(series)
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    series = _to_num(series)
    return series.rolling(window).mean()

def wma(series: pd.Series, window: int) -> pd.Series:
    series = _to_num(series)
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    series = _to_num(series)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = _safe_div(gain, loss)
    out = 100 - (100 / (1 + rs))
    return out

# =========================
# Volatility
# =========================
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h, l, c = _to_num(high), _to_num(low), _to_num(close)
    prev_c = c.shift(1)
    tr1 = h - l
    tr2 = (h - prev_c).abs()
    tr3 = (l - prev_c).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# =========================
# Bands & Channels
# =========================
def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    s = _to_num(series)
    mid = s.rolling(window).mean()
    std = s.rolling(window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = _safe_div(upper - lower, mid)
    # %B = (price - lower) / (upper - lower)
    pctb = _safe_div(s - lower, (upper - lower))
    return mid, upper, lower, width, pctb

# =========================
# MACD & Stochastics
# =========================
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    s = _to_num(series)
    ema_fast = ema(s, fast)
    ema_slow = ema(s, slow)
    line = ema_fast - ema_slow
    signal_line = line.ewm(span=signal, adjust=False).mean()
    hist = line - signal_line
    return line, signal_line, hist

def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3, smooth_k: int = 3):
    h, l, c = _to_num(high), _to_num(low), _to_num(close)
    lowest_low = l.rolling(k_period).min()
    highest_high = h.rolling(k_period).max()
    # %K raw
    k = 100 * _safe_div(c - lowest_low, highest_high - lowest_low)
    if smooth_k and smooth_k > 1:
        k = k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    return k, d

# =========================
# Volume-based
# =========================
def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    c, v = _to_num(close), _to_num(volume)
    direction = np.sign(c.diff()).fillna(0)
    return (direction * v).fillna(0).cumsum()

def cum_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    # Approximate VWAP for 1D/1W using typical price * volume cumulative
    h, l, c, v = _to_num(high), _to_num(low), _to_num(close), _to_num(volume)
    typical = (h + l + c) / 3.0
    pv = (typical * v).cumsum()
    cv = v.cumsum()
    return _safe_div(pv, cv)

# =========================
# Candle anatomy
# =========================
def candle_anatomy(open_, high, low, close):
    o, h, l, c = _to_num(open_), _to_num(high), _to_num(low), _to_num(close)
    body = (c - o).abs()
    # elementwise max/min that preserve Series (avoid numpy.ndarray which lacks .rolling)
    co_max = pd.concat([c, o], axis=1).max(axis=1)
    co_min = pd.concat([c, o], axis=1).min(axis=1)
    upper_wick = h - co_max
    lower_wick = co_min - l
    range_ = h - l
    body_pct = _safe_div(body, range_)
    return body, upper_wick, lower_wick, range_, body_pct

# =========================
# Enrichment entrypoint
# =========================
def enrich_indicators(df: pd.DataFrame, *, ema_fast: int = 20, ema_slow: int = 50, ema_trend: int = 200, bb_window: int = 20, bb_std: float = 2.0, rsi_period: int = 14, atr_period: int = 14) -> pd.DataFrame:
    """Return a copy of df enriched with common indicators.
    Columns added (if input columns exist):
    - ema20, ema50, ema200
    - sma20, sma50
    - rsi14
    - macd_line, macd_signal, macd_hist
    - bb_mid, bb_upper, bb_lower, bb_width, bb_pctb
    - atr14, atr_pct
    - stoch_k, stoch_d
    - obv, cum_vwap
    - body, upper_wick, lower_wick, range, body_pct
    """
    if df is None or df.empty:
        return df

    df = _ensure_sorted(df)
    out = df.copy()

    o = _to_num(out.get("open"))
    h = _to_num(out.get("high"))
    l = _to_num(out.get("low"))
    c = _to_num(out.get("close"))
    # Một số nguồn (nhất là 1W) có thể thiếu volume → đảm bảo cột tồn tại để tránh exception
    if "volume" not in out.columns:
        out["volume"] = np.nan
    v = _to_num(out.get("volume"))
    out["volume"] = v

    # MAs
    out["ema20"] = ema(c, ema_fast)
    out["ema50"] = ema(c, ema_slow)
    out["ema200"] = ema(c, ema_trend)
    out["sma20"] = sma(c, 20)
    out["sma50"] = sma(c, 50)

    # RSI
    out["rsi14"] = rsi(c, rsi_period)

    # MACD
    macd_line, macd_signal, macd_hist = macd(c)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_hist

    # Bollinger
    bb_mid, bb_upper, bb_lower, bb_width, bb_pctb = bollinger(c, window=bb_window, num_std=bb_std)
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower
    out["bb_width"] = bb_width
    out["bb_pctb"] = bb_pctb

    # ATR
    out["atr14"] = atr(h, l, c, period=atr_period)
    out["atr_pct"] = _safe_div(out["atr14"], c) * 100

    # Stoch
    st_k, st_d = stoch(h, l, c)
    out["stoch_k"] = st_k
    out["stoch_d"] = st_d

    # Volume-based
    out["obv"] = obv(c, v)
    out["cum_vwap"] = cum_vwap(h, l, c, v)

    # Candle anatomy
    body, uw, lw, rng, body_pct = candle_anatomy(o, h, l, c)
    out["body"] = body
    out["upper_wick"] = uw
    out["lower_wick"] = lw
    out["range"] = rng
    out["body_pct"] = body_pct

    return out
