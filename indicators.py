# indicators.py
import pandas as pd
import numpy as np

# =========================
# Moving Averages & Oscillators
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low  = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close  = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width_pct = (upper - lower) / mid.replace(0, np.nan) * 100
    return upper, mid, lower, width_pct

def rolling_zscore(series: pd.Series, window=20):
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / sigma.replace(0, np.nan)

# =========================
# Enrich: base indicators
# =========================
def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ema20'] = ema(out['close'], 20)
    out['ema50'] = ema(out['close'], 50)

    ub, mid, lb, width_pct = bollinger(out['close'], 20, 2.0)
    out['bb_upper'], out['bb_mid'], out['bb_lower'], out['bb_width_pct'] = ub, mid, lb, width_pct

    out['rsi14']  = rsi(out['close'], 14)
    out['atr14']  = atr(out, 14)

    out['vol_sma20'] = sma(out['volume'], 20)
    out['vol_ratio'] = out['volume'] / out['vol_sma20'].replace(0, np.nan)

    out['dist_to_ema20_pct'] = (out['close'] - out['ema20']) / out['ema20'].replace(0, np.nan) * 100
    out['dist_to_ema50_pct'] = (out['close'] - out['ema50']) / out['ema50'].replace(0, np.nan) * 100
    return out

# =========================
# Enrich: volume, candle anatomy, SMAs (cho SR mềm)
# =========================
def enrich_more(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Volume features
    out['vol_z20'] = rolling_zscore(out['volume'], 20)
    out['vol_up']  = (out['close'] > out['open']).astype(int)   # nến xanh
    out['vol_dn']  = (out['close'] < out['open']).astype(int)   # nến đỏ

    # Candle anatomy
    body = (out['close'] - out['open']).abs()
    rng  = (out['high'] - out['low']).replace(0, np.nan)
    out['body_pct']       = (body / rng * 100).clip(0, 100)
    out['upper_wick_pct'] = ((out['high'] - out[['open','close']].max(axis=1)) / rng * 100).clip(lower=0)
    out['lower_wick_pct'] = (((out[['open','close']].min(axis=1) - out['low']) / rng) * 100).clip(lower=0)

    # Soft SR components
    out['sma20'] = out['close'].rolling(20).mean()
    out['sma50'] = out['close'].rolling(50).mean()
    return out

# =========================
# Liquidity Zones (Volume Profile)
# =========================
def calc_vp(df: pd.DataFrame, window_bars: int = 120, bins: int = 24, top_k: int = 5):
    """
    Volume profile đơn giản trên 'window_bars' nến gần nhất.
    Dùng HLC3 làm đại diện giá mỗi nến để bucket theo bins.
    Trả về list dict sắp xếp theo volume giảm dần.
    """
    if len(df) == 0:
        return []
    sub = df.tail(window_bars)
    hlc3 = (sub['high'] + sub['low'] + sub['close']) / 3.0

    lo = float(sub['low'].min())
    hi = float(sub['high'].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return []

    edges = np.linspace(lo, hi, bins + 1)
    idx = np.digitize(hlc3.values, edges) - 1
    idx = np.clip(idx, 0, bins - 1)

    vol_bins = np.zeros(bins)
    for i, v in zip(idx, sub['volume'].values):
        if np.isfinite(v):
            vol_bins[i] += v

    zones = []
    for i in range(bins):
        p_lo, p_hi = float(edges[i]), float(edges[i+1])
        zones.append({
            "price_range": (p_lo, p_hi),
            "price_mid": round((p_lo + p_hi) / 2.0, 2),
            "volume_sum": float(vol_bins[i]),
        })
    zones = sorted(zones, key=lambda x: -x["volume_sum"])
    return zones[:top_k]
