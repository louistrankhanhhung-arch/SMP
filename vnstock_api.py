# -*- coding: utf-8 -*-
"""
vnstock_api.py — Data access layer for Vietnam equities (VN30/HOSE/HNX/UPCoM)
----------------------------------------------------------------------------
- Uses unified `vnstock` package (>=3.2.x).
- Fetch OHLCV for 1D & 1W; includes the *running* candle when include_partial=True.
- Normalized columns: ["time","open","high","low","close","volume"]
- Tries multiple sources if a provider returns empty/error: ["VCI","TCBS","MSN"]
- Tries symbol variants: "VCB", "VCB.HOSE", "VCB:HOSE"
- Tries interval aliases: 1D→["1D","1d","D","day","daily"], 1W→["1W","1w","W","week","weekly"]
"""
from __future__ import annotations
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import pandas as pd
import importlib
from datetime import datetime, timedelta, timezone, time as dtime

try:
    import pytz
except Exception:
    pytz = None

# Nếu bạn đang dùng một SDK nào đó (vd vnstock3/ssi/…) để fetch dữ liệu,
# bạn vẫn gọi như cũ ở dưới. Patch này tập trung vào làm sạch & chuẩn hóa.

VN_TZ_NAME = "Asia/Ho_Chi_Minh"
MARKET_CLOSE_HOUR = int(os.getenv("MARKET_CLOSE_HOUR", "15"))  # 15:00 VN
DROP_RUNNING_CANDLE = bool(int(os.getenv("DROP_RUNNING_CANDLE", "1")))  # mặc định bỏ nến chưa chốt
MIN_ROWS_REQUIRED = int(os.getenv("MIN_ROWS_REQUIRED", "25"))  # đủ cho rolling 20 + margin

def _get_tz():
    if pytz is None:
        return None
    try:
        return pytz.timezone(VN_TZ_NAME)
    except Exception:
        return None

def _to_vn_tz(dt: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    tz = _get_tz()
    if isinstance(dt, pd.Series):
        idx = pd.to_datetime(dt, errors="coerce", utc=True)
    else:
        idx = pd.to_datetime(dt, errors="coerce", utc=True)
    if tz is not None:
        try:
            return idx.tz_convert(tz)
        except Exception:
            try:
                # nếu dữ liệu local-naive, gán tz VN
                return idx.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
            except Exception:
                return idx
    return idx

def _coerce_numeric(s: pd.Series) -> pd.Series:
    # xóa dấu phẩy/ngăn cách nghìn nếu có, sau đó ép kiểu số
    if s.dtype == object:
        s = s.astype(str).str.replace(",", "", regex=False).str.replace("_", "", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    # volume âm/close âm => NaN
    out = out.mask(~np.isfinite(out))
    return out

def _sanitize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    # Chuẩn tên cột phổ biến
    cols_map = {
        "date": "time", "time": "time", "datetime": "time",
        "o": "open", "open": "open",
        "h": "high", "high": "high",
        "l": "low", "low": "low",
        "c": "close", "close": "close", "adj_close": "close",
        "v": "volume", "volume": "volume", "vol": "volume",
    }
    # Đổi tên cột theo map nếu phù hợp
    renamed = {}
    for c in df.columns:
        key = c.lower().strip()
        renamed[c] = cols_map.get(key, c)
    df = df.rename(columns=renamed)

    # Chỉ giữ các cột cốt lõi
    keep = [c for c in ["time","open","high","low","close","volume"] if c in df.columns]
    df = df[keep].copy()

    # Chuẩn hóa thời gian về VN timezone
    if "time" in df.columns:
        df["time"] = _to_vn_tz(df["time"])
    else:
        # thiếu trầm trọng: trả khung rỗng đúng schema
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    # Ép số cho OHLCV
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = _coerce_numeric(df[c])
        else:
            df[c] = np.nan

    # Bỏ dòng thiếu toàn bộ OHLC
    df = df.dropna(subset=["open","high","low","close"], how="any")

    # Dedupe + sort theo time
    df = df.drop_duplicates(subset=["time"]).sort_values("time")
    df = df.reset_index(drop=True)

    # Đảm bảo DataFrame có đúng cột & thứ tự
    df = df[["time","open","high","low","close","volume"]]
    return df

def _drop_running_candle_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ nến ngày hiện tại nếu chưa qua giờ đóng cửa (15:00 VN)."""
    if not DROP_RUNNING_CANDLE:
        return df
    if df is None or len(df) == 0:
        return df
    tz = _get_tz()
    now = datetime.now(tz) if tz else datetime.now()
    # Lấy dòng cuối cùng
    last = df.iloc[-1]
    ts: pd.Timestamp = last["time"]
    if pd.isna(ts):
        return df
    try:
        ts_local = ts
        # Nếu ts không có tz, coi như local VN
        if ts_local.tzinfo is None and tz is not None:
            ts_local = ts_local.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
    except Exception:
        ts_local = ts
    if ts_local.date() == now.date():
        # trước 15:00 thì xóa nến hôm nay
        if now.time() < dtime(hour=MARKET_CLOSE_HOUR, minute=0):
            return df.iloc[:-1].copy()
    return df

def _quality_report(df: pd.DataFrame) -> dict:
    issues = []
    ok = True
    need = ["time","open","high","low","close","volume"]
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"ok": False, "issues": ["df_empty"]}
    missing = [c for c in need if c not in df.columns]
    if missing:
        issues.append(f"missing_cols:{missing}")
        ok = False
    if len(df) < MIN_ROWS_REQUIRED:
        issues.append(f"rows_lt_{MIN_ROWS_REQUIRED}({len(df)})")
        ok = False
    # kiểm tra NaN bất thường
    nan_ohlc = df[["open","high","low","close"]].isna().sum().sum()
    if nan_ohlc > 0:
        issues.append(f"nan_ohlc:{int(nan_ohlc)}")
        ok = False
    return {"ok": ok, "issues": issues}

# ---- Backward-compatible aliases (single unified flow) --------------------
# Nếu code cũ của bạn import các hàm dưới đây, chúng sẽ chuyển về fetch_ohlcv/Batch.
def fetch_ohlcv_1d(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Alias: unified path via vnstock, timeframe=1D."""
    df = fetch_ohlcv(symbol, timeframe="1D", limit=600, include_partial=True)
    return df

def fetch_symbols_daily(symbols: list[str], start: Optional[str] = None, end: Optional[str] = None) -> dict[str, pd.DataFrame]:
    """Alias: unified batch with timeframe=1D."""
    return fetch_ohlcv_batch(symbols, timeframe="1D", limit=600, include_partial=True)

# Nếu code cũ của bạn có hàm get_ohlcv hay tương tự, bạn có thể alias:
# get_ohlcv = fetch_ohlcv_1d
    
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
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    rename_map = {
        "time": "time", "date": "time", "TradingDate": "time", "tradingDate": "time",
        "open": "open", "Open": "open", "o": "open",
        "high": "high", "High": "high", "h": "high",
        "low": "low", "Low": "low", "l": "low",
        "close": "close", "Close": "close", "c": "close",
        "volume": "volume", "Volume": "volume", "v": "volume", "value": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    keep = [c for c in ["time","open","high","low","close","volume"] if c in df.columns]
    df = df[keep].copy()
    # Normalize timestamp → VN timezone
    if "time" in df.columns:
        df["time"] = _to_vn_tz(df["time"])
    # Coerce numerics
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Drop rows without time
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
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
    return pd.DataFrame(columns=["time","open","high","low","close","volume"]), {"tried": tried}

def _drop_running_bar_if_needed(df: pd.DataFrame, timeframe: str, include_partial: bool) -> pd.DataFrame:
    if include_partial or len(df) == 0:
        return df
    # For 1D: drop bar if last time is "today" VN time (market not closed/committed yet)
    last_ts = pd.to_datetime(df["time"].iloc[-1])
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
        df = pd.DataFrame(columns=["time","open","high","low","close","volume"])
        if last_err:
            df.attrs["error"] = last_err
        df.attrs["source_tried"] = ",".join(_SUPPORTED_SOURCES)
        df.attrs["debug"] = "no_rows_after_all_fallbacks"
        return df

    # Optionally drop the running bar
    df = _drop_running_bar_if_needed(df, timeframe, include_partial)

    # Final sanitize to our strict schema (time, tz, numerics, dedupe, order)
    df = _sanitize_ohlcv(df)

    # Quality report for clarity (DATA_GAP)
    q = _quality_report(df)
    if not q["ok"]:
        print(f"[{symbol}] DATA_GAP -> {q['issues']}")

    # Attach metadata
    if used_src:
        df.attrs["source_used"] = used_src
    if meta_used:
        for k, v in meta_used.items():
            df.attrs[k] = v

    # 🔧 Hợp thức hóa cột thời gian cho downstream: tạo 'ts' từ 'time'
    try:
        if df is not None and len(df) and "ts" not in df.columns and "time" in df.columns:
            ts = pd.to_datetime(df["time"], errors="coerce")
            # ưu tiên timezone VN nếu có thể (để _last_closed_idx hoạt động chuẩn phiên VN)
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo("Asia/Ho_Chi_Minh")
                if getattr(ts, "dt", None) is not None and ts.dt.tz is None:
                    ts = ts.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
            except Exception:
                pass
            df["ts"] = ts
    except Exception as _e:
        df.attrs["warn_ts_create"] = str(_e)
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
