
#!/usr/bin/env python3
"""
main.py — VN Stocks Signal Scanner (LONG-only)
----------------------------------------------
- Universe from universe.py (override via ENV SYMBOLS="STB,CTG,...")
- Fetch OHLCV 1D & 1W (include running candles)
- Compute features -> evaluate evidence -> decide plan (LONG-only)
- Pretty console logs with TP ladder & direction
- Scheduler: 5 runs per day (Mon–Fri): 09:30, 10:30, 11:30, 14:00, 15:00 Asia/Ho_Chi_Minh
- To avoid congestion: split universe into 7 blocks; each block runs 5 minutes apart

ENV overrides (optional):
- SYMBOLS="STB,CTG,..."            # override universe
- BLOCKS=7                         # number of blocks (default 7)
- BLOCK_DELAY_MIN=5                # minutes between blocks (default 5)
- FAST_START=1                     # run immediately once on launch (for testing)
"""
from __future__ import annotations
import os
import sys
import time
from math import ceil
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

# Local modules
from universe import get_universe_from_env
from vnstock_api import fetch_ohlcv_batch, fetch_ohlcv
from feature_primitives import compute_features_by_tf
from decision_engine import decide

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")

REQUIRED_COLS = ["close","volume","atr14","range","body_pct","rsi14","macd_hist"]

def _df_status(df: pd.DataFrame):
    issues = []
    if not isinstance(df, pd.DataFrame): return False, ["df_none"]
    if len(df) < 25: issues.append(f"rows_lt_25({len(df)})")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing: issues.append(f"missing_cols:{missing}")
    return (len(issues)==0), issues

# ------------------------------
# Logging helpers
# ------------------------------
def now_vn() -> datetime:
    return datetime.now(tz=VN_TZ)

def ts_prefix() -> str:
    return now_vn().strftime("%Y-%m-%d %H:%M:%S")

def log_info(msg: str):
    print(f"{ts_prefix()} INFO {msg}", flush=True)

def log_err(msg: str):
    print(f"{ts_prefix()} ERROR {msg}", flush=True, file=sys.stderr)

# ------------------------------
# Formatting helpers
# ------------------------------
# Number of decimals for price-like fields in logs.
# Can be overridden via env PRICE_DP (default: 1)
try:
    PRICE_DP = int(os.getenv("PRICE_DP", "1"))
except Exception:
    PRICE_DP = 1

def format_plan(sym: str, plan: dict) -> str:
    dec = plan.get("DECISION")
    state = plan.get("STATE")
    direction = plan.get("DIRECTION")
    entry = plan.get("entry"); sl = plan.get("sl")
    tp1, tp2, tp3, tp4, tp5 = (plan.get("tp1"), plan.get("tp2"), plan.get("tp3"), plan.get("tp4"), plan.get("tp5"))
    rr = plan.get("rr"); rr2 = plan.get("rr2")
    miss = plan.get("missing", [])
    conf = plan.get("confirmations", {})
    vline = plan.get("validator_line", "")
    vcheck = plan.get("validator_checklist", "")
    conf_val = plan.get("confidence", None); conf_min = plan.get("min_conf_enter", None)
    vreport = plan.get("validator_report", {})
    data_ok = vreport.get("data_ok", True) if isinstance(vreport, dict) else True
    data_issues = vreport.get("data_issues", []) if isinstance(vreport, dict) else [] 

    if dec == "ENTER":
        return (f"[{sym}] DECISION=ENTER | STATE={state} | DIR={direction} | "
                f"entry={_round(entry, PRICE_DP)} sl={_round(sl, PRICE_DP)} "
                f"TP1={_round(tp1, PRICE_DP)} TP2={_round(tp2, PRICE_DP)} TP3={_round(tp3, PRICE_DP)} TP4={_round(tp4, PRICE_DP)} TP5={_round(tp5, PRICE_DP)} "
                f"rr={_round(rr,2)} rr2={_round(rr2,2)} "
                f"| confirm:V={conf.get('volume',False)} M={conf.get('momentum',False)} C={conf.get('candles',False)}"
                f"{' | validators: ' + vline if vline else ''}"
                f"{' | ' + vcheck if vcheck else ''}"
                f"{(f' | conf={conf_val:.2f}/min={conf_min:.2f}') if (isinstance(conf_val,(int,float)) and isinstance(conf_min,(int,float))) else ''}"
                f"{' | DATA_GAP: ' + ','.join(map(str,data_issues)) if (data_ok is False) else ''}"
    else:
        why = (", ".join(miss)) if miss else "-"
        # NEW: nếu đã có setup kèm DECISION=WAIT -> in gọn setup để trader cân nhắc
        if plan.get("entry") and plan.get("sl") and plan.get("tp1"):
            return (f"[{sym}] DECISION=WAIT (SETUP) | STATE={state} | DIR={direction} | "
                    f"entry={_round(entry)} sl={_round(sl)} "
                    f"TP1={_round(tp1)} TP2={_round(tp2)} TP3={_round(tp3)} TP4={_round(tp4)} TP5={_round(tp5)} "
                    f"rr={_round(rr,2)} rr2={_round(rr2,2)} "
                    f"| reason={why} | confirm:V={conf.get('volume',False)} M={conf.get('momentum',False)} C={conf.get('candles',False)}"
                    f"{' | validators: ' + vline if vline else ''}"
                    f"{' | ' + vcheck if vcheck else ''}"
                    f"{(f' | conf={conf_val:.2f}/min={conf_min:.2f}') if (isinstance(conf_val,(int,float)) and isinstance(conf_min,(int,float))) else ''}"
                    f"{' | DATA_GAP: ' + ','.join(map(str,data_issues)) if (data_ok is False) else ''}"
                   )
        return (f"[{sym}] DECISION={dec} | STATE={state} | DIR={direction} | reason={why} "
                f"| confirm:V={conf.get('volume',False)} M={conf.get('momentum',False)} C={conf.get('candles',False)}"
                f"{' | validators: ' + vline if vline else ''}"
                f"{' | ' + vcheck if vcheck else ''}"
                f"{(f' | conf={conf_val:.2f}/min={conf_min:.2f}') if (isinstance(conf_val,(int,float)) and isinstance(conf_min,(int,float))) else ''}"
                f"{' | DATA_GAP: ' + ','.join(map(str,data_issues)) if (data_ok is False) else ''}")

def _round(x, n=2):
    if x is None:
        return None
    try:
        if n == 0:
            return int(round(float(x)))
        return round(float(x), n)
    except Exception:
        return x

# ------------------------------
# Core scan for a list of symbols
# ------------------------------
def scan_symbols(symbols: List[str]) -> None:
    if not symbols:
        return
    log_info(f"Scanning {len(symbols)} symbols ...")
    # Fetch once per TF for the whole block
    # NEW: tránh dùng nến 1D đang chạy trong giờ giao dịch
    now = now_vn()
    include_partial_daily = False if now.hour < 15 else True
    data_1d = fetch_ohlcv_batch(symbols, timeframe="1D", limit=420, include_partial=include_partial_daily)
    data_1w = fetch_ohlcv_batch(symbols, timeframe="1W", limit=260, include_partial=True)

    for sym in symbols:
        try:
            df1d = data_1d.get(sym, None)
            if df1d is None:
                df1d = pd.DataFrame()
            df1w = data_1w.get(sym, None)
            if df1w is None:
                df1w = pd.DataFrame()
            # Rescue refetch if 1D empty
            if df1d is None or len(df1d) == 0:
                # log debug info from attrs if any
                src_tried = getattr(df1d, "attrs", {}).get("source_tried")
                err = getattr(df1d, "attrs", {}).get("error")
                dbg = getattr(df1d, "attrs", {}).get("debug")
                if src_tried or err or dbg:
                    log_info(f"[{sym}] 1D empty | debug={dbg} | source_tried={src_tried} | error={err}")
                # Try smaller limit & no date cut (still avoid partial during session)
                df1d = fetch_ohlcv(sym, timeframe="1D", limit=120, include_partial=(now.hour >= 15))
                if len(df1d) == 0:
                    # Try exclude running bar (some providers only commit after close)
                    df1d = fetch_ohlcv(sym, timeframe="1D", limit=120, include_partial=False)
            # NEW: nếu vẫn có bar hôm nay và đang trước 15:00 → drop bar chạy
            if df1d is not None and len(df1d) > 0 and now.hour < 15:
                try:
                    last_ts = df1d["time"].iloc[-1]
                    if hasattr(last_ts, "date") and last_ts.date() == now.date():
                        df1d = df1d.iloc[:-1].reset_index(drop=True)
                except Exception as _e:
                    log_info(f"[{sym}] skip-drop-running-bar (info): {_e}")
            if df1d is None or len(df1d) < 30:
                log_info(f"[{sym}] DECISION=WAIT | reason=insufficient_1D_bars({len(df1d) if df1d is not None else 0})")
                continue
            feats = compute_features_by_tf({"1D": df1d, "1W": df1w})
            ev = None  # để decision_engine tự gọi _safe_eval
            plan = decide(feats, ev, sym=sym)
            log_info(format_plan(sym, plan))
            time.sleep(0.1)  # tiny pacing within block
        except Exception as e:
            log_err(f"[{sym}] failed: {e}")

# ------------------------------
# Block scheduler
# ------------------------------
def chunk_blocks(symbols: List[str], blocks: int) -> List[List[str]]:
    n = len(symbols)
    size = ceil(n / max(1, blocks))
    return [symbols[i:i+size] for i in range(0, n, size)]

def run_in_blocks(symbols: List[str], blocks: int, delay_min: int) -> None:
    parts = chunk_blocks(symbols, blocks)
    for i, part in enumerate(parts):
        if not part:
            continue
        log_info(f"Block {i+1}/{len(parts)} → {len(part)} symbols")
        scan_symbols(part)
        if i < len(parts) - 1:
            log_info(f"Sleeping {delay_min} minutes before next block ...")
            time.sleep(delay_min * 60)

# ------------------------------
# Daily schedule (Mon–Fri at 09:30, 10:30, 11:30, 14:00, 15:00 VN time)
# (UTC: 02:30, 03:30, 04:30, 07:00, 08:00)
# ------------------------------
RUN_TIMES = [(9,30), (10,30), (11,30), (14,0), (15,0)]

def is_weekday(dt: datetime) -> bool:
    # Monday=0 ... Sunday=6
    return dt.weekday() < 5

def next_run_after(dt: datetime) -> datetime:
    # Compute the next scheduled run strictly after 'dt'
    # If after last slot today, move to next weekday 11:30
    day = dt.date()
    candidates = []
    for h, m in RUN_TIMES:
        candidates.append(datetime(day.year, day.month, day.day, h, m, tzinfo=VN_TZ))
    future = [c for c in candidates if c > dt]
    if future:
        return min(future)
    # no future today -> move to next weekday
    nxt = dt + timedelta(days=1)
    while not is_weekday(nxt):
        nxt += timedelta(days=1)
    h, m = RUN_TIMES[0]
    return datetime(nxt.year, nxt.month, nxt.day, h, m, tzinfo=VN_TZ)

def main_loop():
    symbols = get_universe_from_env()
    blocks = int(os.getenv("BLOCKS", "7"))
    delay_min = int(os.getenv("BLOCK_DELAY_MIN", "5"))

    # Optional immediate run (useful for testing / first boot)
    if os.getenv("FAST_START", "0") == "1":
        log_info("FAST_START=1 → running an immediate scan ...")
        run_in_blocks(symbols, blocks, delay_min)

    # Regular schedule loop
    while True:
        now = now_vn()
        if is_weekday(now):
            # If current time is very close to a run slot (within 2 minutes), fire immediately
            for (h, m) in RUN_TIMES:
                slot = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if abs((now - slot).total_seconds()) <= 120:
                    log_info(f"Triggered scheduled run @ {slot.strftime('%H:%M')} VN time")
                    run_in_blocks(symbols, blocks, delay_min)
                    # After a run, compute next run to sleep until
                    nxt = next_run_after(now_vn())
                    sleep_s = max(5, int((nxt - now_vn()).total_seconds()) - 2)  # guard
                    log_info(f"Next run at {nxt.strftime('%Y-%m-%d %H:%M')} VN time (sleep {sleep_s//60}m)")
                    time.sleep(sleep_s)
                    break
            else:
                # Not near a slot; sleep until next run
                nxt = next_run_after(now)
                sleep_s = max(5, int((nxt - now).total_seconds()) - 2)
                log_info(f"Idle. Next run at {nxt.strftime('%Y-%m-%d %H:%M')} VN time (sleep {sleep_s//60}m)")
                time.sleep(sleep_s)
        else:
            # Weekend: sleep until next Monday 11:30
            nxt = next_run_after(now)
            sleep_s = max(5, int((nxt - now).total_seconds()) - 2)
            log_info(f"Weekend. Next run at {nxt.strftime('%Y-%m-%d %H:%M')} VN time (sleep {sleep_s//3600}h)")
            time.sleep(sleep_s)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        log_info("Exiting on KeyboardInterrupt.")
    except Exception as e:
        log_err(f"Fatal error: {e}")
        sys.exit(1)
