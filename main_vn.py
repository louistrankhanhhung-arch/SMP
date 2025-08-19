#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: vnstock_api -> indicators -> structure_engine -> gpt_signal_builder

- Khung thời gian: 1W (context) & 1D (khung giao dịch; nến 1D có thể đang chạy)
- Long-only; hiện tại chỉ log ra stdout + file. (Telegram sẽ nối sau)
- Lịch chạy do cron/systemd hẹn giờ.

ENV gợi ý:
  - OPENAI_API_KEY: bắt buộc để gọi GPT
  - SYMBOLS="FPT,SSI,VCB" (tuỳ chọn) hoặc dùng DEFAULT_UNIVERSE trong universe_vn.py
  - VN_EXCHANGE_MAP='{"FPT":"HOSE","SSI":"HOSE"}' (tuỳ chọn) để định dạng giá theo sàn trong gpt_signal_builder

Run:
  python main_vn.py --symbols "FPT,SSI" --logfile logs/signal_$(date +%F).log
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from universe_vn import resolve_symbols, get_universe_from_env
from vnstock_api import fetch_ohlcv
from indicators import enrich_indicators, enrich_more
from structure_engine import build_struct_json
from gpt_signal_builder import make_telegram_signal

# --------------- Logging setup ---------------
def setup_logging(logfile: str | None):
    logfmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format=logfmt, datefmt=datefmt, handlers=handlers)


def _vn_day_progress(now_local_ts) -> float:
    """
    Tính % thời gian giao dịch đã trôi qua trong ngày (VN: 09:00–11:30, 14:00–15:00).
    Bỏ ATO chi tiết; nếu cần có thể điều chỉnh. Trả về [0.05, 1.0].
    """
    import datetime as dt
    total_minutes = 150 + 60  # 09:00-11:30 = 150'; 14:00-15:00 = 60'
    sessions = [(dt.time(9,0), dt.time(11,30)), (dt.time(14,0), dt.time(15,0))]
    passed = 0
    t = now_local_ts.time()

    # tính phút đã qua trong hai phiên
    for start, end in sessions:
        start_dt = dt.datetime.combine(now_local_ts.date(), start, tzinfo=now_local_ts.tzinfo)
        end_dt   = dt.datetime.combine(now_local_ts.date(), end,   tzinfo=now_local_ts.tzinfo)
        if t >= end:
            passed += int((end_dt - start_dt).total_seconds() // 60)
        elif t > start:
            now_dt = dt.datetime.combine(now_local_ts.date(), t, tzinfo=now_local_ts.tzinfo)
            passed += max(0, int((now_dt - start_dt).total_seconds() // 60))

    prog = passed / total_minutes if total_minutes > 0 else 1.0
    # chặn đáy 5% để tránh chia cho số rất nhỏ đầu phiên
    return max(0.05, min(1.0, prog))
# --------------- Utils ---------------
def drop_partial_last_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ nến cuối nếu còn đang chạy (index > now_utc)."""
    if df is None or len(df) == 0:
        return df
    now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    last_ts = df.index[-1]
    try:
        if pd.isna(last_ts.tzinfo):
            last_ts = last_ts.tz_localize("UTC")
    except Exception:
        pass
    return df.iloc[:-1] if last_ts > now_utc else df

def load_and_enrich(symbol: str, tf: str, limit: int = 500) -> pd.DataFrame:
    df = fetch_ohlcv(symbol, timeframe=tf, limit=limit)
    if tf.upper() == "1H":
        df = drop_partial_last_bar(df)
    if df is None or len(df) == 0:
        return df
    df = enrich_indicators(df)
    df = enrich_more(df)
    return df

# --------------- Core per-symbol ---------------
def process_symbol(symbol: str) -> dict:
    try:
        df_w = load_and_enrich(symbol, "1W", limit=400)
        if df_w is None or len(df_w) == 0:
            logging.warning(f"{symbol}: Không có dữ liệu 1W.")
            return {"ok": False, "symbol": symbol, "error": "no_1w"}

        df_d = load_and_enrich(symbol, "1D", limit=400)
        if df_d is None or len(df_d) == 0:
            logging.warning(f"{symbol}: Không có dữ liệu 1D.")
            return {"ok": False, "symbol": symbol, "error": "no_1d"}

        # ---- compute volume progress for live 1D ----
        now_local = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh")
        day_prog = _vn_day_progress(now_local.to_pydatetime())
        try:
            denom = (df_d['vol_sma20'].iloc[-1] or 0) * day_prog
            if denom <= 0:
                vol_progress = None
            else:
                vol_progress = float(df_d['volume'].iloc[-1]) / float(denom)
        except Exception:
            vol_progress = None
    
        # Build struct cho từng TF (context truyền vào TF nhỏ hơn)
        struct_1w = build_struct_json(symbol, "1W", df_w)  # context top
        struct_1d = build_struct_json(symbol, "1D", df_d, context_df=df_w)
        try:
            if vol_progress is not None:
                struct_1d.setdefault('snapshot', {}).setdefault('volume', {})['progress'] = float(vol_progress)
        except Exception:
            pass

        # Gọi GPT để phân loại/ra setup
        gpt = make_telegram_signal(struct_1w, struct_1d)

        # Logging theo yêu cầu
        if not gpt.get("ok"):
            logging.error(f"{symbol}: GPT error: {gpt.get('error')}")
            return {"ok": False, "symbol": symbol, "error": gpt.get("error")}

        decision = (gpt.get("decision") or {}).get("action")
        eta = (gpt.get("decision") or {}).get("tp1_eta_bars")
        logging.info(f"{symbol}: decision={decision}, tp1_eta_bars={eta}, vol_progress={struct_1d.get('snapshot',{}).get('volume',{}).get('progress')}")

        # always log analysis_text
        if gpt.get("analysis_text"):
            for line in str(gpt["analysis_text"]).splitlines():
                logging.info(f"{symbol} | {line}")

        # If GPT đề xuất gửi setup (ENTER hoặc WAIT & ETA<5), telegram_text sẽ có nội dung
        if gpt.get("telegram_text"):
            logging.info(f"{symbol} | SETUP:\n{gpt['telegram_text']}")

        return {"ok": True, "symbol": symbol, "gpt": gpt}

    except Exception as e:
        logging.exception(f"{symbol}: Exception")
        return {"ok": False, "symbol": symbol, "error": str(e)}

# --------------- CLI ---------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default="", help="CSV mã. Nếu bỏ trống dùng env SYMBOLS hoặc DEFAULT_UNIVERSE.")
    parser.add_argument("--logfile", type=str, default=f"logs/signal_{datetime.now().date()}.log")
    args = parser.parse_args()

    setup_logging(args.logfile)
    logging.info("Start VN Signal Pipeline (1W + 1D; live 1D enabled)")

    symbols = resolve_symbols(args.symbols) if args.symbols.strip() else get_universe_from_env()
    logging.info(f"Symbols: {symbols}")

    results = []
    for sym in symbols:
        res = process_symbol(sym)
        results.append(res)

    ok = sum(1 for r in results if r.get("ok"))
    fail = len(results) - ok
    logging.info(f"Done. ok={ok}, fail={fail}")

if __name__ == "__main__":
    main()
