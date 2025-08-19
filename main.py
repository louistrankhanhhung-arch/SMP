
import os, threading, time, logging, json, traceback
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from fastapi import FastAPI
from pydantic import BaseModel

# ====== modules ======
from vnstock_api import fetch_ohlcv
from indicators import enrich_indicators
from structure_engine import build_struct_json
from universe_vn import resolve_symbols
from gpt_signal_builder import make_telegram_signal

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ====== FastAPI app (optional) ======
app = FastAPI()

# ====== Config ======
TIMEZONE = os.getenv("TIMEZONE", "Asia/Ho_Chi_Minh")
RUN_TIMES = os.getenv("RUN_TIMES", "09:30,10:30,11:30,14:00,15:00")  # local time HH:MM
WEEKDAYS_ONLY = os.getenv("WEEKDAYS_ONLY", "1") == "1"
MAX_GPT = int(os.getenv("MAX_GPT", "12"))   # cap per run
LONG_ONLY = os.getenv("LONG_ONLY", "1") == "1"  # force long

# Telegram (optional hooks as in crypto app)
try:
    from telegram_poster import Signal as TgSignal, DailyQuotaPolicy, post_signal
    from notifier import TelegramNotifier, PostRef
    from signal_tracker import SignalTracker
except Exception:
    TgSignal = DailyQuotaPolicy = post_signal = TelegramNotifier = PostRef = SignalTracker = None

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID", "0") or 0)
POLICY_DB = os.getenv("POLICY_DB", "/mnt/data/policy.sqlite3")
POLICY_KEY = os.getenv("POLICY_KEY", "vn")

_BOT = None
_NOTIFIER = None
_TRACKER = None
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID and TgSignal and DailyQuotaPolicy and post_signal:
    try:
        _NOTIFIER = TelegramNotifier(token=TELEGRAM_BOT_TOKEN, default_chat_id=TELEGRAM_CHANNEL_ID)
        _BOT = _NOTIFIER.bot
        _TRACKER = SignalTracker(_NOTIFIER)
        logging.info("[telegram] bot & tracker ready")
    except Exception as e:
        print("[telegram] init error:", e)

# ====== Utils ======
def _build_structs_for(symbols: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            df1h = enrich_indicators(fetch_ohlcv(sym, timeframe="1H", limit=300))
            df1w = enrich_indicators(fetch_ohlcv(sym, timeframe="1W", limit=300))
            df1d = enrich_indicators(fetch_ohlcv(sym, timeframe="1D", limit=300))
            if len(df1h) < 50 or len(df1w) < 50 or len(df1d) < 50:
                print(f"[build] not enough bars for {sym}")
                continue
            s1h = build_struct_json(sym, "1H", df1h)
            s1w = build_struct_json(sym, "1W", df1w)
            s1d = build_struct_json(sym, "1D", df1d)
            out.append({"symbol": sym, "1H": s1h, "1W": s1w, "1D": s1d})
        except Exception as e:
            print(f"[build] error {sym}:", e)
            traceback.print_exc()
    return out

def _fmt_signal_text_enter(symbol: str, decision: Dict[str, Any]) -> str:
    """
    Simple formatter (Long-only)
    """
    entries = decision.get("entries") or []
    tps = decision.get("tps") or []
    sl = decision.get("sl")
    def _fmt(nums):
        if not nums: return "-"
        try:
            return ", ".join(f"{float(x):.0f}" for x in nums)
        except Exception:
            return ", ".join(str(x) for x in nums)
    sl_txt = "-" if sl is None else (f"{float(sl):.0f}" if isinstance(sl,(int,float,str)) else str(sl))
    body = [
        f"Long | {symbol}",
        f"Leverage: -",
        f"Entry: {_fmt(entries)}",
        f"SL: {sl_txt}",
        f"TP: {_fmt(tps)}",
    ]
    return "\\n".join(body)

def scan_once():
    local_tz = ZoneInfo(TIMEZONE)
    ts = datetime.now(local_tz).isoformat()
    symbols = resolve_symbols("")
    print(f"[scan] {ts} symbols={len(symbols)}")

    # limit to MAX_GPT per run for safety
    pick = symbols[:MAX_GPT] if MAX_GPT > 0 else symbols[:]

    structs = _build_structs_for(pick)

    sent = 0
    for sym in pick:
        s1h = next((x["1H"] for x in structs if x["symbol"] == sym), None)
        s1w = next((x["1W"] for x in structs if x["symbol"] == sym), None)
        s1d = next((x["1D"] for x in structs if x["symbol"] == sym), None)
        if not (s1h and s1w and s1d):
            print(f"[scan] missing structs: {sym}")
            continue
        out = make_telegram_signal(s1w, s1d, trigger_1h=s1h)
        if not out.get("ok"):
            print(f"[scan] GPT err {sym}: {out.get('error')}")
            continue
        decision = out.get("decision") or {}
        # ---- Force long-only ----
        decision["side"] = "long"
        act = str(decision.get("action") or "WAIT").upper()
        tele = out.get("telegram_text")
        if act == "ENTER":
            tele = _fmt_signal_text_enter(sym, decision)  # rebuild to remove leverage/decimals
            print("[signal]\\n" + tele)
        else:
            print(f"[signal] {sym} | {act} side=long")
        if out.get("analysis_text"):
            print("[analysis]\\n" + out["analysis_text"])
        sent += 1

        if tele and _BOT and _NOTIFIER and TgSignal and DailyQuotaPolicy and post_signal:
            plan = out.get("plan") or {}
            tg_sig = TgSignal(
                signal_id = plan.get("signal_id") or f"{sym}-{int(time.time())}",
                symbol    = sym,
                timeframe = plan.get("timeframe") or "4H",
                side      = "long",
                strategy  = plan.get("strategy") or "GPT-plan",
                entries   = plan.get("entries") or decision.get("entries") or [],
                sl        = plan.get("sl") if plan.get("sl") is not None else 0.0,
                tps       = plan.get("tps") or decision.get("tps") or [],
                leverage  = None,
                eta       = plan.get("eta"),
            )
            policy = DailyQuotaPolicy(db_path=POLICY_DB, key=POLICY_KEY)
            info = post_signal(bot=_BOT, channel_id=TELEGRAM_CHANNEL_ID, sig=tg_sig, policy=policy)

class ScanNowReq(BaseModel):
    symbols: Optional[List[str]] = None
    max_gpt: Optional[int] = None

@app.post("/scan_now")
def api_scan_now(req: ScanNowReq):
    global MAX_GPT
    if req.max_gpt is not None:
        MAX_GPT = req.max_gpt
    if req.symbols:
        # one-off for provided symbols
        for sym in req.symbols:
            try:
                df1h = enrich_indicators(fetch_ohlcv(sym, "1H", 300))
                df1w = enrich_indicators(fetch_ohlcv(sym, "4H", 300))
                df1d = enrich_indicators(fetch_ohlcv(sym, "1D", 300))
                s1h = build_struct_json(sym, "1H", df1h)
                s1w = build_struct_json(sym, "1W", df1w)
                s1d = build_struct_json(sym, "1D", df1d)
                out = make_telegram_signal(s1w, s1d, trigger_1h=s1h)
                print(json.dumps(out, ensure_ascii=False))
            except Exception as e:
                print(f"[scan_now] error {sym}: {e}")
        return {"ok": True, "count": len(req.symbols)}
    scan_once()
    return {"ok": True}

# ====== Time-based scheduler ======
def _parse_run_times():
    parts = [p.strip() for p in RUN_TIMES.split(",") if p.strip()]
    out = []
    for p in parts:
        hh, mm = p.split(":")
        out.append((int(hh), int(mm)))
    return out

def _is_weekday(dt_local):
    return dt_local.weekday() < 5  # Mon=0..Sun=6

def _next_fire(after: datetime) -> datetime:
    tz = ZoneInfo(TIMEZONE)
    runs = _parse_run_times()
    # Check today
    today = after.date()
    candidates = [after.replace(hour=h, minute=m, second=0, microsecond=0) for (h,m) in runs]
    candidates = [c for c in candidates if c > after]
    if WEEKDAYS_ONLY:
        candidates = [c for c in candidates if _is_weekday(c)]
    if candidates:
        return min(candidates)
    # else go to next day at first run time
    d = today + timedelta(days=1)
    while WEEKDAYS_ONLY and d.weekday() >= 5:
        d += timedelta(days=1)
    first_h, first_m = runs[0]
    return datetime(d.year, d.month, d.day, first_h, first_m, tzinfo=tz)

def _loop():
    tz = ZoneInfo(TIMEZONE)
    while True:
        now = datetime.now(tz)
        nxt = _next_fire(now + timedelta(seconds=1))
        wait = (nxt - now).total_seconds()
        logging.info(f"[scheduler] next run at {nxt.isoformat()} (in {int(wait)}s)")
        time.sleep(max(1, min(wait, 3600)))
        # run
        try:
            if WEEKDAYS_ONLY and not _is_weekday(datetime.now(tz)):
                continue
            scan_once()
        except Exception as e:
            logging.exception(f"[scheduler] run error: {e}")
        # small cool-down
        time.sleep(2)

@app.on_event("startup")
def _on_startup():
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    app.state.scan_thread = t
    logging.info("[scheduler] thread started")

if __name__ == "__main__":
    scan_once()
