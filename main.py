import os, threading, time, logging
import json
import traceback
import math
from typing import List, Dict, Any, Iterable, Optional
from datetime import datetime

from fastapi import FastAPI
from pydantic import BaseModel

# ====== modules trong repo ======
from kucoin_api import fetch_batch
from indicators import enrich_indicators
from structure_engine import build_struct_json
from universe import resolve_symbols
from gpt_signal_builder import make_telegram_signal

# (tuỳ chọn đăng Telegram & nối dây tracker — giữ nguyên nếu đã dùng)
try:
    from telegram_poster import Signal as TgSignal, DailyQuotaPolicy, post_signal
    from notifier import TelegramNotifier, PostRef
    from signal_tracker import SignalTracker
except Exception:
    TgSignal = DailyQuotaPolicy = post_signal = TelegramNotifier = PostRef = SignalTracker = None

# ---- Logging ----
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ====== FastAPI app (chỉ tạo 1 lần) ======
app = FastAPI()

# ====== Cấu hình ======
SCAN_INTERVAL_MIN = int(os.getenv("SCAN_INTERVAL_MIN", "60"))   # vòng lớn mỗi 60 phút
SCAN_TFS = os.getenv("SCAN_TFS", "1H,4H,1D")  # quét 1H/4H/1D
MAX_GPT = int(os.getenv("MAX_GPT", "10"))
EXCHANGE = os.getenv("EXCHANGE", "KUCOIN")

# Block spacing & phân lô trong 1 giờ
BLOCKS_PER_HOUR = int(os.getenv("BLOCKS_PER_HOUR", "3"))
BLOCK_SPACING_MIN = int(os.getenv("BLOCK_SPACING_MIN", "5"))

# Telegram (nếu muốn post)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID = int(os.getenv("TELEGRAM_CHANNEL_ID", "0") or 0)
POLICY_DB = os.getenv("POLICY_DB", "/mnt/data/policy.sqlite3")
POLICY_KEY = os.getenv("POLICY_KEY", "global")

_BOT = None
_NOTIFIER = None
_TRACKER = None

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID and TgSignal and DailyQuotaPolicy and post_signal:
    try:
        _NOTIFIER = TelegramNotifier(token=TELEGRAM_BOT_TOKEN, default_chat_id=TELEGRAM_CHANNEL_ID)
        _BOT = _NOTIFIER.bot  # raw TeleBot instance
        _TRACKER = SignalTracker(_NOTIFIER)
        logging.info("[telegram] bot & tracker ready")
    except Exception as e:
        print("[telegram] init error:", e)
        _NOTIFIER = None
        _BOT = None
        _TRACKER = None

@app.get("/health")
def health():
    return {"ok": True, "ts": time.time()}

# ====== Utils ======
def _chunk(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# --- thay toàn bộ hàm _build_structs_for trong main.py bằng đoạn dưới ---
from kucoin_api import fetch_ohlcv  # đã được import ở đầu file

def _build_structs_for(symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Lấy OHLCV cho mỗi symbol ở 3 khung 1H/4H/1D (không dùng fetch_batch của kucoin_api,
    vì hàm đó nhận 1 symbol + list timeframe).
    """
    out: List[Dict[str, Any]] = []
    for sym in symbols:
        try:
            # lấy dữ liệu từng khung
            df1h = enrich_indicators(fetch_ohlcv(sym, timeframe="1H", limit=300))
            df4h = enrich_indicators(fetch_ohlcv(sym, timeframe="4H", limit=300))
            df1d = enrich_indicators(fetch_ohlcv(sym, timeframe="1D", limit=300))

            if df1h is None or df4h is None or df1d is None:
                print(f"[build] missing df for {sym}")
                continue
            if len(df1h) < 50 or len(df4h) < 50 or len(df1d) < 50:
                print(f"[build] not enough bars for {sym}")
                continue

            s1h = build_struct_json(sym, "1H", df1h)
            s4h = build_struct_json(sym, "4H", df4h)
            s1d = build_struct_json(sym, "1D", df1d)

            out.append({"symbol": sym, "1H": s1h, "4H": s4h, "1D": s1d})
        except Exception as e:
            print(f"[build] error {sym}:", e)
            traceback.print_exc()
    return out

# ====== Round-Robin bền (Cách B) ======
RR_PTR_FILE = os.getenv("RR_PTR_FILE", "/mnt/data/rr_ptr.json")
RR_LOCK = threading.Lock()

def _load_rr_ptr(n_symbols: int) -> int:
    """Đọc con trỏ round-robin từ file; trả 0 nếu chưa có."""
    if n_symbols <= 0:
        return 0
    try:
        if os.path.exists(RR_PTR_FILE):
            with open(RR_PTR_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                ptr = int(data.get("ptr", 0))
                return max(0, ptr) % n_symbols
    except Exception as e:
        logging.warning(f"[rr] load ptr error: {e}")
    return 0

def _save_rr_ptr(ptr: int) -> None:
    """Ghi con trỏ round-robin ra file (best-effort)."""
    try:
        os.makedirs(os.path.dirname(RR_PTR_FILE), exist_ok=True)
        with open(RR_PTR_FILE, "w", encoding="utf-8") as f:
            json.dump({"ptr": int(ptr)}, f)
    except Exception as e:
        logging.warning(f"[rr] save ptr error: {e}")

def _pick_round_robin(symbols: List[str], k: int) -> List[str]:
    """Fallback: chọn k mã theo con trỏ lưu file (bền qua restart)."""
    if not symbols or k <= 0:
        return []
    with RR_LOCK:
        n = len(symbols)
        ptr = _load_rr_ptr(n)
        order = symbols[ptr:] + symbols[:ptr]
        picked = order[:min(k, n)]
        new_ptr = (ptr + len(picked)) % n
        _save_rr_ptr(new_ptr)
    return picked

def _split_blocks(symbols: List[str], blocks: int) -> List[List[str]]:
    """Chia symbols thành `blocks` khối đều nhau (block cuối có thể nhiều/ít hơn 1 đơn vị)."""
    if blocks <= 1 or not symbols:
        return [symbols]
    size = math.ceil(len(symbols) / blocks)
    return [symbols[i*size:(i+1)*size] for i in range(blocks)]

# ====== Scan ======
def scan_once_for_logs(block_idx: Optional[int] = None):
    start_ts = datetime.utcnow().isoformat() + "Z"
    syms = resolve_symbols("")
    if not syms:
        print("[scan] no symbols")
        return

    print(f"[scan] total symbols={len(syms)} exchange={EXCHANGE} tfs={SCAN_TFS}")

    # Chia block cố định theo giờ (3 block): ví dụ 30 mã => 10-10-10
    if block_idx is not None and BLOCKS_PER_HOUR >= 2:
        blocks = _split_blocks(syms, BLOCKS_PER_HOUR)
        if block_idx < 0 or block_idx >= len(blocks):
            print(f"[scan] invalid block_idx={block_idx}, fallback RR")
            pick_syms = _pick_round_robin(syms, MAX_GPT)
            print(f"[scan] candidates(no-filter)={pick_syms} (cap {MAX_GPT}) [RR]")
        else:
            pick_syms = blocks[block_idx][:MAX_GPT]
            print(f"[scan] block {block_idx+1}/{BLOCKS_PER_HOUR} -> {len(pick_syms)} symbols (cap {MAX_GPT})")
    else:
        pick_syms = _pick_round_robin(syms, MAX_GPT)
        print(f"[scan] candidates(no-filter)={pick_syms} (cap {MAX_GPT}) [RR]")

    structs = _build_structs_for(pick_syms)

    sent = 0
    for sym in pick_syms:
        try:
            s1h = next((x["1H"] for x in structs if x["symbol"] == sym), None)
            s4h = next((x["4H"] for x in structs if x["symbol"] == sym), None)
            s1d = next((x["1D"] for x in structs if x["symbol"] == sym), None)
            if not (s1h and s4h and s1d):
                print(f"[scan] missing structs: {sym} (need 1H/4H/1D)")
                continue

            out = make_telegram_signal(s4h, s1d, trigger_1h=s1h)

            if not out.get("ok"):
                print(f"[scan] GPT err {sym}: {out.get('error')}")
                continue

            tele = out.get("telegram_text")
            decision = out.get("decision") or {}
            act = str(decision.get("action") or "N/A").upper()
            side = str(decision.get("side") or "none")
            conf = decision.get("confidence")

            if tele:
                print("[signal]\n" + tele)
            else:
                print(f"[signal] {sym} | {act} side={side} conf={conf}")

            if out.get("analysis_text"):
                print("[analysis]\n" + out["analysis_text"])
            sent += 1

            meta = out.get("meta", {})
            rr = meta.get("rr", {})
            print(f"[meta] {sym} conf={meta.get('confidence')} rr_min={rr.get('rr_min')} rr_max={rr.get('rr_max')} eta={meta.get('eta')}")

            if tele and _BOT and _NOTIFIER and TgSignal and DailyQuotaPolicy and post_signal:
                plan = out.get("plan") or out.get("signal") or {}
                tg_sig = TgSignal(
                    signal_id = plan.get("signal_id") or out.get("signal_id") or f"{sym.replace('/','')}-{int(time.time())}",
                    symbol    = sym.replace("/", ""),
                    timeframe = plan.get("timeframe") or "4H",
                    side      = plan.get("side") or side or "long",
                    strategy  = plan.get("strategy") or "GPT-plan",
                    entries   = plan.get("entries") or [],
                    sl        = plan.get("sl") if plan.get("sl") is not None else 0.0,
                    tps       = plan.get("tps") or [],
                    leverage  = plan.get("leverage"),
                    eta       = plan.get("eta"),
                )
                policy = DailyQuotaPolicy(db_path=POLICY_DB, key=POLICY_KEY)
                info = post_signal(bot=_BOT, channel_id=TELEGRAM_CHANNEL_ID, sig=tg_sig, policy=policy)
                if info and _TRACKER and PostRef:
                    post_ref = PostRef(chat_id=info["chat_id"], message_id=info["message_id"])
                    signal_payload = {
                        "symbol": sym,
                        "side": tg_sig.side,
                        "entries": tg_sig.entries or [],
                        "stop": tg_sig.sl,
                        "tps": tg_sig.tps or [],
                        "leverage": tg_sig.leverage,
                    }
                    sl_mode = (plan.get("sl_mode") or "tick")
                    if sl_mode == "hard":
                        sl_mode = "tick"
                    _TRACKER.register_post(
                        signal_id=tg_sig.signal_id,
                        ref=post_ref,
                        signal=signal_payload,
                        sl_mode=sl_mode,
                    )
        except Exception as e:
            print(f"[scan] error processing {sym}: {e}")
            traceback.print_exc()

    try:
        os.makedirs("/mnt/data/gpt_logs", exist_ok=True)
        with open(f"/mnt/data/gpt_logs/scan_{int(time.time())}.meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "at": start_ts,
                "picked": pick_syms,
                "sent": sent,
                "block_idx": block_idx,
                "blocks_per_hour": BLOCKS_PER_HOUR
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[scan] write log error:", e)

# ====== Scheduler background ======
def _scan_loop():
    logging.info(f"[scheduler] start: interval={SCAN_INTERVAL_MIN} min; blocks_per_hour={BLOCKS_PER_HOUR}; spacing={BLOCK_SPACING_MIN} min")
    while True:
        try:
            if BLOCKS_PER_HOUR <= 1:
                # chế độ cũ: mỗi SCAN_INTERVAL_MIN chạy 1 lần
                scan_once_for_logs()
                logging.info("[scan] done (single)")
                time.sleep(SCAN_INTERVAL_MIN * 60)
            else:
                # chạy block 0..(n-1), cách nhau BLOCK_SPACING_MIN
                for i in range(BLOCKS_PER_HOUR):
                    scan_once_for_logs(block_idx=i)
                    logging.info(f"[scan] done block {i+1}/{BLOCKS_PER_HOUR}")
                    if i < BLOCKS_PER_HOUR - 1:
                        time.sleep(BLOCK_SPACING_MIN * 60)
                # ngủ phần còn lại của giờ
                total_span = (BLOCKS_PER_HOUR - 1) * BLOCK_SPACING_MIN
                remain = max(0, SCAN_INTERVAL_MIN - total_span)
                logging.info(f"[scheduler] sleeping {remain} min to complete the hour")
                time.sleep(remain * 60)
        except Exception as e:
            logging.exception(f"[scan] error: {e}")
            time.sleep(10)  # backoff nhẹ

@app.on_event("startup")
def _on_startup():
    t = threading.Thread(target=_scan_loop, daemon=True)
    t.start()
    app.state.scan_thread = t
    logging.info("[scheduler] thread spawned")

# ====== API ======
class ScanOnceReq(BaseModel):
    symbols: List[str] | None = None
    max_gpt: int | None = None
    block_idx: int | None = None   # NEW: chọn block cụ thể 0..BLOCKS_PER_HOUR-1

@app.post("/scan_once")
def api_scan_once(req: ScanOnceReq):
    global MAX_GPT
    if req.max_gpt is not None:
        MAX_GPT = req.max_gpt
    if req.symbols:
        structs = _build_structs_for(req.symbols)
        for sym in req.symbols:
            s1h = next((x["1H"] for x in structs if x["symbol"] == sym), None)
            s4h = next((x["4H"] for x in structs if x["symbol"] == sym), None)
            s1d = next((x["1D"] for x in structs if x["symbol"] == sym), None)
            out = make_telegram_signal(s4h, s1d, trigger_1h=s1h)
            print(json.dumps(out, ensure_ascii=False))
        return {"ok": True, "count": len(req.symbols)}
    # nếu gọi không truyền symbols: cho phép chọn block thủ công
    if req.block_idx is not None:
        scan_once_for_logs(block_idx=req.block_idx)
    else:
        scan_once_for_logs()
    return {"ok": True}

if __name__ == "__main__":
    # chạy 1 block duy nhất khi chạy trực tiếp (mặc định block 0)
    scan_once_for_logs(block_idx=0 if BLOCKS_PER_HOUR > 1 else None)
