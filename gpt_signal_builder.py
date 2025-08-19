# gpt_signal_builder.py
import os
import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ==== Debug helpers (from performance_logger) ====
# Bật bằng ENV: DEBUG_GPT_INPUT=1 (và tuỳ chọn DEBUG_GPT_DIR)
try:
    from performance_logger import debug_dump_gpt_input, debug_print_gpt_input
except Exception:
    # fallback no-op nếu chưa có helper (tránh lỗi import khi chạy unit test)
    def debug_dump_gpt_input(symbol: str, ctx: Dict[str, Any], tag: str = "ctx") -> None: ...
    def debug_print_gpt_input(ctx: Dict[str, Any]) -> None: ...


# ====== Config ======
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # dùng gpt-4o theo yêu cầu
client = OpenAI()

# ====== Schema output mong đợi từ GPT ======
CLASSIFY_SCHEMA = {
    "symbol": "CTG",
    "decision": "ENTER | WAIT | AVOID",
    "confidence": 0.0,
    "strategy": "trend-follow | breakout | retest | reclaim | range | countertrend",
    "entry": [0.0],
    "entries": [0.0],
    "sl": 0.0,
    "tp": [0.0, 0.0],
    "tps": [0.0, 0.0],
    "reasons": ["..."],
    "trigger_hint": "nếu WAIT: nêu điều kiện kích hoạt",
    "tp1_eta_bars": 3
}
# ====== Helpers ======
def _safe(d: Optional[Dict], *keys, default=None):
    cur = d or {}
    for k in keys:
        if cur is None:
            return default
        cur = cur.get(k)
    return default if cur is None else cur


def _parse_json_from_text(txt: str) -> Dict[str, Any]:
    """
    Cố gắng tách JSON từ nội dung GPT trả về (hỗ trợ có/không code fence).
    """
    t = (txt or "").strip()

    # code fence
    if t.startswith("```"):
        # loại bỏ fence, cả '```json'
        t = t.strip().strip("`")
        if t.lower().startswith("json"):
            p = t.find("{")
            if p != -1:
                t = t[p:]

    # tìm block {...} lớn nhất
    try:
        start, end = t.find("{"), t.rfind("}")
        if start >= 0 and end > start:
            return json.loads(t[start:end + 1])
    except Exception:
        pass

    # fallback: thử parse trực tiếp
    try:
        return json.loads(t)
    except Exception:
        return {}





# ====== Exchange-aware price formatting ======
def _resolve_exchange(symbol: str) -> str:
    # 1) ENV VN_EXCHANGE_MAP (JSON: {"FPT":"HOSE","SSI":"HOSE",...})
    # 2) ENV DEFAULT_EXCHANGE (mặc định "HOSE")
    import os, json
    default_ex = os.getenv("DEFAULT_EXCHANGE", "HOSE").upper()
    try:
        mapping = json.loads(os.getenv("VN_EXCHANGE_MAP","{}"))
        mapping = {k.upper(): (v or "").upper() for k,v in mapping.items()}
    except Exception:
        mapping = {}
    return mapping.get(symbol.replace("/", "").upper(), default_ex)

def _exchange_decimals(exchange: str) -> int:
    # Default decimals by exchange; override via ENV EXCHANGE_DECIMALS_JSON
    import os, json
    default = {"HOSE":0, "HSX":0, "HNX":0, "UPCOM":0, "UPCoM":0, "DERIV":1, "FUTURES":1}
    try:
        cfg = json.loads(os.getenv("EXCHANGE_DECIMALS_JSON","{}"))
        for k,v in cfg.items():
            default[str(k).upper()] = int(v)
    except Exception:
        pass
    return default.get((exchange or "").upper(), 0)

def _fmt_price_by_exchange(x, symbol: str) -> str:
    ex = _resolve_exchange(symbol)
    dec = _exchange_decimals(ex)
    try:
        fx = float(x)
        if dec <= 0:
            return f"{int(round(fx))}"
        return f"{fx:.{dec}f}"
    except Exception:
        return str(x)

def _fmt_price_list_by_exchange(nums, symbol: str) -> str:
    if not nums:
        return "-"
    return ", ".join(_fmt_price_by_exchange(x, symbol) for x in nums)
def _render_simple_signal(symbol: str, decision: Dict[str, Any]) -> str:
    """
    Format Telegram (VN stock):
    {Mã}
    Entry:
    SL:
    TP:
    """
    entries = decision.get("entries") or decision.get("entry") or []
    tps = decision.get("tps") or decision.get("tp") or []
    sl = decision.get("sl")

    def _fmt_list_int(nums):
        if not nums:
            return "-"
        out = []
        for x in nums:
            try:
                out.append(f"{int(float(x))}")
            except Exception:
                out.append(str(x))
        return ", ".join(out)

    def _fmt_one_int(x):
        if x is None:
            return "-"
        try:
            return f"{int(float(x))}"
        except Exception:
            return str(x)

    body = [
        f"{symbol.replace('/', '')}",
        f"Entry: {_fmt_price_list_by_exchange(entries, symbol)}",
        f"SL: {_fmt_price_by_exchange(sl, symbol)}",
        f"TP: {_fmt_price_list_by_exchange(tps, symbol)}",
    ]
    return "\n".join(body)


def _analysis_text(symbol: str, decision: Dict[str, Any]) -> str:
    """
    Text phân tích cho log:
    - ENTER: list reasons gọn.
    - WAIT: nêu reasons + trigger_hint.
    - AVOID: reasons ngắn.
    """
    act = (decision.get("decision") or decision.get("action") or "").upper()
    side = (decision.get("side") or "long").lower()
    reasons = decision.get("reasons") or []
    hint = decision.get("trigger_hint")

    # gom bullets 3–8 dòng tối đa
    bullets = []
    for r in reasons[:8]:
        if isinstance(r, str) and r.strip():
            bullets.append(f"- {r.strip()}")

    if act == "WAIT" and hint:
        bullets.append(f"- Trigger: {hint}")

    hdr = f"[ĐÁNH GIÁ] {symbol} | {act} | side={side} | conf={decision.get('confidence')}"
    if bullets:
        return hdr + "\n" + "\n".join(bullets)
    return hdr


# ====== Prompt xây dựng ======
def build_messages_classify(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    trigger_1h: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Giữ API cũ: trigger_1h là *full struct 1H*.
    """
    ctx = {
        "struct_4h": struct_4h,
        "struct_1d": struct_1d,
        "struct_1h": (trigger_1h or {}),
    }

    # === DEBUG: in & ghi JSON đầu vào GPT khi DEBUG_GPT_INPUT=1 ===
    try:
        sym_for_dump = (
            _safe(struct_4h, "symbol")
            or _safe(struct_1d, "symbol")
            or _safe(trigger_1h, "symbol")
            or "SYMBOL"
        )
        debug_print_gpt_input(ctx)
        debug_dump_gpt_input(sym_for_dump, ctx, tag="ctx")
    except Exception:
        # không để debug làm hỏng flow chính
        pass

    system = {
        "role": "system",
        "content": (
            "Bạn là trader kỹ thuật. Hãy phân loại một mã thành ENTER / WAIT / AVOID dựa trên JSON 3 khung **1D / 1W / 1H (đầy đủ)**.
Quy tắc:
- ENTER: 1D–1W–1H đồng pha và có xác nhận (breakout/reclaim/retest) với volume ủng hộ; R:R hợp lý theo targets/levels → cung cấp Entry/SL/TP.
- WAIT: xu hướng lớn ủng hộ nhưng thiếu xác nhận 1H → chỉ log, kèm trigger_hint (điểm/kịch bản kích hoạt cụ thể).
- AVOID: đi ngược 1D rõ rệt, hoặc R:R xấu/levels tắc/thiếu thanh khoản → log lý do ngắn.
Chỉ tư vấn **LONG** (không đưa ra SHORT, không dùng leverage).
Định dạng giá theo **sàn**: HOSE/HNX/UPCoM dùng **số nguyên VND**; phái sinh (nếu có) dùng **1 chữ số thập phân**; nếu không rõ thì mặc định số nguyên.
JSON đầu ra (tiếng Việt) phải theo schema sau, có trường `tp1_eta_bars` là **số phiên 1D** ước tính để chạm TP1:
" + json.dumps(CLASSIFY_SCHEMA, ensure_ascii=False)
        ),
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Context JSON (1D / 1W / 1H):"},
            {"type": "text", "text": json.dumps(ctx, ensure_ascii=False)},
        ],
    }
    return [system, user]


# ====== Hàm chính: trả về telegram_text/analysis_text theo yêu cầu ======

def make_telegram_signal(
    struct_4h: Dict[str, Any],
    struct_1d: Dict[str, Any],
    trigger_1h: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    - Gọi GPT-4o với 1H/1W/1D đầy đủ. LONG-only. Không leverage.
    - Quy tắc gửi Telegram:
        * ENTER **hoặc** WAIT, nếu `tp1_eta_bars` < 5 phiên → gửi setup (Mã, Entry, SL, TP) + log.
        * WAIT với `tp1_eta_bars` >= 5 hoặc không có → chỉ log hiện trạng + trigger.
        * AVOID → chỉ log hiện trạng + lý do ngắn.
    """
    try:
        msgs = build_messages_classify(struct_4h, struct_1d, trigger_1h=trigger_1h)
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=msgs,
            temperature=0.0,
            max_tokens=1800,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _parse_json_from_text(raw)

        if not isinstance(data, dict) or not data:
            return {"ok": False, "error": "GPT không trả JSON hợp lệ", "raw": raw}

        symbol = (
            data.get("symbol")
            or _safe(struct_4h, "symbol")
            or _safe(struct_1d, "symbol")
            or "SYMBOL"
        )
        decision_str = (data.get("decision") or data.get("action") or "WAIT").upper()
        side = "long"  # long-only

        decision: Dict[str, Any] = {
            "decision": decision_str,
            "side": side,
            "confidence": data.get("confidence"),
            "strategy": data.get("strategy"),
            "entries": data.get("entries") or data.get("entry") or [],
            "sl": data.get("sl"),
            "tps": data.get("tps") or data.get("tp") or [],
            "reasons": data.get("reasons") or (data.get("analysis") and [data["analysis"]] or []),
            "trigger_hint": data.get("trigger_hint"),
            "tp1_eta_bars": _get_tp1_eta_bars(data),
        }

        analysis_text = _analysis_text(symbol, decision)
        telegram_text = None

        eta_bars = decision.get("tp1_eta_bars")
        if decision_str in ("ENTER", "WAIT") and (eta_bars is not None) and (eta_bars < 5):
            telegram_text = _render_simple_signal(symbol, decision)
        elif decision_str == "ENTER":
            # ENTER nhưng không có ETA -> vẫn gửi
            telegram_text = _render_simple_signal(symbol, decision)
        else:
            telegram_text = None

        plan = None
        if telegram_text:
            import time as _t
            plan = {
                "signal_id": f"{symbol.replace('/', '')}-{int(_t.time())}",
                "timeframe": "1D",
                "side": side,
                "strategy": decision.get("strategy") or "GPT-plan",
                "entries": decision.get("entries") or [],
                "sl": decision.get("sl"),
                "tps": decision.get("tps") or [],
                "leverage": None,
                "eta": eta_bars,
            }

        return {
            "ok": True,
            "signal_id": plan and plan["signal_id"],
            "telegram_text": telegram_text,
            "analysis_text": analysis_text,
            "decision": {
                "action": decision["decision"],
                "side": decision["side"],
                "confidence": decision.get("confidence"),
                "strategy": decision.get("strategy"),
                "entries": decision.get("entries"),
                "sl": decision.get("sl"),
                "tps": decision.get("tps"),
                "reasons": decision.get("reasons"),
                "trigger_hint": decision.get("trigger_hint"),
                "tp1_eta_bars": eta_bars,
            },
            "plan": plan,
            "meta": {
                "confidence": decision.get("confidence"),
                "tp1_eta_bars": eta_bars,
            },
        }


    except Exception as e:
        return {"ok": False, "error": str(e)}


def _get_tp1_eta_bars(data: Dict[str, Any]) -> Optional[int]:
    v = data.get("tp1_eta_bars") or data.get("tp1_eta") or data.get("eta_bars")
    try:
        return int(v) if v is not None else None
    except Exception:
        return None
