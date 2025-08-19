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
    "symbol": "BTC/USDT",
    "decision": "ENTER | WAIT | AVOID",
    "side": "long | short",
    "confidence": 0.0,
    "strategy": "trend-follow | breakout | retest | reclaim | range | countertrend",
    "entry": [0.0],   # hoặc "entries"
    "entries": [0.0],
    "sl": 0.0,
    "tp": [0.0, 0.0],  # hoặc "tps"
    "tps": [0.0, 0.0],
    "reasons": ["..."],
    "trigger_hint": "nếu WAIT: nêu điều kiện kích hoạt",
    "leverage": None,
    "eta": None,
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


def _render_simple_signal(symbol: str, decision: Dict[str, Any]) -> str:
    """
    Format Telegram *chỉ dùng khi ENTER*:
    {Direction} | {Mã}
    Leverage:
    Entry:
    SL:
    TP:
    """
    side = (decision.get("side") or "long").lower()
    direction = "Long" if side == "long" else "Short"
    leverage = decision.get("leverage")
    entries = decision.get("entries") or decision.get("entry") or []
    tps = decision.get("tps") or decision.get("tp") or []
    sl = decision.get("sl")

    def _fmt_list(nums):
        if not nums:
            return "-"
        try:
            return ", ".join(f"{float(x):.6f}" for x in nums)
        except Exception:
            return ", ".join(str(x) for x in nums)

    body = [
        f"{direction} | {symbol.replace('/', '')}",
        f"Leverage: {leverage if leverage is not None else '-'}",
        f"Entry: {_fmt_list(entries)}",
        f"SL: {('-' if sl is None else (f'{float(sl):.6f}' if isinstance(sl, (int, float, str)) else str(sl)))}",
        f"TP: {_fmt_list(tps)}",
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
            "Bạn là trader kỹ thuật. Hãy phân loại một mã thành ENTER / WAIT / AVOID "
            "dựa trên JSON 3 khung **1D / 4H / 1H (đầy đủ)**.\n"
            "Quy tắc:\n"
            "- ENTER: 1D–4H–1H đồng pha và có xác nhận (breakout/reclaim/retest) với volume ủng hộ; "
            "R:R hợp lý theo targets/levels → cung cấp Entry/SL/TP.\n"
            "- WAIT: xu hướng lớn ủng hộ nhưng thiếu xác nhận 1H → chỉ log, kèm trigger_hint (điểm/kịch bản kích hoạt cụ thể).\n"
            "- AVOID: đi ngược 1D rõ rệt, hoặc R:R xấu/levels tắc/thiếu thanh khoản → log lý do ngắn.\n"
            "Chấp nhận RSI>70/<30 nếu đi cùng breakout có volume (không loại oan). "
            "Chỉ trả JSON theo schema sau (tiếng Việt):\n"
            + json.dumps(CLASSIFY_SCHEMA, ensure_ascii=False)
        ),
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Context JSON (1D/4H/1H):"},
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
    - Gọi GPT-4o với 1H/4H/1D đầy đủ.
    - Nếu ENTER: tạo telegram_text *đơn giản* (Direction|Mã, Leverage, Entry, SL, TP) + analysis_text để log.
    - Nếu WAIT/AVOID: KHÔNG tạo telegram_text; chỉ trả analysis_text ngắn gọn (WAIT có trigger_hint).
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

        # Chuẩn hoá fields
        symbol = (
            data.get("symbol")
            or _safe(struct_4h, "symbol")
            or _safe(struct_1d, "symbol")
            or "SYMBOL"
        )
        decision_str = (data.get("decision") or data.get("action") or "WAIT").upper()
        side = (data.get("side") or "long").lower()

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
            "leverage": data.get("leverage"),
            "eta": data.get("eta"),
        }

        # Tạo output theo 3 nhánh
        telegram_text = None
        analysis_text = _analysis_text(symbol, decision)

        if decision_str == "ENTER":
            telegram_text = _render_simple_signal(symbol, decision)
        elif decision_str in ("WAIT", "AVOID"):
            telegram_text = None
        else:
            # Bất ngờ/khác → coi như WAIT
            decision["decision"] = "WAIT"
            telegram_text = None

        # Gợi ý "plan" tối giản cho dòng post/track
        plan = None
        if decision["decision"] == "ENTER":
            plan = {
                "signal_id": f"{symbol.replace('/', '')}-{int(time.time())}",
                "timeframe": "4H",
                "side": decision["side"],
                "strategy": decision.get("strategy") or "GPT-plan",
                "entries": decision.get("entries") or [],
                "sl": decision.get("sl"),
                "tps": decision.get("tps") or [],
                "leverage": decision.get("leverage"),
                "eta": decision.get("eta"),
            }

        return {
            "ok": True,
            "signal_id": plan and plan["signal_id"],
            "telegram_text": telegram_text,   # chỉ có khi ENTER
            "analysis_text": analysis_text,   # luôn có
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
                "leverage": decision.get("leverage"),
                "eta": decision.get("eta"),
            },
            "plan": plan,   # chỉ khi ENTER
            "meta": {
                "confidence": decision.get("confidence"),
                "eta": decision.get("eta"),
            },
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
