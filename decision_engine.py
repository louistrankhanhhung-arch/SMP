"""
decision_engine.py — LONG-only trade planner for Vietnam equities
-----------------------------------------------------------------
Contract
--------
decide(features_by_tf: dict, evidence: dict | None = None, *, cfg: dict | None = None) -> dict

Inputs
------
- features_by_tf: output of feature_primitives.compute_features_by_tf
- evidence (optional): output of evidence_evaluators.evaluate; if omitted, we compute internally

Output (example)
----------------
{
  "symbol": "VCB",
  "timeframe_primary": "1D",
  "STATE": "breakout",
  "DIRECTION": "LONG",
  "DECISION": "ENTER" | "WAIT" | "AVOID",
  "entry": 89900.0,
  "entry2": null,
  "sl": 87200.0,
  "tp1": 91000.0, "tp2": ..., "tp3": ..., "tp4": ..., "tp5": ...,
  "rr": 2.1,           # RR to tp1
  "rr2": 3.5,          # RR to tp2 (legacy field)
  "missing": [ ... ],  # reasons for WAIT/AVOID
  "notes":   [ ... ],  # evaluator notes
  "confirmations": {"volume": true, "momentum": false, "candles": true}
}
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np
import os

# Optional import: we only need evaluate if caller didn't pass evidence
try:
    from evidence_evaluators import evaluate as _evaluate
except Exception:
    _evaluate = None  # type: ignore

# --- Logging shim: đảm bảo có log_info/log_warn/log_err ---
try:
    # Nếu bạn có module riêng, import ở đây (tùy codebase của bạn):
    # from log_utils import log_info, log_warn, log_err
    raise ImportError  # ép dùng shim bên dưới nếu chưa có
except Exception:
    import logging, sys
    _logger = logging.getLogger("signals")
    if not _logger.handlers:
        _h = logging.StreamHandler(sys.stdout)
        _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        _logger.addHandler(_h)
        _logger.setLevel(logging.INFO)

    def log_info(msg: str) -> None:
        _logger.info(msg)

    def log_warn(msg: str) -> None:
        _logger.warning(msg)

    def log_err(msg: str) -> None:
        _logger.error(msg)
# --- end shim ---

# --- DEBUG VALIDATORS FLAG ---
try:
    import os
    DEBUG_VALIDATORS = bool(int(os.getenv("DEBUG_VALIDATORS", "0")))
except Exception:
    DEBUG_VALIDATORS = False

def _debug_dump_validators(symbol: str, f1d: dict, confirmations: dict):
    if not DEBUG_VALIDATORS:
        return
    try:
        v = {
            "vol": f1d.get("vol"),
            "vol_ma20": f1d.get("vol_ma20"),
            "vol_ratio": f1d.get("vol_ratio"),
            "vol_z": f1d.get("vol_z"),
            "rsi14": f1d.get("rsi14"),
            "macd_hist": f1d.get("macd_hist"),
            "body_pct": f1d.get("body_pct"),
            "atr14": f1d.get("atr14"),
        }
        conf = confirmations or {}
        log_info(f"[{symbol}] DEBUG_VALIDATORS V={conf.get('volume',False)} M={conf.get('momentum',False)} C={conf.get('candles',False)} | "
                 f"vol={v['vol']} vol_ma20={v['vol_ma20']} vol_ratio={v['vol_ratio']} vol_z={v['vol_z']} "
                 f"rsi14={v['rsi14']} macd_hist={v['macd_hist']} body_pct={v['body_pct']} atr14={v['atr14']}")
    except Exception:
        pass

def _safe_eval(features_all):
    if _evaluate is None:
        log_warn("Evaluator not available: evidence_evaluators.evaluate import failed.")
        return None
    try:
        # TH1: evaluator nhận keyword 'features_by_tf'
        return _evaluate(features_by_tf=features_all)
    except TypeError:
        # TH2: evaluator chỉ nhận 1 positional
        try:
            return _evaluate(features_all)
        except Exception as e:
            log_warn(f"Evaluator raised: {e}")
            return None
    except Exception as e:
        log_warn(f"Evaluator raised: {e}")
        return None

# =========================
# Default configuration
# =========================
DEFAULT_CFG = {
    # Risk buffer & SL logic
    "atr_sl_mult": 1.0,           # SL = swing_low - atr_mult * ATR (clamped)
    "ema_sl_pref": "ema20",       # prefer using ema20 or ema50 as base for SL buffer
    "sl_min_pct": 1.0,            # ensure SL at least 1% below entry
    "sl_max_pct": 8.0,            # cap extreme SL to avoid huge risk
    # Take-profit ladder (multiples of R)
    "tp_multipliers": [1.0, 1.5, 2.0, 2.5, 3.0],
    # Entry policy per state
    "entry_policy": {
        "breakout": "close",          # market @ close
        "reclaim": "close",
        "pullback_to_ema": "close",
        "squeeze_expansion": "close",
        "mean_revert": "close",
        "breakdown": "skip",          # ignored (we are LONG-only)
        "reject": "skip",
        "bullish_potential": "close",
        None: "skip",
    },
    # Minimum confidence required to ENTER
    "min_conf_enter": 0.55,
    # --- New: ngưỡng gating bổ sung (đồng bộ với evidence_evaluators)
    "rsi_breakout_min": 60.0,
    "bb_pctb_breakout_min": 0.80,
    # Require validator OR-gate to pass (volume or momentum/candles)
    "require_validation": True,
    # --- Fallback for early bullish reversal when evaluator returns None/bearish
    "allow_bullish_potential": True,
    "fallback_rsi_trigger": 50.0,
    "fallback_ema20_slope_min": 0.0,
    "weekly_uptrend_check": True,

    # Use primary timeframe for execution
    "primary_tf": "1D",
    # Reclaim proximity tolerance (price not too extended from EMA)
    "max_dist_ema20_pct_for_reclaim": 6.0,
    # Emit setup even if DECISION=WAIT for these states
    "emit_plan_on_wait_states": ["bullish_potential"],
    # -----------------------------
    # RELAXED MODE (with guardrails)
    # -----------------------------
    "relaxed_enabled": True,
    # chỉ cho một số state được nới:
    "relaxed_allowed_states": ["bullish_potential", "pullback_to_ema", "reclaim"],
    # rào chắn:
    "relaxed_min_rr": 1.5,               # RR tới TP1 tối thiểu
    "relaxed_max_dist_ema20": 6.0,       # % cách EMA20 tối đa
    "relaxed_max_pctb": 0.90,            # tránh overextension sát BBupper
    "relaxed_weekly_required": True,     # weekly phải ủng hộ (stacked_bull hoặc slope>0)
    "relaxed_risk_size_cap": 0.5,        # giảm size tối đa còn 50%
}

# --- PATCH: optional DEBUG + format helpers ---
DEBUG_VALIDATORS = bool(int(os.getenv("DEBUG_VALIDATORS", "0")))

def _fmt_val(x, nd=2):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "nan"
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        return f"{float(x):.{nd}f}"
    except Exception:
        return "nan"

def _build_validator_line(rep: dict) -> str:
    """Trả về chuỗi ngắn gọn: V: ... | M: ... | C: ... với actual vs threshold và pass/fail"""
    if not isinstance(rep, dict):
        return "V/M/C: n/a"
    # Volume
    v = rep.get("volume", {})
    v_a = v.get("actuals", {}); v_t = v.get("thresholds", {}); v_sc = v.get("sub_criteria", {})
    v_str = (f"vol_ratio={_fmt_val(v_a.get('vol_ratio'))}>={_fmt_val(v_t.get('vol_ratio_ok'))} "
             f"({ '✓' if v_sc.get('vol_ratio_ok') else '×' }), "
             f"z={_fmt_val(v_a.get('vol_z'))}>={_fmt_val(v_t.get('vol_z_ok'))} "
             f"({ '✓' if v_sc.get('vol_z_ok') else '×' }) → "
             f"{'PASS' if v.get('passed') else 'FAIL'}")
    # Momentum
    m = rep.get("momentum", {})
    m_a = m.get("actuals", {}); m_t = m.get("thresholds", {}); m_sc = m.get("sub_criteria", {})
    m_str = (f"ΔMACD={_fmt_val(m_a.get('macd_delta'), 4)}>{_fmt_val(m_t.get('macd_hist_delta_min'), 4)} "
             f"({ '✓' if m_sc.get('macd_delta_ok') else '×' }), "
             f"RSI={_fmt_val(m_a.get('rsi'))} vs trig={_fmt_val(m_t.get('rsi_fast_trigger'))} "
             f"({ '✓' if m_sc.get('rsi_bias_ok') else '×' }) → "
             f"{'PASS' if m.get('passed') else 'FAIL'}")
    # Candles
    c = rep.get("candles", {})
    c_a = c.get("actuals", {}); c_t = c.get("thresholds", {}); c_sc = c.get("sub_criteria", {})
    # hiển thị range/ATR theo dạng k*ATR
    rng = c_a.get("range"); atr = c_a.get("atr14")
    k = (rng/atr) if (isinstance(rng,(int,float)) and isinstance(atr,(int,float)) and atr not in (0, np.nan)) else np.nan
    c_str = (f"range={_fmt_val(rng)} ≈ {_fmt_val(k)}*ATR (min={_fmt_val(c_t.get('atr_push_min'))}) "
             f"({ '✓' if c_sc.get('push_ok') else '×' }), "
             f"body={_fmt_val(c_a.get('body_pct'))}>={_fmt_val(c_t.get('body_pct_ok'))} "
             f"({ '✓' if c_sc.get('body_ok') else '×' }) → "
             f"{'PASS' if c.get('passed') else 'FAIL'}")
    return f"V: {v_str} | M: {m_str} | C: {c_str}"

def _build_checklist(rep: dict) -> str:
    """Checklist tiêu chí đã ĐẠT/CHƯA ở từng nhánh — dùng icon ✓/×, ngắn gọn."""
    if not isinstance(rep, dict):
        return ""
    items = []
    v_sc = rep.get("volume", {}).get("sub_criteria", {})
    items.append(("vol_ratio_ok", v_sc.get("vol_ratio_ok", False)))
    items.append(("vol_z_ok", v_sc.get("vol_z_ok", False)))
    m_sc = rep.get("momentum", {}).get("sub_criteria", {})
    items.append(("macd_delta_ok", m_sc.get("macd_delta_ok", False)))
    items.append(("rsi_bias_ok", m_sc.get("rsi_bias_ok", False)))
    c_sc = rep.get("candles", {}).get("sub_criteria", {})
    items.append(("push_ok", c_sc.get("push_ok", False)))
    items.append(("body_ok", c_sc.get("body_ok", False)))
    met   = [k for k, ok in items if ok]
    miss  = [k for k, ok in items if not ok]
    return f"met={met} | missing={miss}"


# =========================
# Small helpers
# =========================
def _get(d: dict, k: str, default=np.nan):
    return d.get(k, default) if isinstance(d, dict) else default

def _pct(a, b) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return (a / b - 1.0) * 100.0

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _pick_base_sl(f1d: dict, ema_pref_key: str) -> float:
    # Base = min(ema_pref, lo20, lo50) — honor runtime cfg
    ema_pref = f1d.get(ema_pref_key, np.nan)
    lo20 = _get(f1d, "lo20", np.nan)
    lo50 = _get(f1d, "lo50", np.nan)
    candidates = [x for x in [ema_pref, lo20, lo50] if not np.isnan(x)]
    return float(min(candidates)) if candidates else float("nan")

def _compute_sl(entry: float, f1d: dict, *, atr_mult: float, sl_min_pct: float, sl_max_pct: float, ema_pref_key: str) -> float:
    base = _pick_base_sl(f1d, ema_pref_key)
    atr = _get(f1d, "atr14", np.nan)
    sl = base if not np.isnan(base) else (entry - 2.0 * (entry * sl_min_pct / 100.0))
    if not np.isnan(atr):
        sl = min(sl, entry - atr_mult * atr)
    # enforce min/max percent from entry
    min_gap = entry * (sl_min_pct / 100.0)
    max_gap = entry * (sl_max_pct / 100.0)
    sl = _clamp(sl, entry - max_gap, entry - min_gap)
    return float(sl)

def _tp_from_r(entry: float, sl: float, multipliers: List[float]) -> List[float]:
    R = entry - sl
    return [float(entry + m * R) for m in multipliers]

def _approx_bands(f1d: dict) -> tuple[float, float, float]:
    close = _get(f1d, "close", np.nan)
    ema20 = _get(f1d, "ema20", np.nan)
    atr14 = _get(f1d, "atr14", np.nan)
    if np.isnan(ema20) or np.isnan(atr14):
        return np.nan, np.nan, np.nan
    mid = float(ema20)
    up  = float(ema20 + 2.0 * atr14)
    lo  = float(ema20 - 2.0 * atr14)
    return lo, mid, up

def _dynamic_tp_multipliers(f1d: dict, overext_pct: float = 6.0) -> list[float]:
    close = _get(f1d, "close", np.nan)
    ema20 = _get(f1d, "ema20", np.nan)
    if np.isnan(close) or np.isnan(ema20):
        return [1.0, 1.5, 2.0, 2.5, 3.0]
    dist_pct = abs(close - ema20) / ema20 * 100.0
    if dist_pct >= overext_pct:
        return [0.8, 1.2, 1.8, 2.4, 3.0]
    return [1.0, 1.5, 2.0, 2.5, 3.0]

def _plan_by_state(state: Optional[str], f1d: dict, ev: dict, cfg: dict):
    """
    Trả về: entry, entry2, sl, tps(list 5), risk_size_hint(float), notes(list)
    """
    notes: list[str] = []
    rsh = 1.0  # mặc định full size
    close = _get(f1d, "close", np.nan)
    ema20 = _get(f1d, "ema20", np.nan)
    ema50 = _get(f1d, "ema50", np.nan)
    atr14 = _get(f1d, "atr14", np.nan)
    lo, mid, up = _approx_bands(f1d)
    if np.isnan(close):
        # fallback cứng để không crash
        entry = entry2 = close
        sl = close * (1 - cfg["sl_min_pct"]/100.0)
        tps = _tp_from_r(entry, sl, cfg["tp_multipliers"])
        return entry, None, sl, tps, rsh, ["fallback: NaN close"]

    vol_ok = bool((ev.get("confirmations") or {}).get("volume", False))
    dist20 = _pct(close, ema20) if not np.isnan(ema20) else np.nan
    tp_mult = _dynamic_tp_multipliers(f1d, overext_pct=cfg.get("max_dist_ema20_pct_for_reclaim", 6.0))

    st = (state or "").lower()

    if st in ("breakout", "squeeze_expansion"):
        entry  = float(close)
        entry2 = (close - 0.5 * atr14) if not np.isnan(atr14) else None
        rsh = 0.6 if vol_ok else 0.5
        notes.append("Plan breakout/squeeze: probe at market, add on -0.5*ATR pullback")

    elif st == "reclaim":
        # nếu xa EMA20 thì chờ retest; còn không dùng close
        if not np.isnan(dist20) and dist20 > cfg["max_dist_ema20_pct_for_reclaim"]:
            entry = float(ema20) if not np.isnan(ema20) else float(close)
            entry2 = (ema20 - 0.2 * atr14) if (not np.isnan(ema20) and not np.isnan(atr14)) else None
            notes.append("Plan reclaim: far-from-EMA → wait retest EMA20")
        else:
            entry = float(close)
            entry2 = float(ema20) if not np.isnan(ema20) else None
            notes.append("Plan reclaim: near-EMA → allow market + EMA20 add")
        rsh = 0.8

    elif st == "pullback_to_ema":
        base = mid if not np.isnan(mid) else ema20
        entry  = float(base) if not np.isnan(base) else float(close)
        entry2 = (entry - 0.5 * atr14) if not np.isnan(atr14) else None
        rsh = 1.0
        notes.append("Plan pullback_to_ema: limit at BB mid/EMA20, add at -0.5*ATR")

    elif st == "mean_revert":
        base = mid if not np.isnan(mid) else ema20
        e0 = (close - 0.8 * atr14) if not np.isnan(atr14) else close
        entry  = float(max(base, e0)) if not np.isnan(base) else float(e0)
        entry2 = float(ema20) if not np.isnan(ema20) else None
        rsh = 0.5
        notes.append("Plan mean_revert: avoid chasing; probe lower + add at EMA20")
        tp_mult = _dynamic_tp_multipliers(f1d)  # thường rút ngắn vì overextension

    elif st == "bullish_potential":
        entry  = float(ema20) if not np.isnan(ema20) else float(close)
        entry2 = (entry - 0.3 * atr14) if not np.isnan(atr14) else None
        rsh = 0.5
        notes.append("Plan bullish_potential: cautious near EMA20")

    else:
        # default an toàn
        entry  = float(close)
        entry2 = None
        rsh = 0.8
        notes.append("Plan default: simple market entry (conservative size)")

    sl = _compute_sl(entry, f1d,
                     atr_mult=cfg["atr_sl_mult"],
                     sl_min_pct=cfg["sl_min_pct"],
                     sl_max_pct=cfg["sl_max_pct"],
                     ema_pref_key=cfg["ema_sl_pref"])
    tps = _tp_from_r(entry, sl, tp_mult)
    return float(entry), (float(entry2) if entry2 is not None else None), float(sl), tps, float(rsh), notes

# =========================
# Core planner
# =========================
def _should_enter_long(ev: dict, f1d: dict, cfg: dict, missing: List[str]) -> bool:
    if ev is None:
        missing.append("no_evidence")
        return False
    state = ev.get("state", None)
    if state in ("breakdown", "reject") or state is None:
        missing.append("bearish_or_none_state")
        return False
    if cfg.get("require_validation", True):
        val = ev.get("confirmations", {}) or {}
        st  = (ev.get("state") or "").lower()
        # Mặc định: Volume OR Momentum OR Candles
        gate_ok = bool(any(val.values()))
        # Siết riêng cho breakout: Volume OR (Momentum AND Candles)
        if st == "breakout":
            gate_ok = bool(val.get("volume") or (val.get("momentum") and val.get("candles")))
        # Khuyến nghị: với squeeze_expansion, yêu cầu có Candles (ATR push) hoặc Volume
        if st == "squeeze_expansion":
            gate_ok = bool(val.get("volume") or val.get("candles"))
            if not gate_ok:
                missing.append("breakout_requires_volume OR (momentum AND candles)")
        if not gate_ok:
            missing.append("validators_not_ok")
            return False
    conf = float(ev.get("confidence", 0.0))
    if conf < cfg["min_conf_enter"]:
        missing.append(f"low_confidence<{cfg['min_conf_enter']}")
        return False
    # Thêm filter nhỏ cho breakout: RSI & vị trí trên BB
    if (ev.get("state") or "").lower() == "breakout":
        rsi = float(_get(f1d, "rsi14", float("nan")))
        pctb = float(_get(f1d, "bb_pctb", float("nan")))
        if (not np.isnan(rsi)) and rsi < cfg["rsi_breakout_min"]:
            missing.append("rsi_breakout_min")
            return False
        if (not np.isnan(pctb)) and pctb < cfg["bb_pctb_breakout_min"]:
            missing.append("bb_pctb_breakout_min")
            return False
    # If reclaim too extended away from ema20, skip to avoid chasing
    if state == "reclaim":
        dist20 = _pct(_get(f1d, "close", np.nan), _get(f1d, "ema20", np.nan))
        if not np.isnan(dist20) and dist20 > cfg["max_dist_ema20_pct_for_reclaim"]:
            missing.append("reclaim_too_far_from_ema20")
            return False
    return True

def _require(d, keys, missing):
    import math
    for k in keys:
        v = d.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            missing.append(k)

def decide(features_by_tf: Dict[str, dict], evidence: dict | None = None, *, cfg: dict | None = None, sym: str = "?") -> dict:
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    d1 = features_by_tf.get(cfg["primary_tf"], {})
    df1d = d1.get("df", None)
    f1d = d1.get("features", {}) or {}
    # Compute evidence if not provided
    # Tính evidence an toàn
    ev = evidence
    if ev is None:
        ev = _safe_eval(features_by_tf)  # <- dòng này PHẢI thụt 4 spaces

    # ---------- Fallback: promote to 'bullish_potential' when evaluator is None/bearish ----------
    try:
        if isinstance(ev, dict) is False:
            ev = {}  # normalize
        ev_state = ev.get("state")
        if cfg.get("allow_bullish_potential", True) and (ev_state in (None, "reject", "breakdown")):
            d1 = features_by_tf.get(cfg["primary_tf"], {}) or {}
            f1d = (d1.get("features") or {})
            w1 = features_by_tf.get("1W", {}) or {}
            f1w = (w1.get("features") or {})

            close = float(f1d.get("close", float("nan")))
            ema20 = float(f1d.get("ema20", float("nan")))
            ema20_slope5 = float(f1d.get("ema20_slope5", float("nan")))
            rsi14 = float(f1d.get("rsi14", float("nan")))

            cond_price = (not np.isnan(close) and not np.isnan(ema20) and close > ema20)
            cond_momo = (
                (not np.isnan(ema20_slope5) and ema20_slope5 >= cfg.get("fallback_ema20_slope_min", 0.0))
                or (not np.isnan(rsi14) and rsi14 >= cfg.get("fallback_rsi_trigger", 50.0))
            )

            weekly_ok = True
            if cfg.get("weekly_uptrend_check", True):
                w_stack = bool(f1w.get("stacked_bull", False))
                w_slope = float(f1w.get("ema20_slope5", 0.0))
                weekly_ok = bool(w_stack or (w_slope > 0.0))

            if cond_price and cond_momo and weekly_ok:
                # Promote to a soft-bull state with conservative confidence
                ev = dict(ev or {})
                ev.setdefault("notes", []).append(
                    "fallback_bullish_potential: close>ema20 & (ema20_slope5>=min or RSI>=trigger) & weekly_ok"
                )
                ev["state"] = "bullish_potential"
                ev["direction"] = "LONG"
                ev["confidence"] = max(float(ev.get("confidence", 0.0)), 0.56)
    except Exception:
        # Never break the planner due to fallback errors
        pass

    # Build single unified 'out' (không ghi đè lần 2)
    ev_state = ev.get("state") if isinstance(ev, dict) else None
    ev_dir = ev.get("direction") if isinstance(ev, dict) else None
    # build validator benchmark strings up front so all return paths can include them
    try:
        rep = (ev.get("validator_report") if isinstance(ev, dict) else {}) or {}
        _validator_line = _build_validator_line(rep)
        _validator_checklist = _build_checklist(rep)
    except Exception: _validator_line = ""; _validator_checklist = ""
      
    out: Dict[str, Any] = {
        "symbol": sym,
        "timeframe_primary": cfg["primary_tf"],
        "STATE": ev_state,                  # UPPER for plan view
        "DIRECTION": "LONG",                # LONG-only engine
        "state": ev_state,                  # lower for legacy logs
        "dir":   ev_dir,                    # evaluator’s bias if any
        "DECISION": "WAIT",
        "entry": None, "entry2": None, "sl": None,
        "tp1": None, "tp2": None, "tp3": None, "tp4": None, "tp5": None,
        "rr": None, "rr2": None,
        "missing": [],
        "notes": (ev.get("notes") if isinstance(ev, dict) else []) or [],
        "confirmations": (ev.get("confirmations") if isinstance(ev, dict) else {}) or {},
        # --- NEW: export for top-level logger (main.py) ---
        "validator_line": _validator_line,
        "validator_checklist": _validator_checklist,
        "confidence": float(ev.get("confidence", 0.0)) if isinstance(ev, dict) else 0.0,
        "min_conf_enter": float(cfg.get("min_conf_enter", 0.55)),
        "scores": (ev.get("scores") if isinstance(ev, dict) else {}) or {},
    }

    # Quickly filter: if evaluator says SHORT strongly, we still keep LONG-only;
    # we just do not enter (remain WAIT). We don't change DIRECTION field.
    if isinstance(ev, dict) and ev.get("direction") == "SHORT":
        out["missing"].append("LONG-only engine: evaluator bias SHORT")

    # Decide ENTER/WAIT/AVOID
    # 1) Không có feature 1D -> thiếu dữ liệu
    if not f1d:
        out["missing"].append("missing_features")
        # Đẩy chi tiết lỗi/warning từ feature stage để debug nhanh
        err = d1.get("error")
        if err:
            out["missing"].append(f"feature_error:{err}")
        for w in (d1.get("warnings") or []):
            out["missing"].append(f"warn:{w}")
        _conf = out["confirmations"]
        V_ok = bool(_conf.get("volume", False))
        M_ok = bool(_conf.get("momentum", False))
        C_ok = bool(_conf.get("candles", False))
        out["confirm"] = {"V": V_ok, "M": M_ok, "C": C_ok}
        _debug_dump_validators(out["symbol"], f1d, out.get("confirmations", {}))
        reason = ",".join(out["missing"]) if out.get("missing") else "missing_features"
        notes = "; ".join(out.get("notes", [])) if out.get("notes") else ""
        # still log internally; plus we've exported strings above for main.py
        rep = ev.get("validator_report", {})
        bench_line = _build_validator_line(rep)
        checklist  = _build_checklist(rep)
        log_info(
            f"[{out['symbol']}] DECISION=WAIT | STATE={out.get('STATE')} | DIR={out.get('DIRECTION')} | "
            f"reason={reason} | confirm:V={V_ok} M={M_ok} C={C_ok} | {bench_line} | {checklist} | "
            f"notes={notes} | missing={list(out['missing'])}"
        )
        return out

    # 2) Có feature: cho _should_enter_long tự append các khóa thiếu vào out["missing"]
    can_enter = _should_enter_long((ev or {}), f1d, cfg or {}, out["missing"])
    # -----------------------------
    # RELAXED MODE (nếu gate fail)
    # -----------------------------
    if not can_enter and cfg.get("relaxed_enabled", True):
        st_lower = (out.get("STATE") or "").lower()
        if st_lower in set(cfg.get("relaxed_allowed_states", [])):
            # guardrails 1D/1W
            w1 = features_by_tf.get("1W", {}) or {}
            f1w = (w1.get("features") or {})
            close = float(_get(f1d, "close", float("nan")))
            ema20 = float(_get(f1d, "ema20", float("nan")))
            pctb  = float(_get(f1d, "bb_pctb", float("nan")))
            dist20 = abs(_pct(close, ema20))
            weekly_ok = True
            if cfg.get("relaxed_weekly_required", True):
                weekly_ok = bool(f1w.get("stacked_bull", False) or float(f1w.get("ema20_slope5", 0.0)) > 0.0)
            near_ema = (not np.isnan(dist20)) and (dist20 <= cfg.get("relaxed_max_dist_ema20", 6.0))
            not_overext = (not np.isnan(pctb)) and (pctb <= cfg.get("relaxed_max_pctb", 0.90))
            # preview kế hoạch để kiểm tra RR
            try:
                _entry, _entry2, _sl, _tps, _rsh, _notes = _plan_by_state(out["STATE"], f1d, (ev or {}), cfg)
                R = (_entry - _sl) if (_sl is not None) else float("nan")
                rr1 = ((_tps[0] - _entry) / R) if (R and R > 0) else float("nan")
            except Exception:
                rr1 = float("nan")
            rr_ok = (not np.isnan(rr1)) and (rr1 >= cfg.get("relaxed_min_rr", 1.5))
            relaxed_ok = bool(weekly_ok and near_ema and not_overext and rr_ok)
            if relaxed_ok:
                # bật cờ relaxed, cho phép vào lệnh tiếp tục theo flow chuẩn bên dưới
                out["notes"].append("validators_relaxed: weekly_ok & near_ema20 & not_overext & rr_ok")
                # đánh dấu để hạ size sau khi build plan
                out["__relaxed_pass__"] = True
                can_enter = True

    if not can_enter:
        _conf = out["confirmations"]
        V_ok = bool(_conf.get("volume", False))
        M_ok = bool(_conf.get("momentum", False))
        C_ok = bool(_conf.get("candles", False))
        out["confirm"] = {"V": V_ok, "M": M_ok, "C": C_ok}
        # Khi có setup (WAIT/SETUP), vẫn in benchmark để bạn theo dõi
        if DEBUG_VALIDATORS:
            rep = ev.get("validator_report", {})
            # Defaults to avoid 'nan>=nan' when validator_report is missing
            rep = rep or {}
            rep.setdefault("volume", {}).setdefault("thresholds", {"vol_ratio_ok": 1.2, "vol_z_ok": 0.5})
            rep.setdefault("momentum", {}).setdefault("thresholds", {"macd_hist_delta_min": 0.0, "rsi_fast_trigger": 55.0})
            rep.setdefault("candles", {}).setdefault("thresholds", {"atr_push_min": 0.6, "body_pct_ok": 0.35})
            # Expose to main.format_plan
            out["validator_report"] = rep
            bench_line = _build_validator_line(rep)
            checklist  = _build_checklist(rep)
            log_info(f"[{out['symbol']}] VALIDATORS | {bench_line} | {checklist}")
        _debug_dump_validators(out["symbol"], f1d, out.get("confirmations", {}))

        # NEW: vẫn phát hành setup khi WAIT nếu state được cho phép
        st_lower = (out.get("STATE") or "").lower()
        if st_lower and (st_lower in (cfg.get("emit_plan_on_wait_states") or [])):
            entry_policy = cfg["entry_policy"].get(out["STATE"], "close")
            price = float(_get(f1d, "close", np.nan))
            if entry_policy != "skip" and not np.isnan(price):
                entry, entry2, sl, tps, rsh, extra_notes = _plan_by_state(out["STATE"], f1d, ev or {}, cfg)
                if sl >= entry:  # safety clamp
                    sl = entry * (1.0 - cfg["sl_min_pct"]/100.0)
                out.update({
                    "entry": float(entry),
                    "entry2": (float(entry2) if entry2 is not None else None),
                    "sl": float(sl),
                    "tp1": tps[0], "tp2": tps[1], "tp3": tps[2], "tp4": tps[3], "tp5": tps[4],
                })
                # RR quick refs
                R = entry - sl
                out["rr"]  = float((tps[0] - entry) / R) if R > 0 else None
                out["rr2"] = float((tps[1] - entry) / R) if R > 0 else None
                # Risk-size hint nếu thiếu volume
                try:
                    vol_ok = bool(_conf.get("volume", False))
                    mc_ok  = bool(_conf.get("momentum", False) or _conf.get("candles", False))
                    valid_hint = (0.5 if ((not vol_ok) and mc_ok) else 1.0)
                    out["risk_size_hint"] = float(min(rsh or 1.0, valid_hint))
                except Exception:
                    pass
                out["notes"].append("setup_on_WAIT: emitted plan for bullish_potential (idea only)")
            # ensure validator strings present even if no setup emitted
            out.setdefault("validator_line", _validator_line)
            out.setdefault("validator_checklist", _validator_checklist)  

        reason = ",".join(out["missing"]) if out.get("missing") else "missing_features"
        notes = "; ".join(out.get("notes", [])) if out.get("notes") else ""
        log_info(
            f"[{out['symbol']}] DECISION=WAIT | STATE={out.get('STATE')} | DIR={out.get('DIRECTION')} | "
            f"reason={reason} | confirm:V={V_ok} M={M_ok} C={C_ok} | notes={notes} | missing={list(out['missing'])}"
        )
        return out

    # Build plan
    entry_policy = cfg["entry_policy"].get(out["STATE"], "close")
    price = float(_get(f1d, "close", np.nan))
    if entry_policy == "skip" or np.isnan(price):
        out["missing"].append("entry_policy_skip_or_nan_price")
        return out

    # State-aware planning (flexible entries)
    entry, entry2, sl, tps, rsh, extra_notes = _plan_by_state(out["STATE"], f1d, ev or {}, cfg)
    # Nếu pass nhờ RELAXED → giảm size + đảm bảo có entry2 kiểu pullback
    if out.pop("__relaxed_pass__", False):
        try:
            rsh = float(min(rsh or 1.0, cfg.get("relaxed_risk_size_cap", 0.5)))
            atr14 = _get(f1d, "atr14", np.nan)
            if entry2 is None and not np.isnan(atr14):
                entry2 = float(entry - 0.5 * atr14)  # ép có thang add ở -0.5*ATR
            out["notes"].append("relaxed_mode: cap size to 50% and prefer add @ -0.5*ATR")
        except Exception:
            pass
    if sl >= entry:  # safety
        sl = entry * (1.0 - cfg["sl_min_pct"]/100.0)

    out.update({
        "DECISION": "ENTER",
        "entry": float(entry),
        "entry2": (float(entry2) if entry2 is not None else None),
        "sl": float(sl),
        "tp1": tps[0], "tp2": tps[1], "tp3": tps[2], "tp4": tps[3], "tp5": tps[4],
    })

    # RR to TP1 and TP2 for quick reference (legacy fields rr, rr2)
    R = entry - sl
    out["rr"]  = float((tps[0] - entry) / R) if R > 0 else None
    out["rr2"] = float((tps[1] - entry) / R) if R > 0 else None
    
    # Gộp risk_size_hint từ kế hoạch & từ validator (thiếu Volume)
    plan_hint = rsh
    try:
        confs = out.get("confirmations", {})
        vol_ok = bool(confs.get("volume", False))
        mc_ok  = bool(confs.get("momentum", False) or confs.get("candles", False))
        valid_hint = (0.5 if ((not vol_ok) and mc_ok) else 1.0)
        final_hint = min(plan_hint or 1.0, valid_hint)
        if final_hint != 1.0:
            out["risk_size_hint"] = float(final_hint)
    except Exception:
        pass
    
    # Nhập notes bổ sung từ plan
    if extra_notes:
        out["notes"].extend(extra_notes)
    
    return out

if __name__ == "__main__":
    # minimal smoke test
    try:
        import pandas as pd, numpy as np
        from indicators import enrich_indicators
        from feature_primitives import compute_features_by_tf
        # Synthetic uptrend
        df = pd.DataFrame({
            "ts": pd.date_range("2024-01-01", periods=80, freq="D", tz="Asia/Ho_Chi_Minh"),
            "open": np.linspace(10, 20, 80),
            "high": np.linspace(10.2, 20.2, 80),
            "low":  np.linspace(9.8, 19.8, 80),
            "close": np.linspace(10, 20, 80),
            "volume": np.random.randint(5_000_000, 8_000_000, 80),
        })
        e = enrich_indicators(df)
        feats = compute_features_by_tf({"1D": e, "1W": e.iloc[::5].reset_index(drop=True)})
        ev = None
        if _evaluate:
            ev = _evaluate(feats)
        plan = decide(feats, ev)
        print("Smoke:", plan["DECISION"], plan["STATE"], plan["DIRECTION"], plan["entry"], plan["sl"], plan["tp1"])
    except Exception as ex:
        print("Self-test failed:", ex)
