
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
        None: "skip",
    },
    # Minimum confidence required to ENTER
    "min_conf_enter": 0.55,
    # Require validator OR-gate to pass (volume or momentum/candles)
    "require_validation": True,
    # Use primary timeframe for execution
    "primary_tf": "1D",
    # Reclaim proximity tolerance (price not too extended from EMA)
    "max_dist_ema20_pct_for_reclaim": 6.0,
}

# =========================
# Small helpers
# =========================
def _get(d: dict, k: str, default=np.nan):
    return d.get(k, default) if isinstance(d, dict) else default

def _pct(a, b) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b): return float("nan")
    return (a / b - 1.0) * 100.0

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _pick_base_sl(f1d: dict) -> float:
    # Base = min(ema_pref, lo20, lo50)
    ema_pref = f1d.get(DEFAULT_CFG["ema_sl_pref"], np.nan)
    lo20 = _get(f1d, "lo20", np.nan)
    lo50 = _get(f1d, "lo50", np.nan)
    candidates = [x for x in [ema_pref, lo20, lo50] if not np.isnan(x)]
    return float(min(candidates)) if candidates else float("nan")

def _compute_sl(entry: float, f1d: dict, *, atr_mult: float, sl_min_pct: float, sl_max_pct: float) -> float:
    base = _pick_base_sl(f1d)
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

# =========================
# Core planner
# =========================
def _should_enter_long(ev: dict, f1d: dict, cfg: dict, missing: List[str]) -> bool:
    if ev is None:
        missing.append("no_evidence")
        return False
    state = ev.get("state", None)
    if state in ("breakdown","reject") or state is None:
        missing.append("bearish_or_none_state")
        return False
    if cfg.get("require_validation", True):
        val = ev.get("confirmations", {})
        if not any(val.values()):
            missing.append("validators_not_ok (need Volume OR Momentum/Candles)")
            return False
    conf = float(ev.get("confidence", 0.0))
    if conf < cfg["min_conf_enter"]:
        missing.append(f"low_confidence<{cfg['min_conf_enter']}")
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
    out = {
        "symbol": sym,
        "state": (evidence or {}).get("state"),
        "dir":   (evidence or {}).get("dir"),
        "missing": []
    }
    # Compute evidence if not provided
    ev = evidence
    if ev is None and _evaluate is not None:
        try:
            ev = _evaluate(features_by_tf)
        except Exception:
            ev = None

    out: Dict[str, Any] = {
        "timeframe_primary": cfg["primary_tf"],
        "STATE": ev.get("state") if isinstance(ev, dict) else None,
        "DIRECTION": "LONG",  # LONG-only engine
        "DECISION": "WAIT",
        "entry": None, "entry2": None, "sl": None,
        "tp1": None, "tp2": None, "tp3": None, "tp4": None, "tp5": None,
        "rr": None, "rr2": None,
        "missing": [],
        "notes": ev.get("notes", []) if isinstance(ev, dict) else [],
        "confirmations": ev.get("confirmations", {}) if isinstance(ev, dict) else {},
    }

    # Quickly filter: if evaluator says SHORT strongly, we still keep LONG-only;
    # we just do not enter (remain WAIT). We don't change DIRECTION field.
    if isinstance(ev, dict) and ev.get("direction") == "SHORT":
        out["missing"].append("LONG-only engine: evaluator bias SHORT")

    # Decide ENTER/WAIT/AVOID
    # 1) Không có feature 1D -> thiếu dữ liệu
    if not f1d:
        out["missing"].append("missing_features")
        V_ok = bool((evidence or {}).get("volume_ok", False))
        M_ok = bool((evidence or {}).get("momentum_ok", False))
        C_ok = bool((evidence or {}).get("candle_ok", False))
        out["confirm"] = {"V": V_ok, "M": M_ok, "C": C_ok}
        log_info(f"[{out['symbol']}] DECISION=WAIT | STATE={out.get('state')} | DIR={out.get('dir')} | "
                 f"reason=missing_features | confirm:V={V_ok} M={M_ok} C={C_ok} | missing={list(out['missing'])}")
        return out

    # 2) Có feature: cho _should_enter_long tự append các khóa thiếu vào out["missing"]
    can_enter = _should_enter_long(evidence or {}, f1d, cfg or {}, out["missing"])
    if not can_enter:
        V_ok = bool((evidence or {}).get("volume_ok", False))
        M_ok = bool((evidence or {}).get("momentum_ok", False))
        C_ok = bool((evidence or {}).get("candle_ok", False))
        out["confirm"] = {"V": V_ok, "M": M_ok, "C": C_ok}
        log_info(f"[{out['symbol']}] DECISION=WAIT | STATE={out.get('state')} | DIR={out.get('dir')} | "
                 f"reason=missing_features | confirm:V={V_ok} M={M_ok} C={C_ok} | missing={list(out['missing'])}")
        return out

    # Build plan
    entry_policy = cfg["entry_policy"].get(out["STATE"], "close")
    price = float(_get(f1d, "close", np.nan))
    if entry_policy == "skip" or np.isnan(price):
        out["missing"].append("entry_policy_skip_or_nan_price")
        return out

    entry = price  # simple market entry at scan
    sl = _compute_sl(entry, f1d, atr_mult=cfg["atr_sl_mult"], sl_min_pct=cfg["sl_min_pct"], sl_max_pct=cfg["sl_max_pct"])
    if sl >= entry:  # safety
        sl = entry * (1.0 - cfg["sl_min_pct"]/100.0)

    tps = _tp_from_r(entry, sl, cfg["tp_multipliers"])
    out.update({
        "DECISION": "ENTER",
        "entry": float(entry),
        "sl": float(sl),
        "tp1": tps[0], "tp2": tps[1], "tp3": tps[2], "tp4": tps[3], "tp5": tps[4],
    })
    # RR to TP1 and TP2 for quick reference (legacy fields rr, rr2)
    R = entry - sl
    out["rr"]  = float((tps[0] - entry) / R) if R > 0 else None
    out["rr2"] = float((tps[1] - entry) / R) if R > 0 else None

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
