
"""
evidence_evaluators.py — turn primitive features into tradeable "evidence"
--------------------------------------------------------------------------
Inputs
------
- features_by_tf: dict like the output of feature_primitives.compute_features_by_tf,
  e.g. {"1D": {"df": <enriched_df>, "features": {...}}, "1W": {...}}

Goals
-----
- Detect common trading states for Vietnam stocks:
    * breakout / breakdown
    * reclaim (bear->bull) / reject (bull->bear)
    * pullback_to_ema (trend continuation)
    * mean_revert (extreme to mid)
    * squeeze_expansion (volatility regimes)
- Apply an "OR" gate: Volume OR (Momentum/Candles) to validate a state.
- Produce a compact, deterministic dict to feed decision_engine.

Public API
----------
- evaluate(features_by_tf: dict, *, cfg: dict | None = None) -> dict
  Returns:
  {
    'direction': 'LONG'|'SHORT'|'NEUTRAL',
    'state': 'breakout'|'breakdown'|'reclaim'|'reject'|'pullback_to_ema'|'mean_revert'|'squeeze_expansion'|None,
    'confidence': float in [0,1],
    'scores': {<state>: score_float},
    'confirmations': {'volume': bool, 'momentum': bool, 'candles': bool},
    'notes': [str, ...],
    'by_tf': { '1D': {...}, '1W': {...} }
  }
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, List
import math
import numpy as np
import pandas as pd

# =========================
# Default configuration
# =========================
DEFAULT_CFG = {
    # Distance thresholds (%)
    'near_ema_pct': 1.5,       # pullback to ema20/50 threshold
    'near_sr_pct': 1.0,        # pivot/SR proximity
    'breakout_buffer_pct': 0.5,# must clear hi20 by this percent
    'breakdown_buffer_pct': 0.5,
    'meanrevert_band_edge': 0.10, # bb_pctb < 0.10 (lower) or > 0.90 (upper)
    'atr_push_min': 0.6,       # range >= 0.6 * ATR for meaningful candle
    # Volume / momentum / candles validators
    'vol_ratio_ok': 1.2,       # volume >= 1.2x 20-day avg
    'vol_z_ok': 0.5,           # or zscore >= 0.5
    'macd_hist_delta_min': 0.0,# momentum rising if delta > 0
    'rsi_fast_trigger': 55.0,  # bull if rsi >= 55; bear if <= 45
    'body_pct_ok': 0.35,       # big-bodied candle (body/range)
    # Squeeze
    'bb_width_low': 0.05,      # low vol regime for VN daily
    'bb_width_expand': 0.02,   # expansion threshold day-over-day
}

# =========================
# Small helpers
# =========================
def _get(feats: dict, key: str, default=np.nan):
    return feats.get(key, default)

def _pct_dist(a: float, b: float) -> float:
    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return (a / b - 1.0) * 100.0

def _last_two(series: pd.Series) -> Tuple[float, float]:
    if series is None or len(series) < 2:
        return np.nan, np.nan
    return float(series.iloc[-2]), float(series.iloc[-1])

# =========================
# Core validators (Volume OR Momentum/Candles)
# =========================
def _volume_ok(f1d: dict, cfg: dict) -> bool:
    vol_ratio = _get(f1d, 'vol_ratio', np.nan)
    vol_z = _get(f1d, 'vol_z', np.nan)
    return (not np.isnan(vol_ratio) and vol_ratio >= cfg['vol_ratio_ok']) or                (not np.isnan(vol_z) and vol_z >= cfg['vol_z_ok'])

def _momentum_ok(f1d: dict, prev1d: dict | None, cfg: dict) -> bool:
    macd_now = _get(f1d, 'macd_hist', np.nan)
    rsi_now = _get(f1d, 'rsi14', np.nan)
    macd_prev = _get(prev1d, 'macd_hist', np.nan) if prev1d else np.nan
    # Rising momentum if macd_hist increasing in the desired direction
    rising = (not np.isnan(macd_now) and not np.isnan(macd_prev) and (macd_now - macd_prev) > cfg['macd_hist_delta_min'])
    # RSI bias zone
    rsi_bias = (not np.isnan(rsi_now)) and (rsi_now >= cfg['rsi_fast_trigger'] or rsi_now <= (100 - cfg['rsi_fast_trigger']))
    return bool(rising or rsi_bias)

def _candles_ok(df1d: pd.DataFrame, f1d: dict, cfg: dict) -> bool:
    if df1d is None or len(df1d) < 1:
        return False
    row = df1d.iloc[-1]
    rng = float(row.get('range', np.nan))
    atr14 = float(row.get('atr14', np.nan))
    body_pct = float(row.get('body_pct', np.nan))
    ok_push = (not np.isnan(rng) and not np.isnan(atr14) and rng >= cfg['atr_push_min'] * atr14)
    ok_body = (not np.isnan(body_pct) and body_pct >= cfg['body_pct_ok'])
    return bool(ok_push or ok_body)

def _or_validation(df1d: pd.DataFrame, f1d: dict, prev1d: dict | None, cfg: dict) -> Tuple[bool, dict]:
    vol_ok = _volume_ok(f1d, cfg)
    mom_ok = _momentum_ok(f1d, prev1d, cfg)
    candle_ok = _candles_ok(df1d, f1d, cfg)
    # OR gate: (Volume) OR (Momentum OR Candles)
    valid = bool(vol_ok or mom_ok or candle_ok)
    return valid, {'volume': vol_ok, 'momentum': mom_ok, 'candles': candle_ok}

# =========================
# State detectors
# =========================
def _breakout(df1d: pd.DataFrame, f1d: dict, cfg: dict) -> Tuple[float, str]:
    close = _get(f1d, 'close')
    hi20 = _get(f1d, 'hi20')
    stacked_bull = bool(_get(f1d, 'stacked_bull', False))
    dist = _pct_dist(close, hi20)
    pass_cond = (not np.isnan(dist)) and (dist >= cfg['breakout_buffer_pct']) and stacked_bull
    score = 0.0 if not pass_cond else min(1.0, 0.5 + dist/3.0)  # grows with clearance
    note = f"close>{cfg['breakout_buffer_pct']}% above hi20 & stacked_bull" if pass_cond else "n/a"
    return score, note

def _breakdown(df1d: pd.DataFrame, f1d: dict, cfg: dict) -> Tuple[float, str]:
    close = _get(f1d, 'close')
    lo20 = _get(f1d, 'lo20')
    stacked_bear = bool(_get(f1d, 'stacked_bear', False))
    dist = _pct_dist(close, lo20)
    pass_cond = (not np.isnan(dist)) and (dist <= -cfg['breakdown_buffer_pct']) and stacked_bear
    score = 0.0 if not pass_cond else min(1.0, 0.5 + abs(dist)/3.0)
    note = f"close<{cfg['breakdown_buffer_pct']}% below lo20 & stacked_bear" if pass_cond else "n/a"
    return score, note

def _reclaim(df1d: pd.DataFrame, f1d: dict, prev1d: dict | None, cfg: dict) -> Tuple[float, str]:
    # Price was below pivot/ema20 and now above (bullish reclaim). Opposite is 'reject' (bearish).
    close = _get(f1d, 'close'); prev_close = _get(prev1d, 'close', np.nan) if prev1d else np.nan
    ema20 = _get(f1d, 'ema20'); prev_ema20 = _get(prev1d, 'ema20', np.nan) if prev1d else np.nan
    pivot = _get(f1d, 'pivot'); prev_pivot = _get(prev1d, 'pivot', np.nan) if prev1d else np.nan
    cond = (not np.isnan(prev_close) and not np.isnan(prev_ema20) and prev_close < prev_ema20 and close > ema20) or                (not np.isnan(prev_close) and not np.isnan(prev_pivot) and prev_close < prev_pivot and close > pivot)
    score = 0.65 if cond else 0.0
    note = "bullish reclaim over ema20/pivot" if cond else "n/a"
    return score, note

def _reject(df1d: pd.DataFrame, f1d: dict, prev1d: dict | None, cfg: dict) -> Tuple[float, str]:
    close = _get(f1d, 'close'); prev_close = _get(prev1d, 'close', np.nan) if prev1d else np.nan
    ema20 = _get(f1d, 'ema20'); prev_ema20 = _get(prev1d, 'ema20', np.nan) if prev1d else np.nan
    pivot = _get(f1d, 'pivot'); prev_pivot = _get(prev1d, 'pivot', np.nan) if prev1d else np.nan
    cond = (not np.isnan(prev_close) and not np.isnan(prev_ema20) and prev_close > prev_ema20 and close < ema20) or                (not np.isnan(prev_close) and not np.isnan(prev_pivot) and prev_close > prev_pivot and close < pivot)
    score = 0.65 if cond else 0.0
    note = "bearish reject under ema20/pivot" if cond else "n/a"
    return score, note

def _pullback_to_ema(df1d: pd.DataFrame, f1d: dict, cfg: dict) -> Tuple[float, str]:
    close = _get(f1d, 'close')
    ema20 = _get(f1d, 'ema20')
    ema50 = _get(f1d, 'ema50')
    stacked_bull = bool(_get(f1d, 'stacked_bull', False))
    stacked_bear = bool(_get(f1d, 'stacked_bear', False))
    d20 = abs(_pct_dist(close, ema20)); d50 = abs(_pct_dist(close, ema50))
    # Long in bull trend if close ~ ema20 or ema50; Short in bear symmetric (handled at combine stage)
    near = (not np.isnan(d20) and d20 <= cfg['near_ema_pct']) or (not np.isnan(d50) and d50 <= cfg['near_ema_pct'])
    ok = near and (stacked_bull or stacked_bear)
    score = 0.55 if ok else 0.0
    note = "pullback near ema20/50 within trend" if ok else "n/a"
    return score, note

def _mean_revert(df1d: pd.DataFrame, f1d: dict, cfg: dict) -> Tuple[float, str]:
    pctb = _get(f1d, 'bb_pctb', np.nan)
    zone = (not np.isnan(pctb)) and (pctb <= DEFAULT_CFG['meanrevert_band_edge'] or pctb >= (1.0 - DEFAULT_CFG['meanrevert_band_edge']))
    score = 0.5 if zone else 0.0
    note = "extreme band edge (bb_pctb)" if zone else "n/a"
    return score, note

def _squeeze_expansion(df1d: pd.DataFrame, f1d: dict, prev1d: dict | None, cfg: dict) -> Tuple[float, str]:
    bw = _get(f1d, 'bb_width', np.nan); bw_prev = _get(prev1d, 'bb_width', np.nan) if prev1d else np.nan
    low = (not np.isnan(bw)) and (bw <= cfg['bb_width_low'])
    expanding = (not np.isnan(bw) and not np.isnan(bw_prev)) and ((bw - bw_prev) >= cfg['bb_width_expand'])
    score = 0.45 if (low and expanding) else (0.25 if expanding else 0.0)
    note = "squeeze expanding" if (low or expanding) else "n/a"
    return score, note

# =========================
# Aggregation & direction
# =========================
def _direction_from_scores(scores: Dict[str, float], f1d: dict) -> str:
    # Map states to directional intent
    longish = scores.get('breakout',0)+scores.get('reclaim',0)+scores.get('pullback_to_ema',0)
    shortish = scores.get('breakdown',0)+scores.get('reject',0)+scores.get('mean_revert',0)
    # tie-breakers
    if longish > shortish: return "LONG"
    if shortish > longish: return "SHORT"
    # If equal, bias with trend stack and RSI
    if f1d.get('stacked_bull'): return "LONG"
    if f1d.get('stacked_bear'): return "SHORT"
    rsi = f1d.get('rsi14', np.nan)
    if not np.isnan(rsi):
        if rsi >= 55: return "LONG"
        if rsi <= 45: return "SHORT"
    return "NEUTRAL"

def _confidence(scores: Dict[str, float], validators: Dict[str, bool]) -> float:
    base = max(scores.values()) if scores else 0.0
    bonus = 0.1 * sum(1 for v in validators.values() if v)
    return float(max(0.0, min(1.0, base + bonus)))

# =========================
# Public API
# =========================
def evaluate(features_by_tf: Dict[str, dict], *, cfg: dict | None = None) -> dict:
    cfg = {**DEFAULT_CFG, **(cfg or {})}

    d1 = features_by_tf.get('1D', {})
    w1 = features_by_tf.get('1W', {})

    df1d = d1.get('df', None)
    f1d = d1.get('features', {}) or {}
    # previous bar features (if available)
    prev1d = {}
    if df1d is not None and len(df1d) >= 2:
        prev_row = df1d.iloc[-2]
        prev1d = {
            'close': float(prev_row.get('close', np.nan)),
            'ema20': float(prev_row.get('ema20', np.nan)),
            'pivot': float(prev_row.get('pivot', np.nan)),
            'bb_width': float(prev_row.get('bb_width', np.nan)),
            'macd_hist': float(prev_row.get('macd_hist', np.nan)),
        }

    # Compute state scores
    scores = {}
    notes = []
    s, note = _breakout(df1d, f1d, cfg); scores['breakout'] = s;   (note!='n/a') and notes.append(f"breakout: {note}")
    s, note = _breakdown(df1d, f1d, cfg); scores['breakdown'] = s; (note!='n/a') and notes.append(f"breakdown: {note}")
    s, note = _reclaim(df1d, f1d, prev1d, cfg); scores['reclaim'] = s; (note!='n/a') and notes.append(f"reclaim: {note}")
    s, note = _reject(df1d, f1d, prev1d, cfg); scores['reject'] = s; (note!='n/a') and notes.append(f"reject: {note}")
    s, note = _pullback_to_ema(df1d, f1d, cfg); scores['pullback_to_ema'] = s; (note!='n/a') and notes.append(f"pullback: {note}")
    s, note = _mean_revert(df1d, f1d, cfg); scores['mean_revert'] = s; (note!='n/a') and notes.append(f"mean_revert: {note}")
    s, note = _squeeze_expansion(df1d, f1d, prev1d, cfg); scores['squeeze_expansion'] = s; (note!='n/a') and notes.append(f"squeeze: {note}")

    # Validators (Volume OR Momentum/Candles)
    valid, validators = _or_validation(df1d, f1d, prev1d, cfg)

    # Pick best state
    best_state = None
    best_score = 0.0
    for k, v in scores.items():
        if v > best_score:
            best_state, best_score = k, v

    direction = _direction_from_scores(scores, f1d) if best_score > 0 else "NEUTRAL"
    conf = _confidence(scores, validators) if best_state else 0.0

    return {
        'direction': direction,
        'state': best_state,
        'confidence': conf,
        'scores': scores,
        'confirmations': validators,
        'notes': notes,
        'by_tf': {
            '1D': {'features': f1d, 'n': int(len(df1d)) if df1d is not None else 0},
            '1W': {'features': w1.get('features', {}), 'n': int(len(w1.get('df', []))) if isinstance(w1.get('df'), pd.DataFrame) else 0},
        }
    }

if __name__ == "__main__":
    # quick contract check (no external fetch)
    try:
        import pandas as pd
        # Build a tiny fake df to verify function signatures
        df = pd.DataFrame({
            'ts': pd.date_range("2024-01-01", periods=30, freq="D", tz="Asia/Ho_Chi_Minh"),
            'open': np.linspace(10, 15, 30),
            'high': np.linspace(10.5, 15.5, 30),
            'low': np.linspace(9.5, 14.5, 30),
            'close': np.linspace(10, 15, 30) + np.sin(np.linspace(0, 3.14, 30)),
            'volume': np.random.randint(1_000_000, 2_000_000, 30)
        })
        from indicators import enrich_indicators
        from feature_primitives import compute_features_by_tf
        e = enrich_indicators(df)
        fb = compute_features_by_tf({'1D': e, '1W': e.iloc[::5].reset_index(drop=True)})
        out = evaluate(fb)
        print("Self-test OK:", out['state'], out['direction'], round(out['confidence'],3))
    except Exception as e:
        print("Self-test failed:", e)
