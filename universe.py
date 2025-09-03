# universe.py — Vietnam equities universe
# Compatible with existing main.py import style: from universe import get_universe_from_env
# Override via ENV `SYMBOLS="STB,CTG,VCB"` (comma‑separated).

import os
from typing import List

DEFAULT_UNIVERSE: List[str] = [
    "STB","CTG","VCB","MBB","TCB","BID","ACB",
    "VND","VIX","SSI","VCI","MBS","CTS","BSI",
    "BVH","BMI","BSR","PVD","PVS","PVT","GAS",
    "POW","NT2","REE","DIG","KDH","NLG","HDC",
    "HDG","PDR","KBC","SZC","HHV","VCG","CTD","CII",
    "HPG","HSG","NKG","DGW","MWG","FPT","CMG","MSN",
    "VSC","GMD","CSV","DCM","DPM",
]

def _parse_csv(s: str) -> List[str]:
    """Parse comma‑separated tickers, normalize to UPPER, keep order, drop empties/dupes."""
    seen = set()
    out: List[str] = []
    for raw in (s or "").split(","):
        sym = raw.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out

def get_universe_from_env() -> List[str]:
    """Return SYMBOLS from env if provided, otherwise DEFAULT_UNIVERSE (copy)."""
    env = os.getenv("SYMBOLS", "")
    lst = _parse_csv(env)
    return lst if lst else DEFAULT_UNIVERSE[:]

def resolve_symbols(symbols_param: str) -> List[str]:
    """Prefer explicit query param (comma‑separated), then ENV, then default list."""
    q = _parse_csv(symbols_param)
    return q if q else get_universe_from_env()
