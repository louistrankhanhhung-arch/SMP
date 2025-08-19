
import os

DEFAULT_UNIVERSE = [
    "VCB","STB","MBB","CTG","BID","TCB","LPB","ACB",
    "SSI","VND","VIX","VCI","MBS","BSI","BMI",
    "PVD","PVS","PVT","BSR","GAS","POW","NT2","REE",
    "DIG","KDH","NLG","KBC","HHV","VCG","CTD","CII",
    "DGW","MWG","MSN","FPT","CMG","VSC","GMD",
    "HPG","HSG","NKG","GVR","CSV","DCM","DPM",
]

def _parse_csv(s: str):
    return [x.strip().upper() for x in (s or "").split(",") if x.strip()]

def get_universe_from_env():
    env = os.getenv("SYMBOLS", "")
    lst = _parse_csv(env)
    return lst if lst else DEFAULT_UNIVERSE[:]

def resolve_symbols(symbols_param: str):
    q = _parse_csv(symbols_param)
    return q if q else get_universe_from_env()
