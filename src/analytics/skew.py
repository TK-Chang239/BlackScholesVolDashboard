"""P6: 25-delta skew at the ~30-DTE expiry (SPEC 3 P6). Pure.

Interpolate implied vol in DELTA space, not strike space: the 25-delta
put is "the put that pays off on a ~25%-probability down-move", which is
the same contract regardless of spot level or vol regime. Deltas come
from our own greeks() at each row's own market IV. skew = put − call;
Black-Scholes says it should be zero.
"""
import numpy as np
import pandas as pd

from src.models.black_scholes import greeks

SKEW_KEYS = ["skew_expiry", "skew_dte", "put_iv_25d", "call_iv_25d", "skew_25d"]
TARGET_DELTA = 0.25


def _empty() -> dict:
    out = {k: np.nan for k in SKEW_KEYS}
    out["skew_expiry"] = None
    return out


def _interp_iv_at_delta(g: pd.DataFrame, target: float) -> float:
    g = g.dropna(subset=["iv", "delta"]).sort_values("delta")
    if len(g) < 2 or not (g["delta"].iloc[0] <= target <= g["delta"].iloc[-1]):
        return float("nan")
    return float(np.interp(target, g["delta"].to_numpy(dtype=float),
                           g["iv"].to_numpy(dtype=float)))


def compute_skew_25d(chain_iv: pd.DataFrame, r: float, q: float, cfg: dict) -> dict:
    solved = chain_iv[chain_iv["iv"].notna()]
    if solved.empty:
        return _empty()
    spot = float(solved["spot"].iloc[0])
    target_dte = int(cfg["target_dte"]["atm_panel"])
    available = solved[["expiry", "dte"]].drop_duplicates().reset_index(drop=True)
    pick = available.iloc[(available["dte"] - target_dte).abs().idxmin()]
    g = solved[solved["expiry"] == pick["expiry"]].copy()
    T = float(pick["dte"]) / 365.0
    g["delta"] = np.nan
    for kind in ("call", "put"):
        m = (g["kind"] == kind).to_numpy()
        if m.any():
            g.loc[m, "delta"] = np.atleast_1d(greeks(
                S=spot, K=g.loc[m, "strike"].to_numpy(dtype=float), T=T, r=r,
                sigma=g.loc[m, "iv"].to_numpy(dtype=float), q=q, kind=kind)["delta"])
    puts = g[(g["kind"] == "put") & (g["strike"] < spot)]
    calls = g[(g["kind"] == "call") & (g["strike"] >= spot)]
    put_iv = _interp_iv_at_delta(puts, -TARGET_DELTA)
    call_iv = _interp_iv_at_delta(calls, TARGET_DELTA)
    return {"skew_expiry": pick["expiry"], "skew_dte": int(pick["dte"]),
            "put_iv_25d": put_iv, "call_iv_25d": call_iv, "skew_25d": put_iv - call_iv}
