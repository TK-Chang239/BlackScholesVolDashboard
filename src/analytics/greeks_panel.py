"""P4: Greeks for the ~30-DTE expiry (SPEC 3 P4). Pure, raw derivatives.

Tiles: the converged strike nearest spot per kind. Curves: every converged
strike of that expiry, each evaluated at ITS OWN market IV -- so the
curves are what a desk would actually see, not a flat-vol textbook plot.
Display scaling (theta/day, vega/vol-pt, rho/1%) belongs to src.render.
"""
import numpy as np
import pandas as pd

from src.models.black_scholes import greeks

TILE_COLUMNS = ["kind", "expiry", "dte", "strike", "iv", "delta", "gamma", "vega", "theta", "rho"]
CURVE_COLUMNS = ["kind", "strike", "moneyness", "delta", "gamma"]


def _empty():
    return pd.DataFrame(columns=TILE_COLUMNS), pd.DataFrame(columns=CURVE_COLUMNS)


def compute_greeks_panel(chain_iv: pd.DataFrame, r: float, q: float,
                         cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    solved = chain_iv[chain_iv["iv"].notna()]
    if solved.empty:
        return _empty()
    spot = float(solved["spot"].iloc[0])
    target = int(cfg["target_dte"]["atm_panel"])
    available = solved[["expiry", "dte"]].drop_duplicates().reset_index(drop=True)
    pick = available.iloc[(available["dte"] - target).abs().idxmin()]
    g = solved[solved["expiry"] == pick["expiry"]].copy()
    T = float(pick["dte"]) / 365.0

    frames = []
    for kind in ("call", "put"):
        k = g[g["kind"] == kind].sort_values("strike")
        if k.empty:
            continue
        gr = greeks(S=spot, K=k["strike"].to_numpy(dtype=float), T=T, r=r,
                    sigma=k["iv"].to_numpy(dtype=float), q=q, kind=kind)
        frames.append(pd.DataFrame({
            "kind": kind, "expiry": pick["expiry"], "dte": int(pick["dte"]),
            "strike": k["strike"].to_numpy(dtype=float), "iv": k["iv"].to_numpy(dtype=float),
            "moneyness": k["strike"].to_numpy(dtype=float) / spot,
            **{name: np.atleast_1d(gr[name]) for name in ("delta", "gamma", "vega", "theta", "rho")},
        }))
    full = pd.concat(frames, ignore_index=True)
    idx = [grp.index[(grp["strike"] - spot).abs().argmin()] for _, grp in full.groupby("kind")]
    nearest = full.loc[idx]
    tiles = nearest[TILE_COLUMNS].sort_values("kind").reset_index(drop=True)
    curves = full[CURVE_COLUMNS].sort_values(["kind", "strike"]).reset_index(drop=True)
    return tiles, curves
