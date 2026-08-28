"""P3: ATM IV term structure (SPEC 3 P3). Pure function, tidy output.

ATM IV per expiry: linear interpolation in strike at K = spot, computed
per kind (needs a converged IV strictly below-or-at and strictly above
spot), then averaged across the kinds that could interpolate -- parity
says call and put ATM IVs should agree; averaging smooths quote noise.
"""
import numpy as np
import pandas as pd

TERM_COLUMNS = ["expiry", "dte", "atm_iv"]


def _interp_at(spot: float, g: pd.DataFrame) -> float | None:
    g = g[g["iv"].notna()].sort_values("strike")
    below = g[g["strike"] <= spot].tail(1)
    above = g[g["strike"] > spot].head(1)
    if below.empty or above.empty:
        return None
    k0, v0 = float(below["strike"].iloc[0]), float(below["iv"].iloc[0])
    k1, v1 = float(above["strike"].iloc[0]), float(above["iv"].iloc[0])
    if k1 == k0:
        return v0
    return v0 + (v1 - v0) * (spot - k0) / (k1 - k0)


def compute_term_structure(chain_iv: pd.DataFrame) -> pd.DataFrame:
    if chain_iv.empty:
        return pd.DataFrame(columns=TERM_COLUMNS)
    spot = float(chain_iv["spot"].iloc[0])
    rows = []
    for (expiry, dte), g in chain_iv.groupby(["expiry", "dte"], sort=False):
        kind_ivs = [
            v for kind in ("call", "put")
            if (v := _interp_at(spot, g[g["kind"] == kind])) is not None
        ]
        if kind_ivs:
            rows.append({"expiry": expiry, "dte": int(dte),
                         "atm_iv": float(np.mean(kind_ivs))})
    return (pd.DataFrame(rows, columns=TERM_COLUMNS)
            .sort_values("dte").reset_index(drop=True))
