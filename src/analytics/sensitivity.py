"""P1: input sensitivity of the ~30-DTE ATM call (SPEC 3 P1). Pure.

Every input is bumped by the same ±bump so the comparison is honest. The
`group` column is the panel's real message: sigma, r and q are ESTIMATED
(the market can disagree about them); S and T are OBSERVED (a 20% spot
move is a crash, not a disagreement) -- the render layer gives each group
its own axis so sigma's dominance within the argued group is visible.
"""
import numpy as np
import pandas as pd

from src.models.black_scholes import bs_price

SENSITIVITY_COLUMNS = ["input", "label", "group", "base_value", "down_pct", "up_pct", "span"]
_INPUTS = [
    ("sigma", "Volatility σ", "argued"),
    ("r", "Risk-free rate r", "argued"),
    ("q", "Dividend yield q", "argued"),
    ("S", "Spot S", "observed"),
    ("T", "Time to expiry T", "observed"),
]


def _call(S, sigma, T, r, q, strike):
    return float(bs_price(S=S, K=strike, T=T, r=r, sigma=sigma, q=q, kind="call"))


def compute_sensitivity(spot: float, sigma: float, dte: float, r: float, q: float,
                        bump: float) -> pd.DataFrame:
    base_in = {"S": float(spot), "sigma": float(sigma), "T": float(dte) / 365.0,
               "r": float(r), "q": float(q)}
    base = _call(base_in["S"], base_in["sigma"], base_in["T"], base_in["r"], base_in["q"], spot)
    rows = []
    for name, label, group in _INPUTS:
        pct = {}
        for tag, f in (("down", 1.0 - bump), ("up", 1.0 + bump)):
            kw = dict(base_in)
            kw[name] = base_in[name] * f
            px = _call(kw["S"], kw["sigma"], kw["T"], kw["r"], kw["q"], spot)
            pct[tag] = (px - base) / base if base and np.isfinite(base) else np.nan
        rows.append({"input": name, "label": label, "group": group,
                     "base_value": base_in[name],
                     "down_pct": pct["down"], "up_pct": pct["up"],
                     "span": abs(pct["up"] - pct["down"])})
    df = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
    return df.sort_values("span", ascending=False, na_position="last").reset_index(drop=True)
