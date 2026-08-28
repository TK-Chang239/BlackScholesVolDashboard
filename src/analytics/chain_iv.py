"""Shared analytics step: invert stored chain prices into implied vols.

Price routing per SPEC 2.2: mid for live (yfinance) rows, close otherwise.
All IVs come from our own solver -- vendor_iv is never read. Convergence
stats feed status.json and the Phase 3 exit criterion (>= 95% on liquid
quotes).
"""
import numpy as np
import pandas as pd

from src.data.base import LIVE_SOURCES
from src.models.black_scholes import implied_vol


def compute_chain_iv(chain: pd.DataFrame, r: float, q: float) -> tuple[pd.DataFrame, dict]:
    df = chain.copy()
    live = df["source"].isin(LIVE_SOURCES)
    df["price_used"] = np.where(live, df["mid"], df["close"])
    df["iv"] = np.nan
    if not df.empty:
        t_years = df["dte"].to_numpy(dtype=float) / 365.0
        for kind in ("call", "put"):
            m = (df["kind"] == kind).to_numpy()
            if not m.any():
                continue
            df.loc[m, "iv"] = implied_vol(
                df.loc[m, "price_used"].to_numpy(),
                S=df.loc[m, "spot"].to_numpy(dtype=float),
                K=df.loc[m, "strike"].to_numpy(dtype=float),
                T=t_years[m], r=r, q=q, kind=kind,
            )
    n_priced = int(df["price_used"].notna().sum())
    n_converged = int(df["iv"].notna().sum())
    stats = {
        "n_priced": n_priced,
        "n_converged": n_converged,
        "convergence": (n_converged / n_priced) if n_priced else 0.0,
    }
    return df, stats
