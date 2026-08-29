"""P9: flat-vol Black-Scholes price vs the market, every contract (SPEC 3 P9).

One vol for the whole chain -- today's ~30-DTE ATM IV -- is what the model
literally assumes. The deviation pattern is the smile of P2 re-expressed
in price space; muting within the bid–ask spread mirrors P7.
"""
import numpy as np
import pandas as pd

from src.models.black_scholes import bs_price

MVM_COLUMNS = ["kind", "expiry", "dte", "strike", "moneyness", "market", "model",
               "deviation_pct", "deviation_vol", "quoted", "within_spread"]


def compute_model_vs_market(chain_iv: pd.DataFrame, flat_vol: float,
                            r: float, q: float) -> pd.DataFrame:
    if chain_iv.empty or flat_vol is None or not np.isfinite(flat_vol):
        return pd.DataFrame(columns=MVM_COLUMNS)
    df = chain_iv[chain_iv["price_used"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=MVM_COLUMNS)
    spot = float(df["spot"].iloc[0])
    T = df["dte"].to_numpy(dtype=float) / 365.0
    df["model"] = np.nan
    for kind in ("call", "put"):
        m = (df["kind"] == kind).to_numpy()
        if m.any():
            df.loc[m, "model"] = np.atleast_1d(bs_price(
                S=spot, K=df.loc[m, "strike"].to_numpy(dtype=float), T=T[m], r=r,
                sigma=float(flat_vol), q=q, kind=kind))
    df["market"] = df["price_used"].astype(float)
    df["moneyness"] = df["strike"] / spot
    df["deviation_pct"] = (df["market"] - df["model"]) / df["market"]
    df["deviation_vol"] = df["iv"] - float(flat_vol)
    df["quoted"] = df["bid"].notna() & df["ask"].notna()
    half = (df["ask"] - df["bid"]) / 2.0
    df["within_spread"] = (df["quoted"] & ((df["market"] - df["model"]).abs() <= half)).astype(bool)
    return (df[MVM_COLUMNS].sort_values(["kind", "expiry", "strike"])
            .reset_index(drop=True))
