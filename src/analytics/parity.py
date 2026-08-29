"""P7: put-call parity checker (SPEC 3 P7). Pure, model-free.

C − P = S·e^{−qT} − K·e^{−rT} needs no volatility and no distribution:
it is enforced by arbitrage alone. `deviation` is the dollar gap; a
violation is TRADEABLE only if it exceeds the combined bid–ask spread of
the two legs, so close-based sessions (no quotes) can never flag one and
must say so. Implied carry backs out the r − q the market is using, from
the ATM-interpolated C − P (exactly linear in K, so the interpolation is
exact).
"""
import numpy as np
import pandas as pd

PARITY_COLUMNS = ["expiry", "dte", "strike", "moneyness", "call_price", "put_price",
                  "lhs", "rhs", "deviation", "spread", "tradeable_violation"]
CARRY_COLUMNS = ["expiry", "dte", "implied_carry"]


def compute_parity(chain_iv: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    if chain_iv.empty:
        return pd.DataFrame(columns=PARITY_COLUMNS)
    spot = float(chain_iv["spot"].iloc[0])
    keep = ["expiry", "dte", "strike", "price_used", "bid", "ask"]
    calls = chain_iv[chain_iv["kind"] == "call"][keep].rename(
        columns={"price_used": "call_price", "bid": "call_bid", "ask": "call_ask"})
    puts = chain_iv[chain_iv["kind"] == "put"][keep].rename(
        columns={"price_used": "put_price", "bid": "put_bid", "ask": "put_ask"})
    p = calls.merge(puts, on=["expiry", "dte", "strike"], how="inner")
    p = p.dropna(subset=["call_price", "put_price"]).copy()
    if p.empty:
        return pd.DataFrame(columns=PARITY_COLUMNS)
    T = p["dte"].to_numpy(dtype=float) / 365.0
    p["lhs"] = p["call_price"] - p["put_price"]
    p["rhs"] = spot * np.exp(-q * T) - p["strike"] * np.exp(-r * T)
    p["deviation"] = p["lhs"] - p["rhs"]
    p["spread"] = (p["call_ask"] - p["call_bid"]) + (p["put_ask"] - p["put_bid"])
    p["tradeable_violation"] = (p["deviation"].abs() > p["spread"]) & p["spread"].notna()
    p["tradeable_violation"] = p["tradeable_violation"].astype(bool)
    p["moneyness"] = p["strike"] / spot
    return (p[PARITY_COLUMNS].sort_values(["expiry", "strike"])
            .reset_index(drop=True))


def parity_summary(parity: pd.DataFrame) -> dict:
    n = int(len(parity))
    quoted = parity[parity["spread"].notna()] if n else parity
    n_quoted = int(len(quoted))
    n_viol = int(parity["tradeable_violation"].sum()) if n else 0
    return {
        "n_pairs": n,
        "n_quoted": n_quoted,
        "n_tradeable_violations": n_viol,
        "share_within_spread": (1.0 - n_viol / n_quoted) if n_quoted else np.nan,
        "max_abs_deviation": float(parity["deviation"].abs().max()) if n else np.nan,
    }


def implied_carry(parity: pd.DataFrame, spot: float, r: float) -> pd.DataFrame:
    rows = []
    for (expiry, dte), g in parity.groupby(["expiry", "dte"], sort=False):
        g = g.sort_values("strike")
        ks = g["strike"].to_numpy(dtype=float)
        if len(ks) < 2 or not (ks.min() <= spot <= ks.max()):
            continue
        T = float(dte) / 365.0
        lhs_atm = float(np.interp(spot, ks, g["lhs"].to_numpy(dtype=float)))
        forward = spot + lhs_atm * np.exp(r * T)
        rows.append({"expiry": expiry, "dte": int(dte),
                     "implied_carry": float(np.log(forward / spot) / T)})
    return (pd.DataFrame(rows, columns=CARRY_COLUMNS).sort_values("dte")
            .reset_index(drop=True))


def carry_at_target(carry: pd.DataFrame, target_dte: int) -> tuple[float, float]:
    if carry.empty:
        return (np.nan, np.nan)
    j = int((carry["dte"] - target_dte).abs().idxmin())
    return float(carry.loc[j, "implied_carry"]), float(carry.loc[j, "dte"])
