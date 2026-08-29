"""P7: put-call parity checker (SPEC 3 P7). Pure, model-free.

C - P = S*e^{-qT} - K*e^{-rT} needs no volatility and no distribution:
it is enforced by arbitrage alone. `deviation` is the dollar gap against
*our* spot and carry assumption; a violation is TRADEABLE only if it
exceeds the combined bid-ask spread of the two legs, so close-based
sessions (no quotes) can never flag one and must say so.

That raw test carries a nuisance parameter: the underlying price the
option quotes were actually struck against. SPY options close at 16:15
ET, the stock at 16:00, so our close-based spot can be a dollar away
from the one the market used -- and near the money, a dollar exceeds the
whole combined spread. So the headline test calibrates each expiry's
forward from the market's own ATM C - P (`forward`, `deviation_fwd`)
and then asks whether C - P is the correct affine function of K across
the rest of the ladder: for a fixed expiry parity says
C - P = e^{-rT}(F - K), exactly linear in K with slope -e^{-rT}. That
removes the nuisance parameter and keeps the arbitrage content. The
spot-based deviation stays on the frame and in the summary -- the gap
between the two is itself the measurement.

`implied_forward` reports the calibrated forward and the carry it
implies, ln(F/S)/T, which under our assumptions must equal r - q.

What survives the calibration is mostly not arbitrage either: SPY options
are American, and an early-exercisable in-the-money put is worth more than
the European one parity is written for. `early_exercise_bound` prices the
most that effect can be worth, and `early_exercise_explained` marks the
violations that fit inside it, so the headline count is the residue that
nothing in this file can account for.
"""
import numpy as np
import pandas as pd

PARITY_COLUMNS = ["expiry", "dte", "strike", "moneyness", "call_price", "put_price",
                  "call_oi", "put_oi", "liquid", "lhs", "rhs", "deviation",
                  "forward", "deviation_fwd", "early_exercise_bound",
                  "early_exercise_explained", "spread",
                  "tradeable_violation", "tradeable_violation_fwd"]
CARRY_COLUMNS = ["expiry", "dte", "implied_carry"]
FORWARD_COLUMNS = ["expiry", "dte", "forward", "implied_carry"]


def compute_parity(chain_iv: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    if chain_iv.empty:
        return pd.DataFrame(columns=PARITY_COLUMNS)
    spot = float(chain_iv["spot"].iloc[0])
    keep = ["expiry", "dte", "strike", "price_used", "bid", "ask", "open_interest"]
    calls = chain_iv[chain_iv["kind"] == "call"][keep].rename(
        columns={"price_used": "call_price", "bid": "call_bid", "ask": "call_ask",
                 "open_interest": "call_oi"})
    puts = chain_iv[chain_iv["kind"] == "put"][keep].rename(
        columns={"price_used": "put_price", "bid": "put_bid", "ask": "put_ask",
                 "open_interest": "put_oi"})
    p = calls.merge(puts, on=["expiry", "dte", "strike"], how="inner")
    p = p.dropna(subset=["call_price", "put_price"]).copy()
    if p.empty:
        return pd.DataFrame(columns=PARITY_COLUMNS)
    p["call_oi"] = pd.to_numeric(p["call_oi"], errors="coerce").astype(float)
    p["put_oi"] = pd.to_numeric(p["put_oi"], errors="coerce").astype(float)
    # A pair is liquid when BOTH legs have open interest. A frame with no open
    # interest at all (close-based backfill) would otherwise report zero liquid
    # pairs, so in that case every pair counts as liquid.
    oi_known = bool(p["call_oi"].notna().any() or p["put_oi"].notna().any())
    p["liquid"] = ((p["call_oi"] > 0) & (p["put_oi"] > 0)) if oi_known else True
    p["liquid"] = p["liquid"].astype(bool)

    T = p["dte"].to_numpy(dtype=float) / 365.0
    p["lhs"] = p["call_price"] - p["put_price"]
    p["rhs"] = spot * np.exp(-q * T) - p["strike"] * np.exp(-r * T)
    p["deviation"] = p["lhs"] - p["rhs"]
    p["spread"] = (p["call_ask"] - p["call_bid"]) + (p["put_ask"] - p["put_bid"])
    p["tradeable_violation"] = ((p["deviation"].abs() > p["spread"])
                                & p["spread"].notna()).astype(bool)
    p["moneyness"] = p["strike"] / spot

    fwd = _forward_table(p, spot, r)
    p["forward"] = p["expiry"].map(dict(zip(fwd["expiry"], fwd["forward"]))).astype(float)
    p["deviation_fwd"] = p["lhs"] - np.exp(-r * T) * (p["forward"] - p["strike"])
    p["tradeable_violation_fwd"] = ((p["deviation_fwd"].abs() > p["spread"])
                                    & p["spread"].notna()
                                    & p["deviation_fwd"].notna()).astype(bool)

    # SPY options are American. An American put is worth at least its intrinsic
    # K - S, while its European twin is worth K*e^{-rT} - S*e^{-qT} + C, so the
    # most that early exercise can add to P -- and therefore the most it can
    # push C - P BELOW the parity forward -- is
    #     K*(1 - e^{-rT}) - S*(1 - e^{-qT}),
    # the interest earned on the strike less the dividends forgone by exercising
    # early. It depends only on K, T, r, q, S: no quote enters it. It is a
    # BOUND, not a price: it says what early exercise could be worth at most.
    # For a short T and a fat q it can come out negative -- early exercise is
    # then worth nothing at all -- so clip at zero.
    p["early_exercise_bound"] = np.maximum(
        p["strike"].to_numpy(dtype=float) * (1.0 - np.exp(-r * T))
        - spot * (1.0 - np.exp(-q * T)), 0.0)
    # Reads as "this violation is explained", never "this row is explainable":
    # false wherever there is no tradeable violation to explain. Only a NEGATIVE
    # gap can be an early-exercise story -- a rich call pushes the other way.
    p["early_exercise_explained"] = (
        p["tradeable_violation_fwd"].astype(bool)
        & (p["deviation_fwd"] < 0)
        & (p["deviation_fwd"].abs() <= p["early_exercise_bound"] + p["spread"])
    ).astype(bool)
    return (p[PARITY_COLUMNS].sort_values(["expiry", "strike"])
            .reset_index(drop=True))


def parity_summary(parity: pd.DataFrame) -> dict:
    n = int(len(parity))
    quoted = parity[parity["spread"].notna()] if n else parity
    liquid_quoted = quoted[quoted["liquid"].astype(bool)] if n else parity
    n_quoted = int(len(quoted))
    n_liq_quoted = int(len(liquid_quoted))
    n_viol = int(parity["tradeable_violation"].sum()) if n else 0
    n_viol_fwd = int(liquid_quoted["tradeable_violation_fwd"].sum()) if n else 0
    explained = (liquid_quoted["early_exercise_explained"].astype(bool)
                 if n else pd.Series(dtype=bool))
    n_ee = int(explained.sum()) if n else 0
    return {
        "n_pairs": n,
        "n_quoted": n_quoted,
        "n_liquid": int(parity["liquid"].astype(bool).sum()) if n else 0,
        "n_tradeable_violations": n_viol,
        "n_tradeable_violations_fwd": n_viol_fwd,
        # the two split n_tradeable_violations_fwd exactly
        "n_violations_early_exercise": n_ee,
        "n_violations_unexplained": n_viol_fwd - n_ee,
        "share_within_spread": (1.0 - n_viol / n_quoted) if n_quoted else np.nan,
        "share_within_spread_fwd": ((1.0 - n_viol_fwd / n_liq_quoted)
                                    if n_liq_quoted else np.nan),
        "max_abs_deviation": float(parity["deviation"].abs().max()) if n else np.nan,
        "max_abs_deviation_fwd": (float(parity["deviation_fwd"].abs().max())
                                  if n and parity["deviation_fwd"].notna().any() else np.nan),
    }


def _brackets(g: pd.DataFrame, spot: float) -> bool:
    ks = g["strike"].to_numpy(dtype=float)
    return len(ks) >= 2 and bool(ks.min() <= spot <= ks.max())


def _forward_table(parity: pd.DataFrame, spot: float, r: float) -> pd.DataFrame:
    """Per expiry: interpolate C - P to K = spot and back out F = S + (C-P)e^{rT}.

    C - P is exactly linear in K under parity, so the interpolation is exact
    rather than approximate. Liquid strikes are preferred; an expiry whose
    liquid ladder does not straddle spot falls back to all of its strikes,
    and one that still does not straddle spot is skipped (never extrapolated).
    """
    rows = []
    for (expiry, dte), g in parity.groupby(["expiry", "dte"], sort=False):
        if float(dte) <= 0:
            continue
        sub = g[g["liquid"].astype(bool)] if "liquid" in g.columns else g
        if not _brackets(sub, spot):
            sub = g
        if not _brackets(sub, spot):
            continue
        sub = sub.sort_values("strike")
        T = float(dte) / 365.0
        lhs_atm = float(np.interp(spot, sub["strike"].to_numpy(dtype=float),
                                  sub["lhs"].to_numpy(dtype=float)))
        forward = float(spot + lhs_atm * np.exp(r * T))
        carry = float(np.log(forward / spot) / T) if forward > 0 and spot > 0 else np.nan
        rows.append({"expiry": expiry, "dte": int(dte),
                     "forward": forward, "implied_carry": carry})
    return (pd.DataFrame(rows, columns=FORWARD_COLUMNS).sort_values("dte")
            .reset_index(drop=True))


def implied_forward(parity: pd.DataFrame, spot: float, r: float) -> pd.DataFrame:
    return _forward_table(parity, spot, r)


def implied_carry(parity: pd.DataFrame, spot: float, r: float) -> pd.DataFrame:
    """Backwards-compatible view of `implied_forward`: carry per expiry."""
    return implied_forward(parity, spot, r)[CARRY_COLUMNS]


def carry_at_target(carry: pd.DataFrame, target_dte: int) -> tuple[float, float]:
    if carry.empty:
        return (np.nan, np.nan)
    j = int((carry["dte"] - target_dte).abs().idxmin())
    return float(carry.loc[j, "implied_carry"]), float(carry.loc[j, "dte"])


def carry_reference(carry: pd.DataFrame, min_dte: int = 84) -> tuple[float, float]:
    """The carry to quote: the LONGEST expiry at or beyond `min_dte`.

    A fixed dollar error in the forward divides by T, so the shortest expiry
    is the least informative place to read carry -- a $1 miss at 21 days is
    over two percentage points of "rate", and nothing at all at six months.
    Falls back to the longest expiry available when none qualify.
    """
    if carry.empty:
        return (np.nan, np.nan)
    qualifying = carry[carry["dte"] >= min_dte]
    src = qualifying if len(qualifying) else carry
    j = int(src["dte"].idxmax())
    return float(src.loc[j, "implied_carry"]), float(src.loc[j, "dte"])
