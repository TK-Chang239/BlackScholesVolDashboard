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
removes the nuisance parameter (mostly: the slope still carries our r)
and keeps the arbitrage content. What it cannot see any more is a pure
LEVEL error at the money -- that is exactly what the calibration absorbs.
The spot-based deviation stays on the frame and in the summary -- the gap
between the two is itself the measurement.

The forward itself is estimated robustly: under parity every strike gives
the same F = K + (C - P)e^{rT}, so the near-the-money strikes are many
independent readings of one number and the MEDIAN of them is used. Two ATM
quotes should never be able to move the ruler the whole ladder is measured
against. `implied_forward` reports that forward, how many strikes fed it,
and the carry it implies, ln(F/S)/T, which under our assumptions must
equal r - q.

What survives the calibration is mostly not arbitrage either: SPY options
are American, and an early-exercisable in-the-money put is worth more than
the European one parity is written for. `early_exercise_bound` prices the
most that effect can be worth, and `early_exercise_explained` marks the
in-the-money-put violations that fit inside it, so the headline count is
the residue that nothing in this file can account for.
"""
import numpy as np
import pandas as pd

PARITY_COLUMNS = ["expiry", "dte", "strike", "moneyness", "call_price", "put_price",
                  "call_oi", "put_oi", "liquid", "lhs", "rhs", "deviation",
                  "forward", "deviation_fwd", "early_exercise_bound",
                  "early_exercise_explained", "spread",
                  "tradeable_violation", "tradeable_violation_fwd"]
CARRY_COLUMNS = ["expiry", "dte", "implied_carry"]
FORWARD_COLUMNS = ["expiry", "dte", "forward", "implied_carry", "n_forward_strikes"]


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
    # A pair is liquid when BOTH legs have open interest. A frame carrying NO
    # POSITIVE open interest anywhere is not evidence that nothing trades -- it
    # is the absence of the measurement, whether the column is missing (close-
    # based backfill) or present and zero everywhere (a yfinance payload that
    # dropped the field). Both must fall back to "every pair counts as liquid":
    # taking a zero column at face value drives every liquidity-gated count to
    # zero, which on an unattended cron publishes a comforting all-clear with no
    # signal in it. Zeros on PART of a frame are real and still gate.
    oi_known = bool((p[["call_oi", "put_oi"]] > 0).any().any())
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

    # SPY options are American. The rigorous ceiling on the American-minus-
    # European PUT premium is K*(1 - e^{-rT}): exercising early hands you the
    # strike now instead of at T, so the most the right can ever be worth is the
    # interest earned on K over the remaining life. (Equivalently, the Kim
    # integral  int_0^T (r*K*e^{-rt} - q*S*e^{-qt}) dt  with the exercise-region
    # probabilities set to 1.) Dropping the dividend leg of that integral -- the
    # yield you forgo by giving up the stock -- gives
    #     ee_bound = K*(1 - e^{-rT}) - S*(1 - e^{-qT}),
    # which is STRICTLY TIGHTER than the rigorous ceiling. Using the tighter one
    # makes the classification conservative: it explains away less than it is
    # entitled to, and a gap outside it is therefore not proof of anything.
    # It depends only on K, T, r, q, S: no quote and no volatility enter it. It
    # is a BOUND, not a price. For a short T and a fat q it can come out negative
    # -- early exercise is then worth nothing at all -- so clip at zero.
    p["early_exercise_bound"] = np.maximum(
        p["strike"].to_numpy(dtype=float) * (1.0 - np.exp(-r * T))
        - spot * (1.0 - np.exp(-q * T)), 0.0)
    # Reads as "this violation is explained", never "this row is explainable":
    # false wherever there is no tradeable violation to explain. Only a NEGATIVE
    # gap can be an early-exercise story -- a rich call pushes the other way --
    # and only an IN-THE-MONEY put (K > S) can be exercised early for value at
    # all: the bound grows with K*rT even where the American premium is zero, so
    # without this gate it "explains" out-of-the-money wing puts it has no
    # business explaining.
    p["early_exercise_explained"] = (
        p["tradeable_violation_fwd"].astype(bool)
        & (p["deviation_fwd"] < 0)
        & (p["strike"] > spot)
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


ATM_BAND = 0.02          # |K/S - 1| within which a strike counts as "at the money"
MIN_FORWARD_STRIKES = 3  # below this the median is not worth having; interpolate


def _forward_table(parity: pd.DataFrame, spot: float, r: float) -> pd.DataFrame:
    """Per expiry: back out the forward the market's own C - P implies.

    Under parity EVERY strike gives the same forward, F = K + (C - P)e^{rT},
    so the near-the-money strikes are many independent readings of one number
    and the spread across them is pure noise/staleness. The estimate is the
    MEDIAN of those readings over the liquid strikes within `ATM_BAND` of spot,
    which needs a bad quote on half the ladder to move -- where interpolating
    between the two strikes bracketing spot needs only one. Fewer than
    `MIN_FORWARD_STRIKES` such strikes falls back to that two-point
    interpolation (exact, not approximate: C - P is linear in K).

    Liquid strikes are preferred throughout; an expiry whose liquid ladder does
    not straddle spot falls back to all of its strikes, and one that still does
    not straddle spot is skipped -- the forward is never extrapolated.
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
        strikes = sub["strike"].to_numpy(dtype=float)
        lhs = sub["lhs"].to_numpy(dtype=float)
        atm = np.abs(strikes / spot - 1.0) <= ATM_BAND if spot > 0 else np.zeros(len(sub), bool)
        n_strikes = int(atm.sum())
        if n_strikes >= MIN_FORWARD_STRIKES:
            forward = float(np.median(strikes[atm] + lhs[atm] * np.exp(r * T)))
        else:
            n_strikes = 2       # the two strikes that bracket spot, and only those
            forward = float(spot + float(np.interp(spot, strikes, lhs)) * np.exp(r * T))
        carry = float(np.log(forward / spot) / T) if forward > 0 and spot > 0 else np.nan
        rows.append({"expiry": expiry, "dte": int(dte), "forward": forward,
                     "implied_carry": carry, "n_forward_strikes": n_strikes})
    out = (pd.DataFrame(rows, columns=FORWARD_COLUMNS).sort_values("dte")
           .reset_index(drop=True))
    return out.astype({"n_forward_strikes": int}) if len(out) else out


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


CARRY_REFERENCE_KEYS = ["implied_carry", "dte_min", "dte_max",
                        "carry_lo", "carry_hi", "n_expiries"]


def carry_reference(carry: pd.DataFrame, min_dte: int = 84) -> dict:
    """The carry to quote: the MEDIAN over the expiries at or beyond `min_dte`,
    with the range it hides.

    A fixed dollar error in the forward divides by T, so the shortest expiry
    is the least informative place to read carry -- a $1 miss at 21 days is
    over two percentage points of "rate", and nothing at all at six months.
    That is why short expiries are excluded. But the LONGEST single expiry is
    not the answer either: it is one reading. On 2026-08-28 the 84-, 112- and
    140-day expiries agreed within 3 bp of each other while the 203-day one sat
    38 bp away -- and 203 is exactly what the "longest" rule published. The same
    argument that made `_forward_table` a median over strikes makes this a
    median over expiries.

    The range travels with it: printing the median alone would flatter the
    number by hiding the disagreement, which is the opposite of the point.

    Falls back to the longest expiry available when none qualify (then the
    median is that single reading and lo == hi). All-NaN on an empty frame or
    when no qualifying expiry has a carry.
    """
    empty = {"implied_carry": np.nan, "dte_min": np.nan, "dte_max": np.nan,
             "carry_lo": np.nan, "carry_hi": np.nan, "n_expiries": 0}
    if carry.empty:
        return empty
    qualifying = carry[carry["dte"] >= min_dte]
    src = qualifying if len(qualifying) else carry[carry["dte"] == carry["dte"].max()]
    src = src[src["implied_carry"].notna()]
    if src.empty:
        return empty
    vals = src["implied_carry"].astype(float)
    return {"implied_carry": float(vals.median()),
            "dte_min": float(src["dte"].min()), "dte_max": float(src["dte"].max()),
            "carry_lo": float(vals.min()), "carry_hi": float(vals.max()),
            "n_expiries": int(len(src))}
