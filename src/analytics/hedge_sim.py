"""P8: delta-hedged short-straddle simulation (SPEC 3 P8).

Stateless by construction: nothing here reads or writes sim state. Every
number is recomputed from stored chains + the underlying close series on
every run, so a later fix to this module heals the whole P&L history.

Position convention: SHORT one straddle on ONE share (no 100x contract
multiplier), so every dollar figure is per share. `H` is shares held long;
delta-neutrality against a short straddle means H = delta_call + delta_put.
"""
import datetime as dt

import numpy as np
import pandas as pd

# A monthly ladder puts the nearest expiry within ~15 days of any target, so a
# larger error means the chain is degenerate -- skip the month rather than
# quietly entering a 7-day trade and calling it a 30-day one.
MAX_ENTRY_DTE_ERROR = 15

# Below this share of market-quoted marks a trade is reported as `sparse` and
# kept off the scatter: its P&L is then mostly our own model talking to itself.
MIN_MARKET_MARK_SHARE = 0.7

TRADE_COLUMNS = [
    "entry_date", "expiry", "strike", "dte_at_entry", "entry_iv", "entry_straddle",
    "exit_date", "exit_value", "pnl", "n_days", "n_market_marks", "n_model_marks",
    "market_mark_share", "lifetime_rv", "edge", "status",
]

DAILY_COLUMNS = [
    "date", "entry_date", "expiry", "strike", "spot", "straddle", "call_iv", "put_iv",
    "hedge_shares", "cash", "pv", "pnl_day", "pnl_option", "pnl_hedge",
    "pnl_interest", "pnl_dividend", "pnl_cost", "mark_source",
]


def entry_sessions(session_dates) -> list[dt.date]:
    """The first stored session of each calendar month.

    SPEC says "the first trading day each month". The chain archive does not
    necessarily hold that day, so the rule is the first session we actually
    stored in the month; the page caption states this.
    """
    out: list[dt.date] = []
    seen: set[tuple[int, int]] = set()
    for d in sorted(session_dates):
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def select_straddle(chain_iv: pd.DataFrame, cfg: dict) -> dict | None:
    """The ATM straddle to sell on this session, or None if the chain cannot carry one."""
    structure = cfg["hedge_sim"]["structure"]
    if structure != "short_straddle":
        raise ValueError(f"unsupported hedge_sim.structure: {structure}")
    usable = chain_iv[chain_iv["iv"].notna() & chain_iv["price_used"].notna()]
    if usable.empty:
        return None
    target = int(cfg["hedge_sim"]["entry_dte"])
    available = usable[["expiry", "dte"]].drop_duplicates().sort_values("dte")
    pick = available.iloc[(available["dte"] - target).abs().to_numpy().argmin()]
    if abs(int(pick["dte"]) - target) > MAX_ENTRY_DTE_ERROR:
        return None

    g = usable[usable["expiry"] == pick["expiry"]]
    spot = float(g["spot"].iloc[0])
    legs = g.pivot_table(index="strike", columns="kind",
                         values=["price_used", "iv"], aggfunc="first")
    both = legs.dropna()
    if both.empty or "call" not in legs["price_used"].columns \
            or "put" not in legs["price_used"].columns:
        return None
    strikes = both.index.to_numpy(dtype=float)
    strike = float(strikes[np.abs(strikes - spot).argmin()])
    row = both.loc[strike]
    call_price, put_price = float(row[("price_used", "call")]), float(row[("price_used", "put")])
    call_iv, put_iv = float(row[("iv", "call")]), float(row[("iv", "put")])
    return {
        "expiry": pick["expiry"], "dte": int(pick["dte"]), "strike": strike,
        "call_price": call_price, "put_price": put_price,
        "straddle": call_price + put_price,
        "call_iv": call_iv, "put_iv": put_iv, "iv": 0.5 * (call_iv + put_iv),
    }
