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

from src.models.black_scholes import bs_price, greeks
from src.models.realized_vol import realized_vol_between

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


def hedge_shares(spot: float, strike: float, T: float, r: float, q: float,
                 call_iv: float, put_iv: float) -> float:
    """Shares to hold long so the SHORT straddle is delta-neutral.

    Each leg is evaluated at ITS OWN market implied vol -- the same rule P4
    uses for the Greeks curves -- so the hedge is what a desk reading these
    quotes would actually put on, not a flat-vol textbook number.
    """
    if T <= 0:
        return 0.0
    dc = greeks(S=spot, K=strike, T=T, r=r, sigma=call_iv, q=q, kind="call")["delta"]
    dp = greeks(S=spot, K=strike, T=T, r=r, sigma=put_iv, q=q, kind="put")["delta"]
    return float(dc + dp)


def _quote(chain_iv: pd.DataFrame | None, expiry, strike: float) -> tuple[float, float, float] | None:
    """(straddle value, call_iv, put_iv) for this leg pair, or None if not quoted."""
    if chain_iv is None or chain_iv.empty:
        return None
    g = chain_iv[(chain_iv["expiry"] == expiry) & (chain_iv["strike"] == strike)
                 & chain_iv["iv"].notna() & chain_iv["price_used"].notna()]
    legs = {kind: g[g["kind"] == kind] for kind in ("call", "put")}
    if any(v.empty for v in legs.values()):
        return None
    call, put = legs["call"].iloc[0], legs["put"].iloc[0]
    return (float(call["price_used"]) + float(put["price_used"]),
            float(call["iv"]), float(put["iv"]))


def simulate_trade(entry_date: dt.date, sel: dict, chains: dict, underlying: pd.DataFrame,
                   r: float, q: float, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Replay one short straddle from entry to settlement (or to the last stored session)."""
    frequency = cfg["hedge_sim"]["hedge_frequency"]
    if frequency != "daily":
        raise ValueError(f"unsupported hedge_sim.hedge_frequency: {frequency}")
    bps = float(cfg["hedge_sim"]["transaction_cost_bps"])
    expiry, strike = sel["expiry"], sel["strike"]

    u = underlying.sort_values("date")
    closes = dict(zip(u["date"], u["close"].astype(float)))
    days = [d for d in sorted(closes) if entry_date <= d <= expiry]
    # A holiday expiry never appears in the calendar, so settlement is the last
    # session at or before it -- and only once the calendar has actually reached it.
    settle_date = days[-1] if days and max(closes) >= expiry else None

    spot = closes[entry_date]
    call_iv, put_iv = sel["call_iv"], sel["put_iv"]
    value = float(sel["straddle"])
    shares = hedge_shares(spot, strike, (expiry - entry_date).days / 365.0, r, q, call_iv, put_iv)
    cash = value - shares * spot
    rows = [{
        "date": entry_date, "entry_date": entry_date, "expiry": expiry, "strike": strike,
        "spot": spot, "straddle": value, "call_iv": call_iv, "put_iv": put_iv,
        "hedge_shares": shares, "cash": cash, "pv": cash + shares * spot - value,
        "pnl_day": 0.0, "pnl_option": 0.0, "pnl_hedge": 0.0, "pnl_interest": 0.0,
        "pnl_dividend": 0.0, "pnl_cost": 0.0, "mark_source": "market",
    }]
    n_market = n_model = 0

    for day in days[1:]:
        gap = (day - rows[-1]["date"]).days
        prev = rows[-1]
        new_spot = closes[day]
        if day == settle_date:
            new_value, source, new_shares = abs(new_spot - strike), "settlement", 0.0
        else:
            quoted = _quote(chains.get(day), expiry, strike)
            if quoted is not None:
                new_value, call_iv, put_iv = quoted
                source = "market"
                n_market += 1
            else:
                T = (expiry - day).days / 365.0
                new_value = float(
                    bs_price(new_spot, strike, T, r, call_iv, q, "call")
                    + bs_price(new_spot, strike, T, r, put_iv, q, "put"))
                source = "model"
                n_model += 1
            new_shares = hedge_shares(new_spot, strike, (expiry - day).days / 365.0,
                                      r, q, call_iv, put_iv)

        interest = prev["cash"] * (np.exp(r * gap / 365.0) - 1.0)
        dividend = prev["hedge_shares"] * prev["spot"] * (np.exp(q * gap / 365.0) - 1.0)
        hedge = prev["hedge_shares"] * (new_spot - prev["spot"])
        option = -(new_value - prev["straddle"])
        cost = abs(new_shares - prev["hedge_shares"]) * new_spot * bps / 1e4
        cash = prev["cash"] + interest + dividend - (new_shares - prev["hedge_shares"]) * new_spot - cost
        rows.append({
            "date": day, "entry_date": entry_date, "expiry": expiry, "strike": strike,
            "spot": new_spot, "straddle": new_value, "call_iv": call_iv, "put_iv": put_iv,
            "hedge_shares": new_shares, "cash": cash,
            "pv": cash + new_shares * new_spot - new_value,
            "pnl_day": interest + dividend + hedge + option - cost,
            "pnl_option": option, "pnl_hedge": hedge, "pnl_interest": interest,
            "pnl_dividend": dividend, "pnl_cost": cost, "mark_source": source,
        })
        if day == settle_date:
            n_market += 1     # settlement is a market fact, not a model mark

    daily = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    last = daily.iloc[-1]
    n_days = len(daily) - 1
    share = (n_market / n_days) if n_days else 1.0
    settled = settle_date is not None and last["date"] == settle_date
    status = "open" if not settled else ("settled" if share >= MIN_MARKET_MARK_SHARE else "sparse")
    lifetime_rv = realized_vol_between(underlying, entry_date, last["date"])
    trade = {
        "entry_date": entry_date, "expiry": expiry, "strike": strike,
        "dte_at_entry": int(sel["dte"]), "entry_iv": float(sel["iv"]),
        "entry_straddle": float(sel["straddle"]), "exit_date": last["date"],
        "exit_value": float(last["straddle"]), "pnl": float(last["pv"]),
        "n_days": int(n_days), "n_market_marks": int(n_market), "n_model_marks": int(n_model),
        "market_mark_share": float(share), "lifetime_rv": lifetime_rv,
        "edge": float(sel["iv"]) - lifetime_rv, "status": status,
    }
    return daily, trade
