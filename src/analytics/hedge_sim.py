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
# The share is taken over QUOTABLE sessions only -- see MODEL_MARK_SOURCES.
MIN_MARKET_MARK_SHARE = 0.7

# A mark is either a market fact or our own model, and a MODEL mark comes from
# one of two situations that a single "model" label conflated:
#
#   model_structural -- the session is inside `chain_filter.dte_min`, so the
#       trade's own expiry is absent from every stored chain BY DESIGN. The
#       filter that builds the archive drops it; no amount of backfilling can
#       put it back. Any trade held to expiry ends on a run of these days.
#   model_gap        -- the archive genuinely lacks a usable quote on a session
#       where one could have existed. This is missing data.
#
# Only `model_gap` says anything about how well the archive covers a trade, so
# only it belongs in the denominator of `quotable_mark_share`, the share
# `MIN_MARKET_MARK_SHARE` gates on. Counting the structural days there made the
# threshold unreachable for the ~16-18 day tenors the entry rule actually
# produces (7 quotable marks out of 11 sessions = 0.64), so every such trade was
# permanently `sparse` and the scatter could never populate.
#
# The render layer imports this to keep BOTH kinds out of the daily-P&L
# histogram: the split decides which trades are well enough quoted to plot, not
# whether a given mark carries a vol move (a structural mark carries none).
MODEL_MARK_SOURCES = ("model_structural", "model_gap")

TRADE_COLUMNS = [
    "entry_date", "expiry", "strike", "dte_at_entry", "entry_iv", "entry_straddle",
    "exit_date", "exit_value", "pnl", "n_days", "n_market_marks", "n_model_marks",
    "n_structural_marks", "market_mark_share", "quotable_mark_share",
    "lifetime_rv", "edge", "status",
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
                   r: float, q: float, cfg: dict) -> tuple[pd.DataFrame, dict] | None:
    """Replay one short straddle from entry to settlement (or to the last stored session).

    None when the entry session has no close in `underlying` -- there is no
    spot to open against, so the month is skipped rather than raising.
    """
    frequency = cfg["hedge_sim"]["hedge_frequency"]
    if frequency != "daily":
        raise ValueError(f"unsupported hedge_sim.hedge_frequency: {frequency}")
    bps = float(cfg["hedge_sim"]["transaction_cost_bps"])
    # The archive's own filter: no stored chain ever holds an expiry closer than
    # this, so a missing quote inside it is by construction, not by omission.
    dte_min = int(cfg["chain_filter"]["dte_min"])
    expiry, strike = sel["expiry"], sel["strike"]

    u = underlying.sort_values("date")
    closes = dict(zip(u["date"], u["close"].astype(float)))
    # A stored chain can carry a session the underlying history does not (a
    # vendor gap, a late revision). Unguarded, that one absent close raised a
    # KeyError out of the whole daily run; the replay is meant to tolerate
    # exactly this data condition, so skip the month instead.
    if entry_date not in closes:
        return None
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
    n_market = n_model = n_structural = 0

    for day in days[1:]:
        gap = (day - rows[-1]["date"]).days
        prev = rows[-1]
        new_spot = closes[day]
        dte = (expiry - day).days
        T = dte / 365.0                   # single clock for both the mark and the hedge below
        if day == settle_date:
            new_value, source, new_shares = abs(new_spot - strike), "settlement", 0.0
            n_market += 1     # settlement is a market fact, not a model mark
        else:
            quoted = _quote(chains.get(day), expiry, strike)
            if quoted is not None:
                new_value, call_iv, put_iv = quoted
                source = "market"
                n_market += 1
            else:
                new_value = float(
                    bs_price(new_spot, strike, T, r, call_iv, q, "call")
                    + bs_price(new_spot, strike, T, r, put_iv, q, "put"))
                # `dte < dte_min` is not a judgement about this session's data:
                # the archive's filter removed the expiry, so no chain here or
                # anywhere could have quoted it. Anything else is a real gap.
                structural = dte < dte_min
                source = "model_structural" if structural else "model_gap"
                n_model += 1
                n_structural += int(structural)
            new_shares = hedge_shares(new_spot, strike, T, r, q, call_iv, put_iv)

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

    daily = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    last = daily.iloc[-1]
    n_days = len(daily) - 1
    # Two shares, and the page states both. `market_mark_share` is how much of
    # the trade was modelled AT ALL -- the honest headline. `quotable_mark_share`
    # is how much was modelled AVOIDABLY, over the sessions a stored chain could
    # have quoted this straddle on, and that is the only one a trade can be
    # judged by: the structural days are a property of the archive's filter, not
    # of this trade's coverage.
    n_quotable = n_days - n_structural
    share = (n_market / n_days) if n_days else 1.0
    quotable_share = (n_market / n_quotable) if n_quotable else 1.0
    settled = settle_date is not None and last["date"] == settle_date
    status = ("open" if not settled else
              ("settled" if quotable_share >= MIN_MARKET_MARK_SHARE else "sparse"))
    lifetime_rv = realized_vol_between(underlying, entry_date, last["date"])
    trade = {
        "entry_date": entry_date, "expiry": expiry, "strike": strike,
        "dte_at_entry": int(sel["dte"]), "entry_iv": float(sel["iv"]),
        "entry_straddle": float(sel["straddle"]), "exit_date": last["date"],
        "exit_value": float(last["straddle"]), "pnl": float(last["pv"]),
        "n_days": int(n_days), "n_market_marks": int(n_market), "n_model_marks": int(n_model),
        "n_structural_marks": int(n_structural),
        "market_mark_share": float(share), "quotable_mark_share": float(quotable_share),
        "lifetime_rv": lifetime_rv,
        "edge": float(sel["iv"]) - lifetime_rv, "status": status,
    }
    return daily, trade


def simulate(chains: dict, underlying: pd.DataFrame, r: float, q: float,
             cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Replay every monthly trade the stored archive supports.

    Third return value is the number of months whose first stored session could
    not seed a trade -- no expiry near the target, no strike with both legs
    solved, or no underlying close. A silently dropped month is a selection the
    reader cannot see, and on the real archive it is not hypothetical: a
    holiday-shifted monthly leaves the near expiry out of the ladder and the
    whole month goes missing.
    """
    trades, frames, skipped = [], [], 0
    for entry in entry_sessions(list(chains)):
        sel = select_straddle(chains[entry], cfg)
        result = (simulate_trade(entry, sel, chains, underlying, r, q, cfg)
                  if sel is not None else None)
        if result is None:
            skipped += 1
            continue
        daily, trade = result
        frames.append(daily)
        trades.append(trade)
    daily_all = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=DAILY_COLUMNS))
    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return (trades_df.sort_values("entry_date").reset_index(drop=True), daily_all,
            skipped)


def portfolio_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Total P&L per calendar day across all live trades, and its running sum."""
    cols = ["date", "pnl_day", "pnl_cum", "n_open"]
    if daily.empty:
        return pd.DataFrame(columns=cols)
    g = (daily.groupby("date", as_index=False)
         .agg(pnl_day=("pnl_day", "sum"), n_open=("entry_date", "nunique"))
         .sort_values("date").reset_index(drop=True))
    g["pnl_cum"] = g["pnl_day"].cumsum()
    return g[cols]


def fit_pnl_vs_edge(trades: pd.DataFrame) -> dict:
    """Least-squares line through settled trades: P&L ($) vs edge (VOL POINTS)."""
    out = {"n": 0, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    if trades.empty:
        return out
    s = trades[(trades["status"] == "settled") & trades["edge"].notna()
               & trades["pnl"].notna()]
    out["n"] = int(len(s))
    if len(s) < 2:
        return out
    x = s["edge"].to_numpy(dtype=float) * 100.0
    y = s["pnl"].to_numpy(dtype=float)
    if np.ptp(x) == 0:            # a vertical cloud has no line through it
        return out
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out.update(slope=float(slope), intercept=float(intercept),
               r2=(1.0 - float((resid ** 2).sum()) / ss_tot) if ss_tot > 0 else float("nan"))
    return out


def hedge_summary(trades: pd.DataFrame, port: pd.DataFrame, fit: dict,
                  n_months_skipped: int = 0, cost_bps: float = 0.0,
                  dte_min: int | None = None) -> dict:
    """Everything the page's stat line and status.json need, in one dict.

    "Reached expiry" and "plottable" are DIFFERENT questions and the page needs
    both. `sparse` is a sub-kind of settled, so `n_settled` alone -- which means
    "settled AND carries enough market marks to plot" -- answered "has anything
    finished?" with a no while a trade sat finished in the P&L. `n_reached_expiry`
    answers that one; `n_settled` still gates the scatter and the fit.

    `dte_at_entry_min/max` are here because the entry rule takes the monthly
    expiry NEAREST the target, and a monthly ladder read from the first session
    of a month rarely offers one near it: the realized tenor is bimodal, round-trip
    P&L scales roughly with sqrt(T), and unreported that lands on the scatter as
    unexplained vertical spread.

    `cost_bps` carries `cfg["hedge_sim"]["transaction_cost_bps"]` through so the
    render layer can state the "no spread crossed" clause honestly without
    importing config itself -- `simulate_trade` already charges this cost on
    every hedge trade once it is non-zero, so a config flip must not leave a
    stale claim standing on the page. `dte_min` travels for the same reason:
    the stat line has to name the window inside which a mark can only ever be
    modelled, and that number lives in `cfg["chain_filter"]["dte_min"]`.

    BOTH mark shares are published. `market_mark_share` counts every session, so
    the page can say how much of the P&L is our own model at all;
    `quotable_mark_share` counts only the sessions a stored chain could have
    quoted, which is what a trade's `settled`/`sparse` status is decided on.
    Publishing only the first would hide why a 64%-market trade is plottable;
    publishing only the second would quietly under-report the modelling.
    """
    counts = trades["status"].value_counts().to_dict() if not trades.empty else {}
    open_trades = trades[trades["status"] == "open"] if not trades.empty else trades
    marks = int(trades["n_days"].sum()) if not trades.empty else 0
    n_market = int(trades["n_market_marks"].sum()) if not trades.empty else 0
    n_model = int(trades["n_model_marks"].sum()) if not trades.empty else 0
    n_structural = int(trades["n_structural_marks"].sum()) if not trades.empty else 0
    n_quotable = marks - n_structural
    n_settled = int(counts.get("settled", 0))
    n_sparse = int(counts.get("sparse", 0))
    dtes = trades["dte_at_entry"].dropna() if not trades.empty else None
    return {
        "n_trades": int(len(trades)),
        "n_settled": n_settled,
        "n_open": int(counts.get("open", 0)),
        "n_sparse": n_sparse,
        "n_reached_expiry": n_settled + n_sparse,
        "n_months_skipped": int(n_months_skipped),
        "first_entry": trades["entry_date"].min() if not trades.empty else None,
        "cum_pnl": float(port["pnl_cum"].iloc[-1]) if not port.empty else float("nan"),
        "n_days": int(len(port)),
        "dte_at_entry_min": int(dtes.min()) if dtes is not None and len(dtes) else None,
        "dte_at_entry_max": int(dtes.max()) if dtes is not None and len(dtes) else None,
        "market_mark_share": (float(n_market / marks) if marks else float("nan")),
        "quotable_mark_share": (float(n_market / n_quotable) if n_quotable
                                else float("nan")),
        "n_model_marks": n_model,
        "n_structural_marks": n_structural,
        "n_gap_marks": n_model - n_structural,
        "next_settlement": (open_trades["expiry"].min() if len(open_trades) else None),
        "slope": fit["slope"], "r2": fit["r2"],
        "cost_bps": float(cost_bps),
        "dte_min": None if dte_min is None else int(dte_min),
    }
