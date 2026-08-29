"""P8 hedge-simulation engine tests (SPEC 3 P8). Offline, deterministic."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.hedge_sim import (
    MAX_ENTRY_DTE_ERROR, entry_sessions, select_straddle,
)
from src.analytics.hedge_sim import MIN_MARKET_MARK_SHARE, hedge_shares, simulate_trade

CFG = {
    "symbol": "SPY",
    "chain_filter": {"dte_min": 7, "dte_max": 365},
    "hedge_sim": {"structure": "short_straddle", "entry_dte": 30,
                  "hedge_frequency": "daily", "transaction_cost_bps": 0},
    "rates": {"risk_free_fallback": 0.04, "dividend_yield_fallback": 0.013},
}


def make_chain(session: dt.date, spot: float, expiries: dict[dt.date, float],
               strikes=(95.0, 100.0, 105.0), source="massive-backfill") -> pd.DataFrame:
    """A minimal IV-solved chain: `expiries` maps expiry -> flat IV for that expiry."""
    rows = []
    for expiry, iv in expiries.items():
        for strike in strikes:
            for kind in ("call", "put"):
                intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
                price = intrinsic + 2.0
                rows.append({
                    "snapshot_date": session, "spot": spot, "expiry": expiry,
                    "dte": (expiry - session).days, "strike": strike, "kind": kind,
                    "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                    "close": price, "volume": 10.0, "open_interest": 100.0,
                    "vendor_iv": np.nan, "source": source,
                    "price_used": price, "iv": iv,
                })
    return pd.DataFrame(rows)


class TestEntrySessions:
    def test_one_entry_per_calendar_month(self):
        dates = [dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 30),
                 dt.date(2026, 7, 6), dt.date(2026, 7, 7)]
        assert entry_sessions(dates) == [dt.date(2026, 6, 1), dt.date(2026, 7, 6)]

    def test_first_stored_session_wins_even_when_it_is_not_the_first_trading_day(self):
        # SPEC says "first trading day each month"; the archive may not hold it.
        dates = [dt.date(2026, 8, 14), dt.date(2026, 8, 17)]
        assert entry_sessions(dates) == [dt.date(2026, 8, 14)]

    def test_unsorted_input_is_sorted_first(self):
        dates = [dt.date(2026, 7, 20), dt.date(2026, 7, 2), dt.date(2026, 8, 3)]
        assert entry_sessions(dates) == [dt.date(2026, 7, 2), dt.date(2026, 8, 3)]

    def test_empty(self):
        assert entry_sessions([]) == []


class TestSelectStraddle:
    def test_picks_expiry_nearest_target_dte(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 100.0, {
            dt.date(2026, 8, 21): 0.20,   # 7 dte
            dt.date(2026, 9, 18): 0.18,   # 35 dte -> nearest to 30
            dt.date(2026, 10, 16): 0.19,  # 63 dte
        })
        sel = select_straddle(chain, CFG)
        assert sel["expiry"] == dt.date(2026, 9, 18)
        assert sel["dte"] == 35

    def test_picks_strike_nearest_spot(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 101.0, {dt.date(2026, 9, 18): 0.18})
        sel = select_straddle(chain, CFG)
        assert sel["strike"] == 100.0

    def test_straddle_is_the_sum_of_both_legs_and_iv_is_their_mean(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 100.0, {dt.date(2026, 9, 18): 0.18})
        chain.loc[(chain["strike"] == 100.0) & (chain["kind"] == "put"), "iv"] = 0.22
        sel = select_straddle(chain, CFG)
        assert sel["straddle"] == pytest.approx(sel["call_price"] + sel["put_price"])
        assert sel["call_iv"] == pytest.approx(0.18)
        assert sel["put_iv"] == pytest.approx(0.22)
        assert sel["iv"] == pytest.approx(0.20)

    def test_strike_needs_both_legs_converged(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 100.0, {dt.date(2026, 9, 18): 0.18})
        chain.loc[(chain["strike"] == 100.0) & (chain["kind"] == "put"), "iv"] = np.nan
        sel = select_straddle(chain, CFG)
        assert sel["strike"] in (95.0, 105.0)   # 100 is unusable, a neighbour wins

    def test_returns_none_when_nearest_expiry_is_too_far_from_target(self):
        session = dt.date(2026, 8, 14)
        far = session + dt.timedelta(days=30 + MAX_ENTRY_DTE_ERROR + 1)
        chain = make_chain(session, 100.0, {far: 0.18})
        assert select_straddle(chain, CFG) is None

    def test_returns_none_on_empty_chain(self):
        empty = make_chain(dt.date(2026, 8, 14), 100.0, {dt.date(2026, 9, 18): 0.18}).iloc[0:0]
        assert select_straddle(empty, CFG) is None

    def test_unsupported_structure_raises(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 100.0, {dt.date(2026, 9, 18): 0.18})
        cfg = {**CFG, "hedge_sim": {**CFG["hedge_sim"], "structure": "iron_condor"}}
        with pytest.raises(ValueError, match="iron_condor"):
            select_straddle(chain, cfg)


def make_underlying(start: dt.date, closes: list[float]) -> pd.DataFrame:
    """Consecutive weekday sessions starting at `start` (no weekend logic needed:
    the engine only ever reads the dates present in this frame)."""
    dates = [start + dt.timedelta(days=i) for i in range(len(closes))]
    return pd.DataFrame({"date": dates, "close": closes,
                         "adjusted_close": closes, "volume": [1.0] * len(closes)})


class TestHedgeShares:
    def test_atm_straddle_hedge_is_small_and_positive(self):
        h = hedge_shares(100.0, 100.0, 30 / 365, 0.04, 0.013, 0.20, 0.20)
        assert 0.0 < h < 0.15

    def test_deep_itm_call_dominates(self):
        h = hedge_shares(130.0, 100.0, 30 / 365, 0.04, 0.013, 0.20, 0.20)
        assert h == pytest.approx(1.0, abs=0.02)


class TestSimulateTrade:
    HORIZON = 30      # entry_dte in CFG; select_straddle rejects anything >15 days off it

    def _setup(self, closes, n_chain_days=None, spot0=100.0):
        """A 31-session archive whose single expiry is exactly `HORIZON` days out.

        `closes` is padded by HOLDING ITS LAST VALUE to 31 sessions, so a short
        list describes the opening moves and then a flat tail -- the settlement
        spot is `closes[-1]`.
        """
        entry = dt.date(2026, 6, 1)
        expiry = entry + dt.timedelta(days=self.HORIZON)
        n = self.HORIZON + 1
        series = list(closes) + [closes[-1]] * (n - len(closes))
        u = make_underlying(entry, series)
        n_chain_days = n if n_chain_days is None else n_chain_days
        chains = {}
        for i in range(n_chain_days):
            d = entry + dt.timedelta(days=i)
            chains[d] = make_chain(d, series[i], {expiry: 0.20}, strikes=(95.0, 100.0, 105.0))
        sel = select_straddle(chains[entry], CFG)
        assert sel is not None
        return entry, expiry, sel, chains, u

    def test_day_zero_pv_is_exactly_zero(self):
        entry, expiry, sel, chains, u = self._setup([100.0] * 6)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert daily["pv"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert daily["pnl_day"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert daily["mark_source"].iloc[0] == "market"

    def test_pnl_components_sum_to_daily_pnl_and_to_pv_change(self):
        entry, expiry, sel, chains, u = self._setup(
            [100.0, 101.5, 99.0, 103.0, 98.5, 100.0])
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        parts = (daily["pnl_option"] + daily["pnl_hedge"] + daily["pnl_interest"]
                 + daily["pnl_dividend"] - daily["pnl_cost"])
        assert np.allclose(parts.to_numpy(), daily["pnl_day"].to_numpy(), atol=1e-10)
        assert np.allclose(daily["pv"].diff().dropna().to_numpy(),
                           daily["pnl_day"].iloc[1:].to_numpy(), atol=1e-10)

    def test_settles_at_intrinsic_with_hedge_unwound(self):
        entry, expiry, sel, chains, u = self._setup([100.0, 100.0, 100.0, 107.0])
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        last = daily.iloc[-1]
        assert last["date"] == expiry
        assert last["mark_source"] == "settlement"
        assert last["straddle"] == pytest.approx(abs(107.0 - sel["strike"]))
        assert last["hedge_shares"] == pytest.approx(0.0)
        assert trade["status"] == "settled"
        assert trade["pnl"] == pytest.approx(last["pv"])
        assert trade["exit_date"] == expiry

    def test_trade_is_open_when_underlying_stops_before_expiry(self):
        entry = dt.date(2026, 6, 1)
        expiry = entry + dt.timedelta(days=30)
        u = make_underlying(entry, [100.0, 101.0, 102.0])
        chains = {entry + dt.timedelta(days=i): make_chain(
            entry + dt.timedelta(days=i), 100.0 + i, {expiry: 0.20}) for i in range(3)}
        sel = select_straddle(chains[entry], CFG)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert trade["status"] == "open"
        assert trade["exit_date"] == entry + dt.timedelta(days=2)
        assert not np.isnan(trade["edge"])   # edge is measured to the last mark

    def test_missing_chain_day_is_model_marked_and_counted(self):
        entry, expiry, sel, chains, u = self._setup([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        del chains[entry + dt.timedelta(days=2)]
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        row = daily[daily["date"] == entry + dt.timedelta(days=2)].iloc[0]
        assert row["mark_source"] == "model"
        assert row["straddle"] > 0
        assert trade["n_model_marks"] == 1
        assert trade["n_market_marks"] == trade["n_days"] - 1

    def test_mostly_model_marked_trade_is_reported_sparse(self):
        entry, expiry, sel, chains, u = self._setup([100.0] * 8, n_chain_days=2)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert trade["market_mark_share"] < MIN_MARKET_MARK_SHARE
        assert trade["status"] == "sparse"

    def test_transaction_costs_reduce_pnl(self):
        closes = [100.0, 103.0, 97.0, 104.0, 99.0]
        entry, expiry, sel, chains, u = self._setup(closes)
        free, t_free = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        cfg = {**CFG, "hedge_sim": {**CFG["hedge_sim"], "transaction_cost_bps": 50}}
        costed, t_costed = simulate_trade(entry, sel, chains, u, 0.04, 0.013, cfg)
        assert costed["pnl_cost"].sum() > 0
        assert t_costed["pnl"] < t_free["pnl"]

    def test_short_straddle_makes_money_when_realized_vol_is_far_below_implied(self):
        # Sold at 20 vol, the world does not move at all: the seller keeps the premium.
        entry, expiry, sel, chains, u = self._setup([100.0] * 8)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert trade["pnl"] > 0
        assert trade["lifetime_rv"] == pytest.approx(0.0, abs=1e-9)
        assert trade["edge"] == pytest.approx(trade["entry_iv"] - trade["lifetime_rv"])

    def test_unsupported_hedge_frequency_raises(self):
        entry, expiry, sel, chains, u = self._setup([100.0] * 4)
        cfg = {**CFG, "hedge_sim": {**CFG["hedge_sim"], "hedge_frequency": "weekly"}}
        with pytest.raises(ValueError, match="weekly"):
            simulate_trade(entry, sel, chains, u, 0.04, 0.013, cfg)

    def test_daily_frame_has_the_declared_columns(self):
        from src.analytics.hedge_sim import DAILY_COLUMNS
        entry, expiry, sel, chains, u = self._setup([100.0] * 5)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert list(daily.columns) == DAILY_COLUMNS
