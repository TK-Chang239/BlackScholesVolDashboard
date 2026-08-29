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

    def test_settles_on_last_session_before_a_holiday_expiry_that_is_absent_from_the_calendar(self):
        # `expiry` itself is never a stored session (a market holiday); the archive
        # keeps trading sessions AFTER it. Settlement must still fall back to the
        # last session at or before expiry, not stay "open" just because expiry
        # itself never shows up as a key.
        entry = dt.date(2026, 6, 1)
        horizon = 30
        expiry = entry + dt.timedelta(days=horizon)
        settle_date = expiry - dt.timedelta(days=1)
        pre = [entry + dt.timedelta(days=i) for i in range(horizon)]        # entry..settle_date
        post = [expiry + dt.timedelta(days=i) for i in (1, 2)]              # continues PAST expiry
        pre_closes = [100.0] * (horizon - 1) + [107.0]                      # settle_date moves ITM
        post_closes = [107.0, 107.0]
        u = pd.DataFrame({
            "date": pre + post, "close": pre_closes + post_closes,
            "adjusted_close": pre_closes + post_closes, "volume": [1.0] * (len(pre) + len(post)),
        })
        chains = {d: make_chain(d, c, {expiry: 0.20}, strikes=(95.0, 100.0, 105.0))
                  for d, c in zip(pre, pre_closes)}
        sel = select_straddle(chains[entry], CFG)
        assert sel is not None
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        last = daily.iloc[-1]
        assert last["date"] == settle_date
        assert last["mark_source"] == "settlement"
        assert last["straddle"] == pytest.approx(abs(107.0 - sel["strike"]))
        assert last["hedge_shares"] == pytest.approx(0.0)
        assert trade["status"] == "settled"
        assert trade["exit_date"] == settle_date

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

    def test_chain_present_but_missing_the_quoted_pair_is_model_marked(self):
        # Distinct from a wholly missing chain day: this chain day EXISTS in `chains`,
        # but the ATM strike's call leg has a null iv, so `_quote`'s notna filter
        # (not a missing key) is what forces the fallback to the model mark.
        entry, expiry, sel, chains, u = self._setup([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        d = entry + dt.timedelta(days=2)
        c = chains[d].copy()
        c.loc[(c["strike"] == sel["strike"]) & (c["kind"] == "call"), "iv"] = np.nan
        chains[d] = c
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        row = daily[daily["date"] == d].iloc[0]
        assert row["mark_source"] == "model"
        assert trade["n_model_marks"] == 1
        assert trade["n_market_marks"] == trade["n_days"] - 1

    def test_model_marked_row_carries_forward_the_prior_ivs(self):
        entry, expiry, sel, chains, u = self._setup([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        d = entry + dt.timedelta(days=2)
        del chains[d]
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        prev_row = daily[daily["date"] == d - dt.timedelta(days=1)].iloc[0]
        row = daily[daily["date"] == d].iloc[0]
        assert row["mark_source"] == "model"
        assert row["call_iv"] == pytest.approx(prev_row["call_iv"])
        assert row["put_iv"] == pytest.approx(prev_row["put_iv"])

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
        # `pd.DataFrame(rows, columns=DAILY_COLUMNS)` forces this column list even if a
        # row-dict key were renamed -- that renamed column would just come back all-NaN.
        # Guard against that: every declared column must actually carry data every day.
        assert daily.notna().all().all()


from src.analytics.hedge_sim import (
    TRADE_COLUMNS, fit_pnl_vs_edge, hedge_summary, portfolio_daily, simulate,
)


class TestSimulate:
    def _archive(self, months, closes_per_month=8):
        """One entry per month; each trade's expiry is 5 sessions after entry."""
        chains, closes, dates = {}, [], []
        day = dt.date(2026, 6, 1)
        for _ in range(months * closes_per_month):
            dates.append(day)
            closes.append(100.0)
            day += dt.timedelta(days=4)      # ~4-day steps walk through several months
        u = pd.DataFrame({"date": dates, "close": closes, "adjusted_close": closes,
                          "volume": [1.0] * len(dates)})
        for i, d in enumerate(dates):
            expiry = dates[min(i + 5, len(dates) - 1)]
            chains[d] = make_chain(d, 100.0, {expiry: 0.20})
        return chains, u

    def test_one_trade_per_month_with_declared_columns(self):
        chains, u = self._archive(months=2)
        trades, daily = simulate(chains, u, 0.04, 0.013, CFG)
        assert list(trades.columns) == TRADE_COLUMNS
        assert len(trades) == trades["entry_date"].nunique()
        months = {(d.year, d.month) for d in trades["entry_date"]}
        assert len(months) == len(trades)

    def test_empty_archive_returns_empty_frames_with_columns(self):
        from src.analytics.hedge_sim import DAILY_COLUMNS
        u = pd.DataFrame({"date": [], "close": [], "adjusted_close": [], "volume": []})
        trades, daily = simulate({}, u, 0.04, 0.013, CFG)
        assert list(trades.columns) == TRADE_COLUMNS and trades.empty
        assert list(daily.columns) == DAILY_COLUMNS and daily.empty


class TestPortfolioDaily:
    def test_sums_overlapping_trades_and_accumulates(self):
        daily = pd.DataFrame({
            "date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 2)],
            "entry_date": [dt.date(2026, 6, 1), dt.date(2026, 6, 1), dt.date(2026, 6, 2)],
            "pnl_day": [0.0, 1.5, 0.0],
        })
        port = portfolio_daily(daily)
        assert list(port.columns) == ["date", "pnl_day", "pnl_cum", "n_open"]
        assert port["pnl_day"].tolist() == [0.0, 1.5]
        assert port["pnl_cum"].tolist() == [0.0, 1.5]
        assert port["n_open"].tolist() == [1, 2]

    def test_empty(self):
        port = portfolio_daily(pd.DataFrame(columns=["date", "entry_date", "pnl_day"]))
        assert port.empty and list(port.columns) == ["date", "pnl_day", "pnl_cum", "n_open"]


class TestFit:
    def test_recovers_a_known_line(self):
        trades = pd.DataFrame({
            "edge": [0.01, 0.02, 0.03, 0.04],           # 1..4 vol points
            "pnl": [2.0, 4.0, 6.0, 8.0],                 # slope 2 $/vol point
            "status": ["settled"] * 4,
        })
        fit = fit_pnl_vs_edge(trades)
        assert fit["n"] == 4
        assert fit["slope"] == pytest.approx(2.0)
        assert fit["intercept"] == pytest.approx(0.0, abs=1e-9)
        assert fit["r2"] == pytest.approx(1.0)

    def test_ignores_open_and_sparse_trades(self):
        trades = pd.DataFrame({
            "edge": [0.01, 0.02, 0.03],
            "pnl": [2.0, 4.0, 999.0],
            "status": ["settled", "settled", "open"],
        })
        assert fit_pnl_vs_edge(trades)["n"] == 2

    def test_fewer_than_two_points_gives_no_line(self):
        trades = pd.DataFrame({"edge": [0.01], "pnl": [2.0], "status": ["settled"]})
        fit = fit_pnl_vs_edge(trades)
        assert fit["n"] == 1
        assert np.isnan(fit["slope"]) and np.isnan(fit["r2"])

    def test_nan_edge_is_dropped(self):
        trades = pd.DataFrame({
            "edge": [0.01, np.nan, 0.03], "pnl": [2.0, 4.0, 6.0],
            "status": ["settled"] * 3,
        })
        assert fit_pnl_vs_edge(trades)["n"] == 2


class TestHedgeSummary:
    def test_counts_statuses_and_reports_the_next_settlement(self):
        trades = pd.DataFrame({
            "entry_date": [dt.date(2026, 6, 1), dt.date(2026, 7, 1)],
            "expiry": [dt.date(2026, 7, 17), dt.date(2026, 8, 21)],
            "exit_date": [dt.date(2026, 7, 17), dt.date(2026, 8, 12)],
            "pnl": [1.0, -0.5], "edge": [0.01, 0.02],
            "n_days": [30, 20], "n_market_marks": [27, 20], "n_model_marks": [3, 0],
            "market_mark_share": [0.9, 1.0], "status": ["settled", "open"],
        })
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2)],
                             "pnl_day": [0.0, 1.0], "pnl_cum": [0.0, 1.0], "n_open": [1, 1]})
        s = hedge_summary(trades, port, {"n": 1, "slope": np.nan,
                                         "intercept": np.nan, "r2": np.nan})
        assert (s["n_trades"], s["n_settled"], s["n_open"], s["n_sparse"]) == (2, 1, 1, 0)
        assert s["next_settlement"] == dt.date(2026, 8, 21)
        assert s["cum_pnl"] == pytest.approx(1.0)
        assert s["first_entry"] == dt.date(2026, 6, 1)
        assert s["market_mark_share"] == pytest.approx(47 / 50)

    def test_empty_summary_is_all_zeros_and_nones(self):
        s = hedge_summary(pd.DataFrame(columns=TRADE_COLUMNS),
                          pd.DataFrame(columns=["date", "pnl_day", "pnl_cum", "n_open"]),
                          {"n": 0, "slope": np.nan, "intercept": np.nan, "r2": np.nan})
        assert s["n_trades"] == 0 and s["first_entry"] is None
        assert s["next_settlement"] is None
        assert np.isnan(s["cum_pnl"])
