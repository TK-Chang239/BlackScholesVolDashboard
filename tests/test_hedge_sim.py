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
        # 28 days from expiry, far outside chain_filter.dte_min: the archive
        # COULD have quoted this session and did not, so it is a real gap.
        assert row["mark_source"] == "model_gap"
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
        assert row["mark_source"] == "model_gap"
        assert trade["n_model_marks"] == 1
        assert trade["n_market_marks"] == trade["n_days"] - 1

    def test_model_marked_row_carries_forward_the_prior_ivs(self):
        entry, expiry, sel, chains, u = self._setup([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        d = entry + dt.timedelta(days=2)
        del chains[d]
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        prev_row = daily[daily["date"] == d - dt.timedelta(days=1)].iloc[0]
        row = daily[daily["date"] == d].iloc[0]
        assert row["mark_source"] == "model_gap"
        assert row["call_iv"] == pytest.approx(prev_row["call_iv"])
        assert row["put_iv"] == pytest.approx(prev_row["put_iv"])

    def test_mostly_model_marked_trade_is_reported_sparse(self):
        entry, expiry, sel, chains, u = self._setup([100.0] * 8, n_chain_days=2)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert trade["quotable_mark_share"] < MIN_MARKET_MARK_SHARE
        assert trade["status"] == "sparse"

    # ---- structural vs gap model marks (the defect a real run exposed) ----
    #
    # `chain_filter.dte_min` means a stored chain never holds an expiry closer
    # than that, so over its final `dte_min` days a trade's own expiry is absent
    # from EVERY stored chain by design. Counting those days against the trade
    # made MIN_MARKET_MARK_SHARE unreachable for the ~16-18 day tenors the entry
    # rule actually produces: on the real archive the July 2026 trade topped out
    # at 7/11 = 0.64 and was permanently `sparse`, so the scatter -- the whole
    # point of the panel -- could never populate.

    def _structural_setup(self, horizon=16, dte_min=7):
        """A short-tenor trade whose ONLY unquoted days are inside `dte_min`.

        Every session the archive's filter permits carries a chain; nothing at
        all is missing. `horizon=16` is the tenor the real entry rule delivers
        (the monthly ladder offers ~17 days or ~46, never ~30), which is exactly
        the case the blended share could not clear.
        """
        cfg = {**CFG, "chain_filter": {**CFG["chain_filter"], "dte_min": dte_min},
               "hedge_sim": {**CFG["hedge_sim"], "entry_dte": horizon}}
        entry = dt.date(2026, 6, 1)
        expiry = entry + dt.timedelta(days=horizon)
        u = make_underlying(entry, [100.0] * (horizon + 1))
        chains = {entry + dt.timedelta(days=i):
                  make_chain(entry + dt.timedelta(days=i), 100.0, {expiry: 0.20})
                  for i in range(horizon - dte_min + 1)}    # last stored chain is at dte_min
        sel = select_straddle(chains[entry], cfg)
        assert sel is not None
        return entry, expiry, sel, chains, u, cfg

    def test_a_trade_unquoted_only_inside_dte_min_settles(self):
        entry, expiry, sel, chains, u, cfg = self._structural_setup()
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, cfg)
        assert trade["n_model_marks"] == trade["n_structural_marks"] == 6
        # The blend CANNOT clear the bar: 10 of 16 sessions, and no backfill
        # could add a seventeenth. That is what made this trade unplottable.
        assert trade["market_mark_share"] < MIN_MARKET_MARK_SHARE
        # Judged over the sessions a chain could have quoted, it is perfect.
        assert trade["quotable_mark_share"] == pytest.approx(1.0)
        assert trade["status"] == "settled"

    def test_a_trade_starved_of_chains_is_still_sparse(self):
        # The other side of the same discrimination: this archive holds the entry
        # chain and nothing else, so 9 sessions that COULD have been quoted were
        # not. Excusing the structural days must not excuse these.
        entry, expiry, sel, chains, u, cfg = self._structural_setup()
        daily, trade = simulate_trade(entry, sel, {entry: chains[entry]}, u,
                                      0.04, 0.013, cfg)
        assert trade["n_structural_marks"] == 6      # unchanged: a property of the tenor
        assert trade["n_model_marks"] == 15
        assert trade["quotable_mark_share"] == pytest.approx(0.1)   # settlement only
        assert trade["status"] == "sparse"

    def test_model_marks_are_labelled_structural_or_gap_by_days_to_expiry(self):
        entry, expiry, sel, chains, u, cfg = self._structural_setup()
        daily, _ = simulate_trade(entry, sel, {entry: chains[entry]}, u, 0.04, 0.013, cfg)
        source = dict(zip(daily["date"], daily["mark_source"]))
        assert source[entry + dt.timedelta(days=9)] == "model_gap"          # 7 dte
        assert source[entry + dt.timedelta(days=10)] == "model_structural"  # 6 dte

    def test_the_structural_window_is_read_from_config_not_hard_coded(self):
        # dte_min is a config tunable (SPEC 6). Widening the filter must move the
        # boundary; 7 must never be baked into the engine.
        entry, expiry, sel, chains, u, cfg = self._structural_setup()
        starved = {entry: chains[entry]}
        _, seven = simulate_trade(entry, sel, starved, u, 0.04, 0.013, cfg)
        wider = {**cfg, "chain_filter": {**cfg["chain_filter"], "dte_min": 11}}
        _, eleven = simulate_trade(entry, sel, starved, u, 0.04, 0.013, wider)
        assert seven["n_structural_marks"] == 6      # 6..1 dte
        assert eleven["n_structural_marks"] == 10    # 10..1 dte
        assert seven["n_model_marks"] == eleven["n_model_marks"] == 15

    def test_settlement_stays_a_market_mark_though_its_dte_is_inside_the_window(self):
        # Settlement is intrinsic from the underlying close -- a market fact, at
        # 0 dte. It belongs in the numerator, not in the excluded denominator.
        entry, expiry, sel, chains, u, cfg = self._structural_setup()
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, cfg)
        assert daily["mark_source"].iloc[-1] == "settlement"
        assert trade["n_market_marks"] == trade["n_days"] - trade["n_model_marks"]
        assert trade["n_days"] - trade["n_structural_marks"] == 10   # 9 quoted + settlement

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

    def test_entry_session_absent_from_the_underlying_is_skipped_not_raised(self):
        # F10: a stored chain can carry a session `underlying.parquet` does not.
        # `closes[entry_date]` was an unguarded dict lookup, so one absent close
        # raised a KeyError out of the entire daily run.
        entry, expiry, sel, chains, u = self._setup([100.0] * 6)
        without_entry = u[u["date"] != entry].reset_index(drop=True)
        assert simulate_trade(entry, sel, chains, without_entry, 0.04, 0.013, CFG) is None

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
        trades, daily, skipped = simulate(chains, u, 0.04, 0.013, CFG)
        assert list(trades.columns) == TRADE_COLUMNS
        assert len(trades) == trades["entry_date"].nunique()
        months = {(d.year, d.month) for d in trades["entry_date"]}
        assert len(months) == len(trades)
        assert skipped == 0

    def test_empty_archive_returns_empty_frames_with_columns(self):
        from src.analytics.hedge_sim import DAILY_COLUMNS
        u = pd.DataFrame({"date": [], "close": [], "adjusted_close": [], "volume": []})
        trades, daily, skipped = simulate({}, u, 0.04, 0.013, CFG)
        assert list(trades.columns) == TRADE_COLUMNS and trades.empty
        assert list(daily.columns) == DAILY_COLUMNS and daily.empty
        assert skipped == 0

    def test_a_month_with_no_expiry_near_the_target_is_counted_not_dropped(self):
        # F11: `sel is None` used to `continue` silently, so the sample's own
        # selection was invisible. On the real archive June 2026 goes this way --
        # its monthly is Juneteenth-shifted out of the ladder.
        chains, u = self._archive(months=2)
        entry = min(chains)
        far = entry + dt.timedelta(days=30 + MAX_ENTRY_DTE_ERROR + 1)
        chains[entry] = make_chain(entry, 100.0, {far: 0.20})
        trades, daily, skipped = simulate(chains, u, 0.04, 0.013, CFG)
        assert skipped == 1
        assert len(trades) == 1
        assert entry not in set(trades["entry_date"])

    def test_a_month_whose_entry_close_is_missing_is_counted_not_dropped(self):
        # F10 through the loop: the guard must skip the month AND be visible.
        chains, u = self._archive(months=2)
        entry = min(chains)
        u = u[u["date"] != entry].reset_index(drop=True)
        trades, daily, skipped = simulate(chains, u, 0.04, 0.013, CFG)
        assert skipped == 1
        assert entry not in set(trades["entry_date"])


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
        # The name promised coverage of BOTH exclusions but the frame held no
        # `sparse` row -- over the exact status the page then mis-handled (F1).
        trades = pd.DataFrame({
            "edge": [0.01, 0.02, 0.03, 0.04],
            "pnl": [2.0, 4.0, 999.0, -999.0],
            "status": ["settled", "settled", "open", "sparse"],
        })
        fit = fit_pnl_vs_edge(trades)
        assert fit["n"] == 2
        assert fit["slope"] == pytest.approx(2.0)   # the 999s never entered the line

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


NO_FIT = {"n": 0, "slope": np.nan, "intercept": np.nan, "r2": np.nan}


class TestHedgeSummary:
    def _trades(self, statuses, dtes=(17, 35)):
        n = len(statuses)
        return pd.DataFrame({
            "entry_date": [dt.date(2026, m, 1) for m in range(6, 6 + n)],
            "expiry": [dt.date(2026, m + 1, 17) for m in range(6, 6 + n)],
            "exit_date": [dt.date(2026, m + 1, 12) for m in range(6, 6 + n)],
            "dte_at_entry": list(dtes)[:n],
            "pnl": [1.0, -0.5, 0.25][:n], "edge": [0.01, 0.02, 0.03][:n],
            "n_days": [30, 20, 10][:n], "n_market_marks": [27, 20, 1][:n],
            "n_model_marks": [3, 0, 9][:n], "n_structural_marks": [2, 0, 3][:n],
            "market_mark_share": [0.9, 1.0, 0.1][:n],
            "quotable_mark_share": [27 / 28, 1.0, 1 / 7][:n], "status": list(statuses),
        })

    def test_counts_statuses_and_reports_the_next_settlement(self):
        trades = self._trades(["settled", "open"])
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2)],
                             "pnl_day": [0.0, 1.0], "pnl_cum": [0.0, 1.0], "n_open": [1, 1]})
        s = hedge_summary(trades, port, {"n": 1, "slope": np.nan,
                                         "intercept": np.nan, "r2": np.nan})
        assert (s["n_trades"], s["n_settled"], s["n_open"], s["n_sparse"]) == (2, 1, 1, 0)
        assert s["next_settlement"] == dt.date(2026, 8, 17)
        assert s["cum_pnl"] == pytest.approx(1.0)
        assert s["first_entry"] == dt.date(2026, 6, 1)
        assert s["market_mark_share"] == pytest.approx(47 / 50)

    def test_a_sparse_trade_has_reached_expiry_even_though_it_is_not_plottable(self):
        # F1: `sparse` is a sub-kind of SETTLED. Counting it as neither settled
        # nor open let the page say "none has reached expiry yet" in the same
        # sentence as a cumulative P&L that already contained its round trip.
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1)], "pnl_day": [0.0],
                             "pnl_cum": [2.26], "n_open": [1]})
        s = hedge_summary(self._trades(["sparse", "open"]), port, NO_FIT)
        assert s["n_settled"] == 0          # still not plottable
        assert s["n_sparse"] == 1
        assert s["n_reached_expiry"] == 1   # but it HAS finished

    def test_reached_expiry_counts_settled_and_sparse_together(self):
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1)], "pnl_day": [0.0],
                             "pnl_cum": [1.0], "n_open": [1]})
        s = hedge_summary(self._trades(["settled", "sparse", "open"], dtes=(17, 35, 45)),
                          port, NO_FIT)
        assert (s["n_settled"], s["n_sparse"], s["n_reached_expiry"]) == (1, 1, 2)

    def test_carries_the_tenor_actually_traded(self):
        # F5: the entry rule takes the monthly NEAREST 30 days, which from the
        # first session of a month is systematically ~17 -- and ~45 when that
        # month's monthly is missing. The range has to reach the page.
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1)], "pnl_day": [0.0],
                             "pnl_cum": [1.0], "n_open": [1]})
        s = hedge_summary(self._trades(["settled", "open"], dtes=(17, 45)), port, NO_FIT)
        assert s["dte_at_entry_min"] == 17
        assert s["dte_at_entry_max"] == 45

    def test_skipped_month_count_reaches_the_summary(self):
        s = hedge_summary(pd.DataFrame(columns=TRADE_COLUMNS),
                          pd.DataFrame(columns=["date", "pnl_day", "pnl_cum", "n_open"]),
                          NO_FIT, n_months_skipped=3)
        assert s["n_months_skipped"] == 3

    def test_summary_publishes_nothing_no_reader_consumes(self):
        # F9: these three were computed on every run and read by nobody, under a
        # docstring calling the dict "everything the page and status.json need".
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2)],
                             "pnl_day": [0.0, 1.0], "pnl_cum": [0.0, 1.0], "n_open": [1, 1]})
        s = hedge_summary(self._trades(["settled", "open"]), port, NO_FIT)
        for dead in ("last_mark", "mean_daily_pnl", "sd_daily_pnl"):
            assert dead not in s

    def test_empty_summary_is_all_zeros_and_nones(self):
        s = hedge_summary(pd.DataFrame(columns=TRADE_COLUMNS),
                          pd.DataFrame(columns=["date", "pnl_day", "pnl_cum", "n_open"]),
                          NO_FIT)
        assert s["n_trades"] == 0 and s["first_entry"] is None
        assert s["next_settlement"] is None
        assert np.isnan(s["cum_pnl"])
        assert s["n_reached_expiry"] == 0 and s["n_months_skipped"] == 0
        assert s["dte_at_entry_min"] is None and s["dte_at_entry_max"] is None
        assert np.isnan(s["market_mark_share"]) and np.isnan(s["quotable_mark_share"])
        assert s["n_model_marks"] == s["n_structural_marks"] == s["n_gap_marks"] == 0

    def test_publishes_both_mark_shares_and_the_structural_split(self):
        # Both counts have to reach the page: `market_mark_share` says how much
        # of the P&L is our own model AT ALL, `quotable_mark_share` how much of
        # it was avoidable -- and the second is what the status gate reads.
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1)], "pnl_day": [0.0],
                             "pnl_cum": [1.0], "n_open": [1]})
        s = hedge_summary(self._trades(["settled", "open"]), port, NO_FIT, dte_min=7)
        assert s["market_mark_share"] == pytest.approx(47 / 50)   # 50 sessions
        assert s["quotable_mark_share"] == pytest.approx(47 / 48)  # 2 unquotable
        assert (s["n_model_marks"], s["n_structural_marks"], s["n_gap_marks"]) == (3, 2, 1)
        assert s["dte_min"] == 7

    def test_dte_min_defaults_to_none_so_the_page_can_omit_the_window(self):
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1)], "pnl_day": [0.0],
                             "pnl_cum": [1.0], "n_open": [1]})
        assert hedge_summary(self._trades(["settled"]), port, NO_FIT)["dte_min"] is None

    def test_cost_bps_defaults_to_zero_and_passes_through_when_given(self):
        # M3: `_sim_label` (src/render/stats.py) needs the configured
        # transaction_cost_bps to state "no spread is charged" only when it is
        # actually true, and the render layer must not import config itself --
        # so the value has to travel through this summary dict.
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1)], "pnl_day": [0.0],
                             "pnl_cum": [1.0], "n_open": [1]})
        empty_port = pd.DataFrame(columns=["date", "pnl_day", "pnl_cum", "n_open"])
        s_default = hedge_summary(pd.DataFrame(columns=TRADE_COLUMNS), empty_port, NO_FIT)
        assert s_default["cost_bps"] == 0.0
        s_given = hedge_summary(self._trades(["settled", "open"]), port, NO_FIT,
                                n_months_skipped=0, cost_bps=5.0)
        assert s_given["cost_bps"] == 5.0
