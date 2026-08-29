"""P8 hedge-simulation engine tests (SPEC 3 P8). Offline, deterministic."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.hedge_sim import (
    MAX_ENTRY_DTE_ERROR, entry_sessions, select_straddle,
)

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
