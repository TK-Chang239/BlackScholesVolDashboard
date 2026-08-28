"""Chain filtering: monthly-expiry selection and the SPEC 2.2 filter rules."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.data.base import CHAIN_COLUMNS
from src.data.filters import filter_chain, is_monthly_expiry, select_expiries

TODAY = dt.date(2026, 8, 28)
CFG = {
    "chain_filter": {
        "n_monthly_expiries": 6,
        "dte_min": 7,
        "dte_max": 365,
        "moneyness_min": 0.70,
        "moneyness_max": 1.30,
    }
}


class TestMonthlyExpiry:
    def test_third_fridays_are_monthly(self):
        for d in [dt.date(2026, 9, 18), dt.date(2026, 10, 16), dt.date(2026, 11, 20),
                  dt.date(2026, 12, 18), dt.date(2027, 1, 15), dt.date(2027, 2, 19)]:
            assert is_monthly_expiry(d), d

    def test_non_third_fridays_are_not(self):
        for d in [dt.date(2026, 9, 4),    # first Friday
                  dt.date(2026, 9, 25),   # fourth Friday
                  dt.date(2026, 8, 31),   # a Monday
                  dt.date(2026, 9, 17)]:  # a Thursday
            assert not is_monthly_expiry(d), d


class TestSelectExpiries:
    def test_selects_nearest_six_monthlies_within_dte(self):
        expiries = [
            dt.date(2026, 8, 31), dt.date(2026, 9, 2), dt.date(2026, 9, 4),   # weeklies
            dt.date(2026, 9, 18), dt.date(2026, 10, 16), dt.date(2026, 11, 20),
            dt.date(2026, 12, 18), dt.date(2027, 1, 15), dt.date(2027, 2, 19),
            dt.date(2027, 3, 19),                                             # 7th monthly
            dt.date(2026, 9, 1),                                              # < dte_min once monthly? no: weekly anyway
        ]
        got = select_expiries(expiries, TODAY, CFG)
        assert got == [dt.date(2026, 9, 18), dt.date(2026, 10, 16), dt.date(2026, 11, 20),
                       dt.date(2026, 12, 18), dt.date(2027, 1, 15), dt.date(2027, 2, 19)]

    def test_dte_bounds_respected(self):
        # 2026-09-04 is 7 DTE but weekly; 2026-09-18 is 21 DTE monthly;
        # a monthly < 7 DTE must be dropped: today+3 hypothetical third Friday
        expiries = [dt.date(2026, 9, 18), dt.date(2027, 9, 17)]  # second is 385 DTE > 365
        got = select_expiries(expiries, TODAY, CFG)
        assert got == [dt.date(2026, 9, 18)]

    def test_fewer_than_n_available_returns_what_exists(self):
        got = select_expiries([dt.date(2026, 9, 18)], TODAY, CFG)
        assert got == [dt.date(2026, 9, 18)]


def make_raw(rows):
    cols = ["expiry", "strike", "kind", "bid", "ask", "mid", "close",
            "volume", "open_interest", "vendor_iv", "source"]
    return pd.DataFrame(rows, columns=cols)


SPOT = 770.0
EXP = dt.date(2026, 9, 18)


class TestFilterChain:
    def test_moneyness_bounds(self):
        df = make_raw([
            [EXP, 530.0, "call", 1.0, 1.2, 1.1, 1.1, 10, 10, 0.2, "yfinance"],  # K/S=0.688 < 0.70
            [EXP, 700.0, "call", 1.0, 1.2, 1.1, 1.1, 10, 10, 0.2, "yfinance"],
            [EXP, 1010.0, "call", 1.0, 1.2, 1.1, 1.1, 10, 10, 0.2, "yfinance"],  # 1.312 > 1.30
        ])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert got["strike"].tolist() == [700.0]

    def test_live_liquidity_floor_bid_and_ask(self):
        df = make_raw([
            [EXP, 700.0, "call", 0.0, 1.2, np.nan, 1.1, 10, 10, 0.2, "yfinance"],     # bid=0
            [EXP, 710.0, "call", 1.0, np.nan, np.nan, 1.1, 10, 10, 0.2, "yfinance"],  # no ask
            [EXP, 720.0, "call", 1.0, 1.2, 1.1, 1.1, 0, 0, 0.2, "yfinance"],          # keeps: vol/OI irrelevant live
        ])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert got["strike"].tolist() == [720.0]

    def test_close_only_liquidity_floor_volume_or_oi(self):
        df = make_raw([
            [EXP, 700.0, "call", np.nan, np.nan, np.nan, 1.1, 0, 0, np.nan, "massive-backfill"],   # dead
            [EXP, 710.0, "call", np.nan, np.nan, np.nan, 1.1, 5, 0, np.nan, "massive-backfill"],   # volume
            [EXP, 720.0, "call", np.nan, np.nan, np.nan, 1.1, 0, 9, np.nan, "massive-fallback"],   # OI
            [EXP, 730.0, "call", np.nan, np.nan, np.nan, 1.1, np.nan, np.nan, np.nan, "massive-backfill"],  # NaN==0
        ])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert got["strike"].tolist() == [710.0, 720.0]

    def test_output_schema_and_derived_columns(self):
        df = make_raw([[EXP, 770.0, "put", 10.0, 10.4, 10.2, 10.1, 3, 7, 0.19, "yfinance"]])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert list(got.columns) == CHAIN_COLUMNS
        row = got.iloc[0]
        assert row["snapshot_date"] == TODAY
        assert row["spot"] == SPOT
        assert row["dte"] == (EXP - TODAY).days
        assert got.index.tolist() == [0]

    def test_sorted_by_expiry_kind_strike(self):
        exp2 = dt.date(2026, 10, 16)
        df = make_raw([
            [exp2, 700.0, "call", 1.0, 1.2, 1.1, 1.1, 1, 1, 0.2, "yfinance"],
            [EXP, 710.0, "put", 1.0, 1.2, 1.1, 1.1, 1, 1, 0.2, "yfinance"],
            [EXP, 700.0, "put", 1.0, 1.2, 1.1, 1.1, 1, 1, 0.2, "yfinance"],
            [EXP, 700.0, "call", 1.0, 1.2, 1.1, 1.1, 1, 1, 0.2, "yfinance"],
        ])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert got[["expiry", "kind", "strike"]].values.tolist() == [
            [EXP, "call", 700.0], [EXP, "put", 700.0], [EXP, "put", 710.0], [exp2, "call", 700.0]]
