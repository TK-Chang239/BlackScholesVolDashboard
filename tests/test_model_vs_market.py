"""P9: every contract priced at one flat vol vs the market."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.model_vs_market import MVM_COLUMNS, compute_model_vs_market
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def make_chain(sigma_fn=lambda k: 0.22, source="yfinance", half_spread=0.05,
               strikes=(660.0, 700.0, 740.0, 770.0, 800.0, 840.0, 880.0),
               expiries=((dt.date(2026, 9, 25), 28), (dt.date(2026, 11, 20), 84))):
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                px = float(bs_price(SPOT, strike, dte / 365.0, R, sigma_fn(strike), Q, kind))
                live = source == "yfinance"
                rows.append({"snapshot_date": TODAY, "spot": SPOT, "expiry": expiry, "dte": dte,
                             "strike": strike, "kind": kind,
                             "bid": px - half_spread if live else np.nan,
                             "ask": px + half_spread if live else np.nan,
                             "mid": px if live else np.nan, "close": px,
                             "volume": 10, "open_interest": 100.0, "vendor_iv": np.nan,
                             "source": source})
    out, _ = compute_chain_iv(pd.DataFrame(rows, columns=CHAIN_COLUMNS), R, Q)
    return out


class TestModelVsMarket:
    def test_flat_market_matches_flat_model(self):
        m = compute_model_vs_market(make_chain(), 0.22, R, Q)
        assert list(m.columns) == MVM_COLUMNS and len(m) == 28
        assert np.abs(m["deviation_pct"]).max() < 1e-9
        assert np.abs(m["deviation_vol"]).max() < 1e-6
        assert m["quoted"].all() and m["within_spread"].all()

    def test_skewed_market_is_rich_in_the_put_wing(self):
        skew_fn = lambda k: 0.20 + 0.30 * max(0.0, (SPOT - k) / SPOT)
        m = compute_model_vs_market(make_chain(sigma_fn=skew_fn), 0.20, R, Q)
        wing = m[(m["kind"] == "put") & (m["strike"] == 660.0) & (m["dte"] == 28)].iloc[0]
        assert wing["deviation_pct"] > 0.5            # market far richer than flat-vol model
        assert wing["deviation_vol"] == pytest.approx(skew_fn(660.0) - 0.20, abs=1e-6)
        assert bool(wing["within_spread"]) is False
        atm = m[(m["kind"] == "call") & (m["strike"] == 770.0) & (m["dte"] == 28)].iloc[0]
        assert abs(atm["deviation_pct"]) < 1e-9

    def test_small_deviation_inside_half_spread_is_muted(self):
        m = compute_model_vs_market(make_chain(half_spread=1.0), 0.2205, R, Q)
        assert m["within_spread"].all()

    def test_close_based_rows_are_unquoted_and_never_muted(self):
        m = compute_model_vs_market(make_chain(source="massive-backfill"), 0.22, R, Q)
        assert not m["quoted"].any() and not m["within_spread"].any()
        assert np.abs(m["deviation_pct"]).max() < 1e-9

    def test_nan_flat_vol_or_empty_gives_empty(self):
        assert compute_model_vs_market(make_chain(), float("nan"), R, Q).empty
        e = compute_model_vs_market(make_chain().iloc[0:0], 0.22, R, Q)
        assert e.empty and list(e.columns) == MVM_COLUMNS
