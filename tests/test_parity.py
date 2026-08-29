"""P7: put-call parity deviations, tradeable violations, implied carry."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.parity import (
    PARITY_COLUMNS, carry_at_target, compute_parity, implied_carry, parity_summary,
)
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def make_chain(source="yfinance", spread_frac=0.01,
               strikes=(700.0, 740.0, 760.0, 780.0, 800.0, 840.0),
               expiries=((dt.date(2026, 9, 25), 28), (dt.date(2026, 11, 20), 84))):
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                px = float(bs_price(SPOT, strike, dte / 365.0, R, 0.22, Q, kind))
                live = source == "yfinance"
                rows.append({"snapshot_date": TODAY, "spot": SPOT, "expiry": expiry, "dte": dte,
                             "strike": strike, "kind": kind,
                             "bid": px * (1 - spread_frac) if live else np.nan,
                             "ask": px * (1 + spread_frac) if live else np.nan,
                             "mid": px if live else np.nan, "close": px,
                             "volume": 10, "open_interest": 100.0, "vendor_iv": np.nan,
                             "source": source})
    out, _ = compute_chain_iv(pd.DataFrame(rows, columns=CHAIN_COLUMNS), R, Q)
    return out


class TestParity:
    def test_synthetic_prices_satisfy_parity_to_machine_eps(self):
        p = compute_parity(make_chain(), R, Q)
        assert list(p.columns) == PARITY_COLUMNS
        assert len(p) == 12
        assert np.abs(p["deviation"]).max() < 1e-9
        assert p["spread"].notna().all() and not p["tradeable_violation"].any()
        s = parity_summary(p)
        assert s["n_pairs"] == 12 and s["n_quoted"] == 12
        assert s["n_tradeable_violations"] == 0 and s["share_within_spread"] == 1.0

    def test_violation_beyond_spread_is_flagged(self):
        chain = make_chain()
        m = (chain["kind"] == "put") & (chain["strike"] == 760.0) & (chain["dte"] == 28)
        chain.loc[m, ["mid", "price_used"]] = chain.loc[m, "price_used"] + 5.0   # way outside spread
        p = compute_parity(chain, R, Q)
        row = p[(p["strike"] == 760.0) & (p["dte"] == 28)].iloc[0]
        assert row["deviation"] == pytest.approx(-5.0, abs=1e-9)
        assert bool(row["tradeable_violation"]) is True
        assert parity_summary(p)["n_tradeable_violations"] == 1

    def test_deviation_inside_spread_is_not_tradeable(self):
        chain = make_chain(spread_frac=0.05)
        m = (chain["kind"] == "call") & (chain["strike"] == 780.0) & (chain["dte"] == 28)
        chain.loc[m, "price_used"] = chain.loc[m, "price_used"] + 0.10
        p = compute_parity(chain, R, Q)
        row = p[(p["strike"] == 780.0) & (p["dte"] == 28)].iloc[0]
        assert row["deviation"] == pytest.approx(0.10, abs=1e-9)
        assert bool(row["tradeable_violation"]) is False

    def test_close_based_rows_have_no_spread_and_never_trade(self):
        p = compute_parity(make_chain(source="massive-backfill"), R, Q)
        assert p["spread"].isna().all() and not p["tradeable_violation"].any()
        s = parity_summary(p)
        assert s["n_quoted"] == 0 and np.isnan(s["share_within_spread"])

    def test_unpaired_strikes_are_dropped(self):
        chain = make_chain()
        chain = chain[~((chain["kind"] == "put") & (chain["strike"] == 700.0))]
        p = compute_parity(chain, R, Q)
        assert 700.0 not in set(p["strike"])

    def test_empty(self):
        p = compute_parity(make_chain().iloc[0:0], R, Q)
        assert p.empty and list(p.columns) == PARITY_COLUMNS
        s = parity_summary(p)
        assert s["n_pairs"] == 0 and np.isnan(s["max_abs_deviation"])


class TestImpliedCarry:
    def test_recovers_r_minus_q(self):
        c = implied_carry(compute_parity(make_chain(), R, Q), SPOT, R)
        assert list(c.columns) == ["expiry", "dte", "implied_carry"]
        assert list(c["dte"]) == [28, 84]
        assert c["implied_carry"].to_numpy() == pytest.approx(R - Q, abs=1e-8)
        val, dte = carry_at_target(c, 30)
        assert val == pytest.approx(R - Q, abs=1e-8) and dte == 28

    def test_expiry_not_bracketing_spot_is_skipped(self):
        chain = make_chain(strikes=(800.0, 840.0))     # all above spot
        c = implied_carry(compute_parity(chain, R, Q), SPOT, R)
        assert c.empty
        assert np.isnan(carry_at_target(c, 30)[0])
