"""P7: put-call parity deviations, tradeable violations, implied carry."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.parity import (
    PARITY_COLUMNS, carry_at_target, carry_reference, compute_parity, implied_carry,
    implied_forward, parity_summary,
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


class TestForwardCalibration:
    def test_synthetic_chain_forward_recovers_spot_forward(self):
        p = compute_parity(make_chain(), R, Q)
        f = implied_forward(p, SPOT, R)
        assert list(f.columns) == ["expiry", "dte", "forward", "implied_carry"]
        for _, row in f.iterrows():
            T = row["dte"] / 365.0
            assert row["forward"] == pytest.approx(SPOT * np.exp((R - Q) * T), rel=1e-9)
        assert np.abs(p["deviation_fwd"]).max() < 1e-8      # exact parity -> zero either way

    def test_spot_offset_moves_raw_deviation_but_not_forward_deviation(self):
        # Price the chain off a spot $1 above the one we later assume: this is the
        # real-data situation (options close 16:15, the stock 16:00).
        chain = make_chain()
        offset = 1.0
        T = chain["dte"].to_numpy(dtype=float) / 365.0
        bump = offset * np.exp(-Q * T) * np.where(chain["kind"] == "call", 1.0, -1.0)
        for col in ("price_used", "mid", "close"):
            chain[col] = chain[col] + bump
        p = compute_parity(chain, R, Q)
        # every strike of an expiry now shows the same raw offset (it differs
        # between expiries only by that expiry's e^{-qT} discount factor)...
        for _, g in p.groupby("dte"):
            assert np.abs(g["deviation"] - g["deviation"].iloc[0]).max() < 1e-8
        assert np.abs(p["deviation"]).min() > 0.9
        # ...but calibrating the forward absorbs it exactly
        assert np.abs(p["deviation_fwd"]).max() < 1e-8
        assert not p["tradeable_violation_fwd"].any()

    def test_real_violation_survives_calibration(self):
        chain = make_chain()
        m = (chain["kind"] == "put") & (chain["strike"] == 800.0) & (chain["dte"] == 28)
        chain.loc[m, "price_used"] = chain.loc[m, "price_used"] + 5.0
        p = compute_parity(chain, R, Q)
        row = p[(p["strike"] == 800.0) & (p["dte"] == 28)].iloc[0]
        assert bool(row["tradeable_violation_fwd"]) is True
        # the other strikes of that expiry stay clean (one bad strike must not
        # smear across the expiry through the ATM calibration)
        others = p[(p["dte"] == 28) & (p["strike"] != 800.0)]
        assert not others["tradeable_violation_fwd"].any()

    def test_carry_reference_prefers_a_long_expiry(self):
        f = implied_forward(compute_parity(make_chain(), R, Q), SPOT, R)
        val, dte = carry_reference(f, min_dte=84)
        assert dte == 84 and val == pytest.approx(R - Q, abs=1e-8)
        val2, dte2 = carry_reference(f, min_dte=999)       # nothing qualifies -> longest
        assert dte2 == 84
        assert val2 == pytest.approx(R - Q, abs=1e-8)

    def test_unbracketed_expiry_has_no_forward(self):
        p = compute_parity(make_chain(strikes=(800.0, 840.0)), R, Q)
        assert p["forward"].isna().all() and p["deviation_fwd"].isna().all()
        assert not p["tradeable_violation_fwd"].any()
        assert implied_forward(p, SPOT, R).empty
        assert np.isnan(carry_reference(implied_forward(p, SPOT, R))[0])


class TestLiquidity:
    def test_zero_open_interest_leg_is_not_liquid(self):
        chain = make_chain()
        m = (chain["kind"] == "put") & (chain["strike"] == 700.0)
        chain.loc[m, "open_interest"] = 0.0
        p = compute_parity(chain, R, Q)
        assert not p[p["strike"] == 700.0]["liquid"].any()
        assert p[p["strike"] != 700.0]["liquid"].all()

    def test_frame_without_open_interest_treats_all_as_liquid(self):
        chain = make_chain(source="massive-backfill")
        chain["open_interest"] = np.nan
        p = compute_parity(chain, R, Q)
        assert p["liquid"].all()

    def test_summary_counts_liquid_and_both_violation_kinds(self):
        chain = make_chain()
        chain.loc[(chain["kind"] == "put") & (chain["strike"] == 700.0), "open_interest"] = 0.0
        chain.loc[(chain["kind"] == "put") & (chain["strike"] == 800.0) & (chain["dte"] == 28),
                  "price_used"] += 5.0
        s = parity_summary(compute_parity(chain, R, Q))
        assert s["n_pairs"] == 12 and s["n_liquid"] == 10
        assert s["n_tradeable_violations"] == 1 and s["n_tradeable_violations_fwd"] == 1
        assert s["share_within_spread_fwd"] == pytest.approx(0.9)


class TestEarlyExercise:
    def test_bound_is_interest_on_strike_less_forgone_dividends(self):
        p = compute_parity(make_chain(), R, Q)
        row = p[(p["strike"] == 840.0) & (p["dte"] == 84)].iloc[0]
        T = 84 / 365.0
        expected = 840.0 * (1 - np.exp(-R * T)) - SPOT * (1 - np.exp(-Q * T))
        assert row["early_exercise_bound"] == pytest.approx(expected, rel=1e-12)
        assert (p["early_exercise_bound"] >= 0).all()

    def test_itm_put_gap_within_the_bound_is_explained(self):
        # Make the ITM put richer by an amount an American holder could justify:
        # three quarters of the early-exercise bound ($4.69), inside the bound
        # ($6.25) and well outside the pair's combined spread ($1.71).
        chain = make_chain()
        m = (chain["kind"] == "put") & (chain["strike"] == 840.0) & (chain["dte"] == 84)
        T = 84 / 365.0
        bound = 840.0 * (1 - np.exp(-R * T)) - SPOT * (1 - np.exp(-Q * T))
        chain.loc[m, "price_used"] = chain.loc[m, "price_used"] + bound * 0.75
        p = compute_parity(chain, R, Q)
        row = p[(p["strike"] == 840.0) & (p["dte"] == 84)].iloc[0]
        assert bool(row["tradeable_violation_fwd"]) is True
        assert bool(row["early_exercise_explained"]) is True

    def test_gap_beyond_the_bound_is_not_explained(self):
        chain = make_chain()
        m = (chain["kind"] == "put") & (chain["strike"] == 840.0) & (chain["dte"] == 84)
        chain.loc[m, "price_used"] = chain.loc[m, "price_used"] + 60.0
        p = compute_parity(chain, R, Q)
        row = p[(p["strike"] == 840.0) & (p["dte"] == 84)].iloc[0]
        assert bool(row["tradeable_violation_fwd"]) is True
        assert bool(row["early_exercise_explained"]) is False

    def test_wrong_sign_gap_is_never_explained(self):
        # A call that is too rich pushes deviation POSITIVE; early exercise of a
        # put cannot explain that direction.
        chain = make_chain()
        m = (chain["kind"] == "call") & (chain["strike"] == 840.0) & (chain["dte"] == 84)
        chain.loc[m, "price_used"] = chain.loc[m, "price_used"] + 60.0
        p = compute_parity(chain, R, Q)
        row = p[(p["strike"] == 840.0) & (p["dte"] == 84)].iloc[0]
        assert row["deviation_fwd"] > 0
        assert bool(row["early_exercise_explained"]) is False

    def test_a_small_gap_inside_the_spread_is_no_violation_and_not_explained(self):
        # "explained" reads as "this violation is explained", never as
        # "this row could be explained": a non-violating row is always False.
        p = compute_parity(make_chain(), R, Q)
        assert not p["tradeable_violation_fwd"].any()
        assert not p["early_exercise_explained"].any()
        s = parity_summary(p)
        assert s["n_violations_early_exercise"] == 0
        assert s["n_violations_unexplained"] == 0

    def test_summary_splits_violations_and_they_sum(self):
        chain = make_chain()
        T = 84 / 365.0
        bound = 840.0 * (1 - np.exp(-R * T)) - SPOT * (1 - np.exp(-Q * T))
        chain.loc[(chain["kind"] == "put") & (chain["strike"] == 840.0) & (chain["dte"] == 84),
                  "price_used"] += bound * 0.75
        chain.loc[(chain["kind"] == "put") & (chain["strike"] == 700.0) & (chain["dte"] == 84),
                  "price_used"] += 60.0
        s = parity_summary(compute_parity(chain, R, Q))
        assert s["n_violations_early_exercise"] + s["n_violations_unexplained"] \
            == s["n_tradeable_violations_fwd"]
        assert s["n_violations_early_exercise"] >= 1 and s["n_violations_unexplained"] >= 1

    def test_non_liquid_violations_are_counted_in_neither_bucket(self):
        chain = make_chain()
        T = 84 / 365.0
        bound = 840.0 * (1 - np.exp(-R * T)) - SPOT * (1 - np.exp(-Q * T))
        chain.loc[(chain["kind"] == "put") & (chain["strike"] == 840.0), "open_interest"] = 0.0
        chain.loc[(chain["kind"] == "put") & (chain["strike"] == 840.0) & (chain["dte"] == 84),
                  "price_used"] += bound * 0.75
        s = parity_summary(compute_parity(chain, R, Q))
        assert s["n_violations_early_exercise"] == 0
        assert s["n_violations_unexplained"] == 0
        assert s["n_tradeable_violations_fwd"] == 0
