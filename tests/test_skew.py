"""P6: 25-delta skew, interpolated in delta space from each strike's own IV."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.skew import SKEW_KEYS, compute_skew_25d
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price, greeks

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def make_chain(sigma_fn=lambda k: 0.22,
               strikes=tuple(np.arange(640.0, 901.0, 10.0)),
               expiries=((dt.date(2026, 9, 25), 28), (dt.date(2026, 11, 20), 84))):
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                px = float(bs_price(SPOT, strike, dte / 365.0, R, sigma_fn(strike), Q, kind))
                rows.append({"snapshot_date": TODAY, "spot": SPOT, "expiry": expiry, "dte": dte,
                             "strike": strike, "kind": kind, "bid": px * 0.99, "ask": px * 1.01,
                             "mid": px, "close": px, "volume": 10, "open_interest": 100.0,
                             "vendor_iv": np.nan, "source": "yfinance"})
    out, _ = compute_chain_iv(pd.DataFrame(rows, columns=CHAIN_COLUMNS), R, Q)
    return out


class TestSkew:
    def test_flat_vol_has_zero_skew_at_nearest_expiry(self):
        s = compute_skew_25d(make_chain(), R, Q, cfg())
        assert list(s) == SKEW_KEYS
        assert s["skew_expiry"] == dt.date(2026, 9, 25) and s["skew_dte"] == 28
        assert s["put_iv_25d"] == pytest.approx(0.22, abs=1e-6)
        assert s["call_iv_25d"] == pytest.approx(0.22, abs=1e-6)
        assert abs(s["skew_25d"]) < 1e-9

    def test_put_skew_is_positive_and_bracketed_by_neighbours(self):
        skew_fn = lambda k: 0.20 + 0.30 * max(0.0, (SPOT - k) / SPOT)
        s = compute_skew_25d(make_chain(sigma_fn=skew_fn), R, Q, cfg())
        assert s["skew_25d"] > 0.01
        assert s["call_iv_25d"] == pytest.approx(0.20, abs=1e-6)     # flat on the call side
        # the 25-delta put sits somewhere below spot: its IV must lie between the
        # smallest and largest OTM-put vols in the chain
        assert 0.20 < s["put_iv_25d"] < skew_fn(640.0)

    def test_interpolates_at_delta_not_strike(self):
        # Two chains with the same strikes but different vol levels have different
        # 25-delta strikes; the interpolated IV must follow the skew function at the
        # strike whose own-IV delta is -0.25, to first order.
        skew_fn = lambda k: 0.15 + 0.40 * max(0.0, (SPOT - k) / SPOT)
        chain = make_chain(sigma_fn=skew_fn)
        s = compute_skew_25d(chain, R, Q, cfg())
        g = chain[(chain["dte"] == 28) & (chain["kind"] == "put") & (chain["strike"] < SPOT)]
        deltas = greeks(SPOT, g["strike"].to_numpy(float), 28 / 365.0, R,
                        g["iv"].to_numpy(float), Q, "put")["delta"]
        k_star = float(np.interp(-0.25, np.sort(deltas), g["strike"].to_numpy(float)[np.argsort(deltas)]))
        assert s["put_iv_25d"] == pytest.approx(skew_fn(k_star), abs=0.003)

    def test_unbracketed_side_is_nan(self):
        chain = make_chain(strikes=(760.0, 770.0, 780.0))   # OTM puts: only 760 -> no bracket
        s = compute_skew_25d(chain, R, Q, cfg())
        assert np.isnan(s["put_iv_25d"]) and np.isnan(s["skew_25d"])

    def test_empty_chain(self):
        s = compute_skew_25d(make_chain().iloc[0:0], R, Q, cfg())
        assert s["skew_expiry"] is None and np.isnan(s["skew_25d"])
