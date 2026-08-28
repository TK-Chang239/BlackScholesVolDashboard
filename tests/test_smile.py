"""P2 smile frame: expiry targeting, OTM convention, tidy output."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.smile import compute_smile
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098
CFG = {"target_dte": {"smile_expiries": [30, 90, 180]}}

EXPIRIES = [  # (expiry, dte) — six monthlies like a real stored chain
    (dt.date(2026, 9, 18), 21), (dt.date(2026, 10, 16), 49),
    (dt.date(2026, 11, 20), 84), (dt.date(2026, 12, 18), 112),
    (dt.date(2027, 1, 15), 140), (dt.date(2027, 2, 19), 175),
]


def chain_with_skew():
    rows = []
    for expiry, dte in EXPIRIES:
        for strike in (700.0, 740.0, 770.0, 800.0, 840.0):
            for kind in ("call", "put"):
                sigma = 0.20 + 0.25 * max(0.0, (SPOT - strike) / SPOT)
                price = bs_price(S=SPOT, K=strike, T=dte / 365.0, r=R,
                                 sigma=sigma, q=Q, kind=kind)
                rows.append({
                    "snapshot_date": TODAY, "spot": SPOT, "expiry": expiry,
                    "dte": dte, "strike": strike, "kind": kind,
                    "bid": price * 0.99, "ask": price * 1.01, "mid": price,
                    "close": price, "volume": 10, "open_interest": 100,
                    "vendor_iv": np.nan, "source": "yfinance",
                })
    df = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    out, _ = compute_chain_iv(df, R, Q)
    return out


class TestSmile:
    def test_selects_nearest_expiry_per_target(self):
        got = compute_smile(chain_with_skew(), CFG)
        # nearest to 30 -> 21d (9/18); to 90 -> 84d (11/20); to 180 -> 175d (2/19)
        assert sorted(got["dte"].unique()) == [21, 84, 175]

    def test_otm_convention(self):
        got = compute_smile(chain_with_skew(), CFG)
        puts, calls = got[got["kind"] == "put"], got[got["kind"] == "call"]
        assert (puts["strike"] < SPOT).all()
        assert (calls["strike"] >= SPOT).all()
        # exactly one row per strike per expiry after OTM split
        assert not got.duplicated(["expiry", "strike"]).any()

    def test_downward_put_skew_visible(self):
        got = compute_smile(chain_with_skew(), CFG)
        one = got[got["dte"] == 21].sort_values("strike")
        assert one["iv"].iloc[0] > one["iv"].iloc[-1]  # 700-strike IV > 840-strike IV

    def test_output_schema_and_moneyness(self):
        got = compute_smile(chain_with_skew(), CFG)
        assert list(got.columns) == ["expiry", "dte", "strike", "moneyness", "kind", "iv"]
        assert got["moneyness"].between(0.70, 1.30).all()
        assert got.index.tolist() == list(range(len(got)))
        assert got["iv"].notna().all()

    def test_duplicate_targets_collapse(self):
        # only two expiries stored: targets 30/90/180 must not repeat one
        df = chain_with_skew()
        df = df[df["dte"].isin([21, 84])]
        got = compute_smile(df, CFG)
        assert sorted(got["dte"].unique()) == [21, 84]
