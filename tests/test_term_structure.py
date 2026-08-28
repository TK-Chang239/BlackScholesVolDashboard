"""P3 term structure: strike-space ATM interpolation per expiry."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.term_structure import compute_term_structure
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def make_chain(expiries, strikes=(740.0, 760.0, 780.0, 800.0), sigma_by_dte=None):
    sigma_by_dte = sigma_by_dte or {}
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                sigma = sigma_by_dte.get(dte, 0.22)
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


class TestTermStructure:
    def test_flat_vol_interpolates_exactly(self):
        chain = make_chain([(dt.date(2026, 9, 18), 21), (dt.date(2026, 11, 20), 84)],
                           sigma_by_dte={21: 0.18, 84: 0.24})
        got = compute_term_structure(chain)
        assert got["dte"].tolist() == [21, 84]
        assert got["atm_iv"].iloc[0] == pytest.approx(0.18, abs=1e-5)
        assert got["atm_iv"].iloc[1] == pytest.approx(0.24, abs=1e-5)

    def test_interpolation_is_linear_between_bracketing_strikes(self):
        # give the two strikes around spot different known vols; ATM must be
        # the strike-linear blend at spot=770 between K=760 and K=780
        chain = make_chain([(dt.date(2026, 9, 18), 21)], strikes=(760.0, 780.0))
        # overwrite IVs directly with a synthetic wedge: iv = a + b*K
        chain["iv"] = 0.10 + (chain["strike"] - 760.0) * 0.001  # 760->0.10, 780->0.12
        got = compute_term_structure(chain)
        assert got["atm_iv"].iloc[0] == pytest.approx(0.11, abs=1e-12)

    def test_unbracketed_expiry_dropped(self):
        chain = make_chain([(dt.date(2026, 9, 18), 21)], strikes=(780.0, 800.0, 820.0))
        got = compute_term_structure(chain)  # all strikes above spot: no bracket
        assert got.empty

    def test_one_kind_bracketing_suffices(self):
        chain = make_chain([(dt.date(2026, 9, 18), 21)], strikes=(760.0, 780.0))
        chain.loc[chain["kind"] == "put", "iv"] = np.nan  # puts unusable
        got = compute_term_structure(chain)
        assert len(got) == 1 and got["atm_iv"].notna().all()

    def test_schema_and_sorting(self):
        chain = make_chain([(dt.date(2026, 11, 20), 84), (dt.date(2026, 9, 18), 21)])
        got = compute_term_structure(chain)
        assert list(got.columns) == ["expiry", "dte", "atm_iv"]
        assert got["dte"].tolist() == sorted(got["dte"].tolist())
        assert got.index.tolist() == list(range(len(got)))

    def test_empty_input(self):
        chain = make_chain([(dt.date(2026, 9, 18), 21)]).iloc[0:0]
        got = compute_term_structure(chain)
        assert got.empty and list(got.columns) == ["expiry", "dte", "atm_iv"]
