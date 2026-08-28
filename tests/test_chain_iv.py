"""chain_iv: source-routed pricing + IV recovery of known synthetic vols."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def synthetic_chain(source="yfinance", sigma_fn=lambda k, t: 0.22):
    """Chain whose prices come from bs_price with known per-row vols."""
    rows = []
    for expiry, dte in ((dt.date(2026, 9, 18), 21), (dt.date(2026, 11, 20), 84)):
        for strike in (700.0, 740.0, 770.0, 800.0, 840.0):
            for kind in ("call", "put"):
                sigma = sigma_fn(strike, dte)
                price = bs_price(S=SPOT, K=strike, T=dte / 365.0, r=R,
                                 sigma=sigma, q=Q, kind=kind)
                is_live = source == "yfinance"
                rows.append({
                    "snapshot_date": TODAY, "spot": SPOT, "expiry": expiry,
                    "dte": dte, "strike": strike, "kind": kind,
                    "bid": price * 0.99 if is_live else np.nan,
                    "ask": price * 1.01 if is_live else np.nan,
                    "mid": price if is_live else np.nan,
                    "close": price * 1.02 if is_live else price,
                    "volume": 10, "open_interest": 100,
                    "vendor_iv": 0.99,  # poison: must never be read
                    "source": source,
                })
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


class TestPriceRouting:
    def test_live_rows_use_mid_not_close(self):
        df, _ = compute_chain_iv(synthetic_chain("yfinance"), R, Q)
        # close was poisoned at 1.02x; recovery to the exact vol proves mid was used
        assert np.allclose(df["iv"], 0.22, atol=1e-6)
        assert np.allclose(df["price_used"], df["mid"])

    def test_backfill_rows_use_close(self):
        df, _ = compute_chain_iv(synthetic_chain("massive-backfill"), R, Q)
        assert np.allclose(df["iv"], 0.22, atol=1e-6)
        assert np.allclose(df["price_used"], df["close"])


class TestRecovery:
    def test_recovers_a_skewed_smile(self):
        skew = lambda k, t: 0.20 + 0.30 * max(0.0, (770.0 - k) / 770.0)
        df, stats = compute_chain_iv(synthetic_chain(sigma_fn=skew), R, Q)
        for _, row in df.iterrows():
            assert row["iv"] == pytest.approx(skew(row["strike"], row["dte"]), abs=1e-6)
        assert stats["convergence"] == 1.0

    def test_vendor_iv_never_consulted(self):
        df, _ = compute_chain_iv(synthetic_chain(), R, Q)
        assert not np.allclose(df["iv"], df["vendor_iv"])


class TestStats:
    def test_unpriceable_row_counts(self):
        chain = synthetic_chain()
        chain.loc[0, "mid"] = np.nan          # live row with no mid -> no usable price
        chain.loc[1, "mid"] = 1e9             # absurd price -> no-arb NaN from solver
        df, stats = compute_chain_iv(chain, R, Q)
        assert np.isnan(df.loc[0, "iv"]) and np.isnan(df.loc[1, "iv"])
        assert stats["n_priced"] == len(chain) - 1        # NaN price excluded
        assert stats["n_converged"] == len(chain) - 2     # absurd price counted, failed
        assert stats["convergence"] == pytest.approx((len(chain) - 2) / (len(chain) - 1))

    def test_empty_chain(self):
        df, stats = compute_chain_iv(synthetic_chain().iloc[0:0], R, Q)
        assert df.empty and stats == {"n_priced": 0, "n_converged": 0, "convergence": 0.0}

    def test_input_not_mutated(self):
        chain = synthetic_chain()
        before = chain.copy()
        compute_chain_iv(chain, R, Q)
        pd.testing.assert_frame_equal(chain, before)
