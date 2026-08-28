"""bs_price tests: textbook anchors, identities, boundaries, NaN contract.

Textbook anchor: Hull's standard example — S=42, K=40, r=10%, sigma=20%,
T=0.5, q=0: call = 4.7594, put = 0.8086 (4 dp).
"""
import numpy as np
import pytest

from src.models.black_scholes import bs_price


HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)

# Shared grid for identity tests: realistic SPY-ish ranges (SPEC §2.2 filter)
S0 = 100.0
GRID_K = np.linspace(70.0, 130.0, 13)
GRID_T = np.array([7 / 365, 30 / 365, 0.25, 0.5, 1.0, 2.0])[:, None]
GRID = dict(S=S0, K=GRID_K, T=GRID_T, r=0.04, sigma=0.22, q=0.013)


class TestTextbookValues:
    def test_hull_call(self):
        assert bs_price(**HULL, kind="call") == pytest.approx(4.7594, abs=1e-4)

    def test_hull_put(self):
        assert bs_price(**HULL, kind="put") == pytest.approx(0.8086, abs=1e-4)


class TestIdentities:
    def test_put_call_parity(self):
        """C - P == S e^{-qT} - K e^{-rT}, an internal identity (machine eps)."""
        c = bs_price(**GRID, kind="call")
        p = bs_price(**GRID, kind="put")
        fwd = S0 * np.exp(-GRID["q"] * GRID_T) - GRID_K * np.exp(-0.04 * GRID_T)
        np.testing.assert_allclose(c - p, np.broadcast_to(fwd, c.shape), atol=1e-10)

    def test_dividend_yield_is_spot_shift(self):
        """BSM(S,...,q) == BS(S e^{-qT},...,q=0) exactly — validates q handling."""
        q, T = 0.013, GRID_T
        with_q = bs_price(S=S0, K=GRID_K, T=T, r=0.04, sigma=0.22, q=q, kind="call")
        shifted = bs_price(S=S0 * np.exp(-q * T), K=GRID_K, T=T, r=0.04, sigma=0.22, q=0.0, kind="call")
        np.testing.assert_allclose(with_q, shifted, atol=1e-12)


class TestBoundaries:
    def test_sigma_zero_gives_discounted_forward_intrinsic(self):
        got = bs_price(S=S0, K=GRID_K, T=0.5, r=0.04, sigma=0.0, q=0.013, kind="call")
        want = np.maximum(S0 * np.exp(-0.013 * 0.5) - GRID_K * np.exp(-0.04 * 0.5), 0.0)
        np.testing.assert_allclose(got, want, atol=1e-10)

    def test_t_zero_gives_intrinsic(self):
        assert bs_price(S=105.0, K=100.0, T=0.0, r=0.04, sigma=0.2, kind="call") == pytest.approx(5.0)
        assert bs_price(S=95.0, K=100.0, T=0.0, r=0.04, sigma=0.2, kind="call") == pytest.approx(0.0)
        assert bs_price(S=100.0, K=100.0, T=0.0, r=0.04, sigma=0.2, kind="call") == pytest.approx(0.0)

    def test_no_arbitrage_bounds(self):
        c = bs_price(**GRID, kind="call")
        lo = np.maximum(S0 * np.exp(-GRID["q"] * GRID_T) - GRID_K * np.exp(-0.04 * GRID_T), 0.0)
        hi = S0 * np.exp(-GRID["q"] * GRID_T)
        assert np.all(c >= np.broadcast_to(lo, c.shape) - 1e-12)
        assert np.all(c <= np.broadcast_to(hi, c.shape) + 1e-12)

    def test_monotone_increasing_in_vol(self):
        sig = np.linspace(0.05, 1.0, 20)
        prices = bs_price(S=100.0, K=110.0, T=0.25, r=0.04, sigma=sig, kind="call")
        assert np.all(np.diff(prices) > 0)


class TestContract:
    def test_invalid_numeric_inputs_yield_nan_not_raise(self):
        bad = [
            dict(S=-1.0), dict(S=0.0), dict(K=-5.0), dict(K=0.0),
            dict(T=-0.1), dict(sigma=-0.2), dict(S=np.nan), dict(r=np.inf),
        ]
        base = dict(S=100.0, K=100.0, T=0.5, r=0.04, sigma=0.2)
        for override in bad:
            out = bs_price(**{**base, **override}, kind="call")
            assert np.isnan(out), f"expected NaN for {override}"

    def test_nan_is_elementwise(self):
        out = bs_price(S=100.0, K=np.array([100.0, -1.0, 110.0]), T=0.5, r=0.04, sigma=0.2, kind="call")
        assert not np.isnan(out[0]) and np.isnan(out[1]) and not np.isnan(out[2])

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError):
            bs_price(S=100.0, K=100.0, T=0.5, r=0.04, sigma=0.2, kind="cal")

    def test_scalar_in_float_out(self):
        out = bs_price(S=100.0, K=100.0, T=0.5, r=0.04, sigma=0.2, kind="call")
        assert isinstance(out, float)

    def test_broadcast_shape(self):
        out = bs_price(S=100.0, K=GRID_K, T=GRID_T, r=0.04, sigma=0.2, kind="call")
        assert out.shape == (GRID_T.size, GRID_K.size)
