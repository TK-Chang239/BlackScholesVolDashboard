"""greeks tests: Hull's published table + finite differences of our own pricer.

Hull anchor (Ch. 19 example): S=49, K=50, r=5%, sigma=20%, T=20/52, q=0:
delta=0.522, gamma=0.066, vega=12.1, theta=-4.31/yr, rho=8.91 (as published).
Finite differences against bs_price are the tight check.
"""
import numpy as np
import pytest

from src.models.black_scholes import bs_price, greeks

HULL = dict(S=49.0, K=50.0, T=20 / 52, r=0.05, sigma=0.20)
BASE = dict(S=100.0, K=105.0, T=0.4, r=0.04, sigma=0.25, q=0.013)


class TestHullTable:
    def test_call_greeks_match_published(self):
        g = greeks(**HULL, kind="call")
        # Tolerances = Hull's published rounding precision; the tight
        # correctness check is TestFiniteDifferences below.
        assert g["delta"] == pytest.approx(0.522, abs=1e-3)
        assert g["gamma"] == pytest.approx(0.066, abs=1e-3)
        assert g["vega"] == pytest.approx(12.1, abs=5e-2)
        assert g["theta"] == pytest.approx(-4.31, abs=1e-2)
        assert g["rho"] == pytest.approx(8.91, abs=1e-2)


class TestFiniteDifferences:
    """Central differences of bs_price, both kinds, with q != 0."""

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_delta(self, kind):
        h = BASE["S"] * 1e-5
        up = bs_price(**{**BASE, "S": BASE["S"] + h}, kind=kind)
        dn = bs_price(**{**BASE, "S": BASE["S"] - h}, kind=kind)
        assert greeks(**BASE, kind=kind)["delta"] == pytest.approx((up - dn) / (2 * h), rel=1e-6)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_gamma(self, kind):
        h = BASE["S"] * 1e-4
        up = bs_price(**{**BASE, "S": BASE["S"] + h}, kind=kind)
        mid = bs_price(**BASE, kind=kind)
        dn = bs_price(**{**BASE, "S": BASE["S"] - h}, kind=kind)
        fd = (up - 2 * mid + dn) / h**2
        assert greeks(**BASE, kind=kind)["gamma"] == pytest.approx(fd, rel=1e-4)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_vega(self, kind):
        h = 1e-6
        up = bs_price(**{**BASE, "sigma": BASE["sigma"] + h}, kind=kind)
        dn = bs_price(**{**BASE, "sigma": BASE["sigma"] - h}, kind=kind)
        assert greeks(**BASE, kind=kind)["vega"] == pytest.approx((up - dn) / (2 * h), rel=1e-6)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_theta_is_calendar_derivative(self, kind):
        """theta = dP/dt = -dP/dT: price at shorter expiry minus longer."""
        h = 1e-6
        shorter = bs_price(**{**BASE, "T": BASE["T"] - h}, kind=kind)
        longer = bs_price(**{**BASE, "T": BASE["T"] + h}, kind=kind)
        assert greeks(**BASE, kind=kind)["theta"] == pytest.approx((shorter - longer) / (2 * h), rel=1e-5)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_rho(self, kind):
        h = 1e-6
        up = bs_price(**{**BASE, "r": BASE["r"] + h}, kind=kind)
        dn = bs_price(**{**BASE, "r": BASE["r"] - h}, kind=kind)
        assert greeks(**BASE, kind=kind)["rho"] == pytest.approx((up - dn) / (2 * h), rel=1e-5)


class TestShapes:
    def test_atm_call_delta_near_half_and_gamma_peaks_atm(self):
        strikes = np.linspace(70.0, 130.0, 61)
        g = greeks(S=100.0, K=strikes, T=30 / 365, r=0.04, sigma=0.2, q=0.013, kind="call")
        atm = np.argmin(np.abs(strikes - 100.0))
        assert 0.45 < g["delta"][atm] < 0.60          # ~N(d1), slightly above 0.5
        assert np.argmax(g["gamma"]) in range(atm - 2, atm + 3)  # gamma peaks ~ATM
        assert np.all(np.diff(g["delta"]) < 0)        # call delta falls as K rises

    def test_put_call_delta_relation(self):
        """delta_call - delta_put == e^{-qT} (parity differentiated in S)."""
        c = greeks(**BASE, kind="call")["delta"]
        p = greeks(**BASE, kind="put")["delta"]
        assert c - p == pytest.approx(np.exp(-BASE["q"] * BASE["T"]), abs=1e-12)

    def test_invalid_inputs_yield_nan(self):
        g = greeks(S=100.0, K=100.0, T=-0.5, r=0.04, sigma=0.2, kind="call")
        assert all(np.isnan(v) for v in g.values())

    def test_vectorized_output_shape(self):
        g = greeks(S=100.0, K=np.linspace(70, 130, 13), T=0.25, r=0.04, sigma=0.2, kind="put")
        assert g["delta"].shape == (13,)


def test_bad_kind_raises():
    with pytest.raises(ValueError):
        greeks(S=100.0, K=100.0, T=0.5, r=0.04, sigma=0.2, kind="cal")
