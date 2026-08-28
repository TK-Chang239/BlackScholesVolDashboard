"""implied_vol tests: round-trip recovery, corner contract, no-arb NaNs.

SPEC §4 contract: exact recovery (1e-6) on the normal grid; near-degenerate
corners must be accurate-or-NaN — never silently wrong.
"""
import numpy as np
import pytest

from src.models.black_scholes import bs_price, implied_vol

R, Q = 0.04, 0.013
S0 = 100.0


def roundtrip(sigma, K, T, kind):
    price = bs_price(S=S0, K=K, T=T, r=R, sigma=sigma, q=Q, kind=kind)
    return implied_vol(price, S=S0, K=K, T=T, r=R, q=Q, kind=kind)


class TestNormalGridExactRecovery:
    """Liquid regime: recovery to 1e-6, no NaNs.

    Grid bounds matter: sigma-space accuracy of a price-space Newton stop is
    ~ price_tol / vega, so the 1e-6 guarantee only holds where vega is
    healthy (roughly |d1| < 4). Short-dated far wings live in the corner
    contract below instead — same reason real chains' wing IVs are noisy.
    """

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_roundtrip_grid(self, kind):
        sigma = np.array([0.15, 0.25, 0.40, 0.60, 1.00])[:, None, None]
        K = np.linspace(85.0, 115.0, 7)[None, :, None]
        T = np.array([30 / 365, 90 / 365, 0.5, 1.0])[None, None, :]
        got = roundtrip(sigma, K, T, kind)
        assert not np.any(np.isnan(got)), "no NaNs allowed on the normal grid"
        np.testing.assert_allclose(got, np.broadcast_to(sigma, got.shape), atol=1e-6)


class TestCornerContract:
    """Degenerate corners: recover to 1e-6 OR return NaN — never wrong."""

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_corners_accurate_or_nan(self, kind):
        sigma = np.array([0.05, 0.20, 0.90, 3.00])[:, None, None]
        K = np.array([70.0, 100.0, 130.0, 300.0])[None, :, None]
        T = np.array([1 / 365, 7 / 365, 2.0])[None, None, :]
        got = roundtrip(sigma, K, T, kind)
        want = np.broadcast_to(sigma, got.shape)
        ok = np.isnan(got) | (np.abs(got - want) < 1e-6)
        assert np.all(ok), f"silently-wrong IVs at {np.argwhere(~ok)}"

    def test_deep_otm_tiny_vega_does_not_return_garbage(self):
        """The naive-Newton killer: vega ~ 0, price ~ 1e-30."""
        price = bs_price(S=S0, K=300.0, T=0.05, r=R, sigma=0.30, q=Q, kind="call")
        got = implied_vol(price, S=S0, K=300.0, T=0.05, r=R, q=Q, kind="call")
        assert np.isnan(got) or abs(got - 0.30) < 1e-6


class TestNoArbitrageBounds:
    def test_price_above_upper_bound_is_nan(self):
        assert np.isnan(implied_vol(101.0, S=S0, K=100.0, T=0.5, r=R, q=Q, kind="call"))

    def test_price_below_intrinsic_is_nan(self):
        # discounted forward intrinsic of this ITM call is ~ 20.9; 15 violates it
        assert np.isnan(implied_vol(15.0, S=S0, K=80.0, T=0.5, r=R, q=Q, kind="call"))

    def test_nonpositive_or_nan_price_is_nan(self):
        for bad in (0.0, -1.0, np.nan):
            assert np.isnan(implied_vol(bad, S=S0, K=100.0, T=0.5, r=R, q=Q, kind="call"))

    def test_zero_t_is_nan(self):
        assert np.isnan(implied_vol(5.0, S=S0, K=100.0, T=0.0, r=R, q=Q, kind="call"))


class TestVectorization:
    def test_mixed_good_and_bad_elementwise(self):
        K = np.array([100.0, 110.0, 100.0])
        price = np.array(
            [bs_price(S=S0, K=100.0, T=0.5, r=R, sigma=0.2, q=Q, kind="call"), 200.0, -1.0]
        )
        got = implied_vol(price, S=S0, K=K, T=0.5, r=R, q=Q, kind="call")
        assert abs(got[0] - 0.2) < 1e-6
        assert np.isnan(got[1]) and np.isnan(got[2])

    def test_scalar_in_float_out(self):
        price = bs_price(S=S0, K=105.0, T=0.5, r=R, sigma=0.25, q=Q, kind="put")
        out = implied_vol(price, S=S0, K=105.0, T=0.5, r=R, q=Q, kind="put")
        assert isinstance(out, float) and abs(out - 0.25) < 1e-6
