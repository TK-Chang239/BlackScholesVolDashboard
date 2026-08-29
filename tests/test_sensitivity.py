"""P1: ±bump sensitivity of a ~30-DTE ATM call to each BS input."""
import numpy as np
import pytest

from src.analytics.sensitivity import SENSITIVITY_COLUMNS, compute_sensitivity
from src.models.black_scholes import bs_price

SPOT, SIGMA, DTE, R, Q = 770.0, 0.12, 30, 0.0415, 0.0098


class TestSensitivity:
    def test_shape_and_groups(self):
        s = compute_sensitivity(SPOT, SIGMA, DTE, R, Q, bump=0.20)
        assert list(s.columns) == SENSITIVITY_COLUMNS
        assert set(s["input"]) == {"sigma", "S", "r", "q", "T"}
        assert set(s[s["group"] == "argued"]["input"]) == {"sigma", "r", "q"}
        assert set(s[s["group"] == "observed"]["input"]) == {"S", "T"}
        assert s["span"].is_monotonic_decreasing

    def test_values_are_relative_to_base_price(self):
        s = compute_sensitivity(SPOT, SIGMA, DTE, R, Q, bump=0.20).set_index("input")
        base = bs_price(SPOT, SPOT, DTE / 365.0, R, SIGMA, Q, "call")
        up = bs_price(SPOT, SPOT, DTE / 365.0, R, SIGMA * 1.2, Q, "call")
        assert s.loc["sigma", "up_pct"] == pytest.approx((up - base) / base)
        assert s.loc["sigma", "base_value"] == SIGMA

    def test_sigma_dominates_the_argued_group_and_rate_is_tiny(self):
        s = compute_sensitivity(SPOT, SIGMA, DTE, R, Q, bump=0.20).set_index("input")
        assert s.loc["sigma", "up_pct"] == pytest.approx(0.20, abs=0.03)   # ~linear ATM
        assert s.loc["sigma", "span"] > 5 * s.loc["r", "span"]
        assert s.loc["sigma", "span"] > 5 * s.loc["q", "span"]
        assert s.loc["r", "span"] < 0.05

    def test_spot_is_the_observed_giant(self):
        s = compute_sensitivity(SPOT, SIGMA, DTE, R, Q, bump=0.20).set_index("input")
        assert s.loc["S", "span"] > s.loc["sigma", "span"]          # why the groups exist
        assert s.loc["S", "down_pct"] == pytest.approx(-1.0, abs=0.02)  # 20% crash: worthless

    def test_nan_sigma_gives_nan_rows_not_raise(self):
        s = compute_sensitivity(SPOT, float("nan"), DTE, R, Q, bump=0.20)
        assert len(s) == 5 and s["up_pct"].isna().all()
