"""Realized vol: log returns on adjusted close, sample std, sqrt(252)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.models.realized_vol import (
    MIN_FORWARD_RETURNS, forward_realized_vol, log_returns, trailing_realized_vol,
)

START = dt.date(2026, 1, 5)


def business_days(n):
    return list(pd.bdate_range(START, periods=n).date)


def frame(adjusted, close=None):
    dates = business_days(len(adjusted))
    close = list(adjusted) if close is None else close
    return pd.DataFrame({"date": dates, "close": close,
                         "adjusted_close": list(adjusted),
                         "volume": [1_000_000] * len(adjusted)})


class TestLogReturns:
    def test_first_is_nan_then_log_ratio(self):
        r = log_returns(pd.Series([100.0, 110.0, 99.0]))
        assert np.isnan(r.iloc[0])
        assert r.iloc[1] == pytest.approx(np.log(1.1))
        assert r.iloc[2] == pytest.approx(np.log(0.9))


class TestTrailing:
    def test_constant_price_is_zero_vol(self):
        rv = trailing_realized_vol(frame([100.0] * 30), window=20)
        assert rv.name == "rv_20d"
        assert rv.iloc[:20].isna().all()          # needs 20 returns = 21 closes
        assert np.allclose(rv.iloc[20:], 0.0)

    def test_alternating_returns_match_closed_form(self):
        x = 0.01
        prices = [100.0]
        for i in range(40):
            prices.append(prices[-1] * np.exp(x if i % 2 == 0 else -x))
        rv = trailing_realized_vol(frame(prices), window=20, annualization_days=252)
        # 20 returns of +x/-x: mean 0, sample var = 20x^2/19
        expected = x * np.sqrt(20 / 19) * np.sqrt(252)
        assert rv.iloc[-1] == pytest.approx(expected, rel=1e-9)

    def test_uses_adjusted_close_not_close(self):
        # ex-dividend day: raw close drops 2%, adjusted series is flat
        close = [100.0] * 15 + [98.0] * 15
        rv = trailing_realized_vol(frame([100.0] * 30, close=close), window=20)
        assert np.allclose(rv.dropna(), 0.0)

    def test_indexed_by_date_and_sorted(self):
        df = frame([100.0, 101.0, 102.0, 101.0, 103.0]).iloc[::-1]  # reversed input
        rv = trailing_realized_vol(df, window=2)
        assert list(rv.index) == business_days(5)
        assert np.isnan(rv.iloc[0]) and np.isnan(rv.iloc[1])
        assert not np.isnan(rv.iloc[2])


class TestForward:
    def test_matured_dates_only(self):
        df = frame(np.linspace(100.0, 120.0, 60))
        fwd = forward_realized_vol(df, horizon_days=30)
        assert fwd.name == "fwd_rv_30d"
        last = df["date"].iloc[-1]
        for date, val in fwd.items():
            matured = (date + dt.timedelta(days=30)) <= last
            assert (not np.isnan(val)) == matured, date

    def test_value_matches_hand_computation(self):
        rng = np.random.default_rng(7)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
        df = frame(prices)
        fwd = forward_realized_vol(df, horizon_days=30, annualization_days=252)
        t = df["date"].iloc[5]
        end = t + dt.timedelta(days=30)
        sub = df[(df["date"] > t) & (df["date"] <= end)]
        # returns *into* those dates: include the close on t as the base
        base = df[df["date"] == t]["adjusted_close"].iloc[0]
        series = np.concatenate([[base], sub["adjusted_close"].to_numpy()])
        rets = np.diff(np.log(series))
        expected = np.std(rets, ddof=1) * np.sqrt(252)
        assert fwd.loc[t] == pytest.approx(expected, rel=1e-9)

    def test_too_few_returns_is_nan(self):
        # 3-day horizon holds at most 3 returns < MIN_FORWARD_RETURNS
        assert MIN_FORWARD_RETURNS > 3
        fwd = forward_realized_vol(frame([100.0 + i for i in range(40)]), horizon_days=3)
        assert fwd.isna().all()
