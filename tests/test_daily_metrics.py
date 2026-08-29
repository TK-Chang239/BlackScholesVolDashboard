"""daily_metrics: ATM-IV-in-DTE interpolation, session rows, RV refresh, upsert."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analytics.daily_metrics import (
    IV_COLUMNS, interp_atm_iv, metric_columns, refresh_rv_columns,
    rv_columns, session_metrics_row, upsert_session,
)
from src.data import storage


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def term(rows):
    return pd.DataFrame(rows, columns=["expiry", "dte", "atm_iv"])


def underlying(n=80, start=dt.date(2026, 5, 1)):
    dates = list(pd.bdate_range(start, periods=n).date)
    rng = np.random.default_rng(3)
    px = 700.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": dates, "close": px, "adjusted_close": px,
                         "volume": [1] * n})


class TestColumns:
    def test_shipped_config_names(self):
        assert rv_columns(cfg()) == ["rv_20d", "rv_60d", "fwd_rv_30d"]
        assert metric_columns(cfg()) == IV_COLUMNS + ["rv_20d", "rv_60d", "fwd_rv_30d"]


class TestInterpAtmIv:
    def test_bracketed_is_linear_in_dte(self):
        t = term([(dt.date(2026, 9, 18), 20, 0.10), (dt.date(2026, 10, 16), 40, 0.20)])
        iv, eff = interp_atm_iv(t, 30)
        assert iv == pytest.approx(0.15) and eff == 30

    def test_exact_expiry_returns_it(self):
        t = term([(dt.date(2026, 9, 18), 30, 0.13)])
        assert interp_atm_iv(t, 30) == (pytest.approx(0.13), 30)

    def test_nearest_within_tolerance(self):
        t = term([(dt.date(2026, 9, 18), 22, 0.11), (dt.date(2026, 11, 20), 85, 0.14)])
        iv, eff = interp_atm_iv(t, 30)     # bracketed -> interpolate, not nearest
        assert 0.11 < iv < 0.14 and eff == 30
        t2 = term([(dt.date(2026, 9, 25), 38, 0.12)])
        assert interp_atm_iv(t2, 30) == (pytest.approx(0.12), 38)

    def test_out_of_range_is_nan(self):
        t = term([(dt.date(2026, 11, 20), 85, 0.14)])
        iv, eff = interp_atm_iv(t, 30)
        assert np.isnan(iv) and np.isnan(eff)
        iv, eff = interp_atm_iv(term([]), 30)
        assert np.isnan(iv) and np.isnan(eff)

    def test_nan_atm_iv_rows_ignored(self):
        t = term([(dt.date(2026, 9, 18), 20, np.nan), (dt.date(2026, 10, 16), 40, 0.20)])
        iv, eff = interp_atm_iv(t, 30)
        assert iv == pytest.approx(0.20) and eff == 40


class TestSessionRow:
    def test_row_keys_and_values(self):
        t = term([(dt.date(2026, 9, 18), 20, 0.10), (dt.date(2026, 10, 16), 40, 0.20)])
        row = session_metrics_row(dt.date(2026, 8, 28), 771.1, "yfinance", t,
                                  {"convergence": 0.993}, cfg())
        assert list(row) == IV_COLUMNS
        assert row["date"] == dt.date(2026, 8, 28)
        assert row["atm_iv_30d"] == pytest.approx(0.15)
        assert row["atm_iv_30d_dte"] == 30
        assert row["iv_convergence"] == 0.993 and row["source"] == "yfinance"


class TestUpsert:
    def test_new_row_appends_with_nan_rv(self):
        m = pd.DataFrame(columns=metric_columns(cfg()))
        row = session_metrics_row(dt.date(2026, 8, 28), 771.1, "yfinance",
                                  term([]), {"convergence": 0.0}, cfg())
        out = upsert_session(m, row, cfg())
        assert list(out.columns) == metric_columns(cfg()) and len(out) == 1
        assert out["rv_20d"].isna().all()

    def test_same_date_replaces_and_sorts(self):
        m = pd.DataFrame(columns=metric_columns(cfg()))
        for d, s in ((dt.date(2026, 8, 28), 771.1), (dt.date(2026, 8, 27), 766.0),
                     (dt.date(2026, 8, 28), 772.2)):
            m = upsert_session(m, session_metrics_row(d, s, "yfinance", term([]),
                                                      {"convergence": 1.0}, cfg()), cfg())
        assert list(m["date"]) == [dt.date(2026, 8, 27), dt.date(2026, 8, 28)]
        assert m["spot"].iloc[-1] == 772.2


class TestRefreshRv:
    def test_fills_rv_for_every_row_and_matures_forward(self):
        u = underlying(n=80)
        m = pd.DataFrame(columns=metric_columns(cfg()))
        early, late = u["date"].iloc[30], u["date"].iloc[-1]
        for d in (early, late):
            m = upsert_session(m, session_metrics_row(d, 700.0, "yfinance", term([]),
                                                      {"convergence": 1.0}, cfg()), cfg())
        out = refresh_rv_columns(m, u, cfg())
        assert list(out.columns) == metric_columns(cfg())
        assert out["rv_20d"].notna().all() and out["rv_60d"].isna().iloc[0]  # 30 < 60 returns
        assert out["fwd_rv_30d"].notna().iloc[0]      # early row has matured
        assert out["fwd_rv_30d"].isna().iloc[1]       # last row cannot have

    def test_refresh_is_idempotent_and_overwrites_stale(self):
        u = underlying(n=80)
        m = pd.DataFrame(columns=metric_columns(cfg()))
        m = upsert_session(m, session_metrics_row(u["date"].iloc[30], 700.0, "yfinance",
                                                  term([]), {"convergence": 1.0}, cfg()), cfg())
        m.loc[0, "rv_20d"] = 9.99                      # stale/wrong value
        once = refresh_rv_columns(m, u, cfg())
        twice = refresh_rv_columns(once, u, cfg())
        assert once["rv_20d"].iloc[0] != 9.99
        pd.testing.assert_frame_equal(once, twice)


class TestSkewColumns:
    def test_columns_include_skew(self):
        assert IV_COLUMNS[-2:] == ["skew_25d", "skew_25d_dte"]
        assert metric_columns(cfg())[:8] == IV_COLUMNS

    def test_row_without_skew_is_nan(self):
        row = session_metrics_row(dt.date(2026, 8, 28), 771.1, "yfinance", term([]),
                                  {"convergence": 1.0}, cfg())
        assert list(row) == IV_COLUMNS
        assert np.isnan(row["skew_25d"]) and np.isnan(row["skew_25d_dte"])

    def test_row_with_skew(self):
        skew = {"skew_expiry": dt.date(2026, 9, 25), "skew_dte": 28,
                "put_iv_25d": 0.25, "call_iv_25d": 0.20, "skew_25d": 0.05}
        row = session_metrics_row(dt.date(2026, 8, 28), 771.1, "yfinance", term([]),
                                  {"convergence": 1.0}, cfg(), skew=skew)
        assert row["skew_25d"] == pytest.approx(0.05) and row["skew_25d_dte"] == 28


class TestStorage:
    def test_read_missing_gives_empty_with_columns(self, tmp_path):
        df = storage.read_daily_metrics(tmp_path, metric_columns(cfg()))
        assert df.empty and list(df.columns) == metric_columns(cfg())

    def test_write_read_roundtrip_adds_missing_columns(self, tmp_path):
        m = pd.DataFrame({"date": [dt.date(2026, 8, 28)], "spot": [771.1]})
        p = storage.write_daily_metrics(m, tmp_path)
        assert p == tmp_path / "data" / "daily_metrics.parquet"
        back = storage.read_daily_metrics(tmp_path, metric_columns(cfg()))
        assert list(back.columns) == metric_columns(cfg())
        assert back["date"].iloc[0] == dt.date(2026, 8, 28)
        assert back["rv_20d"].isna().all()

    def test_write_rejects_duplicate_dates(self, tmp_path):
        m = pd.DataFrame({"date": [dt.date(2026, 8, 28)] * 2, "spot": [1.0, 2.0]})
        with pytest.raises(ValueError):
            storage.write_daily_metrics(m, tmp_path)

    def test_read_underlying(self, tmp_path):
        storage.upsert_underlying(underlying(n=5), tmp_path)
        assert len(storage.read_underlying(tmp_path)) == 5
