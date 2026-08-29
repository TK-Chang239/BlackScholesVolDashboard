"""P5: implied vs realized series and the forward-RV summary stat."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analytics.daily_metrics import metric_columns
from src.analytics.iv_rv import IV_RV_COLUMNS, iv_rv_series, iv_rv_summary, iv_rv_summary_html


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def metrics(rows):
    df = pd.DataFrame(rows)
    return df.reindex(columns=metric_columns(cfg()))


D = [dt.date(2026, 8, 20) + dt.timedelta(days=i) for i in range(5)]


class TestSeries:
    def test_columns_spread_and_running_mean(self):
        m = metrics([
            {"date": D[0], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": 0.11},
            {"date": D[1], "atm_iv_30d": 0.14, "rv_20d": 0.10, "fwd_rv_30d": np.nan},
            {"date": D[2], "atm_iv_30d": np.nan, "rv_20d": 0.10, "fwd_rv_30d": 0.09},
        ])
        s = iv_rv_series(m, cfg())
        assert list(s.columns) == IV_RV_COLUMNS
        assert list(s["date"]) == [D[0], D[1]]                # NaN-IV row dropped
        assert s["spread"].to_numpy() == pytest.approx([0.02, 0.04])
        assert s["spread_running_mean"].to_numpy() == pytest.approx([0.02, 0.03])

    def test_running_mean_skips_nan_rv(self):
        m = metrics([
            {"date": D[0], "atm_iv_30d": 0.12, "rv_20d": np.nan},
            {"date": D[1], "atm_iv_30d": 0.14, "rv_20d": 0.10},
        ])
        s = iv_rv_series(m, cfg())
        assert np.isnan(s["spread_running_mean"].iloc[0])
        assert s["spread_running_mean"].iloc[1] == pytest.approx(0.04)

    def test_empty(self):
        s = iv_rv_series(metrics([]), cfg())
        assert s.empty and list(s.columns) == IV_RV_COLUMNS


class TestSummary:
    def test_counts_and_share(self):
        m = metrics([
            {"date": D[0], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": 0.11},  # IV > fwd
            {"date": D[1], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": 0.13},  # IV < fwd
            {"date": D[2], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": 0.10},  # IV > fwd
            {"date": D[3], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": np.nan},
        ])
        summ = iv_rv_summary(iv_rv_series(m, cfg()))
        assert summ["history_since"] == D[0] and summ["n_sessions"] == 4
        assert summ["evaluable_days"] == 3
        assert summ["share_iv_above_fwd_rv"] == pytest.approx(2 / 3)
        assert summ["mean_spread_trailing"] == pytest.approx(0.02)
        html = iv_rv_summary_html(summ)
        assert "3 sessions" in html and "2 of them" in html and "too few" in html

    def test_single_evaluable_day_is_not_a_rate(self):
        m = metrics([{"date": D[0], "atm_iv_30d": 0.161, "rv_20d": 0.10, "fwd_rv_30d": 0.1115}])
        summ = iv_rv_summary(iv_rv_series(m, cfg()))
        assert summ["evaluable_days"] == 1
        assert summ["last_evaluable_iv"] == pytest.approx(0.161)
        assert summ["last_evaluable_fwd_rv"] == pytest.approx(0.1115)
        html = iv_rv_summary_html(summ)
        assert "100%" not in html
        assert "1 session" in html and "16.1%" in html and "11.2%" in html
        assert "not a rate" in html

    def test_percentage_sentence_at_ge_10_days(self):
        dates = [dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(10)]
        rows = [{"date": d, "atm_iv_30d": 0.12, "rv_20d": 0.10,
                 "fwd_rv_30d": 0.11 if i % 2 == 0 else 0.13}
                for i, d in enumerate(dates)]
        summ = iv_rv_summary(iv_rv_series(metrics(rows), cfg()))
        assert summ["evaluable_days"] == 10
        html = iv_rv_summary_html(summ)
        assert "evaluable" in html and "days" in html
        assert f"{summ['share_iv_above_fwd_rv']:.0%}" in html
        assert dates[0].isoformat() in html

    def test_no_evaluable_days(self):
        m = metrics([{"date": D[0], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": np.nan}])
        summ = iv_rv_summary(iv_rv_series(m, cfg()))
        assert summ["evaluable_days"] == 0 and np.isnan(summ["share_iv_above_fwd_rv"])
        assert "not yet" in iv_rv_summary_html(summ)

    def test_empty_summary(self):
        summ = iv_rv_summary(iv_rv_series(metrics([]), cfg()))
        assert summ["history_since"] is None and summ["n_sessions"] == 0
        assert "no history" in iv_rv_summary_html(summ).lower()
