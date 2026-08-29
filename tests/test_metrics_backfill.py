"""Replay stored chains -> daily_metrics.parquet (stateless rebuild)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import storage
from src.data.base import CHAIN_COLUMNS
from src.data.metrics_backfill import rebuild_daily_metrics
from src.models.black_scholes import bs_price

R, Q = 0.0415, 0.0098


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def chain(session, spot, sigma, source, expiries=((30, 0), (60, 0))):
    rows = []
    for dte, _ in expiries:
        expiry = session + dt.timedelta(days=dte)
        for strike in (spot - 20.0, spot - 10.0, spot, spot + 10.0, spot + 20.0):
            for kind in ("call", "put"):
                px = float(bs_price(spot, strike, dte / 365.0, R, sigma, Q, kind))
                live = source == "yfinance"
                rows.append({
                    "snapshot_date": session, "spot": spot, "expiry": expiry, "dte": dte,
                    "strike": strike, "kind": kind,
                    "bid": px - 0.05 if live else np.nan, "ask": px + 0.05 if live else np.nan,
                    "mid": px if live else np.nan, "close": px,
                    "volume": 10, "open_interest": 100.0, "vendor_iv": np.nan,
                    "source": source})
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def seed(root):
    dates = list(pd.bdate_range(dt.date(2026, 6, 1), periods=70).date)
    px = np.linspace(740.0, 770.0, 70)
    storage.upsert_underlying(pd.DataFrame({"date": dates, "close": px,
                                            "adjusted_close": px, "volume": [1] * 70}), root)
    s1, s2 = dates[10], dates[-1]
    storage.write_chain(chain(s1, float(px[10]), 0.18, "massive-backfill"), s1, root)
    storage.write_chain(chain(s2, float(px[-1]), 0.24, "yfinance"), s2, root)
    return s1, s2


class TestRebuild:
    def test_one_row_per_chain_with_recovered_atm_iv(self, tmp_path):
        s1, s2 = seed(tmp_path)
        out = rebuild_daily_metrics(tmp_path, R, Q, cfg())
        assert list(out["date"]) == [s1, s2]
        assert out["atm_iv_30d"].to_numpy() == pytest.approx([0.18, 0.24], abs=1e-5)
        assert list(out["source"]) == ["massive-backfill", "yfinance"]
        assert out["iv_convergence"].to_numpy() == pytest.approx([1.0, 1.0])
        assert out["rv_20d"].notna().iloc[1]
        assert out["fwd_rv_30d"].notna().iloc[0] and out["fwd_rv_30d"].isna().iloc[1]
        assert storage.daily_metrics_path(tmp_path).exists()

    def test_write_false_leaves_disk_untouched(self, tmp_path):
        seed(tmp_path)
        rebuild_daily_metrics(tmp_path, R, Q, cfg(), write=False)
        assert not storage.daily_metrics_path(tmp_path).exists()

    def test_rerun_is_idempotent(self, tmp_path):
        seed(tmp_path)
        a = rebuild_daily_metrics(tmp_path, R, Q, cfg())
        b = rebuild_daily_metrics(tmp_path, R, Q, cfg())
        pd.testing.assert_frame_equal(a, b)
