"""Backfill loop: trading-day iteration, resumability, empty-day handling."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import storage
from src.data.backfill import backfill
from src.data.base import UNDERLYING_COLUMNS

CFG_PATH = Path("config.yaml")


def cfg():
    import yaml
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


DATES = [dt.date(2026, 8, 24), dt.date(2026, 8, 25), dt.date(2026, 8, 26)]


class FakeEODHD:
    def get_underlying_history(self, symbol, start=None):
        return pd.DataFrame({
            "date": DATES,
            "close": [765.0, 768.0, 770.4],
            "adjusted_close": [765.0, 768.0, 770.4],
            "volume": [1, 1, 1],
        })[UNDERLYING_COLUMNS]


class FakeMassive:
    def __init__(self, empty_on=None):
        self.requested = []
        self.empty_on = empty_on or set()

    def get_historical_chain(self, symbol, snapshot_date, spot, cfg):
        self.requested.append((snapshot_date, spot))
        if snapshot_date in self.empty_on:
            return pd.DataFrame(columns=["expiry", "strike", "kind", "bid", "ask", "mid",
                                         "close", "volume", "open_interest", "vendor_iv", "source"])
        exp = dt.date(2026, 9, 18)
        return pd.DataFrame({
            "expiry": [exp], "strike": [700.0], "kind": ["call"],
            "bid": [np.nan], "ask": [np.nan], "mid": [np.nan],
            "close": [71.0], "volume": [10], "open_interest": [np.nan],
            "vendor_iv": [np.nan], "source": ["massive-backfill"],
        })


class TestBackfill:
    def test_iterates_trading_days_with_per_date_spot(self, tmp_path):
        m = FakeMassive()
        summary = backfill(m, FakeEODHD(), cfg(), tmp_path,
                           start=DATES[0], end=DATES[-1], log=lambda *_: None)
        assert [d for d, _ in m.requested] == DATES
        assert [s for _, s in m.requested] == [765.0, 768.0, 770.4]
        assert summary["dates_written"] == 3 and summary["dates_skipped"] == 0
        for d in DATES:
            assert storage.chain_exists(d, tmp_path)

    def test_resumability_skips_existing_files(self, tmp_path):
        m1 = FakeMassive()
        backfill(m1, FakeEODHD(), cfg(), tmp_path, start=DATES[0], end=DATES[0], log=lambda *_: None)
        m2 = FakeMassive()
        summary = backfill(m2, FakeEODHD(), cfg(), tmp_path,
                           start=DATES[0], end=DATES[-1], log=lambda *_: None)
        assert [d for d, _ in m2.requested] == DATES[1:]  # first date skipped
        assert summary["dates_skipped"] == 1 and summary["dates_written"] == 2

    def test_empty_day_recorded_not_written(self, tmp_path):
        m = FakeMassive(empty_on={DATES[1]})
        summary = backfill(m, FakeEODHD(), cfg(), tmp_path,
                           start=DATES[0], end=DATES[-1], log=lambda *_: None)
        assert summary["dates_empty"] == 1
        assert not storage.chain_exists(DATES[1], tmp_path)

    def test_date_window_respected(self, tmp_path):
        m = FakeMassive()
        backfill(m, FakeEODHD(), cfg(), tmp_path,
                 start=DATES[1], end=DATES[1], log=lambda *_: None)
        assert [d for d, _ in m.requested] == [DATES[1]]
