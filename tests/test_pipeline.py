"""SPEC 4 pipeline test: full daily run against fake providers, no network.

Also covers storage primitives: atomic chain write, underlying upsert,
schema validation, and the failure policy (no partial writes).
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import storage
from src.data.base import CHAIN_COLUMNS, UNDERLYING_COLUMNS
from src.run_daily import run

TODAY = dt.date(2026, 8, 28)


def real_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def underlying_frame():
    return pd.DataFrame({
        "date": [dt.date(2026, 8, 27), TODAY],
        "close": [771.1, 770.0],
        "adjusted_close": [771.1, 770.0],
        "volume": [39000000, 40500000],
    })[UNDERLYING_COLUMNS]


def chain_frame(source):
    exp = dt.date(2026, 9, 18)
    quoted = source == "yfinance"
    return pd.DataFrame({
        "expiry": [exp, exp],
        "strike": [700.0, 700.0],
        "kind": ["call", "put"],
        "bid": [71.0, 1.9] if quoted else [np.nan] * 2,
        "ask": [71.4, 2.1] if quoted else [np.nan] * 2,
        "mid": [71.2, 2.0] if quoted else [np.nan] * 2,
        "close": [71.3, 2.05],
        "volume": [120, 80],
        "open_interest": [5000, 3100],
        "vendor_iv": [0.213, 0.242],
        "source": source,
    })


class FakeEODHD:
    def get_underlying_history(self, symbol, start=None):
        return underlying_frame()
    def get_risk_free_rate(self):
        return 0.0415
    def get_dividend_yield(self, spot, today=None, symbol="SPY"):
        return 0.0098


class FakeLive:
    fail = False
    def get_option_chain(self, symbol, snapshot_date, spot, cfg):
        if self.fail:
            raise RuntimeError("yahoo broke")
        return chain_frame("yfinance")


class FakeFallback:
    called = False
    def get_option_chain(self, symbol, snapshot_date, spot, cfg):
        self.called = True
        return chain_frame("massive-fallback")


class TestStorage:
    def test_chain_write_roundtrip_and_path(self, tmp_path):
        df = chain_frame("yfinance")
        df["snapshot_date"] = TODAY
        df["spot"] = 770.0
        df["dte"] = 21
        df = df[CHAIN_COLUMNS]
        p = storage.write_chain(df, TODAY, tmp_path)
        assert p == tmp_path / "data" / "chains" / "2026-08-28.parquet"
        assert storage.chain_exists(TODAY, tmp_path)
        back = pd.read_parquet(p)
        assert list(back.columns) == CHAIN_COLUMNS and len(back) == 2

    def test_chain_write_rejects_wrong_schema(self, tmp_path):
        with pytest.raises(ValueError):
            storage.write_chain(pd.DataFrame({"x": [1]}), TODAY, tmp_path)

    def test_underlying_upsert_dedupes_new_wins(self, tmp_path):
        storage.upsert_underlying(underlying_frame(), tmp_path)
        revised = underlying_frame()
        revised.loc[revised["date"] == TODAY, "close"] = 769.5
        p = storage.upsert_underlying(revised, tmp_path)
        back = pd.read_parquet(p)
        assert len(back) == 2
        assert back.loc[back["date"] == TODAY, "close"].iloc[0] == 769.5


class TestDailyRun:
    def test_happy_path_yfinance(self, tmp_path):
        fallback = FakeFallback()
        status = run(FakeEODHD(), FakeLive(), fallback, real_cfg(), tmp_path, today=TODAY)
        assert not fallback.called
        assert status["source"] == "yfinance"
        assert status["snapshot_date"] == "2026-08-28"
        assert status["spot"] == 770.0
        assert status["rows_stored"] == 2
        assert status["risk_free_rate"] == 0.0415
        assert status["dividend_yield"] == 0.0098
        assert storage.chain_exists(TODAY, tmp_path)
        on_disk = json.loads((tmp_path / "docs" / "status.json").read_text())
        assert on_disk == status
        stored = pd.read_parquet(storage.chain_path(TODAY, tmp_path))
        assert (stored["source"] == "yfinance").all()

    def test_fallback_on_live_failure(self, tmp_path):
        live = FakeLive(); live.fail = True
        fallback = FakeFallback()
        status = run(FakeEODHD(), live, fallback, real_cfg(), tmp_path, today=TODAY)
        assert fallback.called
        assert status["source"] == "massive-fallback"
        stored = pd.read_parquet(storage.chain_path(TODAY, tmp_path))
        assert (stored["source"] == "massive-fallback").all()

    def test_both_sources_failing_writes_nothing(self, tmp_path):
        live = FakeLive(); live.fail = True
        class DeadFallback:
            def get_option_chain(self, *a, **k):
                raise RuntimeError("also down")
        with pytest.raises(RuntimeError):
            run(FakeEODHD(), live, DeadFallback(), real_cfg(), tmp_path, today=TODAY)
        assert not storage.chain_exists(TODAY, tmp_path)
        assert not (tmp_path / "docs" / "status.json").exists()

    def test_empty_filtered_chain_is_a_failure(self, tmp_path):
        class EmptyLive:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return chain_frame("yfinance").iloc[0:0]
        class DeadFallback:
            def get_option_chain(self, *a, **k):
                raise RuntimeError("down")
        with pytest.raises(RuntimeError):
            run(FakeEODHD(), EmptyLive(), DeadFallback(), real_cfg(), tmp_path, today=TODAY)
        assert not storage.chain_exists(TODAY, tmp_path)
