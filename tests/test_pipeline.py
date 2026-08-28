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
    return underlying_frame_ending(TODAY)


def underlying_frame_ending(end_date):
    return pd.DataFrame({
        "date": [end_date - dt.timedelta(days=1), end_date],
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
        # deliberate (controller ruling): underlying history is independently valid market data;
        # SPEC 2.1's no-partial-write guarantee covers only the chain file and status.json
        assert (tmp_path / "data" / "underlying.parquet").exists()

    def test_empty_filtered_live_chain_falls_back(self, tmp_path):
        class EmptyLive:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return chain_frame("yfinance").iloc[0:0]
        fallback = FakeFallback()
        status = run(FakeEODHD(), EmptyLive(), fallback, real_cfg(), tmp_path, today=TODAY)
        assert fallback.called
        assert status["source"] == "massive-fallback"
        assert storage.chain_exists(TODAY, tmp_path)
        stored = pd.read_parquet(storage.chain_path(TODAY, tmp_path))
        assert (stored["source"] == "massive-fallback").all()

    def test_both_paths_empty_is_a_failure(self, tmp_path):
        class EmptyLive:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return chain_frame("yfinance").iloc[0:0]
        class EmptyFallback:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return chain_frame("massive-fallback").iloc[0:0]
        with pytest.raises(RuntimeError):
            run(FakeEODHD(), EmptyLive(), EmptyFallback(), real_cfg(), tmp_path, today=TODAY)
        assert not storage.chain_exists(TODAY, tmp_path)
        assert not (tmp_path / "docs" / "status.json").exists()

    def test_late_provider_failure_writes_no_chain_or_status(self, tmp_path):
        """Regression: late provider calls (get_risk_free_rate, etc) must happen
        before write_chain to ensure no partial writes when they fail."""
        class BrokenEODHD:
            def get_underlying_history(self, symbol, start=None):
                return underlying_frame()
            def get_risk_free_rate(self):
                raise RuntimeError("rate service down")
            def get_dividend_yield(self, spot, today=None, symbol="SPY"):
                return 0.0098
        with pytest.raises(RuntimeError):
            run(BrokenEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert not storage.chain_exists(TODAY, tmp_path)
        assert not (tmp_path / "docs" / "status.json").exists()


class TestSessionDateLabeling:
    """snapshot_date must be the market session the underlying describes,
    never wall-clock `today` — matters off-hours and under a cron that runs
    before the session it's labeling has fully rolled over."""

    def test_labels_with_prior_session_when_underlying_lags_today(self, tmp_path):
        prior_session = TODAY - dt.timedelta(days=1)

        class LaggingEODHD(FakeEODHD):
            def get_underlying_history(self, symbol, start=None):
                return underlying_frame_ending(prior_session)

        fallback = FakeFallback()
        status = run(LaggingEODHD(), FakeLive(), fallback, real_cfg(), tmp_path, today=TODAY)
        assert not fallback.called
        assert status["snapshot_date"] == prior_session.isoformat()
        assert storage.chain_exists(prior_session, tmp_path)
        assert not storage.chain_exists(TODAY, tmp_path)

    def test_stale_underlying_raises_and_writes_nothing(self, tmp_path):
        stale_session = TODAY - dt.timedelta(days=10)

        class StaleEODHD(FakeEODHD):
            def get_underlying_history(self, symbol, start=None):
                return underlying_frame_ending(stale_session)

        with pytest.raises(RuntimeError):
            run(StaleEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert not storage.chain_exists(stale_session, tmp_path)
        assert not storage.chain_exists(TODAY, tmp_path)
        assert not (tmp_path / "docs" / "status.json").exists()


class TestRenderStage:
    def test_run_renders_page_and_reports_convergence(self, tmp_path):
        status = run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "<html" in page and "vol-lens" in page
        # the vendored bundle inlines an unused default (topojsonURL) that
        # literally contains this substring; strip it before checking that
        # *our own markup* makes no live external CDN reference (mirrors
        # test_render.py::TestPage::test_self_contained_no_external_refs).
        from src.render.page import _BUNDLE
        own_markup = page.replace(_BUNDLE.read_text(), "")
        assert "https://cdn.plot.ly" not in own_markup
        assert 0.0 <= status["iv_convergence"] <= 1.0
        assert status["panels_rendered"] == ["P2", "P3"]
        # status still written last and consistent with the page
        on_disk = json.loads((tmp_path / "docs" / "status.json").read_text())
        assert on_disk == status

    def test_yesterday_overlay_uses_prior_chain_when_present(self, tmp_path):
        # compute_term_structure's ATM interpolation (src/analytics/term_structure.py
        # ::_interp_at) needs a converged IV strictly at-or-below AND strictly above
        # spot; the shared single-strike chain_frame() fixture (700 only, always below
        # spot) can never straddle it, so it can never yield a non-empty term
        # structure regardless of how run() is wired. Give the FIRST (prior-day) run a
        # two-strike chain that brackets spot instead -- local to this test only;
        # chain_frame()/FakeLive/FakeFallback used by every other test are untouched.
        from src.models.black_scholes import bs_price

        def bracketing_chain(spot, expiry, dte):
            T = dte / 365.0
            rows = []
            for strike in (spot - 20.0, spot + 20.0):
                for kind in ("call", "put"):
                    price = float(bs_price(spot, strike, T, 0.0415, 0.20, 0.0098, kind))
                    rows.append({
                        "expiry": expiry, "strike": strike, "kind": kind,
                        "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                        "close": price, "volume": 100, "open_interest": 500,
                        "vendor_iv": 0.20, "source": "yfinance",
                    })
            return pd.DataFrame(rows)

        class BracketingLive:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return bracketing_chain(spot, dt.date(2026, 9, 18), 21)

        run(FakeEODHD(), BracketingLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        # second run for the "next" session: underlying now ends a day later
        class NextDayEODHD(FakeEODHD):
            def get_underlying_history(self, symbol, start=None):
                df = underlying_frame()
                extra = pd.DataFrame({"date": [TODAY + dt.timedelta(days=1)],
                                      "close": [772.0], "adjusted_close": [772.0],
                                      "volume": [1]})
                return pd.concat([df, extra], ignore_index=True)
        status = run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(),
                     tmp_path, today=TODAY + dt.timedelta(days=1))
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "yesterday" in page
        assert status["snapshot_date"] == (TODAY + dt.timedelta(days=1)).isoformat()

    def test_render_failure_leaves_page_and_status_untouched(self, tmp_path, monkeypatch):
        run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        page_before = (tmp_path / "docs" / "index.html").read_text()
        status_before = (tmp_path / "docs" / "status.json").read_text()
        import src.run_daily as rd
        monkeypatch.setattr(rd, "render_page",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        class NextDayEODHD(FakeEODHD):
            def get_underlying_history(self, symbol, start=None):
                df = underlying_frame()
                extra = pd.DataFrame({"date": [TODAY + dt.timedelta(days=1)],
                                      "close": [772.0], "adjusted_close": [772.0],
                                      "volume": [1]})
                return pd.concat([df, extra], ignore_index=True)
        with pytest.raises(RuntimeError):
            run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(),
                tmp_path, today=TODAY + dt.timedelta(days=1))
        assert (tmp_path / "docs" / "index.html").read_text() == page_before
        assert (tmp_path / "docs" / "status.json").read_text() == status_before
        # and no new chain file landed for the failed session either
        assert not storage.chain_exists(TODAY + dt.timedelta(days=1), tmp_path)
