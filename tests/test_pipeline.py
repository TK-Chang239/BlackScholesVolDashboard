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
from src import run_daily
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


def flat_chain(n_pairs, source="yfinance"):
    """A trivial CHAIN_COLUMNS-conforming frame with `n_pairs` call+put strikes,
    written straight to disk to seed a "stored chain already exists" fixture
    without going through run()/filter_chain."""
    exp = dt.date(2026, 9, 18)
    rows = []
    for i in range(n_pairs):
        strike = 700.0 + i
        for kind in ("call", "put"):
            rows.append({
                "expiry": exp, "strike": strike, "kind": kind,
                "bid": 1.0, "ask": 1.2, "mid": 1.1, "close": 1.1,
                "volume": 10, "open_interest": 10, "vendor_iv": 0.2,
                "source": source,
            })
    df = pd.DataFrame(rows)
    df["snapshot_date"] = TODAY
    df["spot"] = 770.0
    df["dte"] = 21
    return df[CHAIN_COLUMNS]


class TestOverwriteShrinkGuard:
    """F7 regression guard: refuse to publish when the freshly filtered chain
    is drastically smaller than the stored chain it would replace (the
    eddf247 pre-market defect: 1,452 stored rows -> 38 filtered rows, silently
    accepted because 38 is not empty)."""

    def test_fires_on_large_shrink(self, tmp_path):
        storage.write_chain(flat_chain(20), TODAY, tmp_path)   # 40 rows stored
        with pytest.raises(RuntimeError) as excinfo:
            run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        msg = str(excinfo.value)
        assert "2026-08-28" in msg              # names the session
        assert "40" in msg and "2" in msg       # both row counts (stored, new)
        assert "re-run" in msg.lower()          # what to do about it
        assert "delete" in msg.lower()          # ...or accept it deliberately

    def test_first_ever_write_is_not_gated(self, tmp_path):
        assert not storage.chain_exists(TODAY, tmp_path)
        status = run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert status["rows_stored"] == 2
        assert storage.chain_exists(TODAY, tmp_path)

    def test_same_size_or_larger_chain_is_not_gated(self, tmp_path):
        storage.write_chain(flat_chain(1), TODAY, tmp_path)    # 2 rows stored
        status = run(FakeEODHD(), WideLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert status["rows_stored"] > 2
        stored_after = pd.read_parquet(storage.chain_path(TODAY, tmp_path))
        assert len(stored_after) == status["rows_stored"]

    def test_nothing_is_written_when_it_fires(self, tmp_path):
        # a real successful run establishes a large stored chain plus every
        # other downstream artifact
        run(FakeEODHD(), WideLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        chain_before = storage.chain_path(TODAY, tmp_path).read_bytes()
        metrics_before = storage.daily_metrics_path(tmp_path).read_bytes()
        page_before = (tmp_path / "docs" / "index.html").read_bytes()
        status_before = (tmp_path / "docs" / "status.json").read_bytes()

        # second run for the SAME session with a drastically smaller live
        # chain (2 rows vs the 42 just stored) must raise before writing
        with pytest.raises(RuntimeError):
            run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)

        # non-vacuous: the two runs feed genuinely different data (42-row wide
        # chain vs 2-row chain_frame), so any write at all -- even a partial
        # one -- would change these bytes, not merely leave a file "present"
        assert storage.chain_path(TODAY, tmp_path).read_bytes() == chain_before
        assert storage.daily_metrics_path(tmp_path).read_bytes() == metrics_before
        assert (tmp_path / "docs" / "index.html").read_bytes() == page_before
        assert (tmp_path / "docs" / "status.json").read_bytes() == status_before


class TestThinFirstWriteGuard:
    """The overwrite guard compares a session only against ITSELF, so it
    returns early on a first-ever write -- and a scheduled run almost always
    writes a session for the first time. Runs 33601964465 and its successor
    published 29-row chains for 2026-09-01 and 2026-09-02 that way, against
    ~1,440 rows on each neighbouring session, and neither was refused: the
    guard and the off-hours triage built on it were both inert for the cron,
    which is the only caller that matters in production.

    A chain far below what recent sessions hold is a defect whether or not
    that session has been stored before."""

    def _seed_recent(self, tmp_path, n_pairs, count=5):
        """`count` stored sessions ending two days before TODAY, each holding
        `n_pairs` call+put pairs. They deliberately sit outside the two dates
        in `underlying_frame`, so they are visible to the guard without also
        becoming the run's `prior_dates` term-structure comparison -- this
        tests the guard, not the previous-session panel."""
        for i in range(count):
            storage.write_chain(flat_chain(n_pairs), TODAY - dt.timedelta(days=2 + i), tmp_path)

    def test_fires_when_a_first_write_is_far_below_recent_sessions(self, tmp_path):
        self._seed_recent(tmp_path, 20)                  # 5 sessions x 40 rows
        assert not storage.chain_exists(TODAY, tmp_path)  # genuinely a first write
        with pytest.raises(run_daily.ChainRetentionRefusal):
            run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)

    def test_the_refusal_names_the_recent_typical_size_and_the_new_count(self, tmp_path):
        self._seed_recent(tmp_path, 20)
        with pytest.raises(run_daily.ChainRetentionRefusal) as excinfo:
            run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        msg = str(excinfo.value)
        assert "2026-08-28" in msg                 # names the session
        assert "40" in msg and "2" in msg          # typical recent size, and this one
        assert "recent" in msg.lower()             # says what it compared against

    def test_a_first_write_in_line_with_recent_sessions_is_accepted(self, tmp_path):
        self._seed_recent(tmp_path, 1)                   # 5 sessions x 2 rows
        status = run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert status["rows_stored"] == 2
        assert storage.chain_exists(TODAY, tmp_path)

    def test_too_little_history_to_judge_permits_the_write(self, tmp_path):
        # Below the sample floor there is no "typical" to measure against, and
        # refusing here would block the archive's own first sessions.
        self._seed_recent(tmp_path, 20, count=2)
        status = run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert status["rows_stored"] == 2

    def test_nothing_is_written_when_it_fires(self, tmp_path):
        self._seed_recent(tmp_path, 20)
        with pytest.raises(run_daily.ChainRetentionRefusal):
            run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert not storage.chain_exists(TODAY, tmp_path)
        assert not (tmp_path / "docs" / "status.json").exists()
        assert not storage.daily_metrics_path(tmp_path).exists()


class WideFallback:
    """Massive standing in for a session the live book could not supply."""
    called = False
    def get_option_chain(self, symbol, snapshot_date, spot, cfg):
        self.called = True
        return wide_chain(spot, dt.date(2026, 9, 25), 28, source="massive-fallback")


class NeverCalledFallback:
    def get_option_chain(self, *a, **k):
        raise AssertionError("the fallback must not be reached for a healthy live book")


class TestThinLiveBookFallsThroughToMassive:
    """The cron has been dispatched ~5h late on every observed night, landing
    near 03:00 ET where no book exists. The retention guard stops that from
    corrupting the archive, but refusing is all it can do -- the page then
    stops advancing, which trades a wrong dashboard for a frozen one.

    A thin live book is not a reason to publish nothing. It is a reason to
    ask the other vendor, exactly as an empty one already does: Massive
    serves the closed session and the run publishes real rows at any hour."""

    def _seed_recent(self, tmp_path, n_pairs=20, count=5):
        for i in range(count):
            storage.write_chain(flat_chain(n_pairs), TODAY - dt.timedelta(days=2 + i), tmp_path)

    def test_a_thin_live_book_falls_through_to_massive(self, tmp_path):
        self._seed_recent(tmp_path)                  # 5 sessions x 40 rows
        fallback = WideFallback()
        status = run(FakeEODHD(), FakeLive(), fallback, real_cfg(), tmp_path, today=TODAY)
        assert fallback.called, "a 2-row book against a 40-row normal never asked Massive"
        assert status["source"] == "massive-fallback"

    def test_the_stored_chain_is_the_massive_one_not_the_thin_live_one(self, tmp_path):
        self._seed_recent(tmp_path)
        status = run(FakeEODHD(), FakeLive(), WideFallback(), real_cfg(), tmp_path, today=TODAY)
        stored = pd.read_parquet(storage.chain_path(TODAY, tmp_path))
        # non-vacuous: the live book carried 2 rows and a "yfinance" source, so
        # neither assertion can pass on the frame that would have been stored
        assert (stored["source"] == "massive-fallback").all()
        assert len(stored) > 2
        assert status["rows_stored"] == len(stored)

    def test_a_healthy_live_book_never_reaches_the_fallback(self, tmp_path):
        self._seed_recent(tmp_path)
        status = run(FakeEODHD(), WideLive(), NeverCalledFallback(), real_cfg(), tmp_path, today=TODAY)
        assert status["source"] == "yfinance"

    def test_both_sources_thin_still_refuses(self, tmp_path):
        # what the fallback cannot fix, the guard must still catch
        self._seed_recent(tmp_path)
        with pytest.raises(run_daily.ChainRetentionRefusal):
            run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)

    def test_too_little_history_publishes_the_thin_live_chain(self, tmp_path):
        # No "normal" to measure against, so there is nothing to call thin and
        # no reason to spend a Massive fetch.
        self._seed_recent(tmp_path, count=2)
        fallback = WideFallback()
        status = run(FakeEODHD(), FakeLive(), fallback, real_cfg(), tmp_path, today=TODAY)
        assert not fallback.called
        assert status["source"] == "yfinance"
        assert status["rows_stored"] == 2


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
        assert status["panels_rendered"] == sorted(
            ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8a", "P8b", "P8c", "P9"])
        # P8 hedge-sim keys (Phase 6). This fixture opens NO trade, and the
        # reason is not moneyness: `select_straddle` has no at-the-money
        # requirement, it just takes the strike nearest spot, so a lone 700
        # strike would be selected whatever it is. What actually happens is that
        # 700 is ~9% IN the money for the call, and its $71.20 quote sits below
        # the European no-arbitrage floor S·e^-qT - K·e^-rT = $71.23 at
        # r=0.0415, q=0.0098, T=21/365 -- so the call's IV never converges, no
        # strike carries both legs, and `select_straddle` returns None. Still
        # the right place to assert the KEYS exist: the shape of status.json
        # must hold whether or not a trade could be opened. A live trade
        # through the same wiring is covered by
        # test_wide_chain_seeds_a_hedge_trade_end_to_end below.
        for key in ("hedge_trades", "hedge_trades_settled", "hedge_trades_open",
                    "hedge_trades_sparse", "hedge_trades_reached_expiry",
                    "hedge_months_skipped", "hedge_cum_pnl", "hedge_sessions",
                    "hedge_dte_at_entry_min", "hedge_dte_at_entry_max",
                    "hedge_market_mark_share", "hedge_quotable_mark_share",
                    "hedge_model_marks", "hedge_model_marks_structural",
                    "hedge_model_marks_gap",
                    "hedge_slope_per_vol_point", "hedge_r2"):
            assert key in status
        assert status["hedge_trades"] == 0
        # the month WAS considered and rejected -- that must be visible, not silent
        assert status["hedge_months_skipped"] == 1
        from src.analytics.chain_iv import compute_chain_iv
        stored = pd.read_parquet(storage.chain_path(TODAY, tmp_path))
        solved, _ = compute_chain_iv(stored, 0.0415, 0.0098)
        call = solved[solved["kind"] == "call"].iloc[0]
        assert pd.isna(call["iv"])          # the stated cause, not an assumed one
        assert solved[solved["kind"] == "put"]["iv"].notna().all()   # the put is fine
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
        assert "prev session" in page
        assert status["snapshot_date"] == (TODAY + dt.timedelta(days=1)).isoformat()

    def test_close_based_prior_chain_labels_overlay(self, tmp_path):
        # Same shape as the prior-chain-overlay test above, but the prior
        # session's chain came from the massive fallback (not yfinance), so
        # the overlay label must flag it as close-based rather than implying
        # a live quote timestamp.
        from src.models.black_scholes import bs_price

        def bracketing_chain(spot, expiry, dte, source):
            T = dte / 365.0
            rows = []
            for strike in (spot - 20.0, spot + 20.0):
                for kind in ("call", "put"):
                    price = float(bs_price(spot, strike, T, 0.0415, 0.20, 0.0098, kind))
                    rows.append({
                        "expiry": expiry, "strike": strike, "kind": kind,
                        "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                        "close": price, "volume": 100, "open_interest": 500,
                        "vendor_iv": 0.20, "source": source,
                    })
            return pd.DataFrame(rows)

        class FailingLive:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                raise RuntimeError("yahoo broke")

        class BracketingFallback:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return bracketing_chain(spot, dt.date(2026, 9, 18), 21, "massive-fallback")

        run(FakeEODHD(), FailingLive(), BracketingFallback(), real_cfg(), tmp_path, today=TODAY)

        class NextDayEODHD(FakeEODHD):
            def get_underlying_history(self, symbol, start=None):
                df = underlying_frame()
                extra = pd.DataFrame({"date": [TODAY + dt.timedelta(days=1)],
                                      "close": [772.0], "adjusted_close": [772.0],
                                      "volume": [1]})
                return pd.concat([df, extra], ignore_index=True)
        run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(),
            tmp_path, today=TODAY + dt.timedelta(days=1))
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "close-based" in page

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


class TestPhase5Stage:
    def test_status_and_page_carry_skew_parity_heatmap(self, tmp_path):
        status = run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        # single-strike fixture: one parity pair, no skew bracket, no flat vol -> P9 empty
        assert status["parity_pairs"] == 1
        assert status["parity_liquid_pairs"] == 1
        assert status["parity_tradeable_violations"] in (0, 1)
        # one strike cannot bracket spot -> no forward, so nothing is calibratable
        assert status["parity_tradeable_violations_fwd"] == 0
        assert status["parity_violations_early_exercise"] == 0
        assert status["parity_violations_unexplained"] == 0
        assert status["implied_carry"] is None
        assert status["implied_carry_dte_min"] is None
        assert status["implied_carry_dte_max"] is None
        assert status["implied_carry_expiries"] == 0
        assert status["implied_carry_statistic"] is None
        assert status["skew_25d"] is None
        m = pd.read_parquet(storage.daily_metrics_path(tmp_path))
        assert "skew_25d" in m.columns and np.isnan(m["skew_25d"].iloc[0])
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "Put-call parity checker" in page and "Model-vs-market" in page
        assert "25-delta skew" in page
        assert "arrives in Phase 5" not in page
        # Phase 6 retires the Q4 placeholder: the P8 panels render (in their
        # own empty state, since this single-strike fixture is ~9% OTM of spot
        # and can never seed an at-the-money straddle) rather than a
        # "arrives in Phase N" note.
        assert "arrives in Phase 6" not in page
        assert "Cumulative P&L" in page

    def test_wide_chain_seeds_a_hedge_trade_end_to_end(self, tmp_path):
        """F7: run_daily -> replay_hedge_sim -> hedge_figures -> render_page with a
        REAL trade in it. Every other pipeline test drives that path in its empty
        state, which is why F1, F2 and F4 shipped undetected.
        """
        status = run(FakeEODHD(), WideLive(), FakeFallback(), real_cfg(), tmp_path,
                     today=TODAY)
        assert status["hedge_trades"] == 1
        assert status["hedge_trades_open"] == 1
        assert status["hedge_trades_settled"] == 0
        assert status["hedge_trades_reached_expiry"] == 0
        assert status["hedge_trades_sparse"] == 0
        assert status["hedge_months_skipped"] == 0
        assert status["hedge_sessions"] == 1
        # the tenor the sim actually traded, not config's entry_dte: 30
        assert status["hedge_dte_at_entry_min"] == 28
        assert status["hedge_dte_at_entry_max"] == 28
        # the structural/gap split reaches status.json: this trade is one session
        # old, so nothing is modelled either avoidably or by design yet
        assert status["hedge_model_marks"] == 0
        assert status["hedge_model_marks_structural"] == 0
        assert status["hedge_model_marks_gap"] == 0
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "1 simulated trade since 2026-08-28" in page
        assert "sold 28 days from expiry" in page
        assert "none has reached expiry yet" in page
        # the SPEC 3 P8 disclosure line reached the page and does not claim a mid
        assert "close-based" in page
        assert "sold at the mid" not in page
        on_disk = json.loads((tmp_path / "docs" / "status.json").read_text())
        assert on_disk == status

    def test_wide_chain_records_skew_and_zero_violations(self, tmp_path):
        from src.models.black_scholes import bs_price

        def wide_chain(spot, expiry, dte):
            T = dte / 365.0
            rows = []
            for strike in [spot + d for d in range(-100, 101, 10)]:
                for kind in ("call", "put"):
                    price = float(bs_price(spot, strike, T, 0.0415, 0.20, 0.0098, kind))
                    rows.append({"expiry": expiry, "strike": float(strike), "kind": kind,
                                 "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                                 "close": price, "volume": 100, "open_interest": 500,
                                 "vendor_iv": 0.20, "source": "yfinance"})
            return pd.DataFrame(rows)

        class WideLive:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return wide_chain(spot, dt.date(2026, 9, 25), 28)

        status = run(FakeEODHD(), WideLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert status["skew_25d"] == pytest.approx(0.0, abs=1e-6)   # flat vol -> no skew
        assert status["parity_pairs"] == 21 and status["parity_tradeable_violations"] == 0
        assert status["parity_liquid_pairs"] == 21
        assert status["parity_tradeable_violations_fwd"] == 0
        # the synthetic chain satisfies parity exactly: nothing to explain either way
        assert status["parity_violations_early_exercise"] == 0
        assert status["parity_violations_unexplained"] == 0
        assert status["implied_carry"] == pytest.approx(0.0415 - 0.0098, abs=1e-6)
        # only one expiry available and it is below min_dte -> the longest-expiry
        # fallback, so the median is that single reading and the range is degenerate
        assert status["implied_carry_dte_min"] == 28
        assert status["implied_carry_dte_max"] == 28
        assert status["implied_carry_expiries"] == 1
        assert status["implied_carry_lo"] == status["implied_carry_hi"] == status["implied_carry"]
        # status.json must say which statistic `implied_carry` is, not leave
        # the reader to infer it from the keys around it
        assert status["implied_carry_statistic"] == (
            "implied carry at the longest available expiry (28d); no expiry reaches 84d")
        m = pd.read_parquet(storage.daily_metrics_path(tmp_path))
        assert m["skew_25d_dte"].iloc[0] == 28
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "0 tradeable violations" in page
        assert "Vol points" in page                          # P9 toggle rendered


def wide_chain(spot, expiry, dte, source="yfinance"):
    """A ladder that brackets spot on both sides, so P6 gets a real skew point."""
    from src.models.black_scholes import bs_price
    T = dte / 365.0
    rows = []
    for strike in [spot + d for d in range(-100, 101, 10)]:
        for kind in ("call", "put"):
            price = float(bs_price(spot, strike, T, 0.0415, 0.20, 0.0098, kind))
            rows.append({"expiry": expiry, "strike": float(strike), "kind": kind,
                         "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                         "close": price, "volume": 100, "open_interest": 500,
                         "vendor_iv": 0.20, "source": source})
    return pd.DataFrame(rows)


class WideLive:
    def get_option_chain(self, symbol, snapshot_date, spot, cfg):
        return wide_chain(spot, dt.date(2026, 9, 25), 28)


class TestAnnotationResilience:
    """A4: `data/annotations.yaml` is designed to be hand-edited ("adding a news
    note is a 1-line commit"), so a typo in it is the EXPECTED failure -- and it
    used to abort the run after the whole compute stage but before any write,
    costing that session's chain, metrics row, page and status. A session's
    quotes cannot be re-fetched later. Annotations are decoration."""

    def _write(self, tmp_path, text):
        (tmp_path / "data").mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / "annotations.yaml").write_text(text)

    GOOD = "annotations:\n  - {date: 2026-08-28, note: FOMC statement}\n"
    BAD = ("annotations:\n  - {date: 2026-08-28, note: FOMC statement}\n"
           "  - {date: 2026-08-27}\n")          # no 'note': one hand-edit typo

    def test_a_valid_annotation_reaches_the_page(self, tmp_path):
        # the positive control for the test below: this note IS renderable
        self._write(tmp_path, self.GOOD)
        run(FakeEODHD(), WideLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert "FOMC statement" in (tmp_path / "docs" / "index.html").read_text()

    def test_a_malformed_annotations_file_does_not_cost_the_session(self, tmp_path, capsys):
        self._write(tmp_path, self.BAD)
        status = run(FakeEODHD(), WideLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        # every write still happened
        assert storage.chain_exists(TODAY, tmp_path)
        assert storage.daily_metrics_path(tmp_path).exists()
        assert (tmp_path / "docs" / "index.html").exists()
        assert json.loads((tmp_path / "docs" / "status.json").read_text()) == status
        # ...and P6 rendered, just without the notes
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "25-delta skew" in page
        assert "FOMC statement" not in page
        # the failure is not silent
        err = capsys.readouterr().err
        assert "annotations.yaml" in err

    def test_load_annotations_itself_still_raises(self, tmp_path):
        # strictness stays where it is useful: CI and test_shipped_file_loads
        # must still catch a bad file that someone committed
        from src.analytics.annotations import load_annotations
        self._write(tmp_path, self.BAD)
        with pytest.raises(ValueError):
            load_annotations(tmp_path / "data" / "annotations.yaml")


class TestDailyMetricsStage:
    def test_first_run_writes_one_metrics_row(self, tmp_path):
        status = run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        m = pd.read_parquet(storage.daily_metrics_path(tmp_path))
        assert len(m) == 1 and m["date"].iloc[0] == TODAY
        assert m["source"].iloc[0] == "yfinance"
        assert "rv_20d" in m.columns and "fwd_rv_30d" in m.columns
        # single-strike fixture cannot bracket spot -> no ATM IV, and that is recorded honestly
        assert np.isnan(m["atm_iv_30d"].iloc[0]) and status["atm_iv_30d"] is None
        assert status["daily_metrics_rows"] == 1
        assert status["history_since"] == TODAY.isoformat()

    def test_second_session_appends_and_rerun_replaces(self, tmp_path):
        run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        class NextDayEODHD(FakeEODHD):
            def get_underlying_history(self, symbol, start=None):
                df = underlying_frame()
                extra = pd.DataFrame({"date": [TODAY + dt.timedelta(days=1)],
                                      "close": [772.0], "adjusted_close": [772.0],
                                      "volume": [1]})
                return pd.concat([df, extra], ignore_index=True)
        s2 = run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path,
                 today=TODAY + dt.timedelta(days=1))
        s3 = run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path,
                 today=TODAY + dt.timedelta(days=1))
        m = pd.read_parquet(storage.daily_metrics_path(tmp_path))
        assert list(m["date"]) == [TODAY, TODAY + dt.timedelta(days=1)]
        assert s2["daily_metrics_rows"] == 2 and s3["daily_metrics_rows"] == 2

    def test_bracketing_chain_records_atm_iv_and_renders_tiles(self, tmp_path):
        from src.models.black_scholes import bs_price

        def bracketing_chain(spot, expiry, dte):
            T = dte / 365.0
            rows = []
            for strike in (spot - 20.0, spot, spot + 20.0):
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
                return bracketing_chain(spot, dt.date(2026, 9, 25), 28)

        status = run(FakeEODHD(), BracketingLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert status["atm_iv_30d"] == pytest.approx(0.20, abs=1e-4)
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "class='tiles'" in page and "Delta" in page
        # tornado rendered with data; plotly's default (no-orjson) JSON engine
        # ASCII-escapes the non-ASCII sigma glyph in the "Volatility <sigma>"
        # label inside the embedded figure JSON (renders correctly in-browser
        # via JSON.parse, but never appears as the literal glyph in the raw
        # HTML text) -- same "Volatility" substring precedent already used by
        # test_render.py's sensitivity-figure test.
        assert "Volatility" in page
        assert "accumulating since" in page

    def test_render_failure_leaves_metrics_untouched(self, tmp_path, monkeypatch):
        run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        before = storage.daily_metrics_path(tmp_path).read_bytes()
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
            run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path,
                today=TODAY + dt.timedelta(days=1))
        assert storage.daily_metrics_path(tmp_path).read_bytes() == before


class TestRefusalTriage:
    """A retention refusal is a defect only if a full book existed to fetch.

    GitHub dispatched scheduled run 33477912528 five hours after its 01:30 UTC
    slot, landing at 02:30 ET against a shut book, where `filter_chain`'s
    `bid > 0 & ask > 0` rule cut the chain to 32 rows against 1440 stored. The
    guard was right to refuse; the red run was the false alarm, and a guard
    that cries wolf on every late cron teaches the reader to ignore red runs.
    """

    LATE_CRON = dt.datetime(2026, 9, 1, 6, 30, 42, tzinfo=dt.timezone.utc)   # 02:30 ET Tue
    AFTER_CLOSE = dt.datetime(2026, 9, 1, 2, 16, 5, tzinfo=dt.timezone.utc)  # 22:16 ET Mon
    MIDSESSION = dt.datetime(2026, 9, 1, 17, 0, tzinfo=dt.timezone.utc)      # 13:00 ET Tue
    PRE_MARKET = dt.datetime(2026, 8, 31, 9, 45, tzinfo=dt.timezone.utc)     # 05:45 ET Mon
    WINTER_CRON = dt.datetime(2026, 1, 15, 2, 17, tzinfo=dt.timezone.utc)    # 21:17 EST Wed
    SATURDAY = dt.datetime(2026, 9, 5, 6, 30, tzinfo=dt.timezone.utc)        # 02:30 ET Sat

    def test_the_late_cron_that_actually_failed_is_not_expected_to_see_a_book(self):
        assert not run_daily.book_should_be_full(self.LATE_CRON)

    def test_the_post_close_window_the_cron_targets_is(self):
        assert run_daily.book_should_be_full(self.AFTER_CLOSE)

    def test_the_new_cron_slot_is_after_the_close_in_winter_too(self):
        # 02:17 UTC is 22:17 ET under EDT but 21:17 ET under EST; both must
        # count as post-close, or the guard would go quiet half the year.
        assert run_daily.book_should_be_full(self.WINTER_CRON)

    def test_a_shrink_during_the_session_is_never_excused(self):
        assert run_daily.book_should_be_full(self.MIDSESSION)

    def test_the_original_pre_market_defect_is_also_recognised_as_off_hours(self):
        # 05:45 ET, the incident MIN_CHAIN_RETENTION was written for.
        assert not run_daily.book_should_be_full(self.PRE_MARKET)

    def test_nothing_has_traded_since_friday_by_saturday_morning(self):
        assert not run_daily.book_should_be_full(self.SATURDAY)

    def test_the_guard_raises_a_type_the_caller_can_single_out(self):
        assert issubclass(run_daily.ChainRetentionRefusal, RuntimeError)

    def test_the_guard_actually_raises_that_type(self, tmp_path):
        storage.write_chain(flat_chain(20), TODAY, tmp_path)
        with pytest.raises(run_daily.ChainRetentionRefusal):
            run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)


class TestMainTriagesTheRefusal:
    """`main` is where a refusal becomes an exit code, so the branch lives here
    and not in `run`, which offline replay and every test call directly."""

    @staticmethod
    def _wire(monkeypatch, tmp_path, outcome, book_full):
        monkeypatch.setattr(run_daily, "_providers", lambda cfg, root: (None, None, None))
        monkeypatch.setattr(run_daily, "book_should_be_full", lambda now: book_full)

        def fake_run(eodhd, live, fallback, cfg, root, today=None):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(run_daily, "run", fake_run)
        out = tmp_path / "gh_output"
        out.write_text("")
        monkeypatch.setenv("GITHUB_OUTPUT", str(out))
        return out

    def test_an_off_hours_refusal_exits_clean_and_reports_nothing_published(
            self, monkeypatch, tmp_path, capsys):
        refusal = run_daily.ChainRetentionRefusal("32 rows vs 1440, below the 50% floor")
        out = self._wire(monkeypatch, tmp_path, refusal, book_full=False)
        run_daily.main()                      # must not raise
        assert "published=false" in out.read_text()
        captured = capsys.readouterr().out
        assert "::warning" in captured        # visible on the run, not silent
        assert "%25" in captured              # the literal % is escaped for GitHub

    def test_a_refusal_while_the_book_was_full_still_fails_loudly(
            self, monkeypatch, tmp_path):
        refusal = run_daily.ChainRetentionRefusal("32 rows vs 1440")
        self._wire(monkeypatch, tmp_path, refusal, book_full=True)
        with pytest.raises(run_daily.ChainRetentionRefusal):
            run_daily.main()

    def test_an_unrelated_failure_is_never_excused_by_the_clock(
            self, monkeypatch, tmp_path):
        self._wire(monkeypatch, tmp_path, RuntimeError("both sources empty"),
                   book_full=False)
        with pytest.raises(RuntimeError, match="both sources empty"):
            run_daily.main()

    def test_a_normal_run_reports_that_it_published(self, monkeypatch, tmp_path, capsys):
        out = self._wire(monkeypatch, tmp_path, {"rows_stored": 1440}, book_full=True)
        run_daily.main()
        assert "published=true" in out.read_text()
        assert "1440" in capsys.readouterr().out

    def test_no_github_output_file_is_not_a_crash(self, monkeypatch, tmp_path):
        self._wire(monkeypatch, tmp_path, {"rows_stored": 1440}, book_full=True)
        monkeypatch.delenv("GITHUB_OUTPUT")
        run_daily.main()          # running locally must still work
