"""P8 replay driver: stored parquet in, simulation out. Offline."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.data.hedge_replay import replay_hedge_sim

CFG = {
    "symbol": "SPY",
    "chain_filter": {"dte_min": 7, "dte_max": 365},
    "hedge_sim": {"structure": "short_straddle", "entry_dte": 30,
                  "hedge_frequency": "daily", "transaction_cost_bps": 0},
}


def write_archive(root, dates, expiry, spot=100.0, underlying_dates=None):
    """`underlying_dates` defaults to `dates`; pass it when the close series has
    to run past the last STORED chain (chain_filter.dte_min truncates it)."""
    (root / "data" / "chains").mkdir(parents=True)
    for d in dates:
        rows = []
        for strike in (95.0, 100.0, 105.0):
            for kind in ("call", "put"):
                intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
                price = intrinsic + 3.0
                rows.append({
                    "snapshot_date": d, "spot": spot, "expiry": expiry,
                    "dte": (expiry - d).days, "strike": strike, "kind": kind,
                    "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                    "close": price, "volume": 10.0, "open_interest": 100.0,
                    "vendor_iv": np.nan, "source": "massive-backfill",
                })
        pd.DataFrame(rows).to_parquet(root / "data" / "chains" / f"{d.isoformat()}.parquet")
    u_dates = list(dates if underlying_dates is None else underlying_dates)
    u = pd.DataFrame({"date": u_dates, "close": [spot] * len(u_dates),
                      "adjusted_close": [spot] * len(u_dates),
                      "volume": [1.0] * len(u_dates)})
    u.to_parquet(root / "data" / "underlying.parquet")


def test_replays_the_archive_and_returns_every_key(tmp_path):
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(6)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 7, 1))
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    assert set(out) == {"trades", "daily", "portfolio", "fit", "summary"}
    assert len(out["trades"]) == 1
    assert out["trades"]["entry_date"].iloc[0] == dates[0]
    assert out["summary"]["n_open"] == 1
    assert len(out["portfolio"]) == 6
    # the keys the page and status.json read, all the way through the driver
    s = out["summary"]
    assert s["n_reached_expiry"] == 0 and s["n_months_skipped"] == 0
    assert s["dte_at_entry_min"] == s["dte_at_entry_max"] == 30
    assert s["cost_bps"] == 0.0    # CFG's transaction_cost_bps, threaded through


def test_the_configured_transaction_cost_reaches_the_summary(tmp_path):
    # M3: `_sim_label` (src/render/stats.py) states "no spread is charged" only
    # when the config actually says so; that means this driver has to carry
    # cfg["hedge_sim"]["transaction_cost_bps"] into the summary rather than
    # leaving the render layer to import config for itself.
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(6)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 7, 1))
    cfg = {**CFG, "hedge_sim": {**CFG["hedge_sim"], "transaction_cost_bps": 5}}
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, cfg)
    assert out["summary"]["cost_bps"] == 5.0


def test_a_month_that_cannot_seed_a_trade_is_counted_in_the_summary(tmp_path):
    # F11: the only expiry in this archive is 61 days out, past
    # MAX_ENTRY_DTE_ERROR, so the month is rejected. That has to be visible --
    # on the real archive June 2026 goes exactly this way.
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(4)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 8, 1))
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    assert out["trades"].empty
    assert out["summary"]["n_months_skipped"] == 1


def test_a_chain_session_the_underlying_does_not_cover_is_skipped_not_raised(tmp_path):
    # F10: `closes[entry_date]` was unguarded, so one absent close raised a
    # KeyError out of the whole daily run.
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(6)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 7, 1))
    u = pd.read_parquet(tmp_path / "data" / "underlying.parquet")
    u[u["date"] != dates[0]].to_parquet(tmp_path / "data" / "underlying.parquet")
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    assert out["trades"].empty
    assert out["summary"]["n_months_skipped"] == 1


def test_ignores_non_date_filenames(tmp_path):
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(4)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 7, 1))
    (tmp_path / "data" / "chains" / "notes.parquet").write_bytes(b"not a parquet")
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    assert len(out["trades"]) == 1


def test_extra_chains_are_visible_before_the_file_is_written(tmp_path):
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(4)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 7, 1))
    today = dates[-1] + dt.timedelta(days=1)
    stored = pd.read_parquet(tmp_path / "data" / "chains" / f"{dates[-1].isoformat()}.parquet")
    fresh = stored.assign(snapshot_date=today, dte=lambda d: (dt.date(2026, 7, 1) - today).days)
    from src.analytics.chain_iv import compute_chain_iv
    solved, _ = compute_chain_iv(fresh, 0.04, 0.013)
    u = pd.read_parquet(tmp_path / "data" / "underlying.parquet")
    u = pd.concat([u, u.tail(1).assign(date=today)], ignore_index=True)
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG,
                           extra_chains={today: solved}, underlying=u)
    assert out["portfolio"]["date"].max() == today


def test_extra_chains_override_disk_for_same_date(tmp_path):
    """Verify extra_chains takes precedence over disk for the same date.

    This tests the SPEC 2.1 failure policy: today's chain is computed before
    being written to disk, so if a same-dated file already exists, the injected
    version must override it. The test passes an already-stored date with
    modified option prices (higher) and verifies the simulation uses the
    injected version (higher entry price) not the disk version.
    """
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(4)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 7, 1), spot=100.0)

    # Read the first date's chain from disk
    first_date = dates[0]
    stored = pd.read_parquet(tmp_path / "data" / "chains" / f"{first_date.isoformat()}.parquet")

    # Modify it: increase option prices (straddle will cost more to enter)
    modified = stored.assign(
        close=lambda d: d["close"] + 5.0,
        bid=lambda d: d["bid"] + 5.0,
        ask=lambda d: d["ask"] + 5.0,
        mid=lambda d: d["mid"] + 5.0,
    )

    from src.analytics.chain_iv import compute_chain_iv
    solved_modified, _ = compute_chain_iv(modified, 0.04, 0.013)

    # Run with disk version (no extra_chains)
    out_disk = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)

    # Run with injected version that has higher option prices
    out_injected = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG,
                                    extra_chains={first_date: solved_modified})

    # Both should have the same number of trades (one entry per valid session)
    assert len(out_disk["trades"]) == len(out_injected["trades"]) == 1

    # Entry straddle cost should be higher in injected version (higher option prices)
    entry_straddle_injected = out_injected["trades"]["entry_straddle"].iloc[0]
    entry_straddle_disk = out_disk["trades"]["entry_straddle"].iloc[0]
    assert entry_straddle_injected > entry_straddle_disk, \
        f"extra_chains version ({entry_straddle_injected}) should have higher " \
        f"entry_straddle than disk version ({entry_straddle_disk})"


def test_empty_archive_is_not_an_error(tmp_path):
    (tmp_path / "data" / "chains").mkdir(parents=True)
    pd.DataFrame({"date": [], "close": [], "adjusted_close": [], "volume": []}) \
        .to_parquet(tmp_path / "data" / "underlying.parquet")
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    assert out["trades"].empty
    assert out["summary"]["n_trades"] == 0


def test_a_trade_unquoted_only_inside_dte_min_settles_through_the_driver(tmp_path):
    """The defect a real run exposed, end to end through the replay driver.

    `chain_filter.dte_min = 7` keeps a 16-day trade's own expiry out of every
    stored chain over its final week, so at best 10 of its 16 sessions can carry
    a market mark -- 0.625, permanently under MIN_MARKET_MARK_SHARE. Judged over
    the sessions a chain COULD have quoted it is 10/10, and it belongs on the
    scatter. On the real archive this is the 2026-07-01 -> 2026-07-17 trade.
    """
    from src.analytics.hedge_sim import MIN_MARKET_MARK_SHARE

    dte_min = CFG["chain_filter"]["dte_min"]
    entry = dt.date(2026, 6, 1)
    expiry = entry + dt.timedelta(days=16)
    sessions = [entry + dt.timedelta(days=i) for i in range(17)]
    stored = [d for d in sessions if (expiry - d).days >= dte_min]
    write_archive(tmp_path, stored, expiry=expiry, underlying_dates=sessions)

    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    trade = out["trades"].iloc[0]
    assert trade["n_model_marks"] == trade["n_structural_marks"] == 6
    assert trade["market_mark_share"] < MIN_MARKET_MARK_SHARE     # the old, blended rule
    assert trade["quotable_mark_share"] == 1.0
    assert trade["status"] == "settled"
    s = out["summary"]
    assert s["n_settled"] == 1 and s["n_sparse"] == 0
    assert s["dte_min"] == dte_min          # cfg["chain_filter"]["dte_min"], threaded through
    assert s["n_structural_marks"] == 6 and s["n_gap_marks"] == 0


def test_a_trade_the_archive_never_covers_stays_sparse_through_the_driver(tmp_path):
    # The discrimination the threshold was written for: same tenor, but the
    # archive holds only the entry session, so nine quotable days are simply
    # missing. On the real archive this is the 2024-09-03 trade.
    entry = dt.date(2026, 6, 1)
    expiry = entry + dt.timedelta(days=16)
    sessions = [entry + dt.timedelta(days=i) for i in range(17)]
    write_archive(tmp_path, [entry], expiry=expiry, underlying_dates=sessions)

    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    trade = out["trades"].iloc[0]
    assert trade["n_structural_marks"] == 6
    assert trade["quotable_mark_share"] == pytest.approx(0.1)
    assert trade["status"] == "sparse"
    assert out["summary"]["n_sparse"] == 1 and out["summary"]["n_settled"] == 0
