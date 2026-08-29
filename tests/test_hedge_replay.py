"""P8 replay driver: stored parquet in, simulation out. Offline."""
import datetime as dt

import numpy as np
import pandas as pd

from src.data.hedge_replay import replay_hedge_sim

CFG = {
    "symbol": "SPY",
    "chain_filter": {"dte_min": 7, "dte_max": 365},
    "hedge_sim": {"structure": "short_straddle", "entry_dte": 30,
                  "hedge_frequency": "daily", "transaction_cost_bps": 0},
}


def write_archive(root, dates, expiry, spot=100.0):
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
    u = pd.DataFrame({"date": dates, "close": [spot] * len(dates),
                      "adjusted_close": [spot] * len(dates), "volume": [1.0] * len(dates)})
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
