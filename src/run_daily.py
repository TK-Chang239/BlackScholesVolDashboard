"""Daily orchestrator: fetch -> filter -> compute -> render -> store (SPEC 2.1/2.2).

The ONLY place vendors are composed. Failure policy: any unrecoverable
error raises, leaving no chain file, docs/index.html, or status.json
untouched, so the Action exits nonzero and yesterday's dashboard stays live.
A thin live book is handled before that policy applies: if the filtered live
chain falls far below what recent sessions hold, `run` discards it and fetches
the session from Massive instead, the same branch an empty frame already takes.
`_check_overwrite_shrink` then guards what survives -- it raises, before any
write, if the chain STILL holds far fewer rows than the archive says a session
should, so a degraded chain can never enter the archive, neither by overwriting
a good one nor by being the first thing ever stored for its session.

That refusal is the one failure `main` does not always report as one. Nothing
is written either way, but whether it means something is broken depends on
whether a book existed to fetch -- see `book_should_be_full`.
"""
import datetime as dt
import os
import statistics
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from src.analytics.annotations import COLUMNS as ANNOTATION_COLUMNS, load_annotations
from src.analytics.chain_iv import compute_chain_iv
from src.analytics.daily_metrics import (
    metric_columns, refresh_rv_columns, session_metrics_row, upsert_session,
)
from src.analytics.greeks_panel import compute_greeks_panel
from src.analytics.iv_rv import iv_rv_series, iv_rv_summary, iv_rv_summary_html
from src.analytics.model_vs_market import compute_model_vs_market
from src.analytics.parity import carry_reference, compute_parity, implied_forward, parity_summary
from src.analytics.sensitivity import compute_sensitivity
from src.analytics.skew import compute_skew_25d
from src.analytics.smile import compute_smile
from src.analytics.term_structure import compute_term_structure
from src.data import storage
from src.data.filters import filter_chain
from src.data.hedge_replay import replay_hedge_sim
from src.render.figures import (
    build_greeks_curves_figure, build_iv_rv_figure, build_model_vs_market_figure,
    build_parity_figure, build_sensitivity_figure, build_skew_figure,
    build_smile_figure, build_term_structure_figure,
)
from src.render.hedge_figures import (
    build_hedge_histogram_figure, build_hedge_pnl_figure, build_hedge_scatter_figure,
)
from src.render.page import render_page
from src.render.stats import hedge_summary_html, parity_summary_html
from src.render.tiles import greek_tiles_html

STALENESS_DAYS = 5
CARRY_MIN_DTE = 84       # below this a dollar of forward error is a huge "rate" error

# `data/chains/` is not a cache: P8's hedge replay and every panel that reads a
# prior session depend on it holding the best data this pipeline ever captured
# for that date. A freshly filtered chain that keeps fewer than half the rows
# of the one already on disk is treated as off-hours/degraded data (e.g. a
# pre-market run where almost nothing has a bid) rather than a legitimate
# update, and publishing it would silently corrupt the stored history instead
# of just one day's page. Half is a deliberately blunt line: ordinary
# day-to-day liquidity swings do not cut a chain's row count in half, but a
# closed book does.
MIN_CHAIN_RETENTION = 0.5

# What a first-ever write is measured against, since it has no stored version
# of itself to compare with. Five sessions is about a trading week: long enough
# that one odd day cannot drag the median, short enough to track the current
# expiry cycle rather than last quarter's. Below MIN_RECENT_SESSIONS there is
# no "normal" to speak of and the guard declines to judge -- otherwise it would
# refuse the archive's own opening sessions.
RECENT_SESSIONS = 5
MIN_RECENT_SESSIONS = 3

MARKET_TZ = ZoneInfo("America/New_York")

# The chain fetch is a LIVE snapshot labelled with the last completed session,
# so what it returns depends on when the job runs. `filter_chain` keeps only
# `bid > 0 & ask > 0`, and outside the hours a book exists almost nothing is
# quoted. From the regular open onward there is a book to fetch -- during the
# session it is live, and after the 16:00 close the day's quotes persist -- so
# a shrink from here on is a real defect. Before the open there is nothing to
# see and a shrink says only that the job ran early (or, as in run
# 33477912528, that GitHub dispatched the cron five hours late).
BOOK_FULL_FROM_ET = dt.time(9, 30)


class ChainRetentionRefusal(RuntimeError):
    """`_check_overwrite_shrink` declined to replace a stored chain.

    A distinct type so `main` can tell "the book was shut" apart from a
    genuine failure. It stays a `RuntimeError` because every caller other than
    `main` -- offline replay, the tests -- should keep treating it as one.
    """


def book_should_be_full(now_utc: dt.datetime) -> bool:
    """Was there a full book to fetch at `now_utc`, so a thin one is a defect?

    Weekends are excluded because nothing has traded since Friday's close.
    Market holidays are deliberately not modelled: a holiday evening still
    serves the previous session's quotes, and buying a market calendar to
    silence a once-a-year false alarm costs more than the alarm.
    """
    now_et = now_utc.astimezone(MARKET_TZ)
    if now_et.weekday() >= 5:
        return False
    return now_et.time() >= BOOK_FULL_FROM_ET


def _recent_chain_rows(session_date: dt.date, root: Path) -> list[int]:
    """Row counts of the most recent stored chains before `session_date`.

    Reads Parquet footers rather than the frames: this runs on every daily
    run and only the row count is wanted.
    """
    chains = Path(root) / "data" / "chains"
    if not chains.is_dir():
        return []
    earlier = []
    for path in chains.glob("*.parquet"):
        try:
            day = dt.date.fromisoformat(path.stem)
        except ValueError:
            continue                       # .tmp.parquet and anything hand-dropped
        if day < session_date:
            earlier.append((day, path))
    earlier.sort(reverse=True)
    return [pq.ParquetFile(p).metadata.num_rows for _, p in earlier[:RECENT_SESSIONS]]


def _thin_against_recent(chain: pd.DataFrame, session_date: dt.date,
                        root: Path) -> tuple[int, int] | None:
    """`(typical rows, sessions compared)` when `chain` sits far below what
    recent sessions hold; `None` when it does not, or when there is too little
    history to have an opinion.

    One definition of "far below normal", shared by the two callers that need
    it: `run`, which reaches for the other vendor, and `_check_overwrite_shrink`,
    which refuses when neither vendor could do better. Two copies of this
    threshold would drift apart.
    """
    recent = _recent_chain_rows(session_date, root)
    if len(recent) < MIN_RECENT_SESSIONS:
        return None
    typical = int(statistics.median(recent))
    if len(chain) < MIN_CHAIN_RETENTION * typical:
        return typical, len(recent)
    return None


def _check_overwrite_shrink(chain: pd.DataFrame, session_date: dt.date, root: Path) -> None:
    """Refuse to store a chain far smaller than the archive says it should be.

    Runs in the compute stage, before any write (SPEC 2.1): raising here
    leaves the stored chain, daily_metrics, page and status all untouched, so
    yesterday's dashboard stays live.

    Two reference points, because a session has a stored version of itself
    only on a re-run. When one exists it is the comparison. When none does --
    the case for essentially every scheduled run, which writes its session for
    the first time -- recent sessions stand in for it. Without that second
    arm the guard was inert exactly where it was needed: runs 33601964465 and
    its successor each stored a 29-row chain beside ~1,440-row neighbours and
    neither was refused.

    Raises `ChainRetentionRefusal` rather than a bare `RuntimeError` so `main`
    can decide whether it deserves a non-zero exit; every other caller can go
    on treating it as the `RuntimeError` it still is.
    """
    new_rows = len(chain)
    if not storage.chain_exists(session_date, root):
        thin = _thin_against_recent(chain, session_date, root)
        if thin is None:
            return
        typical, sessions = thin
        raise ChainRetentionRefusal(
            f"refusing to store a first chain for session "
            f"{session_date.isoformat()}: the freshly filtered chain has "
            f"{new_rows} rows against a typical {typical} across the "
            f"{sessions} most recent stored sessions, below the "
            f"{MIN_CHAIN_RETENTION:.0%} floor. `run` already tried the Massive "
            "fallback and it came back no better, so this is not merely a thin "
            "live book. Nothing has been stored. If a chain this thin is "
            "genuinely what you want kept, store it deliberately with "
            "scripts/backfill.py."
        )
    stored_rows = len(pd.read_parquet(storage.chain_path(session_date, root)))
    if new_rows < MIN_CHAIN_RETENTION * stored_rows:
        raise ChainRetentionRefusal(
            f"refusing to overwrite the stored chain for session "
            f"{session_date.isoformat()}: the freshly filtered chain has "
            f"{new_rows} rows vs {stored_rows} currently stored in "
            f"data/chains/{session_date.isoformat()}.parquet, below the "
            f"{MIN_CHAIN_RETENTION:.0%} retention floor. This usually means "
            "the pipeline ran off-hours against a thin book. Re-run after the "
            "session closes so a fuller chain is captured, or if this smaller "
            "chain is genuinely what you want kept, delete the stored file "
            "first to accept the loss deliberately."
        )


def _carry_statistic(carry: dict) -> str | None:
    """One sentence naming which statistic `implied_carry` is.

    Without it `status.json` publishes a number and five keys around it and
    leaves the reader to infer whether the middle one is a mean, a median or
    a single reading.
    """
    if not carry["n_expiries"]:
        return None
    if carry["dte_min"] >= CARRY_MIN_DTE:
        return (f"median implied carry over the {carry['n_expiries']} expiries "
                f"with dte >= {CARRY_MIN_DTE}")
    return (f"implied carry at the longest available expiry ({carry['dte_max']:.0f}d); "
            f"no expiry reaches {CARRY_MIN_DTE}d")


def _round_or_none(value: float, digits: int = 6) -> float | None:
    return round(float(value), digits) if np.isfinite(value) else None


def _int_or_none(value: float) -> int | None:
    return int(value) if np.isfinite(value) else None


def run(eodhd, live, fallback, cfg: dict, root: Path, today: dt.date | None = None) -> dict:
    """Run one daily snapshot.

    `snapshot_date` (used for the chain fetch, filtering, storage path, and
    status.json's "snapshot_date") is the market session date, derived from
    the freshly-fetched underlying history -- never wall-clock `today`. This
    matters off-hours and under a cron that runs before the session it is
    labeling has fully rolled over: labeling with wall-clock `today` would
    tag a chain with a date the market never actually traded on.

    `today` (wall clock, defaulting to `dt.date.today()`) is used only for
    the staleness guard and `last_success_utc`.
    """
    today = today or dt.date.today()
    symbol = cfg["symbol"]

    underlying = eodhd.get_underlying_history(symbol)
    storage.upsert_underlying(underlying, root)
    sorted_underlying = underlying.sort_values("date")
    session_date = sorted_underlying["date"].iloc[-1]
    if (today - session_date).days > STALENESS_DAYS:
        raise RuntimeError(
            f"underlying data is stale: latest session {session_date.isoformat()} "
            f"is more than {STALENESS_DAYS} days before today {today.isoformat()}"
        )
    spot = float(sorted_underlying["close"].iloc[-1])

    chain = None
    source = None
    try:
        raw = live.get_option_chain(symbol, session_date, spot, cfg)
        chain = filter_chain(raw, spot, session_date, cfg)
        source = "yfinance"
    except Exception as live_err:
        print(f"live chain failed ({live_err!r}); falling back to Massive", file=sys.stderr)

    # A thin live book is not a reason to publish nothing, it is a reason to
    # ask the other vendor -- exactly as an empty one already is. GitHub has
    # dispatched this cron ~5h late on every observed night, landing near
    # 03:00 ET where nothing is quoted; without this the guard refuses and the
    # page simply stops advancing, trading a wrong dashboard for a frozen one.
    # Massive serves the closed session, so the run publishes real rows at any
    # hour. Discarding the thin frame here lets the existing branch below do
    # the fetch.
    if chain is not None and not chain.empty:
        thin = _thin_against_recent(chain, session_date, root)
        if thin:
            typical, sessions = thin
            print(f"live filtered chain is thin ({len(chain)} rows against a typical "
                  f"{typical} across {sessions} recent sessions); falling back to Massive",
                  file=sys.stderr)
            chain = None

    if chain is None or chain.empty:
        if chain is not None:
            print("live filtered chain is empty; falling back to Massive", file=sys.stderr)
        raw = fallback.get_option_chain(symbol, session_date, spot, cfg)
        chain = filter_chain(raw, spot, session_date, cfg)
        source = "massive-fallback"

    if chain.empty:
        raise RuntimeError("both sources produced an empty filtered chain")

    _check_overwrite_shrink(chain, session_date, root)

    # Get all status values from providers BEFORE any writes (failure policy: no partial writes).
    risk_free_rate = eodhd.get_risk_free_rate()
    dividend_yield = eodhd.get_dividend_yield(spot, today=today, symbol=symbol)

    # ---- compute stage: everything derived before anything is written ----
    chain_iv, iv_stats = compute_chain_iv(chain, risk_free_rate, dividend_yield)
    smile = compute_smile(chain_iv, cfg)
    ts_today = compute_term_structure(chain_iv)

    ts_prev = None
    prev_label = "previous session"
    prior_dates = sorted(
        d for d in sorted_underlying["date"] if d < session_date
        and storage.chain_exists(d, root)
    )
    if prior_dates:
        prev_chain = pd.read_parquet(storage.chain_path(prior_dates[-1], root))
        prev_iv, _ = compute_chain_iv(prev_chain, risk_free_rate, dividend_yield)
        ts_prev = compute_term_structure(prev_iv)
        prev_label = f"prev session {prior_dates[-1].isoformat()}"
        if not (prev_chain["source"] == "yfinance").all():
            prev_label += " (close-based)"

    # ---- P4 Greeks (today + previous session for the 1-day change) ----
    tiles, curves = compute_greeks_panel(chain_iv, risk_free_rate, dividend_yield, cfg)
    tiles_prev = None
    if prior_dates:
        tiles_prev, _ = compute_greeks_panel(prev_iv, risk_free_rate, dividend_yield, cfg)

    # ---- P6 skew (into this session's daily_metrics row) ----
    skew = compute_skew_25d(chain_iv, risk_free_rate, dividend_yield, cfg)

    # ---- daily_metrics: this session's row + RV refresh over the whole history ----
    metrics = storage.read_daily_metrics(root, metric_columns(cfg))
    metrics = upsert_session(
        metrics, session_metrics_row(session_date, spot, source, ts_today, iv_stats, cfg,
                                     skew=skew), cfg)
    metrics = refresh_rv_columns(metrics, storage.read_underlying(root), cfg)
    this_row = metrics[metrics["date"] == session_date].iloc[0]
    atm_iv_30d = float(this_row["atm_iv_30d"])
    atm_dte = float(this_row["atm_iv_30d_dte"])

    # ---- P1 sensitivity: prefer the 30-DTE ATM IV; else the chain's median IV ----
    sigma_p1 = atm_iv_30d if np.isfinite(atm_iv_30d) else float(np.nanmedian(chain_iv["iv"])) \
        if chain_iv["iv"].notna().any() else float("nan")
    dte_p1 = atm_dte if np.isfinite(atm_dte) else float(cfg["target_dte"]["atm_panel"])
    sens = compute_sensitivity(spot, sigma_p1, dte_p1, risk_free_rate, dividend_yield,
                               cfg["sensitivity"]["bump"])

    # ---- P5 ----
    series = iv_rv_series(metrics, cfg)
    summary = iv_rv_summary(series)

    # ---- P7 parity + implied carry ----
    parity = compute_parity(chain_iv, risk_free_rate, dividend_yield)
    psum = parity_summary(parity)
    # Report carry from the long expiries: a fixed dollar of forward error
    # divided by a three-week T looks like a huge rate error and says nothing.
    # And report their MEDIAN with its range, not the single longest reading.
    forwards = implied_forward(parity, spot, risk_free_rate)
    carry = carry_reference(forwards, min_dte=CARRY_MIN_DTE)

    # ---- P9 model vs market at the same flat vol P1 uses ----
    mvm = compute_model_vs_market(chain_iv, sigma_p1, risk_free_rate, dividend_yield)

    # ---- P8 hedge simulation: a full stateless replay of the stored archive.
    # Today's chain is not on disk yet (SPEC 2.1: compute before write), so it is
    # handed in directly -- otherwise the current session would be invisible to
    # the sim until tomorrow.
    hedge = replay_hedge_sim(root, risk_free_rate, dividend_yield, cfg,
                             extra_chains={session_date: chain_iv},
                             underlying=sorted_underlying)
    hsum = hedge["summary"]

    # P6 annotations are decoration and the file is designed to be hand-edited
    # ("adding a news note is a 1-line commit"), so a typo in it is the EXPECTED
    # failure -- and this call sits after the whole compute stage but before any
    # write. Letting it raise costs the session's chain parquet, metrics row,
    # page and status for a missing colon, and a session's quotes cannot be
    # re-fetched later. `load_annotations` keeps raising, so CI and
    # `test_shipped_file_loads` still catch a bad file that was committed.
    annotations_path = Path(root) / "data" / "annotations.yaml"
    try:
        annotations = load_annotations(annotations_path)
    except Exception as ann_err:
        print(f"annotations file {annotations_path} failed to load ({ann_err!r}); "
              "rendering P6 without notes", file=sys.stderr)
        annotations = pd.DataFrame(columns=ANNOTATION_COLUMNS)

    figures = {
        "P1": build_sensitivity_figure(sens, cfg["sensitivity"]["bump"]),
        "P2": build_smile_figure(smile, spot),
        "P3": build_term_structure_figure(ts_today, ts_prev, previous_label=prev_label),
        "P4": build_greeks_curves_figure(curves, spot),
        "P5": build_iv_rv_figure(series, summary, cfg),
        "P6": build_skew_figure(metrics, annotations),
        "P7": build_parity_figure(parity, spot),
        "P8a": build_hedge_pnl_figure(hedge["portfolio"], hedge["daily"]),
        "P8b": build_hedge_scatter_figure(hedge["trades"], hedge["fit"],
                                          next_settlement=hsum["next_settlement"]),
        "P8c": build_hedge_histogram_figure(hedge["daily"]),
        "P9": build_model_vs_market_figure(mvm, sigma_p1),
    }
    extras = {"P4": greek_tiles_html(tiles, tiles_prev, prev_label),
              "P5": iv_rv_summary_html(summary),
              "P7": parity_summary_html(psum, carry, risk_free_rate, dividend_yield),
              "P8a": hedge_summary_html(hsum)}

    status = {
        "last_success_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": session_date.isoformat(),
        "source": source,
        "rows_stored": int(len(chain)),
        "spot": spot,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "iv_convergence": round(iv_stats["convergence"], 4),
        "atm_iv_30d": round(atm_iv_30d, 6) if np.isfinite(atm_iv_30d) else None,
        "daily_metrics_rows": int(len(metrics)),
        "history_since": (metrics["date"].iloc[0].isoformat() if len(metrics) else None),
        "skew_25d": (round(float(skew["skew_25d"]), 6)
                     if np.isfinite(skew["skew_25d"]) else None),
        "parity_pairs": psum["n_pairs"],
        "parity_liquid_pairs": psum["n_liquid"],
        "parity_tradeable_violations": psum["n_tradeable_violations"],
        "parity_tradeable_violations_fwd": psum["n_tradeable_violations_fwd"],
        "parity_violations_early_exercise": psum["n_violations_early_exercise"],
        "parity_violations_unexplained": psum["n_violations_unexplained"],
        # `implied_carry` is the MEDIAN over the expiries at or beyond 84 DTE;
        # the four keys beside it say which expiries and how far apart they were,
        # so the file describes its own summary statistic rather than presenting
        # one reading as the answer.
        "implied_carry": _round_or_none(carry["implied_carry"]),
        "implied_carry_lo": _round_or_none(carry["carry_lo"]),
        "implied_carry_hi": _round_or_none(carry["carry_hi"]),
        "implied_carry_dte_min": _int_or_none(carry["dte_min"]),
        "implied_carry_dte_max": _int_or_none(carry["dte_max"]),
        "implied_carry_expiries": int(carry["n_expiries"]),
        "implied_carry_statistic": _carry_statistic(carry),
        "hedge_trades": hsum["n_trades"],
        # `settled` means "reached expiry AND plottable"; `sparse` trades reached
        # expiry too. Publishing only the first told a reader of status.json that
        # nothing had finished while a finished trade sat in `hedge_cum_pnl`.
        "hedge_trades_settled": hsum["n_settled"],
        "hedge_trades_reached_expiry": hsum["n_reached_expiry"],
        "hedge_trades_open": hsum["n_open"],
        "hedge_trades_sparse": hsum["n_sparse"],
        "hedge_months_skipped": hsum["n_months_skipped"],
        "hedge_sessions": hsum["n_days"],
        # The entry rule takes the monthly expiry NEAREST the target, which from
        # the first session of a month is rarely near it. Publish the tenor the
        # sim actually traded rather than leaving `entry_dte: 30` to imply it.
        "hedge_dte_at_entry_min": hsum["dte_at_entry_min"],
        "hedge_dte_at_entry_max": hsum["dte_at_entry_max"],
        "hedge_cum_pnl": _round_or_none(hsum["cum_pnl"], 4),
        # Two shares, because they answer different questions. The first counts
        # every session, so a reader sees how much of the P&L is our own model at
        # all. The second counts only sessions a stored chain could have quoted:
        # `chain_filter.dte_min` keeps a trade's own expiry out of every chain
        # over its final days, so those sessions are unquotable by construction
        # and cannot be evidence of thin coverage. The second is what decides
        # `settled` vs `sparse`; publishing only the first made the split
        # invisible and the threshold look arbitrary.
        "hedge_market_mark_share": _round_or_none(hsum["market_mark_share"], 4),
        "hedge_quotable_mark_share": _round_or_none(hsum["quotable_mark_share"], 4),
        "hedge_model_marks": hsum["n_model_marks"],
        "hedge_model_marks_structural": hsum["n_structural_marks"],
        "hedge_model_marks_gap": hsum["n_gap_marks"],
        "hedge_slope_per_vol_point": _round_or_none(hsum["slope"], 4),
        "hedge_r2": _round_or_none(hsum["r2"], 4),
        "hedge_next_settlement": (hsum["next_settlement"].isoformat()
                                  if hsum["next_settlement"] is not None else None),
        "panels_rendered": sorted(figures),
    }
    html = render_page(figures, status, extras=extras)

    # ---- write stage: chain -> daily_metrics -> page -> status, each atomic ----
    storage.write_chain(chain, session_date, root)
    storage.write_daily_metrics(metrics, root)
    storage.write_text(html, Path(root) / "docs" / "index.html")
    storage.write_status(status, root)
    return status


def _providers(cfg: dict, root: Path):
    """The real vendor trio, split out so `main`'s triage below can be tested
    without credentials or a network."""
    from src.data.envfile import get_secret
    from src.data.eodhd_provider import EODHDProvider
    from src.data.massive_provider import MassiveProvider
    from src.data.yfinance_provider import YFinanceProvider

    return (EODHDProvider(get_secret("EODHD_API_TOKEN", root / ".env"), cfg),
            YFinanceProvider(cfg),
            MassiveProvider(get_secret("MASSIVE_API_KEY", root / ".env"), cfg))


def _github_output(key: str, value: str) -> None:
    """Publish a step output under Actions; a no-op anywhere else."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        f.write(f"{key}={value}\n")


def _annotation(message: str) -> str:
    """Escape a message for a `::warning::` workflow command. The refusal text
    quotes a literal `%` (the retention floor), which GitHub would eat."""
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main() -> None:
    import json

    root = Path(__file__).resolve().parent.parent
    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    eodhd, live, fallback = _providers(cfg, root)

    try:
        status = run(eodhd, live, fallback, cfg, root)
    except ChainRetentionRefusal as refusal:
        # During the hours a book exists a refusal is a real defect and must
        # stay red. Outside them it says only that the job ran when there was
        # nothing to fetch: the stored chain is untouched and yesterday's
        # dashboard is still live, so failing here would report a problem that
        # does not exist and teach the reader to ignore red runs.
        if book_should_be_full(dt.datetime.now(dt.timezone.utc)):
            raise
        print(f"::warning title=Daily snapshot skipped::{_annotation(str(refusal))}")
        print(f"skipped: ran outside market hours, nothing published; {refusal}",
              file=sys.stderr)
        _github_output("published", "false")
        return

    _github_output("published", "true")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
