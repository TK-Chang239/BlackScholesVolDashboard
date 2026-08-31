"""Daily orchestrator: fetch -> filter -> compute -> render -> store (SPEC 2.1/2.2).

The ONLY place vendors are composed. Failure policy: any unrecoverable
error raises, leaving no chain file, docs/index.html, or status.json
untouched, so the Action exits nonzero and yesterday's dashboard stays live.
`_check_overwrite_shrink` is part of that same policy: it raises, before any
write, if a freshly filtered chain would replace an already-stored one with
far fewer rows (e.g. an off-hours run against a thin book), so a degraded
chain can never silently overwrite a good one in the archive.
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


def _check_overwrite_shrink(chain: pd.DataFrame, session_date: dt.date, root: Path) -> None:
    """Refuse to replace a stored chain with a drastically smaller one.

    Runs in the compute stage, before any write (SPEC 2.1): raising here
    leaves the stored chain, daily_metrics, page and status all untouched, so
    the job exits non-zero and yesterday's dashboard stays live. Only applies
    when a chain for `session_date` is already on disk -- a first-ever write
    for a session has nothing to compare against and must proceed.
    """
    if not storage.chain_exists(session_date, root):
        return
    stored_rows = len(pd.read_parquet(storage.chain_path(session_date, root)))
    new_rows = len(chain)
    if new_rows < MIN_CHAIN_RETENTION * stored_rows:
        raise RuntimeError(
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


def main() -> None:
    from src.data.envfile import get_secret
    from src.data.eodhd_provider import EODHDProvider
    from src.data.massive_provider import MassiveProvider
    from src.data.yfinance_provider import YFinanceProvider

    root = Path(__file__).resolve().parent.parent
    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    eodhd = EODHDProvider(get_secret("EODHD_API_TOKEN", root / ".env"), cfg)
    live = YFinanceProvider(cfg)
    fallback = MassiveProvider(get_secret("MASSIVE_API_KEY", root / ".env"), cfg)
    status = run(eodhd, live, fallback, cfg, root)
    import json
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
