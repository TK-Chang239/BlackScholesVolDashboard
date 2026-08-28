"""Daily orchestrator: fetch -> filter -> compute -> render -> store (SPEC 2.1/2.2).

The ONLY place vendors are composed. Failure policy: any unrecoverable
error raises, leaving no chain file, docs/index.html, or status.json
untouched, so the Action exits nonzero and yesterday's dashboard stays live.
"""
import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.smile import compute_smile
from src.analytics.term_structure import compute_term_structure
from src.data import storage
from src.data.filters import filter_chain
from src.render.figures import build_smile_figure, build_term_structure_figure
from src.render.page import render_page

STALENESS_DAYS = 5


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

    # Get all status values from providers BEFORE any writes (failure policy: no partial writes).
    risk_free_rate = eodhd.get_risk_free_rate()
    dividend_yield = eodhd.get_dividend_yield(spot, today=today, symbol=symbol)

    # ---- compute stage: everything derived before anything is written ----
    chain_iv, iv_stats = compute_chain_iv(chain, risk_free_rate, dividend_yield)
    smile = compute_smile(chain_iv, cfg)
    ts_today = compute_term_structure(chain_iv)

    ts_prev = None
    prior_dates = sorted(
        d for d in sorted_underlying["date"] if d < session_date
        and storage.chain_exists(d, root)
    )
    if prior_dates:
        prev_chain = pd.read_parquet(storage.chain_path(prior_dates[-1], root))
        prev_iv, _ = compute_chain_iv(prev_chain, risk_free_rate, dividend_yield)
        ts_prev = compute_term_structure(prev_iv)

    figures = {"P2": build_smile_figure(smile, spot),
               "P3": build_term_structure_figure(ts_today, ts_prev)}

    status = {
        "last_success_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": session_date.isoformat(),
        "source": source,
        "rows_stored": int(len(chain)),
        "spot": spot,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "iv_convergence": round(iv_stats["convergence"], 4),
        "panels_rendered": ["P2", "P3"],
    }
    html = render_page(figures, status)

    # ---- write stage: chain -> page -> status, each atomic ----
    storage.write_chain(chain, session_date, root)
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
