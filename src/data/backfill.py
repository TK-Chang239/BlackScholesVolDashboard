"""One-time (resumable) historical backfill from Massive (SPEC 2.2).

Resumability = chain files already on disk are skipped, so an interrupted
run (rate limit, ctrl-C, crash) just continues on the next invocation.
Trading days come from EODHD's underlying history -- no exchange-calendar
dependency needed.
"""
import datetime as dt
from pathlib import Path

from src.data import storage
from src.data.filters import filter_chain


def backfill(massive, eodhd, cfg: dict, root: Path,
             start: dt.date, end: dt.date, log=print) -> dict:
    symbol = cfg["symbol"]
    underlying = eodhd.get_underlying_history(symbol, start=start - dt.timedelta(days=7))
    storage.upsert_underlying(underlying, root)
    by_date = underlying.set_index("date")["close"].to_dict()
    days = [d for d in sorted(by_date) if start <= d <= end]

    summary = {"dates_total": len(days), "dates_skipped": 0,
               "dates_written": 0, "dates_empty": 0, "rows_written": 0}
    for day in days:
        if storage.chain_exists(day, root):
            summary["dates_skipped"] += 1
            continue
        spot = float(by_date[day])
        raw = massive.get_historical_chain(symbol, day, spot, cfg)
        chain = filter_chain(raw, spot, day, cfg)
        if chain.empty:
            summary["dates_empty"] += 1
            log(f"{day}: empty after filtering — nothing stored")
            continue
        storage.write_chain(chain, day, root)
        summary["dates_written"] += 1
        summary["rows_written"] += len(chain)
        log(f"{day}: {len(chain)} rows")
    return summary
