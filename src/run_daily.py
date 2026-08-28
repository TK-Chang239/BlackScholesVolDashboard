"""Daily orchestrator: fetch -> filter -> store -> status (SPEC 2.1/2.2).

The ONLY place vendors are composed. Failure policy: any unrecoverable
error raises, leaving no chain file and status.json untouched, so the
Action exits nonzero and yesterday's dashboard stays live.
"""
import datetime as dt
import sys
from pathlib import Path

import yaml

from src.data import storage
from src.data.filters import filter_chain


def run(eodhd, live, fallback, cfg: dict, root: Path, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    symbol = cfg["symbol"]

    underlying = eodhd.get_underlying_history(symbol)
    storage.upsert_underlying(underlying, root)
    spot = float(underlying.sort_values("date")["close"].iloc[-1])

    try:
        raw = live.get_option_chain(symbol, today, spot, cfg)
        source = "yfinance"
    except Exception as live_err:
        print(f"live chain failed ({live_err!r}); falling back to Massive", file=sys.stderr)
        raw = fallback.get_option_chain(symbol, today, spot, cfg)
        source = "massive-fallback"

    chain = filter_chain(raw, spot, today, cfg)
    if chain.empty:
        raise RuntimeError(f"filtered chain is empty (source={source}) — refusing to store")
    storage.write_chain(chain, today, root)

    status = {
        "last_success_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": today.isoformat(),
        "source": source,
        "rows_stored": int(len(chain)),
        "spot": spot,
        "risk_free_rate": eodhd.get_risk_free_rate(),
        "dividend_yield": eodhd.get_dividend_yield(spot, today=today, symbol=symbol),
    }
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
