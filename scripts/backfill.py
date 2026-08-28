"""CLI for the historical backfill. Usage:
    python scripts/backfill.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
Defaults: end = yesterday, start = end - 730 days. Re-run to resume.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.backfill import backfill
from src.data.envfile import get_secret
from src.data.eodhd_provider import EODHDProvider
from src.data.massive_provider import MassiveProvider


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=dt.date.fromisoformat)
    ap.add_argument("--end", type=dt.date.fromisoformat)
    args = ap.parse_args()
    end = args.end or dt.date.today() - dt.timedelta(days=1)
    start = args.start or end - dt.timedelta(days=730)

    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    eodhd = EODHDProvider(get_secret("EODHD_API_TOKEN", root / ".env"), cfg)
    massive = MassiveProvider(get_secret("MASSIVE_API_KEY", root / ".env"), cfg)
    summary = backfill(massive, eodhd, cfg, root, start, end)
    print(summary)


if __name__ == "__main__":
    main()
