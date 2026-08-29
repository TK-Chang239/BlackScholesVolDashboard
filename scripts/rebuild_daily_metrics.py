"""Rebuild data/daily_metrics.parquet from stored chains. No network.

r and q come from docs/status.json (the last real run) when present, else
config.yaml fallbacks. Usage: .venv/bin/python scripts/rebuild_daily_metrics.py
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.metrics_backfill import rebuild_daily_metrics  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    r, q = cfg["rates"]["risk_free_fallback"], cfg["rates"]["dividend_yield_fallback"]
    status_path = root / "docs" / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        r, q = status.get("risk_free_rate", r), status.get("dividend_yield", q)
    out = rebuild_daily_metrics(root, r, q, cfg)
    print(f"rebuilt {len(out)} rows -> data/daily_metrics.parquet (r={r:.4%}, q={q:.4%})")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
