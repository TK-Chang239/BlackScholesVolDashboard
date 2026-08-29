"""Rebuild data/daily_metrics.parquet by replaying every stored chain.

Stateless by design (same philosophy as SPEC 3 P8): the metrics file is a
pure function of data/chains/ + data/underlying.parquet + (r, q, cfg), so
any analytics fix heals the whole history on the next rebuild. r and q
are the caller's current values -- a documented approximation.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.daily_metrics import (
    metric_columns, refresh_rv_columns, session_metrics_row, upsert_session,
)
from src.analytics.term_structure import compute_term_structure
from src.data import storage


def chain_source_label(chain: pd.DataFrame) -> str:
    if (chain["source"] == "yfinance").all():
        return "yfinance"
    return str(chain["source"].mode().iloc[0])


def rebuild_daily_metrics(root: Path, r: float, q: float, cfg: dict,
                          write: bool = True) -> pd.DataFrame:
    root = Path(root)
    metrics = pd.DataFrame(columns=metric_columns(cfg))
    for path in sorted((root / "data" / "chains").glob("*.parquet")):
        session = dt.date.fromisoformat(path.stem)
        chain = pd.read_parquet(path)
        chain_iv, stats = compute_chain_iv(chain, r, q)
        term = compute_term_structure(chain_iv)
        row = session_metrics_row(session, float(chain["spot"].iloc[0]),
                                  chain_source_label(chain), term, stats, cfg)
        metrics = upsert_session(metrics, row, cfg)
    metrics = refresh_rv_columns(metrics, storage.read_underlying(root), cfg)
    if write:
        storage.write_daily_metrics(metrics, root)
    return metrics
