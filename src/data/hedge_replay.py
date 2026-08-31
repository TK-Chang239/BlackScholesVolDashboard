"""Replay the P8 hedge simulation over the stored chain archive (SPEC 3 P8).

Same shape as src/data/metrics_backfill.py: read every stored chain, solve
its IVs with the shared solver, hand the result to a pure analytics engine.
Nothing is persisted -- the whole P&L history is a function of data/chains/
+ data/underlying.parquet + (r, q, cfg), recomputed on every run.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.hedge_sim import (
    fit_pnl_vs_edge, hedge_summary, portfolio_daily, simulate,
)
from src.data import storage

_CHAIN_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"


def replay_hedge_sim(root: Path, r: float, q: float, cfg: dict,
                     extra_chains: dict | None = None,
                     underlying: pd.DataFrame | None = None) -> dict:
    root = Path(root)
    chains: dict[dt.date, pd.DataFrame] = {}
    for path in sorted((root / "data" / "chains").glob(_CHAIN_GLOB)):
        session = dt.date.fromisoformat(path.stem)
        solved, _ = compute_chain_iv(pd.read_parquet(path), r, q)
        chains[session] = solved
    # Today's chain is computed before it is written (SPEC 2.1 failure policy),
    # so the caller passes it in rather than the sim missing the current session.
    chains.update(extra_chains or {})

    if underlying is None:
        underlying = storage.read_underlying(root)
    trades, daily, months_skipped = simulate(chains, underlying, r, q, cfg)
    port = portfolio_daily(daily)
    fit = fit_pnl_vs_edge(trades)
    cost_bps = float(cfg["hedge_sim"]["transaction_cost_bps"])
    # The stat line names the window inside which an expiry is absent from every
    # stored chain by design; that number is config's, not the render layer's.
    dte_min = int(cfg["chain_filter"]["dte_min"])
    return {"trades": trades, "daily": daily, "portfolio": port, "fit": fit,
            "summary": hedge_summary(trades, port, fit, months_skipped, cost_bps,
                                     dte_min)}
