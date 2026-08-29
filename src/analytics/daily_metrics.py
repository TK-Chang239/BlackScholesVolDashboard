"""daily_metrics: one row per session (SPEC 2.2 storage, 3 P5). Pure.

IV-side columns are facts about that session's chain, computed once.
RV-side columns are recomputed for EVERY row from the full underlying
history on every run, so forward RV back-fills as days mature. Parquet
I/O lives in src.data.storage; replay of stored chains in
src.data.metrics_backfill.
"""
import datetime as dt

import numpy as np
import pandas as pd

from src.models.realized_vol import forward_realized_vol, trailing_realized_vol

IV_COLUMNS = ["date", "spot", "source", "atm_iv_30d", "atm_iv_30d_dte", "iv_convergence"]
ATM_NEAREST_TOLERANCE_DAYS = 10


def rv_columns(cfg: dict) -> list[str]:
    rv = cfg["realized_vol"]
    return [f"rv_{w}d" for w in rv["windows"]] + [f"fwd_rv_{rv['forward_horizon_days']}d"]


def metric_columns(cfg: dict) -> list[str]:
    return IV_COLUMNS + rv_columns(cfg)


def interp_atm_iv(term: pd.DataFrame, target_dte: int) -> tuple[float, float]:
    """ATM IV at `target_dte`: linear in DTE when bracketed by stored
    expiries, the nearest expiry when within tolerance, else NaN. Returns
    (atm_iv, effective_dte) so callers can label what they actually got."""
    t = term.dropna(subset=["atm_iv"]).sort_values("dte") if len(term) else term
    if t is None or t.empty:
        return (np.nan, np.nan)
    dtes = t["dte"].to_numpy(dtype=float)
    ivs = t["atm_iv"].to_numpy(dtype=float)
    if dtes.min() <= target_dte <= dtes.max():
        return float(np.interp(target_dte, dtes, ivs)), float(target_dte)
    j = int(np.argmin(np.abs(dtes - target_dte)))
    if abs(dtes[j] - target_dte) <= ATM_NEAREST_TOLERANCE_DAYS:
        return float(ivs[j]), float(dtes[j])
    return (np.nan, np.nan)


def session_metrics_row(session_date: dt.date, spot: float, source: str,
                        term: pd.DataFrame, iv_stats: dict, cfg: dict) -> dict:
    atm_iv, eff_dte = interp_atm_iv(term, int(cfg["target_dte"]["atm_panel"]))
    return {
        "date": session_date, "spot": float(spot), "source": source,
        "atm_iv_30d": atm_iv, "atm_iv_30d_dte": eff_dte,
        "iv_convergence": float(iv_stats["convergence"]),
    }


def upsert_session(metrics: pd.DataFrame, row: dict, cfg: dict) -> pd.DataFrame:
    cols = metric_columns(cfg)
    new = pd.DataFrame([row]).reindex(columns=cols)
    base = metrics.reindex(columns=cols)
    merged = pd.concat([base, new], ignore_index=True) if len(base) else new
    return (merged.drop_duplicates("date", keep="last")
                  .sort_values("date").reset_index(drop=True)[cols])


def refresh_rv_columns(metrics: pd.DataFrame, underlying: pd.DataFrame,
                       cfg: dict) -> pd.DataFrame:
    rv_cfg = cfg["realized_vol"]
    ann = rv_cfg["annualization_days"]
    series = [trailing_realized_vol(underlying, w, ann) for w in rv_cfg["windows"]]
    series.append(forward_realized_vol(underlying, rv_cfg["forward_horizon_days"], ann))
    rv = pd.concat(series, axis=1).reset_index()          # date + rv columns
    out = metrics.drop(columns=rv_columns(cfg), errors="ignore")
    out = out.merge(rv, on="date", how="left")
    return out[metric_columns(cfg)].sort_values("date").reset_index(drop=True)
