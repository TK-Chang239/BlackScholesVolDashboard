"""P5: implied vs realized vol (SPEC 3 P5). Pure.

Trailing RV is the live visual; FORWARD RV is the honest forecast target:
the vol that actually happened over the ~30 days after each IV quote. The
summary stat is computed against forward RV only.
"""
import numpy as np
import pandas as pd

IV_RV_COLUMNS = ["date", "atm_iv", "rv_trailing", "fwd_rv", "spread", "spread_running_mean"]


def iv_rv_series(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rv_cfg = cfg["realized_vol"]
    trailing = f"rv_{rv_cfg['windows'][0]}d"
    forward = f"fwd_rv_{rv_cfg['forward_horizon_days']}d"
    if metrics.empty:
        return pd.DataFrame(columns=IV_RV_COLUMNS)
    s = pd.DataFrame({
        "date": metrics["date"],
        "atm_iv": metrics["atm_iv_30d"].astype(float),
        "rv_trailing": metrics[trailing].astype(float),
        "fwd_rv": metrics[forward].astype(float),
    })
    s = s[s["atm_iv"].notna()].sort_values("date").reset_index(drop=True)
    s["spread"] = s["atm_iv"] - s["rv_trailing"]
    s["spread_running_mean"] = s["spread"].expanding().mean()   # NaNs skipped by pandas
    return s[IV_RV_COLUMNS]


def iv_rv_summary(series: pd.DataFrame) -> dict:
    if series.empty:
        return {"history_since": None, "n_sessions": 0, "evaluable_days": 0,
                "share_iv_above_fwd_rv": np.nan, "mean_spread_trailing": np.nan,
                "last_evaluable_iv": np.nan, "last_evaluable_fwd_rv": np.nan}
    ev = series[series["atm_iv"].notna() & series["fwd_rv"].notna()]
    return {
        "history_since": series["date"].iloc[0],
        "n_sessions": int(len(series)),
        "evaluable_days": int(len(ev)),
        "share_iv_above_fwd_rv": float((ev["atm_iv"] > ev["fwd_rv"]).mean()) if len(ev) else np.nan,
        "mean_spread_trailing": float(series["spread"].mean()) if series["spread"].notna().any() else np.nan,
        "last_evaluable_iv": float(ev["atm_iv"].iloc[-1]) if len(ev) else np.nan,
        "last_evaluable_fwd_rv": float(ev["fwd_rv"].iloc[-1]) if len(ev) else np.nan,
    }


def iv_rv_summary_html(summary: dict) -> str:
    if summary["n_sessions"] == 0:
        return "<p class='stat'>No history yet — the series starts with the first stored session.</p>"
    since = summary["history_since"].isoformat()
    n = summary["evaluable_days"]
    if n == 0:
        return (f"<p class='stat'>History since {since} ({summary['n_sessions']} sessions); "
                "no day is old enough for its forward realized vol to be known yet — "
                "the forecast scorecard is not yet evaluable.</p>")
    if n == 1:
        return (f"<p class='stat'>Only 1 session is old enough to score so far: implied "
                f"{summary['last_evaluable_iv']:.1%} vs {summary['last_evaluable_fwd_rv']:.1%} "
                "subsequently realized — one data point, not a rate.</p>")
    if n < 10:
        k = round(summary["share_iv_above_fwd_rv"] * n)
        return (f"<p class='stat'>Only {n} sessions are old enough to score so far: implied vol "
                f"exceeded subsequently-realized vol on {k} of them — too few to call a rate.</p>")
    return (f"<p class='stat'>Implied vol exceeded subsequently-realized vol on "
            f"{summary['share_iv_above_fwd_rv']:.0%} of {n} evaluable "
            f"days (history since {since}, {summary['n_sessions']} sessions).</p>")
