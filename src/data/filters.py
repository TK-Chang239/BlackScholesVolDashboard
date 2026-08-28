"""Pure chain-filtering functions (SPEC 2.2). No I/O, no vendor knowledge."""
import datetime as dt
from typing import Iterable

import pandas as pd

from src.data.base import CHAIN_COLUMNS, LIVE_SOURCES


def is_monthly_expiry(d: dt.date) -> bool:
    """Standard monthly = third Friday of its month."""
    return d.weekday() == 4 and 15 <= d.day <= 21


def select_expiries(expiries: Iterable[dt.date], today: dt.date, cfg: dict) -> list[dt.date]:
    f = cfg["chain_filter"]
    monthlies = sorted(
        e for e in set(expiries)
        if is_monthly_expiry(e) and f["dte_min"] <= (e - today).days <= f["dte_max"]
    )
    return monthlies[: f["n_monthly_expiries"]]


def filter_chain(df: pd.DataFrame, spot: float, today: dt.date, cfg: dict) -> pd.DataFrame:
    f = cfg["chain_filter"]
    df = df.copy()
    moneyness = df["strike"] / spot
    keep = (moneyness >= f["moneyness_min"]) & (moneyness <= f["moneyness_max"])

    live = df["source"].isin(LIVE_SOURCES)
    liquid_live = (df["bid"] > 0) & (df["ask"] > 0)
    liquid_close = (df["volume"].fillna(0) > 0) | (df["open_interest"].fillna(0) > 0)
    keep &= liquid_live.where(live, liquid_close)

    out = df.loc[keep].copy()
    out["snapshot_date"] = today
    out["spot"] = float(spot)
    out["dte"] = out["expiry"].map(lambda e: (e - today).days)
    out = out[CHAIN_COLUMNS].sort_values(["expiry", "kind", "strike"]).reset_index(drop=True)
    return out
