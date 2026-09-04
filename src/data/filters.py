"""Pure chain-filtering functions (SPEC 2.2). No I/O, no vendor knowledge."""
import datetime as dt
from typing import Iterable

import pandas as pd

from src.data.base import CHAIN_COLUMNS, LIVE_SOURCES

# The NYSE's first Juneteenth closure. The federal holiday was signed into law
# in June 2021, but the exchange stayed open that year and first observed it in
# 2022 (Monday the 20th, June 19 being a Sunday).
NYSE_FIRST_JUNETEENTH_YEAR = 2022


def _easter(year: int) -> dt.date:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day = divmod(h + ll - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _third_friday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def _exchange_closed(friday: dt.date) -> bool:
    """Is the NYSE shut on this third Friday?

    Only two holidays can land on one, and the list is closed by construction
    rather than by maintenance. Every other NYSE holiday is a fixed date
    outside the window (New Year, Independence Day, Christmas), a Monday (MLK,
    Washington, Memorial, Labor), or the fourth Thursday (Thanksgiving). The
    two that are:

    - Good Friday, a Friday by definition, falling between March 20 and April
      23; it is the third Friday whenever it lands on the 15th to the 21st.
    - Juneteenth, observed on the Friday when June 19 is a Friday or a
      Saturday. A Sunday June 19 is observed on the Monday and leaves the
      Friday open. The NYSE first closed for it in 2022; before that the
      holiday did not exist for this market, so reading it back through
      history would invent shifts that never happened.

    Unscheduled closures (a national day of mourning) are not modelled: they
    are not knowable in advance, and the vendor's contract reference is the
    authority on what actually expired.
    """
    if friday == _easter(friday.year) - dt.timedelta(days=2):
        return True
    if friday.year < NYSE_FIRST_JUNETEENTH_YEAR:
        return False
    juneteenth = dt.date(friday.year, 6, 19)
    if juneteenth.weekday() == 4:            # holiday falls on the Friday itself
        return friday == juneteenth
    if juneteenth.weekday() == 5:            # Saturday -> observed the Friday before
        return friday == juneteenth - dt.timedelta(days=1)
    return False


def is_monthly_expiry(d: dt.date) -> bool:
    """Standard monthly = third Friday of its month.

    Except when the exchange is shut that Friday, in which case the contract
    settles on the preceding session and the Thursday is the only date the
    vendor ever lists -- verified against Massive's contract reference, which
    carries 2026-06-18 and 2025-04-17 but neither of their Fridays. Reading
    the rule as "third Friday" alone dropped those months out of the ladder
    entirely, taking P5, P6 and two whole hedge trades with them.

    The shift is computed against the third Friday itself, not against a "15th
    to 21st" day window: when the third Friday IS the 15th (Good Friday 2022,
    2033) the contract settles on the 14th, and a day window drops the month
    on the floor -- the same failure this fix exists to remove.
    """
    friday = _third_friday(d.year, d.month)
    if _exchange_closed(friday):
        return d == friday - dt.timedelta(days=1)
    return d == friday


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
