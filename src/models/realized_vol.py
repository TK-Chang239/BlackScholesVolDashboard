"""Realized volatility from adjusted closes (SPEC 2.3, 3 P5). Hand-built.

Close-to-close LOG returns on ADJUSTED close -- unadjusted closes would
book every dividend as a fake down-move -- sample std (ddof=1), annualized
by sqrt(annualization_days). Two views of the same quantity:

  trailing_realized_vol  what happened over the last `window` returns
                         ending at t (the live visual on P5)
  forward_realized_vol   what subsequently happened over (t, t+horizon]
                         calendar days -- NaN until that horizon has fully
                         matured. This is the honest forecast target for
                         a ~horizon-DTE implied vol quoted on day t.
"""
import numpy as np
import pandas as pd

MIN_FORWARD_RETURNS = 5


def log_returns(adjusted_close: pd.Series) -> pd.Series:
    return np.log(adjusted_close.astype(float)).diff()


def _sorted(underlying: pd.DataFrame) -> pd.DataFrame:
    return underlying.sort_values("date").reset_index(drop=True)


def trailing_realized_vol(underlying: pd.DataFrame, window: int,
                          annualization_days: int = 252) -> pd.Series:
    df = _sorted(underlying)
    rets = log_returns(df["adjusted_close"])
    rv = rets.rolling(window).std(ddof=1) * np.sqrt(annualization_days)
    return pd.Series(rv.to_numpy(), index=pd.Index(df["date"], name="date"),
                     name=f"rv_{window}d")


def forward_realized_vol(underlying: pd.DataFrame, horizon_days: int = 30,
                         annualization_days: int = 252) -> pd.Series:
    df = _sorted(underlying)
    dates = pd.to_datetime(df["date"])
    rets = log_returns(df["adjusted_close"]).to_numpy()
    last = dates.iloc[-1]
    out = np.full(len(df), np.nan)
    for i, t in enumerate(dates):
        end = t + pd.Timedelta(days=horizon_days)
        if end > last:
            break                       # nothing later can be matured either
        in_window = ((dates > t) & (dates <= end)).to_numpy()
        r = rets[in_window]
        if len(r) >= MIN_FORWARD_RETURNS:
            out[i] = np.std(r, ddof=1) * np.sqrt(annualization_days)
    return pd.Series(out, index=pd.Index(df["date"], name="date"),
                     name=f"fwd_rv_{horizon_days}d")
