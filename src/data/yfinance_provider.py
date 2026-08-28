"""Live SPY chain via yfinance (SPEC 2.2: the only source with real bid/ask).

Unofficial API: any breakage raises out of here so run_daily can fall back
to Massive and flag the day (SPEC 2.1/2.2). The yfinance import lives
inside the factory default so the test stub never touches the real library.
"""
import datetime as dt

import numpy as np
import pandas as pd

from src.data.filters import select_expiries

PRE_FILTER_COLUMNS = ["expiry", "strike", "kind", "bid", "ask", "mid", "close",
                      "volume", "open_interest", "vendor_iv", "source"]


def _default_ticker_factory(symbol: str):
    import yfinance as yf
    return yf.Ticker(symbol)


class YFinanceProvider:
    def __init__(self, cfg: dict, ticker_factory=None):
        self._cfg = cfg
        self._factory = ticker_factory or _default_ticker_factory

    def get_option_chain(self, symbol: str, snapshot_date: dt.date,
                         spot: float, cfg: dict) -> pd.DataFrame:
        t = self._factory(symbol)
        all_expiries = [dt.date.fromisoformat(e) for e in t.options]
        selected = select_expiries(all_expiries, snapshot_date, cfg)
        frames = []
        for expiry in selected:
            oc = t.option_chain(expiry.isoformat())
            for kind, raw in (("call", oc.calls), ("put", oc.puts)):
                if raw.empty:
                    continue
                df = pd.DataFrame({
                    "expiry": expiry,
                    "strike": raw["strike"].astype(float),
                    "kind": kind,
                    "bid": raw["bid"].astype(float),
                    "ask": raw["ask"].astype(float),
                    "close": raw["lastPrice"].astype(float),
                    "volume": raw["volume"],
                    "open_interest": raw["openInterest"],
                    "vendor_iv": raw["impliedVolatility"].astype(float),
                })
                quoted = (df["bid"] > 0) & (df["ask"] > 0)
                df["mid"] = np.where(quoted, (df["bid"] + df["ask"]) / 2.0, np.nan)
                df["source"] = "yfinance"
                frames.append(df)
        out = pd.concat(frames, ignore_index=True)
        return out[PRE_FILTER_COLUMNS]
