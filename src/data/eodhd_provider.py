"""EODHD: underlying history, risk-free rate, dividend yield (SPEC 2.2).

Rates degrade to config fallbacks on any failure -- their precision barely
matters and the sensitivity panel exists to demonstrate that (SPEC 2.2).
The underlying fetch does NOT degrade: without it there is no spot and the
daily run must fail loudly (SPEC 2.1).
"""
import datetime as dt

import pandas as pd
import requests

from src.data.base import UNDERLYING_COLUMNS

BASE = "https://eodhd.com/api"
TIMEOUT = 30


class EODHDProvider:
    def __init__(self, token: str, cfg: dict, session: requests.Session | None = None):
        self._token = token
        self._cfg = cfg
        self._session = session or requests.Session()

    def _get_json(self, path: str, **params):
        params.setdefault("api_token", self._token)
        params.setdefault("fmt", "json")
        r = self._session.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def get_underlying_history(self, symbol: str, start: dt.date | None = None) -> pd.DataFrame:
        params = {}
        if start is not None:
            params["from"] = str(start)
        rows = self._get_json(f"eod/{symbol}{self._cfg['exchange_suffix']}", **params)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df[UNDERLYING_COLUMNS].sort_values("date").reset_index(drop=True)

    def get_risk_free_rate(self) -> float:
        try:
            rows = self._get_json("eod/US3M.INDX")
            return float(rows[-1]["close"]) / 100.0
        except Exception:
            return float(self._cfg["rates"]["risk_free_fallback"])

    def get_dividend_yield(self, spot: float, today: dt.date | None = None,
                           symbol: str = "SPY") -> float:
        today = today or dt.date.today()
        try:
            rows = self._get_json(
                f"div/{symbol}{self._cfg['exchange_suffix']}",
                **{"from": str(today - dt.timedelta(days=400))},
            )
            window_start = today - dt.timedelta(days=365)
            total = sum(
                float(r["value"]) for r in rows
                if window_start <= dt.date.fromisoformat(r["date"]) <= today
            )
            if total <= 0:
                return float(self._cfg["rates"]["dividend_yield_fallback"])
            return total / float(spot)
        except Exception:
            return float(self._cfg["rates"]["dividend_yield_fallback"])
