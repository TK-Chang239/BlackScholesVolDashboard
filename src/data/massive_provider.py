"""Massive (ex-Polygon) options: snapshot fallback + historical backfill.

Starter-tier facts (verified 2026-08-28, see LEARNING_LOG Phase 0):
- chain snapshot: day OHLC, open_interest, vendor IV/greeks on liquid
  strikes; NO bid/ask (higher tier) -> bid/ask/mid stored as NaN.
- historical: reference (as_of, both expired flags) + per-contract daily
  aggregates; no historical open interest -> NaN, volume-only liquidity.
- errors and paywalls arrive inside HTTP 200 bodies: check body["status"].
"""
import datetime as dt
import time

import numpy as np
import pandas as pd
import requests

from src.data.filters import select_expiries

BASE = "https://api.polygon.io"
TIMEOUT = 30
RETRY_SLEEPS = (2, 8)  # seconds between attempts 1->2 and 2->3 (3 attempts total)

PRE_FILTER_COLUMNS = ["expiry", "strike", "kind", "bid", "ask", "mid", "close",
                      "volume", "open_interest", "vendor_iv", "source"]


class MassiveProvider:
    def __init__(self, api_key: str, cfg: dict, session: requests.Session | None = None):
        self._key = api_key
        self._cfg = cfg
        self._session = session or requests.Session()

    def _get_json(self, url: str, **params):
        headers = {"Authorization": f"Bearer {self._key}"}
        # Retry only transport-level failures (timeouts, connection resets).
        # A bad body status (e.g. a NOT_AUTHORIZED paywall) is not a transient
        # condition -- it must fail immediately, never be retried.
        for attempt in range(3):
            try:
                r = self._session.get(url, params=params, headers=headers, timeout=TIMEOUT)
                break
            except requests.exceptions.RequestException:
                if attempt == 2:
                    raise
                time.sleep(RETRY_SLEEPS[attempt])
        body = r.json()
        status = body.get("status") if isinstance(body, dict) else None
        if r.status_code != 200 or status not in ("OK", "DELAYED"):
            raise RuntimeError(f"Massive API error: HTTP {r.status_code}, status={status!r}, "
                               f"message={body.get('message') if isinstance(body, dict) else body!r}")
        return body

    def _strike_bounds(self, spot: float, cfg: dict) -> tuple[float, float]:
        f = cfg["chain_filter"]
        return spot * f["moneyness_min"], spot * f["moneyness_max"]

    # -- fallback: today's chain from the snapshot ---------------------------

    def get_option_chain(self, symbol: str, snapshot_date: dt.date,
                         spot: float, cfg: dict) -> pd.DataFrame:
        lo, hi = self._strike_bounds(spot, cfg)
        f = cfg["chain_filter"]
        url = f"{BASE}/v3/snapshot/options/{symbol}"
        params = {
            "limit": 250,
            "strike_price.gte": lo, "strike_price.lte": hi,
            "expiration_date.gte": str(snapshot_date + dt.timedelta(days=f["dte_min"])),
            "expiration_date.lte": str(snapshot_date + dt.timedelta(days=f["dte_max"])),
        }
        results = []
        while url:
            body = self._get_json(url, **params)
            results.extend(body.get("results") or [])
            url = body.get("next_url")
            params = {}  # next_url carries the cursor; only the key is re-added
        rows = []
        for res in results:
            det = res.get("details", {})
            day = res.get("day", {})
            rows.append({
                "expiry": dt.date.fromisoformat(det["expiration_date"]),
                "strike": float(det["strike_price"]),
                "kind": det["contract_type"],
                "bid": np.nan, "ask": np.nan, "mid": np.nan,
                "close": day.get("close", np.nan),
                "volume": day.get("volume", np.nan),
                "open_interest": res.get("open_interest", np.nan),
                "vendor_iv": res.get("implied_volatility", np.nan),
                "source": "massive-fallback",
            })
        df = pd.DataFrame(rows, columns=PRE_FILTER_COLUMNS)
        keep = set(select_expiries(df["expiry"].unique(), snapshot_date, cfg))
        return df[df["expiry"].isin(keep)].reset_index(drop=True)

    # -- backfill: a past date's chain from reference + aggregates -----------

    def get_historical_chain(self, symbol: str, snapshot_date: dt.date,
                             spot: float, cfg: dict) -> pd.DataFrame:
        lo, hi = self._strike_bounds(spot, cfg)
        f = cfg["chain_filter"]
        contracts = []
        for expired in ("false", "true"):
            url = f"{BASE}/v3/reference/options/contracts"
            params = {
                "underlying_ticker": symbol, "as_of": str(snapshot_date),
                "expired": expired, "limit": 1000,
                "strike_price.gte": lo, "strike_price.lte": hi,
                "expiration_date.gte": str(snapshot_date + dt.timedelta(days=f["dte_min"])),
                "expiration_date.lte": str(snapshot_date + dt.timedelta(days=f["dte_max"])),
            }
            while url:
                body = self._get_json(url, **params)
                contracts.extend(body.get("results") or [])
                url = body.get("next_url")
                params = {}
        seen: dict[str, dict] = {c["ticker"]: c for c in contracts}
        keep_exp = set(select_expiries(
            {dt.date.fromisoformat(c["expiration_date"]) for c in seen.values()},
            snapshot_date, cfg))
        rows = []
        for tk, c in sorted(seen.items()):
            expiry = dt.date.fromisoformat(c["expiration_date"])
            if expiry not in keep_exp:
                continue
            aggs = self._get_json(
                f"{BASE}/v2/aggs/ticker/{tk}/range/1/day/{snapshot_date}/{snapshot_date}")
            if not aggs.get("resultsCount"):
                continue  # no trade that day: nothing honest to store
            bar = aggs["results"][0]
            rows.append({
                "expiry": expiry,
                "strike": float(c["strike_price"]),
                "kind": c["contract_type"],
                "bid": np.nan, "ask": np.nan, "mid": np.nan,
                "close": float(bar["c"]),
                "volume": bar.get("v", np.nan),
                "open_interest": np.nan,  # not available historically on this tier
                "vendor_iv": np.nan,
                "source": "massive-backfill",
            })
        return pd.DataFrame(rows, columns=PRE_FILTER_COLUMNS)
