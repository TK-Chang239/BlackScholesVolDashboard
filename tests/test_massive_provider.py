"""MassiveProvider against fixture JSON with a mocked session. No network."""
import datetime as dt
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data.massive_provider import MassiveProvider

FIX = Path("tests/fixtures")
TODAY = dt.date(2026, 8, 28)
CFG = {
    "chain_filter": {
        "n_monthly_expiries": 6, "dte_min": 7, "dte_max": 365,
        "moneyness_min": 0.70, "moneyness_max": 1.30,
    }
}


def load(name):
    return json.loads((FIX / name).read_text())


def session_returning(payloads):
    s = MagicMock()
    resps = []
    for p in payloads:
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = p
        resps.append(r)
    s.get.side_effect = resps
    return s


class TestSnapshotFallback:
    def test_normalizes_and_keeps_only_selected_monthlies(self):
        s = session_returning([load("massive_snapshot_page.json")])
        p = MassiveProvider("key", CFG, session=s)
        df = p.get_option_chain("SPY", TODAY, spot=770.0, cfg=CFG)
        assert set(df["expiry"]) == {dt.date(2026, 9, 18)}  # weekly 09-04 dropped
        call = df[df["kind"] == "call"].iloc[0]
        assert call["close"] == 71.3
        assert call["open_interest"] == 5000
        assert call["vendor_iv"] == pytest.approx(0.213)
        assert np.isnan(call["bid"]) and np.isnan(call["ask"]) and np.isnan(call["mid"])
        assert (df["source"] == "massive-fallback").all()

    def test_pagination_follows_next_url(self):
        page1 = dict(load("massive_snapshot_page.json"))
        page1["next_url"] = "https://api.polygon.io/v3/snapshot/options/SPY?cursor=abc"
        page2 = {"status": "OK", "results": []}
        s = session_returning([page1, page2])
        p = MassiveProvider("key", CFG, session=s)
        p.get_option_chain("SPY", TODAY, spot=770.0, cfg=CFG)
        assert s.get.call_count == 2
        second_url = s.get.call_args_list[1][0][0]
        assert "cursor=abc" in second_url

    def test_not_authorized_raises(self):
        s = session_returning([{"status": "NOT_AUTHORIZED", "message": "upgrade"}])
        p = MassiveProvider("key", CFG, session=s)
        with pytest.raises(RuntimeError, match="NOT_AUTHORIZED"):
            p.get_option_chain("SPY", TODAY, spot=770.0, cfg=CFG)


class TestHistoricalBackfill:
    def test_builds_rows_from_reference_plus_aggs(self):
        # calls: reference(expired=false), reference(expired=true), aggs x2 monthlies
        empty_ref = {"status": "OK", "results": []}
        s = session_returning([
            load("massive_reference.json"), empty_ref,
            load("massive_aggs.json"), load("massive_aggs.json"),
        ])
        p = MassiveProvider("key", CFG, session=s)
        df = p.get_historical_chain("SPY", TODAY, spot=770.0, cfg=CFG)
        assert set(df["expiry"]) == {dt.date(2026, 9, 18)}   # weekly dropped pre-aggs
        assert len(df) == 2                                   # call + put
        assert (df["source"] == "massive-backfill").all()
        assert df["close"].tolist() == [71.3, 71.3]
        assert df["volume"].tolist() == [120, 120]
        assert df["open_interest"].isna().all()
        assert df["bid"].isna().all() and df["mid"].isna().all()
        # aggs called once per surviving contract, never for the weekly
        agg_urls = [c[0][0] for c in s.get.call_args_list[2:]]
        assert all("/range/1/day/" in u for u in agg_urls)
        assert not any("SPY260904" in u for u in agg_urls)

    def test_no_trade_day_skips_contract(self):
        empty_ref = {"status": "OK", "results": []}
        no_trades = {"status": "OK", "resultsCount": 0, "results": []}
        s = session_returning([
            load("massive_reference.json"), empty_ref,
            load("massive_aggs.json"), no_trades,
        ])
        p = MassiveProvider("key", CFG, session=s)
        df = p.get_historical_chain("SPY", TODAY, spot=770.0, cfg=CFG)
        assert len(df) == 1  # the no-trade contract produced no row
