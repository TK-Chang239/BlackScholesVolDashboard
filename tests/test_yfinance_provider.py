"""YFinanceProvider with a stubbed yfinance Ticker. No network."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.data.yfinance_provider import YFinanceProvider

TODAY = dt.date(2026, 8, 28)
CFG = {
    "chain_filter": {
        "n_monthly_expiries": 2, "dte_min": 7, "dte_max": 365,
        "moneyness_min": 0.70, "moneyness_max": 1.30,
    }
}


def yf_frame(rows):
    return pd.DataFrame(rows, columns=[
        "contractSymbol", "strike", "bid", "ask", "lastPrice",
        "volume", "openInterest", "impliedVolatility"])


class StubTicker:
    options = ("2026-09-04", "2026-09-18", "2026-10-16", "2026-11-20")

    def __init__(self):
        self.requested = []

    def option_chain(self, expiry):
        self.requested.append(expiry)

        class OC:
            calls = yf_frame([["C1", 700.0, 71.0, 71.4, 71.2, 10, 100, 0.21]])
            puts = yf_frame([["P1", 700.0, 1.9, 2.1, 2.0, 5, 50, 0.24]])
        return OC()


class TestYFinance:
    def test_fetches_only_selected_monthlies(self):
        stub = StubTicker()
        p = YFinanceProvider(CFG, ticker_factory=lambda sym: stub)
        df = p.get_option_chain("SPY", TODAY, spot=770.0, cfg=CFG)
        # n=2 monthlies from {09-18, 10-16, 11-20}; weekly 09-04 never fetched
        assert stub.requested == ["2026-09-18", "2026-10-16"]
        assert set(df["expiry"]) == {dt.date(2026, 9, 18), dt.date(2026, 10, 16)}

    def test_row_normalization(self):
        p = YFinanceProvider(CFG, ticker_factory=lambda sym: StubTicker())
        df = p.get_option_chain("SPY", TODAY, spot=770.0, cfg=CFG)
        call = df[(df["kind"] == "call") & (df["expiry"] == dt.date(2026, 9, 18))].iloc[0]
        assert call["bid"] == 71.0 and call["ask"] == 71.4
        assert call["mid"] == pytest.approx(71.2)
        assert call["close"] == 71.2  # lastPrice
        assert call["open_interest"] == 100
        assert call["vendor_iv"] == pytest.approx(0.21)
        assert call["source"] == "yfinance"

    def test_mid_nan_when_unquoted(self):
        class Stub(StubTicker):
            def option_chain(self, expiry):
                class OC:
                    calls = yf_frame([["C1", 700.0, 0.0, 1.2, 1.1, 1, 1, 0.2]])
                    puts = yf_frame([])
                return OC()
        p = YFinanceProvider(CFG, ticker_factory=lambda sym: Stub())
        df = p.get_option_chain("SPY", TODAY, spot=770.0, cfg=CFG)
        assert np.isnan(df["mid"].iloc[0])  # bid=0 -> no honest mid

    def test_vendor_failure_raises(self):
        class Boom:
            @property
            def options(self):
                raise RuntimeError("yahoo changed something")
        p = YFinanceProvider(CFG, ticker_factory=lambda sym: Boom())
        with pytest.raises(Exception):
            p.get_option_chain("SPY", TODAY, spot=770.0, cfg=CFG)

    def test_put_row_normalization(self):
        p = YFinanceProvider(CFG, ticker_factory=lambda sym: StubTicker())
        df = p.get_option_chain("SPY", TODAY, spot=770.0, cfg=CFG)
        put = df[(df["kind"] == "put") & (df["expiry"] == dt.date(2026, 9, 18))].iloc[0]
        assert put["kind"] == "put"
        assert put["bid"] == 1.9
        assert put["ask"] == 2.1
        assert put["mid"] == pytest.approx(2.0)
        assert put["close"] == 2.0  # lastPrice
        assert put["open_interest"] == 50
        assert put["vendor_iv"] == pytest.approx(0.24)
        assert put["source"] == "yfinance"

    def test_empty_chain_raises(self):
        class EmptyStub(StubTicker):
            def option_chain(self, expiry):
                class OC:
                    calls = yf_frame([])
                    puts = yf_frame([])
                return OC()
        p = YFinanceProvider(CFG, ticker_factory=lambda sym: EmptyStub())
        with pytest.raises(RuntimeError, match="no option rows"):
            p.get_option_chain("SPY", TODAY, spot=770.0, cfg=CFG)
