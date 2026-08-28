"""EODHDProvider against fixture JSON; envfile parsing. No network."""
import datetime as dt
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.base import UNDERLYING_COLUMNS
from src.data.envfile import get_secret, load_env
from src.data.eodhd_provider import EODHDProvider

FIX = Path("tests/fixtures")
CFG = {
    "exchange_suffix": ".US",
    "rates": {"risk_free_fallback": 0.04, "dividend_yield_fallback": 0.013},
}


def fake_session(*payloads):
    """Session whose successive .get calls return these JSON payloads."""
    s = MagicMock()
    responses = []
    for p in payloads:
        r = MagicMock()
        r.status_code = 200
        r.json.return_value = p
        responses.append(r)
    s.get.side_effect = responses
    return s


def load(name):
    return json.loads((FIX / name).read_text())


class TestEnvfile:
    def test_load_env_parses_and_ignores_noise(self, tmp_path):
        p = tmp_path / ".env"
        p.write_text("# comment\nEODHD_API_TOKEN=abc123\n\nMASSIVE_API_KEY=xyz=with=equals\n")
        env = load_env(p)
        assert env == {"EODHD_API_TOKEN": "abc123", "MASSIVE_API_KEY": "xyz=with=equals"}

    def test_get_secret_prefers_os_environ(self, tmp_path, monkeypatch):
        p = tmp_path / ".env"
        p.write_text("K=file\n")
        monkeypatch.setenv("K", "environ")
        assert get_secret("K", p) == "environ"
        monkeypatch.delenv("K")
        assert get_secret("K", p) == "file"

    def test_get_secret_missing_raises_keyerror(self, tmp_path):
        with pytest.raises(KeyError):
            get_secret("NOPE", tmp_path / ".env")


class TestUnderlying:
    def test_history_shape_and_types(self):
        s = fake_session(load("eodhd_eod.json"))
        p = EODHDProvider("tok", CFG, session=s)
        df = p.get_underlying_history("SPY")
        assert list(df.columns) == UNDERLYING_COLUMNS
        assert df["date"].tolist() == [dt.date(2026, 8, 26), dt.date(2026, 8, 27), dt.date(2026, 8, 28)]
        assert df["close"].iloc[-1] == 770.0
        # request went to the right place with the token
        url = s.get.call_args[0][0]
        assert "eod/SPY.US" in url
        assert s.get.call_args[1]["params"]["api_token"] == "tok"


class TestRates:
    def test_risk_free_from_us3m(self):
        s = fake_session(load("eodhd_us3m.json"))
        p = EODHDProvider("tok", CFG, session=s)
        assert p.get_risk_free_rate() == pytest.approx(0.0415)

    def test_risk_free_fallback_on_error(self):
        s = MagicMock()
        s.get.side_effect = ConnectionError("down")
        p = EODHDProvider("tok", CFG, session=s)
        assert p.get_risk_free_rate() == 0.04

    def test_dividend_yield_trailing_12m_only(self):
        s = fake_session(load("eodhd_div.json"))
        p = EODHDProvider("tok", CFG, session=s)
        # today=2026-08-28: trailing 365d window excludes 2025-06-20's 1.75
        q = p.get_dividend_yield(spot=770.0, today=dt.date(2026, 8, 28))
        assert q == pytest.approx((1.80 + 2.00 + 1.85 + 1.90) / 770.0)

    def test_dividend_yield_fallback_on_empty(self):
        s = fake_session([])
        p = EODHDProvider("tok", CFG, session=s)
        assert p.get_dividend_yield(770.0) == 0.013
