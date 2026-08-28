# vol-lens Phase 2 Data Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the hybrid data layer — yfinance live chain, Massive backfill/fallback, EODHD underlying/dividends/rates — with filtering, Parquet storage, a thin daily orchestrator, and a resumable backfill, ending with one real stored snapshot and a real backfill run.

**Architecture:** Three vendor providers with a uniform chain contract (`CHAIN_COLUMNS`), pure config-driven filtering, and append-only Parquet storage under `data/`. `src/run_daily.py` wires them: EODHD underlying first (canonical spot), then yfinance chain with automatic Massive-snapshot fallback, filter, atomic write, `docs/status.json` last. Backfill reconstructs per-date chains from Massive reference+aggregates; resumability = chain files already on disk are skipped. All tests run offline against committed fixtures (SPEC §4); real network runs happen once, at the end, to satisfy the Phase 2 exit criterion.

**Tech Stack:** Python 3.13 local / 3.12 CI · pandas · pyarrow · requests · yfinance · pytest

**Spec:** `SPEC.md` — §2.1 (failure policy, secrets), §2.2 (data layer, the binding section), §4 (fixture pipeline test), §5 Phase 2 exit criterion: "≥ 1 real daily snapshot stored; backfill run attempted and depth documented."

## Global Constraints

- Nothing outside `src/data/` may know which vendor serves which call (SPEC §2.2); `src/run_daily.py` is the single composition point.
- Every stored chain row records its `source`: `"yfinance"` | `"massive-fallback"` | `"massive-backfill"` (SPEC §2.2).
- Chain filter values from `config.yaml` verbatim: 6 monthly expiries, DTE 7–365, moneyness K/S 0.70–1.30 (SPEC §2.2).
- Liquidity floor: live rows drop bid = 0 or missing ask; close-only rows drop zero-volume AND zero-OI (OI treated as 0 when absent) (SPEC §2.2).
- Vendor IV is stored for cross-checking, **never** used in computations (SPEC §2.2). Vendor Greeks are NOT stored — yfinance has none and no panel consumes them; recording this trim here (YAGNI).
- Secrets: `EODHD_API_TOKEN`, `MASSIVE_API_KEY` from env or gitignored `.env`; never in code, never committed (SPEC §2.1). yfinance needs no key.
- Failure policy: a failed fetch raises; the run exits nonzero having written no chain file and left `docs/status.json` untouched (SPEC §2.1). Writes are atomic (tmp + rename).
- All pytest tests run without network or API keys, against fixtures in `tests/fixtures/` (SPEC §4). `filterwarnings = ["error"]` stays on — stub out anything noisy.
- Model-layer note (deferred minor from Phase 1): if you touch `src/models/black_scholes.py`, rename `_d1_d2`'s argument order to `(S, K, T, r, sigma, q)`. No Phase 2 task touches it.
- TDD per task; commit at task end. Python: `.venv/bin/pytest`, `.venv/bin/python`.

---

## File Structure

```
src/data/
├── __init__.py            # Task 1 (empty)
├── base.py                # Task 1 — CHAIN_COLUMNS schema constant, UNDERLYING_COLUMNS
├── filters.py             # Task 1 — monthly-expiry logic, expiry selection, chain filter
├── envfile.py             # Task 2 — minimal .env reader (no python-dotenv dep)
├── eodhd_provider.py      # Task 2 — underlying history, risk-free rate, dividend yield
├── yfinance_provider.py   # Task 3 — live chain via yfinance
├── massive_provider.py    # Task 4 — snapshot fallback + historical backfill chains
├── storage.py             # Task 5 — Parquet writes (atomic), underlying upsert, status.json
└── backfill.py            # Task 6 — resumable backfill loop
src/run_daily.py           # Task 5 — daily orchestrator (fetch → filter → store → status)
scripts/backfill.py        # Task 6 — CLI wrapper
tests/
├── fixtures/              # Task 2/4 — committed vendor-response JSON
│   ├── eodhd_eod.json  eodhd_div.json  eodhd_us3m.json
│   ├── massive_snapshot_page.json  massive_reference.json  massive_aggs.json
├── test_filters.py        # Task 1
├── test_eodhd_provider.py # Task 2
├── test_yfinance_provider.py  # Task 3
├── test_massive_provider.py   # Task 4
├── test_pipeline.py       # Task 5 — the SPEC §4 fixture pipeline test
└── test_backfill.py       # Task 6
data/chains/.gitkeep       # Task 5
```

One vendor per file (each has distinct API quirks); `filters.py` is pure functions; `storage.py` owns every disk write. There is no formal Protocol class: the "single DataProvider interface" (SPEC §2.2) is the uniform `get_option_chain(symbol, snapshot_date, spot, cfg) -> DataFrame[CHAIN_COLUMNS]` signature shared by both chain providers plus EODHD's three market-data methods, with `run_daily.py` as the only composition point. A Protocol would add a class no test needs.

**Canonical spot decision (binding for all tasks):** the spot used for moneyness filtering and stored in every chain row is **EODHD's close for the snapshot date** — one source for both live and fallback/backfill paths, so filtering is consistent across sources. yfinance's live price is never used as spot.

---

### Task 1: Schema and chain filtering

**Files:**
- Create: `src/data/__init__.py` (empty), `src/data/base.py`, `src/data/filters.py`
- Modify: `requirements.txt`
- Test: `tests/test_filters.py`

**Interfaces:**
- Consumes: `config.yaml` keys `chain_filter.{n_monthly_expiries, dte_min, dte_max, moneyness_min, moneyness_max}` (existing).
- Produces (used by every later task):
  - `base.CHAIN_COLUMNS: list[str]` = `["snapshot_date", "spot", "expiry", "dte", "strike", "kind", "bid", "ask", "mid", "close", "volume", "open_interest", "vendor_iv", "source"]`
  - `base.UNDERLYING_COLUMNS: list[str]` = `["date", "close", "adjusted_close", "volume"]`
  - `filters.is_monthly_expiry(d: dt.date) -> bool`
  - `filters.select_expiries(expiries: Iterable[dt.date], today: dt.date, cfg: dict) -> list[dt.date]` — monthlies within DTE bounds, nearest `n_monthly_expiries`, sorted ascending
  - `filters.filter_chain(df: pd.DataFrame, spot: float, today: dt.date, cfg: dict) -> pd.DataFrame` — input needs columns `expiry, strike, kind, bid, ask, mid, close, volume, open_interest, vendor_iv, source`; output is exactly `CHAIN_COLUMNS`, sorted by (expiry, kind, strike), fresh index

- [ ] **Step 1: Add Phase 2 deps**

Append to `requirements.txt` (yfinance is already present):

```
pandas>=2.1
pyarrow>=15.0
```

Run: `.venv/bin/pip install -r requirements.txt`

- [ ] **Step 2: Write the failing tests**

`tests/test_filters.py`:

```python
"""Chain filtering: monthly-expiry selection and the SPEC 2.2 filter rules."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.data.base import CHAIN_COLUMNS
from src.data.filters import filter_chain, is_monthly_expiry, select_expiries

TODAY = dt.date(2026, 8, 28)
CFG = {
    "chain_filter": {
        "n_monthly_expiries": 6,
        "dte_min": 7,
        "dte_max": 365,
        "moneyness_min": 0.70,
        "moneyness_max": 1.30,
    }
}


class TestMonthlyExpiry:
    def test_third_fridays_are_monthly(self):
        for d in [dt.date(2026, 9, 18), dt.date(2026, 10, 16), dt.date(2026, 11, 20),
                  dt.date(2026, 12, 18), dt.date(2027, 1, 15), dt.date(2027, 2, 19)]:
            assert is_monthly_expiry(d), d

    def test_non_third_fridays_are_not(self):
        for d in [dt.date(2026, 9, 4),    # first Friday
                  dt.date(2026, 9, 25),   # fourth Friday
                  dt.date(2026, 8, 31),   # a Monday
                  dt.date(2026, 9, 17)]:  # a Thursday
            assert not is_monthly_expiry(d), d


class TestSelectExpiries:
    def test_selects_nearest_six_monthlies_within_dte(self):
        expiries = [
            dt.date(2026, 8, 31), dt.date(2026, 9, 2), dt.date(2026, 9, 4),   # weeklies
            dt.date(2026, 9, 18), dt.date(2026, 10, 16), dt.date(2026, 11, 20),
            dt.date(2026, 12, 18), dt.date(2027, 1, 15), dt.date(2027, 2, 19),
            dt.date(2027, 3, 19),                                             # 7th monthly
            dt.date(2026, 9, 1),                                              # < dte_min once monthly? no: weekly anyway
        ]
        got = select_expiries(expiries, TODAY, CFG)
        assert got == [dt.date(2026, 9, 18), dt.date(2026, 10, 16), dt.date(2026, 11, 20),
                       dt.date(2026, 12, 18), dt.date(2027, 1, 15), dt.date(2027, 2, 19)]

    def test_dte_bounds_respected(self):
        # 2026-09-04 is 7 DTE but weekly; 2026-09-18 is 21 DTE monthly;
        # a monthly < 7 DTE must be dropped: today+3 hypothetical third Friday
        expiries = [dt.date(2026, 9, 18), dt.date(2027, 9, 17)]  # second is 385 DTE > 365
        got = select_expiries(expiries, TODAY, CFG)
        assert got == [dt.date(2026, 9, 18)]

    def test_fewer_than_n_available_returns_what_exists(self):
        got = select_expiries([dt.date(2026, 9, 18)], TODAY, CFG)
        assert got == [dt.date(2026, 9, 18)]


def make_raw(rows):
    cols = ["expiry", "strike", "kind", "bid", "ask", "mid", "close",
            "volume", "open_interest", "vendor_iv", "source"]
    return pd.DataFrame(rows, columns=cols)


SPOT = 770.0
EXP = dt.date(2026, 9, 18)


class TestFilterChain:
    def test_moneyness_bounds(self):
        df = make_raw([
            [EXP, 530.0, "call", 1.0, 1.2, 1.1, 1.1, 10, 10, 0.2, "yfinance"],  # K/S=0.688 < 0.70
            [EXP, 700.0, "call", 1.0, 1.2, 1.1, 1.1, 10, 10, 0.2, "yfinance"],
            [EXP, 1010.0, "call", 1.0, 1.2, 1.1, 1.1, 10, 10, 0.2, "yfinance"],  # 1.312 > 1.30
        ])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert got["strike"].tolist() == [700.0]

    def test_live_liquidity_floor_bid_and_ask(self):
        df = make_raw([
            [EXP, 700.0, "call", 0.0, 1.2, np.nan, 1.1, 10, 10, 0.2, "yfinance"],     # bid=0
            [EXP, 710.0, "call", 1.0, np.nan, np.nan, 1.1, 10, 10, 0.2, "yfinance"],  # no ask
            [EXP, 720.0, "call", 1.0, 1.2, 1.1, 1.1, 0, 0, 0.2, "yfinance"],          # keeps: vol/OI irrelevant live
        ])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert got["strike"].tolist() == [720.0]

    def test_close_only_liquidity_floor_volume_or_oi(self):
        df = make_raw([
            [EXP, 700.0, "call", np.nan, np.nan, np.nan, 1.1, 0, 0, np.nan, "massive-backfill"],   # dead
            [EXP, 710.0, "call", np.nan, np.nan, np.nan, 1.1, 5, 0, np.nan, "massive-backfill"],   # volume
            [EXP, 720.0, "call", np.nan, np.nan, np.nan, 1.1, 0, 9, np.nan, "massive-fallback"],   # OI
            [EXP, 730.0, "call", np.nan, np.nan, np.nan, 1.1, np.nan, np.nan, np.nan, "massive-backfill"],  # NaN==0
        ])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert got["strike"].tolist() == [710.0, 720.0]

    def test_output_schema_and_derived_columns(self):
        df = make_raw([[EXP, 770.0, "put", 10.0, 10.4, 10.2, 10.1, 3, 7, 0.19, "yfinance"]])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert list(got.columns) == CHAIN_COLUMNS
        row = got.iloc[0]
        assert row["snapshot_date"] == TODAY
        assert row["spot"] == SPOT
        assert row["dte"] == (EXP - TODAY).days
        assert got.index.tolist() == [0]

    def test_sorted_by_expiry_kind_strike(self):
        exp2 = dt.date(2026, 10, 16)
        df = make_raw([
            [exp2, 700.0, "call", 1.0, 1.2, 1.1, 1.1, 1, 1, 0.2, "yfinance"],
            [EXP, 710.0, "put", 1.0, 1.2, 1.1, 1.1, 1, 1, 0.2, "yfinance"],
            [EXP, 700.0, "put", 1.0, 1.2, 1.1, 1.1, 1, 1, 0.2, "yfinance"],
            [EXP, 700.0, "call", 1.0, 1.2, 1.1, 1.1, 1, 1, 0.2, "yfinance"],
        ])
        got = filter_chain(df, SPOT, TODAY, CFG)
        assert got[["expiry", "kind", "strike"]].values.tolist() == [
            [EXP, "call", 700.0], [EXP, "put", 700.0], [EXP, "put", 710.0], [exp2, "call", 700.0]]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_filters.py -v`
Expected: ImportError (module doesn't exist).

- [ ] **Step 4: Implement**

`src/data/base.py`:

```python
"""Canonical schemas for the data layer (SPEC 2.2).

The "single DataProvider interface" is a signature contract, not a class:
chain providers implement
    get_option_chain(symbol, snapshot_date, spot, cfg) -> DataFrame[CHAIN_COLUMNS-input]
and src/run_daily.py is the only place vendors are composed/routed.
"""

CHAIN_COLUMNS = [
    "snapshot_date", "spot", "expiry", "dte", "strike", "kind",
    "bid", "ask", "mid", "close", "volume", "open_interest",
    "vendor_iv", "source",
]

UNDERLYING_COLUMNS = ["date", "close", "adjusted_close", "volume"]

LIVE_SOURCES = ("yfinance",)  # rows with real bid/ask quotes
```

`src/data/filters.py`:

```python
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
    liquid_live = (df["bid"] > 0) & df["ask"].notna()
    liquid_close = (df["volume"].fillna(0) > 0) | (df["open_interest"].fillna(0) > 0)
    keep &= liquid_live.where(live, liquid_close)

    out = df.loc[keep].copy()
    out["snapshot_date"] = today
    out["spot"] = float(spot)
    out["dte"] = out["expiry"].map(lambda e: (e - today).days)
    out = out[CHAIN_COLUMNS].sort_values(["expiry", "kind", "strike"]).reset_index(drop=True)
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_filters.py -v`
Expected: all pass. Then full suite: `.venv/bin/pytest -q` → 43 prior + new, all green.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/data/ tests/test_filters.py
git commit -m "feat: chain schema and SPEC-2.2 filtering (monthly expiries, moneyness, per-source liquidity)"
```

---

### Task 2: EODHDProvider + .env reader

**Files:**
- Create: `src/data/envfile.py`, `src/data/eodhd_provider.py`, `tests/fixtures/eodhd_eod.json`, `tests/fixtures/eodhd_div.json`, `tests/fixtures/eodhd_us3m.json`
- Test: `tests/test_eodhd_provider.py`

**Interfaces:**
- Consumes: `base.UNDERLYING_COLUMNS`; `config.yaml` keys `rates.risk_free_fallback`, `rates.dividend_yield_fallback`, `exchange_suffix`.
- Produces:
  - `envfile.load_env(path: str | Path = ".env") -> dict[str, str]` — parses `KEY=VALUE` lines (ignores blanks/comments), does NOT mutate `os.environ`
  - `envfile.get_secret(name: str, env_path: str | Path = ".env") -> str` — `os.environ[name]` if set, else from file, else raises `KeyError` with a helpful message
  - `EODHDProvider(token: str, cfg: dict, session: requests.Session | None = None)` with:
    - `get_underlying_history(symbol: str, start: dt.date | None = None) -> pd.DataFrame[UNDERLYING_COLUMNS]` (date ascending, `date` as `dt.date`)
    - `get_risk_free_rate() -> float` — latest `US3M.INDX` close / 100; on any failure returns `cfg["rates"]["risk_free_fallback"]`
    - `get_dividend_yield(spot: float) -> float` — sum of dividends in the trailing 365 days / spot; on any failure returns `cfg["rates"]["dividend_yield_fallback"]`

- [ ] **Step 1: Write fixtures**

`tests/fixtures/eodhd_eod.json`:

```json
[
  {"date": "2026-08-26", "open": 768.1, "high": 771.0, "low": 767.2, "close": 770.4, "adjusted_close": 770.4, "volume": 41000000},
  {"date": "2026-08-27", "open": 770.5, "high": 772.3, "low": 769.0, "close": 771.1, "adjusted_close": 771.1, "volume": 39000000},
  {"date": "2026-08-28", "open": 771.2, "high": 773.0, "low": 769.8, "close": 770.0, "adjusted_close": 770.0, "volume": 40500000}
]
```

`tests/fixtures/eodhd_div.json` (four quarterly SPY-style distributions; one older than 365d from 2026-08-28 to prove the window):

```json
[
  {"date": "2025-06-20", "declarationDate": "2025-06-18", "value": 1.75, "currency": "USD"},
  {"date": "2025-09-19", "declarationDate": "2025-09-17", "value": 1.80, "currency": "USD"},
  {"date": "2025-12-19", "declarationDate": "2025-12-17", "value": 2.00, "currency": "USD"},
  {"date": "2026-03-20", "declarationDate": "2026-03-18", "value": 1.85, "currency": "USD"},
  {"date": "2026-06-19", "declarationDate": "2026-06-17", "value": 1.90, "currency": "USD"}
]
```

`tests/fixtures/eodhd_us3m.json`:

```json
[
  {"date": "2026-08-27", "close": 4.12, "adjusted_close": 4.12},
  {"date": "2026-08-28", "close": 4.15, "adjusted_close": 4.15}
]
```

- [ ] **Step 2: Write the failing tests**

`tests/test_eodhd_provider.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_eodhd_provider.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement**

`src/data/envfile.py`:

```python
"""Minimal .env reader: no dependency, no os.environ mutation."""
import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip()
    return out


def get_secret(name: str, env_path: str | Path = ".env") -> str:
    if name in os.environ and os.environ[name].strip():
        return os.environ[name].strip()
    env = load_env(env_path)
    if name in env and env[name]:
        return env[name]
    raise KeyError(f"{name} not set in environment or {env_path}")
```

`src/data/eodhd_provider.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_eodhd_provider.py -v`
Expected: all pass. Full suite green.

- [ ] **Step 6: Commit**

```bash
git add src/data/envfile.py src/data/eodhd_provider.py tests/fixtures/eodhd_*.json tests/test_eodhd_provider.py
git commit -m "feat: EODHD provider (underlying, risk-free, dividend yield) + .env reader"
```

---

### Task 3: YFinanceProvider

**Files:**
- Create: `src/data/yfinance_provider.py`
- Test: `tests/test_yfinance_provider.py`

**Interfaces:**
- Consumes: `filters.select_expiries`; a `ticker_factory` injection point for tests.
- Produces: `YFinanceProvider(cfg: dict, ticker_factory=None)` with
  `get_option_chain(symbol: str, snapshot_date: dt.date, spot: float, cfg: dict) -> pd.DataFrame` — pre-filter columns (`expiry, strike, kind, bid, ask, mid, close, volume, open_interest, vendor_iv, source`), `source="yfinance"`, one row per contract for the selected monthly expiries. Raises on any vendor failure (run_daily catches and falls back — SPEC §2.2).

- [ ] **Step 1: Write the failing tests**

`tests/test_yfinance_provider.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_yfinance_provider.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/data/yfinance_provider.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_yfinance_provider.py -v` then full suite.
Expected: all pass, no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/data/yfinance_provider.py tests/test_yfinance_provider.py
git commit -m "feat: yfinance live-chain provider with injected ticker factory"
```

---

### Task 4: MassiveProvider (snapshot fallback + historical backfill)

**Files:**
- Create: `src/data/massive_provider.py`, `tests/fixtures/massive_snapshot_page.json`, `tests/fixtures/massive_reference.json`, `tests/fixtures/massive_aggs.json`
- Test: `tests/test_massive_provider.py`

**Interfaces:**
- Consumes: `filters.select_expiries`; `PRE_FILTER_COLUMNS` shape from Task 3 (re-declared locally — no import between vendor modules).
- Produces: `MassiveProvider(api_key: str, cfg: dict, session=None)` with
  - `get_option_chain(symbol, snapshot_date, spot, cfg) -> pd.DataFrame` — pre-filter columns, `source="massive-fallback"`, from the paginated snapshot; bid/ask/mid all NaN; `close` = `day.close`; `vendor_iv` from `implied_volatility` when present
  - `get_historical_chain(symbol, snapshot_date, spot, cfg) -> pd.DataFrame` — pre-filter columns, `source="massive-backfill"`, from reference (as_of, expired both true and false) + per-contract daily aggregates; `open_interest` NaN (not available historically — the volume-only liquidity floor case)
  - Both raise `RuntimeError` when the API answers `status: "NOT_AUTHORIZED"`.

- [ ] **Step 1: Write fixtures**

`tests/fixtures/massive_snapshot_page.json` (one page, no next_url; two contracts on the 2026-09-18 monthly, one on a weekly that must be dropped by expiry selection):

```json
{
  "status": "OK",
  "results": [
    {
      "details": {"ticker": "O:SPY260918C00700000", "contract_type": "call",
                  "expiration_date": "2026-09-18", "strike_price": 700},
      "day": {"close": 71.3, "volume": 120},
      "open_interest": 5000,
      "implied_volatility": 0.213,
      "greeks": {"delta": 0.91}
    },
    {
      "details": {"ticker": "O:SPY260918P00700000", "contract_type": "put",
                  "expiration_date": "2026-09-18", "strike_price": 700},
      "day": {"close": 2.05, "volume": 80},
      "open_interest": 3100,
      "implied_volatility": 0.242
    },
    {
      "details": {"ticker": "O:SPY260904C00770000", "contract_type": "call",
                  "expiration_date": "2026-09-04", "strike_price": 770},
      "day": {"close": 5.1, "volume": 900},
      "open_interest": 7000,
      "implied_volatility": 0.11
    }
  ]
}
```

`tests/fixtures/massive_reference.json`:

```json
{
  "status": "OK",
  "results": [
    {"ticker": "O:SPY260918C00700000", "contract_type": "call",
     "expiration_date": "2026-09-18", "strike_price": 700, "underlying_ticker": "SPY"},
    {"ticker": "O:SPY260918P00700000", "contract_type": "put",
     "expiration_date": "2026-09-18", "strike_price": 700, "underlying_ticker": "SPY"},
    {"ticker": "O:SPY260904C00770000", "contract_type": "call",
     "expiration_date": "2026-09-04", "strike_price": 770, "underlying_ticker": "SPY"}
  ]
}
```

`tests/fixtures/massive_aggs.json` (response for one contract's single-day range):

```json
{
  "status": "OK",
  "resultsCount": 1,
  "results": [{"t": 1787342400000, "o": 70.9, "c": 71.3, "h": 71.5, "l": 70.2, "v": 120, "n": 12}]
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_massive_provider.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_massive_provider.py -v`
Expected: ImportError.

- [ ] **Step 4: Implement**

`src/data/massive_provider.py`:

```python
"""Massive (ex-Polygon) options: snapshot fallback + historical backfill.

Starter-tier facts (verified 2026-08-28, see LEARNING_LOG Phase 0):
- chain snapshot: day OHLC, open_interest, vendor IV/greeks on liquid
  strikes; NO bid/ask (higher tier) -> bid/ask/mid stored as NaN.
- historical: reference (as_of, both expired flags) + per-contract daily
  aggregates; no historical open interest -> NaN, volume-only liquidity.
- errors and paywalls arrive inside HTTP 200 bodies: check body["status"].
"""
import datetime as dt

import numpy as np
import pandas as pd
import requests

from src.data.filters import select_expiries

BASE = "https://api.polygon.io"
TIMEOUT = 30

PRE_FILTER_COLUMNS = ["expiry", "strike", "kind", "bid", "ask", "mid", "close",
                      "volume", "open_interest", "vendor_iv", "source"]


class MassiveProvider:
    def __init__(self, api_key: str, cfg: dict, session: requests.Session | None = None):
        self._key = api_key
        self._cfg = cfg
        self._session = session or requests.Session()

    def _get_json(self, url: str, **params):
        params.setdefault("apiKey", self._key)
        r = self._session.get(url, params=params, timeout=TIMEOUT)
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_massive_provider.py -v` then full suite.
Expected: all pass, no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/data/massive_provider.py tests/fixtures/massive_*.json tests/test_massive_provider.py
git commit -m "feat: Massive provider — paginated snapshot fallback and historical backfill chains"
```

---

### Task 5: Storage + daily orchestrator (the SPEC §4 fixture pipeline test)

**Files:**
- Create: `src/data/storage.py`, `src/run_daily.py`, `data/chains/.gitkeep`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `CHAIN_COLUMNS`, `UNDERLYING_COLUMNS`, `filter_chain`, the three providers' signatures (Tasks 1–4), `envfile.get_secret`.
- Produces:
  - `storage.write_chain(df, snapshot_date: dt.date, root: Path) -> Path` — validates `list(df.columns) == CHAIN_COLUMNS` (raises `ValueError` otherwise), atomic write to `data/chains/YYYY-MM-DD.parquet`
  - `storage.chain_path(snapshot_date, root) -> Path` and `storage.chain_exists(snapshot_date, root) -> bool`
  - `storage.upsert_underlying(df_new, root: Path) -> Path` — merge with existing `data/underlying.parquet` on `date` (new rows win), sorted, deduped
  - `storage.write_status(status: dict, root: Path) -> Path` — `docs/status.json`, pretty-printed
  - `run_daily.run(eodhd, live, fallback, cfg, root: Path, today: dt.date | None = None) -> dict` — the returned dict IS what was written to status.json
  - `run_daily.main()` — constructs real providers from secrets and calls `run` (the only place vendors are wired; SPEC §2.2)

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:

```python
"""SPEC 4 pipeline test: full daily run against fake providers, no network.

Also covers storage primitives: atomic chain write, underlying upsert,
schema validation, and the failure policy (no partial writes).
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import storage
from src.data.base import CHAIN_COLUMNS, UNDERLYING_COLUMNS
from src.run_daily import run

TODAY = dt.date(2026, 8, 28)


def real_cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def underlying_frame():
    return pd.DataFrame({
        "date": [dt.date(2026, 8, 27), TODAY],
        "close": [771.1, 770.0],
        "adjusted_close": [771.1, 770.0],
        "volume": [39000000, 40500000],
    })[UNDERLYING_COLUMNS]


def chain_frame(source):
    exp = dt.date(2026, 9, 18)
    quoted = source == "yfinance"
    return pd.DataFrame({
        "expiry": [exp, exp],
        "strike": [700.0, 700.0],
        "kind": ["call", "put"],
        "bid": [71.0, 1.9] if quoted else [np.nan] * 2,
        "ask": [71.4, 2.1] if quoted else [np.nan] * 2,
        "mid": [71.2, 2.0] if quoted else [np.nan] * 2,
        "close": [71.3, 2.05],
        "volume": [120, 80],
        "open_interest": [5000, 3100],
        "vendor_iv": [0.213, 0.242],
        "source": source,
    })


class FakeEODHD:
    def get_underlying_history(self, symbol, start=None):
        return underlying_frame()
    def get_risk_free_rate(self):
        return 0.0415
    def get_dividend_yield(self, spot, today=None, symbol="SPY"):
        return 0.0098


class FakeLive:
    fail = False
    def get_option_chain(self, symbol, snapshot_date, spot, cfg):
        if self.fail:
            raise RuntimeError("yahoo broke")
        return chain_frame("yfinance")


class FakeFallback:
    called = False
    def get_option_chain(self, symbol, snapshot_date, spot, cfg):
        self.called = True
        return chain_frame("massive-fallback")


class TestStorage:
    def test_chain_write_roundtrip_and_path(self, tmp_path):
        df = chain_frame("yfinance")
        df["snapshot_date"] = TODAY
        df["spot"] = 770.0
        df["dte"] = 21
        df = df[CHAIN_COLUMNS]
        p = storage.write_chain(df, TODAY, tmp_path)
        assert p == tmp_path / "data" / "chains" / "2026-08-28.parquet"
        assert storage.chain_exists(TODAY, tmp_path)
        back = pd.read_parquet(p)
        assert list(back.columns) == CHAIN_COLUMNS and len(back) == 2

    def test_chain_write_rejects_wrong_schema(self, tmp_path):
        with pytest.raises(ValueError):
            storage.write_chain(pd.DataFrame({"x": [1]}), TODAY, tmp_path)

    def test_underlying_upsert_dedupes_new_wins(self, tmp_path):
        storage.upsert_underlying(underlying_frame(), tmp_path)
        revised = underlying_frame()
        revised.loc[revised["date"] == TODAY, "close"] = 769.5
        p = storage.upsert_underlying(revised, tmp_path)
        back = pd.read_parquet(p)
        assert len(back) == 2
        assert back.loc[back["date"] == TODAY, "close"].iloc[0] == 769.5


class TestDailyRun:
    def test_happy_path_yfinance(self, tmp_path):
        fallback = FakeFallback()
        status = run(FakeEODHD(), FakeLive(), fallback, real_cfg(), tmp_path, today=TODAY)
        assert not fallback.called
        assert status["source"] == "yfinance"
        assert status["snapshot_date"] == "2026-08-28"
        assert status["spot"] == 770.0
        assert status["rows_stored"] == 2
        assert status["risk_free_rate"] == 0.0415
        assert status["dividend_yield"] == 0.0098
        assert storage.chain_exists(TODAY, tmp_path)
        on_disk = json.loads((tmp_path / "docs" / "status.json").read_text())
        assert on_disk == status
        stored = pd.read_parquet(storage.chain_path(TODAY, tmp_path))
        assert (stored["source"] == "yfinance").all()

    def test_fallback_on_live_failure(self, tmp_path):
        live = FakeLive(); live.fail = True
        fallback = FakeFallback()
        status = run(FakeEODHD(), live, fallback, real_cfg(), tmp_path, today=TODAY)
        assert fallback.called
        assert status["source"] == "massive-fallback"
        stored = pd.read_parquet(storage.chain_path(TODAY, tmp_path))
        assert (stored["source"] == "massive-fallback").all()

    def test_both_sources_failing_writes_nothing(self, tmp_path):
        live = FakeLive(); live.fail = True
        class DeadFallback:
            def get_option_chain(self, *a, **k):
                raise RuntimeError("also down")
        with pytest.raises(RuntimeError):
            run(FakeEODHD(), live, DeadFallback(), real_cfg(), tmp_path, today=TODAY)
        assert not storage.chain_exists(TODAY, tmp_path)
        assert not (tmp_path / "docs" / "status.json").exists()

    def test_empty_filtered_chain_is_a_failure(self, tmp_path):
        class EmptyLive:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return chain_frame("yfinance").iloc[0:0]
        class DeadFallback:
            def get_option_chain(self, *a, **k):
                raise RuntimeError("down")
        with pytest.raises(RuntimeError):
            run(FakeEODHD(), EmptyLive(), DeadFallback(), real_cfg(), tmp_path, today=TODAY)
        assert not storage.chain_exists(TODAY, tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/data/storage.py`:

```python
"""All disk writes for the data layer. Atomic (tmp + rename), schema-checked."""
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

from src.data.base import CHAIN_COLUMNS


def chain_path(snapshot_date: dt.date, root: Path) -> Path:
    return Path(root) / "data" / "chains" / f"{snapshot_date.isoformat()}.parquet"


def chain_exists(snapshot_date: dt.date, root: Path) -> bool:
    return chain_path(snapshot_date, root).exists()


def _atomic_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def write_chain(df: pd.DataFrame, snapshot_date: dt.date, root: Path) -> Path:
    if list(df.columns) != CHAIN_COLUMNS:
        raise ValueError(f"chain schema mismatch: {list(df.columns)} != {CHAIN_COLUMNS}")
    return _atomic_parquet(df, chain_path(snapshot_date, root))


def upsert_underlying(df_new: pd.DataFrame, root: Path) -> Path:
    path = Path(root) / "data" / "underlying.parquet"
    if path.exists():
        merged = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
    else:
        merged = df_new.copy()
    merged = (merged.drop_duplicates("date", keep="last")
                    .sort_values("date").reset_index(drop=True))
    return _atomic_parquet(merged, path)


def write_status(status: dict, root: Path) -> Path:
    path = Path(root) / "docs" / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, path)
    return path
```

`src/run_daily.py`:

```python
"""Daily orchestrator: fetch -> filter -> store -> status (SPEC 2.1/2.2).

The ONLY place vendors are composed. Failure policy: any unrecoverable
error raises, leaving no chain file and status.json untouched, so the
Action exits nonzero and yesterday's dashboard stays live.
"""
import datetime as dt
import sys
from pathlib import Path

import yaml

from src.data import storage
from src.data.filters import filter_chain


def run(eodhd, live, fallback, cfg: dict, root: Path, today: dt.date | None = None) -> dict:
    today = today or dt.date.today()
    symbol = cfg["symbol"]

    underlying = eodhd.get_underlying_history(symbol)
    storage.upsert_underlying(underlying, root)
    spot = float(underlying.sort_values("date")["close"].iloc[-1])

    try:
        raw = live.get_option_chain(symbol, today, spot, cfg)
        source = "yfinance"
    except Exception as live_err:
        print(f"live chain failed ({live_err!r}); falling back to Massive", file=sys.stderr)
        raw = fallback.get_option_chain(symbol, today, spot, cfg)
        source = "massive-fallback"

    chain = filter_chain(raw, spot, today, cfg)
    if chain.empty:
        raise RuntimeError(f"filtered chain is empty (source={source}) — refusing to store")
    storage.write_chain(chain, today, root)

    status = {
        "last_success_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": today.isoformat(),
        "source": source,
        "rows_stored": int(len(chain)),
        "spot": spot,
        "risk_free_rate": eodhd.get_risk_free_rate(),
        "dividend_yield": eodhd.get_dividend_yield(spot, today=today, symbol=symbol),
    }
    storage.write_status(status, root)
    return status


def main() -> None:
    from src.data.envfile import get_secret
    from src.data.eodhd_provider import EODHDProvider
    from src.data.massive_provider import MassiveProvider
    from src.data.yfinance_provider import YFinanceProvider

    root = Path(__file__).resolve().parent.parent
    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    eodhd = EODHDProvider(get_secret("EODHD_API_TOKEN", root / ".env"), cfg)
    live = YFinanceProvider(cfg)
    fallback = MassiveProvider(get_secret("MASSIVE_API_KEY", root / ".env"), cfg)
    status = run(eodhd, live, fallback, cfg, root)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    import json
    main()
```

(Note: `json` is used only in `main`; keep the local import as shown or move it to the top — either is fine, but the file must import cleanly under `pytest` without `yfinance` installed side effects. The top-level imports must remain exactly `datetime, sys, pathlib, yaml, storage, filter_chain` so the fixture pipeline test never touches vendor libraries.)

Create empty `data/chains/.gitkeep`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -v` then the full suite.
Expected: all pass, no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/data/storage.py src/run_daily.py data/chains/.gitkeep tests/test_pipeline.py
git commit -m "feat: storage layer and daily orchestrator with fallback + fixture pipeline test"
```

---

### Task 6: Resumable backfill + REAL runs (Phase 2 exit criterion)

**Files:**
- Create: `src/data/backfill.py`, `scripts/backfill.py`
- Modify: `LEARNING_LOG.md` (document real-run results)
- Test: `tests/test_backfill.py`

**Interfaces:**
- Consumes: `storage.chain_exists/write_chain/upsert_underlying`, `filter_chain`, `MassiveProvider.get_historical_chain`, `EODHDProvider.get_underlying_history`, `envfile.get_secret`.
- Produces:
  - `backfill.backfill(massive, eodhd, cfg, root: Path, start: dt.date, end: dt.date, log=print) -> dict` — returns `{"dates_total", "dates_skipped", "dates_written", "dates_empty", "rows_written"}`
  - `scripts/backfill.py` CLI: `--start YYYY-MM-DD --end YYYY-MM-DD` (defaults: end = yesterday, start = end − 730 days)

**Note for the executor:** Steps 6–7 hit real APIs and need `.env` + network (disable sandbox for those commands, as in prior real-run tasks). Everything before them is offline TDD.

- [ ] **Step 1: Write the failing tests**

`tests/test_backfill.py`:

```python
"""Backfill loop: trading-day iteration, resumability, empty-day handling."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import storage
from src.data.backfill import backfill
from src.data.base import UNDERLYING_COLUMNS

CFG_PATH = Path("config.yaml")


def cfg():
    import yaml
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


DATES = [dt.date(2026, 8, 24), dt.date(2026, 8, 25), dt.date(2026, 8, 26)]


class FakeEODHD:
    def get_underlying_history(self, symbol, start=None):
        return pd.DataFrame({
            "date": DATES,
            "close": [765.0, 768.0, 770.4],
            "adjusted_close": [765.0, 768.0, 770.4],
            "volume": [1, 1, 1],
        })[UNDERLYING_COLUMNS]


class FakeMassive:
    def __init__(self, empty_on=None):
        self.requested = []
        self.empty_on = empty_on or set()

    def get_historical_chain(self, symbol, snapshot_date, spot, cfg):
        self.requested.append((snapshot_date, spot))
        if snapshot_date in self.empty_on:
            return pd.DataFrame(columns=["expiry", "strike", "kind", "bid", "ask", "mid",
                                         "close", "volume", "open_interest", "vendor_iv", "source"])
        exp = dt.date(2026, 9, 18)
        return pd.DataFrame({
            "expiry": [exp], "strike": [700.0], "kind": ["call"],
            "bid": [np.nan], "ask": [np.nan], "mid": [np.nan],
            "close": [71.0], "volume": [10], "open_interest": [np.nan],
            "vendor_iv": [np.nan], "source": ["massive-backfill"],
        })


class TestBackfill:
    def test_iterates_trading_days_with_per_date_spot(self, tmp_path):
        m = FakeMassive()
        summary = backfill(m, FakeEODHD(), cfg(), tmp_path,
                           start=DATES[0], end=DATES[-1], log=lambda *_: None)
        assert [d for d, _ in m.requested] == DATES
        assert [s for _, s in m.requested] == [765.0, 768.0, 770.4]
        assert summary["dates_written"] == 3 and summary["dates_skipped"] == 0
        for d in DATES:
            assert storage.chain_exists(d, tmp_path)

    def test_resumability_skips_existing_files(self, tmp_path):
        m1 = FakeMassive()
        backfill(m1, FakeEODHD(), cfg(), tmp_path, start=DATES[0], end=DATES[0], log=lambda *_: None)
        m2 = FakeMassive()
        summary = backfill(m2, FakeEODHD(), cfg(), tmp_path,
                           start=DATES[0], end=DATES[-1], log=lambda *_: None)
        assert [d for d, _ in m2.requested] == DATES[1:]  # first date skipped
        assert summary["dates_skipped"] == 1 and summary["dates_written"] == 2

    def test_empty_day_recorded_not_written(self, tmp_path):
        m = FakeMassive(empty_on={DATES[1]})
        summary = backfill(m, FakeEODHD(), cfg(), tmp_path,
                           start=DATES[0], end=DATES[-1], log=lambda *_: None)
        assert summary["dates_empty"] == 1
        assert not storage.chain_exists(DATES[1], tmp_path)

    def test_date_window_respected(self, tmp_path):
        m = FakeMassive()
        backfill(m, FakeEODHD(), cfg(), tmp_path,
                 start=DATES[1], end=DATES[1], log=lambda *_: None)
        assert [d for d, _ in m.requested] == [DATES[1]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_backfill.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/data/backfill.py`:

```python
"""One-time (resumable) historical backfill from Massive (SPEC 2.2).

Resumability = chain files already on disk are skipped, so an interrupted
run (rate limit, ctrl-C, crash) just continues on the next invocation.
Trading days come from EODHD's underlying history -- no exchange-calendar
dependency needed.
"""
import datetime as dt
from pathlib import Path

from src.data import storage
from src.data.filters import filter_chain


def backfill(massive, eodhd, cfg: dict, root: Path,
             start: dt.date, end: dt.date, log=print) -> dict:
    symbol = cfg["symbol"]
    underlying = eodhd.get_underlying_history(symbol, start=start - dt.timedelta(days=7))
    storage.upsert_underlying(underlying, root)
    by_date = underlying.set_index("date")["close"].to_dict()
    days = [d for d in sorted(by_date) if start <= d <= end]

    summary = {"dates_total": len(days), "dates_skipped": 0,
               "dates_written": 0, "dates_empty": 0, "rows_written": 0}
    for day in days:
        if storage.chain_exists(day, root):
            summary["dates_skipped"] += 1
            continue
        spot = float(by_date[day])
        raw = massive.get_historical_chain(symbol, day, spot, cfg)
        chain = filter_chain(raw, spot, day, cfg)
        if chain.empty:
            summary["dates_empty"] += 1
            log(f"{day}: empty after filtering — nothing stored")
            continue
        storage.write_chain(chain, day, root)
        summary["dates_written"] += 1
        summary["rows_written"] += len(chain)
        log(f"{day}: {len(chain)} rows")
    return summary
```

`scripts/backfill.py`:

```python
"""CLI for the historical backfill. Usage:
    python scripts/backfill.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
Defaults: end = yesterday, start = end - 730 days. Re-run to resume.
"""
import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.data.backfill import backfill
from src.data.envfile import get_secret
from src.data.eodhd_provider import EODHDProvider
from src.data.massive_provider import MassiveProvider


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=dt.date.fromisoformat)
    ap.add_argument("--end", type=dt.date.fromisoformat)
    args = ap.parse_args()
    end = args.end or dt.date.today() - dt.timedelta(days=1)
    start = args.start or end - dt.timedelta(days=730)

    with open(root / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    eodhd = EODHDProvider(get_secret("EODHD_API_TOKEN", root / ".env"), cfg)
    massive = MassiveProvider(get_secret("MASSIVE_API_KEY", root / ".env"), cfg)
    summary = backfill(massive, eodhd, cfg, root, start, end)
    print(summary)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_backfill.py -v` then the full suite.
Expected: all pass, no warnings.

- [ ] **Step 5: Commit the code**

```bash
git add src/data/backfill.py scripts/backfill.py tests/test_backfill.py
git commit -m "feat: resumable Massive backfill with per-date spot and skip-existing resume"
```

- [ ] **Step 6: REAL RUN — one daily snapshot (Phase 2 exit criterion, part 1)**

Run (network + `.env` required): `.venv/bin/python -m src.run_daily`
Expected: prints a status dict with `"source": "yfinance"` (or `"massive-fallback"` if Yahoo is having a day — either satisfies the criterion), `rows_stored` in the hundreds; `data/chains/<today>.parquet`, `data/underlying.parquet`, and `docs/status.json` exist. Sanity-check the stored file: `.venv/bin/python -c "import pandas as pd, glob; df = pd.read_parquet(sorted(glob.glob('data/chains/*.parquet'))[-1]); print(df['source'].unique(), len(df), df['expiry'].nunique())"` — expect 1 source, 6 expiries.

- [ ] **Step 7: REAL RUN — backfill attempt (Phase 2 exit criterion, part 2)**

Start with a deliberately small window to gauge duration and behavior:
`.venv/bin/python scripts/backfill.py --start <today-minus-14d> --end <yesterday>`
Then, if the per-day timing is acceptable (expect minutes per day; the full 730-day run is hours and can be resumed anytime), launch the wider run — or leave the wide run for the user to run overnight and note that in the report. Record actual depth achieved: re-run with an old `--start` (e.g. 2 years back) for ONE day (`--start X --end X`) to find where Massive data runs out.

- [ ] **Step 8: Document the runs in LEARNING_LOG.md**

Append a dated entry **"Phase 2: first real data"** under `## Entries`: which source served the first live snapshot, rows/expiries stored, how long a backfill day takes, the deepest date Massive returned data for, and anything surprising (rate-limit messages, empty days, filter kill-rates). Honest numbers, not summaries.

- [ ] **Step 9: Commit data + log**

```bash
git add data/ docs/status.json LEARNING_LOG.md
git commit -m "feat: first real daily snapshot + backfill run; Phase 2 exit criteria documented"
```

---

## Self-Review Notes (already applied)

- SPEC §2.2 coverage: providers (T2–T4), filtering (T1), storage/parquet (T5), source labeling (T1/T3/T4), canonical-spot decision (header), backfill + resumability (T6), status.json (T5), failure policy (T5 tests), fixture-only CI tests (all). `daily_metrics.parquet` is **deliberately absent** — SPEC assigns its computation to the analytics layer (Phase 4+); nothing in Phase 2 produces metrics.
- The `run_daily.py` local `import json` note exists because the pipeline test must import the module without vendor libraries loading at import time; implementers should keep vendor imports inside `main()`.
- Type consistency: `PRE_FILTER_COLUMNS` is declared identically in Tasks 3 and 4 (deliberate duplication — vendor modules must not import each other; `filter_chain` re-validates by selecting `CHAIN_COLUMNS`).
- GitHub Actions `daily.yml` is Phase 7 per SPEC §5 — not in this plan.
