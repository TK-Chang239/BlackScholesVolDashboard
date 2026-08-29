# vol-lens Phase 4 — Realized Vol, Sensitivity & Greeks Panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first *time-series* layer (`data/daily_metrics.parquet`, realized vol) and render three new panels from real stored history — P5 implied-vs-realized vol (Q3), P1 input sensitivity (Q1), P4 Greeks (Q1) — so the dashboard answers Q1 and Q3 with evidence, not placeholders.

**Architecture:** A hand-built `realized_vol` module (trailing and forward, on adjusted close) joins the model layer. A new `daily_metrics` analytics module produces one row per session (30-DTE ATM IV interpolated from the P3 term structure, convergence, source) and refreshes the realized-vol columns from the full underlying history on every run so forward RV back-fills as days mature. A one-time, re-runnable rebuild script replays every stored chain through the same functions so P5 starts populated (same stateless philosophy as P8). `sensitivity` and `greeks_panel` are pure per-session analytics over the existing `chain_iv` frame. The render layer gains a tornado, a two-subplot Greeks curve figure, an HTML stat-tile block, and the IV-vs-RV chart; `run_daily` wires it all in with the same compute-everything-then-write-everything ordering (chain → daily_metrics → page → status).

**Tech Stack:** NumPy/pandas · existing `src.models.black_scholes` (`bs_price`, `greeks`, `implied_vol`) · plotly (`make_subplots`) · pyarrow parquet · pytest (offline, warnings-as-errors) · `.venv/bin/python`, `.venv/bin/pytest`

**Spec:** `SPEC.md` — §2.2 (`daily_metrics` storage), §2.3 (`realized_vol.py`; Greeks convention: raw derivatives in the model, display scaling in render), §2.4 (pure analytics), §2.5 (page/captions/mobile), §3 P1/P4/P5 (binding), §2.1 failure policy, §5 Phase 4 exit criterion: **"Panels render from real stored history."**

## Global Constraints

- No pricing libraries. Realized vol, sensitivity, and Greeks all come from this repo's own code (`src/models/`). Vendor IV/Greeks are never read (SPEC §2.2/§2.3).
- Realized vol: close-to-close **log** returns on **`adjusted_close`**, sample std (`ddof=1`), annualized by `sqrt(cfg["realized_vol"]["annualization_days"])` (=252); windows from `cfg["realized_vol"]["windows"]` (=[20, 60]); forward horizon `cfg["realized_vol"]["forward_horizon_days"]` (=30, **calendar** days, added to `config.yaml` in Task 1) (SPEC §2.3, §3 P5).
- Greeks convention (SPEC §2.3): the model layer returns raw calculus derivatives; **the render layer** divides theta by 365 (per day), vega by 100 (per vol point), rho by 100 (per 1%). Analytics frames carry raw values.
- `T = dte / 365.0` everywhere in analytics (Phase 3 constraint, unchanged).
- Analytics functions are pure `(DataFrame(s), cfg) → tidy DataFrame`, no I/O (SPEC §2.4). Disk access lives in `src/data/storage.py` (parquet) and `src/data/metrics_backfill.py` (replay).
- `daily_metrics.parquet`: one row per session keyed by `date`; upserts dedupe on `date` keeping the newest row (mirrors `upsert_underlying`). IV-side columns are per-session facts; RV-side columns are recomputed for **every** row on every run (SPEC §3 P5 "back-filled as days mature").
- `r` and `q` used for replaying historical chains (rebuild script, prior-day overlays, 1-day Greek changes) are the *current* run's values — documented approximation, same as Phase 3's overlay constraint. P1 exists precisely to show it is immaterial.
- Failure policy (SPEC §2.1): ALL computation completes before ANY write; write order **chain → daily_metrics → index.html → status.json**, each atomic. Any failure leaves the previous page, metrics, and status live.
- One self-contained `docs/index.html` < 5 MB; no external requests; mobile-readable; every panel gets its 2–3 sentence caption (texts in Task 6 — use verbatim) (SPEC §2.5).
- Section placement (SPEC §2.5): **Q1 → P1, P4**; **Q3 → P5**; Q2 keeps P2/P3; Q4/Q5 keep their "arrives in Phase N" lines.
- P1 deviation, deliberately: SPEC §3 P1 says ±20% on every input and "σ visibly dominates". A ±20% *spot* bump on a 30-DTE ATM call moves the price by an order of magnitude more than σ does (it is a crash, not a disagreement). The computation follows the spec exactly (all five inputs, ±20%); the chart splits the inputs into two subplots with independent axes — "inputs the market argues about" (σ, r, q) and "inputs everyone observes" (S, T) — so σ visibly dominates *its* group and the caption states why S is excluded from the argument. Do not "fix" this by shrinking the S bump.
- pytest offline, warnings-as-errors; TDD per task; commit per task. Run the full suite before each commit: `.venv/bin/pytest -q`.

---

## File Structure

```
config.yaml                          # Task 1 — add realized_vol.forward_horizon_days, sensitivity.bump
src/models/realized_vol.py           # Task 1 — trailing + forward realized vol (hand-built)
src/analytics/daily_metrics.py       # Task 2 — session row, ATM-IV interpolation in DTE, RV refresh, upsert
src/data/storage.py                  # Task 2 — modify: read/write daily_metrics, read_underlying
src/data/metrics_backfill.py         # Task 3 — replay stored chains -> daily_metrics (I/O)
scripts/rebuild_daily_metrics.py     # Task 3 — thin CLI over metrics_backfill
src/analytics/sensitivity.py         # Task 4 — P1 ±bump table
src/analytics/greeks_panel.py        # Task 5 — P4 tiles + delta/gamma curves
src/analytics/iv_rv.py               # Task 6 — P5 tidy series + summary stat
src/render/figures.py                # Tasks 4–6 — modify: tornado, greeks curves, IV-vs-RV figures
src/render/tiles.py                  # Task 5 — Greek stat tiles HTML (display scaling lives here)
src/render/page.py                   # Task 6 — modify: Q1/Q3 wiring, captions, extras slot, GitHub link
src/run_daily.py                     # Task 7 — modify: metrics + three panels, write order
tests/test_realized_vol.py           # Task 1
tests/test_daily_metrics.py          # Task 2
tests/test_metrics_backfill.py       # Task 3
tests/test_sensitivity.py            # Task 4
tests/test_greeks_panel.py           # Task 5
tests/test_iv_rv.py                  # Task 6
tests/test_render.py                 # Tasks 4–6 — extend
tests/test_pipeline.py               # Task 7 — extend
LEARNING_LOG.md                      # Task 8 — teaching entries
```

Test convention (unchanged from Phase 3): synthetic chains are built through `bs_price` with *known* vols so recovery is the assertion; each test file defines its own fixture helper (deliberate duplication — files stay independently runnable).

---

### Task 1: `realized_vol` — trailing and forward realized volatility

**Files:**
- Create: `src/models/realized_vol.py`
- Modify: `config.yaml` (add two keys)
- Test: `tests/test_realized_vol.py`

**Interfaces:**
- Consumes: underlying frames with `UNDERLYING_COLUMNS = ["date", "close", "adjusted_close", "volume"]` (`date` is a `datetime.date`).
- Produces:
  - `log_returns(adjusted_close: pd.Series) -> pd.Series` (first element NaN).
  - `trailing_realized_vol(underlying: pd.DataFrame, window: int, annualization_days: int = 252) -> pd.Series` — indexed by `date`, name `f"rv_{window}d"`, NaN until `window` returns exist.
  - `forward_realized_vol(underlying: pd.DataFrame, horizon_days: int = 30, annualization_days: int = 252) -> pd.Series` — indexed by `date`, name `f"fwd_rv_{horizon_days}d"`; value at date t = annualized std of log returns on dates in `(t, t + horizon_days]` (calendar days); NaN when `t + horizon_days` is after the last available date (not matured) or fewer than `MIN_FORWARD_RETURNS = 5` returns fall in the window.

- [ ] **Step 1: Add config keys**

In `config.yaml`, change the `realized_vol` block and add a `sensitivity` block:

```yaml
realized_vol:                   # SPEC §2.3, §3 P5
  windows: [20, 60]
  annualization_days: 252
  forward_horizon_days: 30      # P5 forward RV: vol realized over (t, t+30 calendar days]

sensitivity:                    # SPEC §3 P1
  bump: 0.20                    # ±20% perturbation of each input
```

- [ ] **Step 2: Write the failing tests**

`tests/test_realized_vol.py`:

```python
"""Realized vol: log returns on adjusted close, sample std, sqrt(252)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.models.realized_vol import (
    MIN_FORWARD_RETURNS, forward_realized_vol, log_returns, trailing_realized_vol,
)

START = dt.date(2026, 1, 5)


def business_days(n):
    return list(pd.bdate_range(START, periods=n).date)


def frame(adjusted, close=None):
    dates = business_days(len(adjusted))
    close = list(adjusted) if close is None else close
    return pd.DataFrame({"date": dates, "close": close,
                         "adjusted_close": list(adjusted),
                         "volume": [1_000_000] * len(adjusted)})


class TestLogReturns:
    def test_first_is_nan_then_log_ratio(self):
        r = log_returns(pd.Series([100.0, 110.0, 99.0]))
        assert np.isnan(r.iloc[0])
        assert r.iloc[1] == pytest.approx(np.log(1.1))
        assert r.iloc[2] == pytest.approx(np.log(0.9))


class TestTrailing:
    def test_constant_price_is_zero_vol(self):
        rv = trailing_realized_vol(frame([100.0] * 30), window=20)
        assert rv.name == "rv_20d"
        assert rv.iloc[:20].isna().all()          # needs 20 returns = 21 closes
        assert np.allclose(rv.iloc[20:], 0.0)

    def test_alternating_returns_match_closed_form(self):
        x = 0.01
        prices = [100.0]
        for i in range(40):
            prices.append(prices[-1] * np.exp(x if i % 2 == 0 else -x))
        rv = trailing_realized_vol(frame(prices), window=20, annualization_days=252)
        # 20 returns of +x/-x: mean 0, sample var = 20x^2/19
        expected = x * np.sqrt(20 / 19) * np.sqrt(252)
        assert rv.iloc[-1] == pytest.approx(expected, rel=1e-9)

    def test_uses_adjusted_close_not_close(self):
        # ex-dividend day: raw close drops 2%, adjusted series is flat
        close = [100.0] * 15 + [98.0] * 15
        rv = trailing_realized_vol(frame([100.0] * 30, close=close), window=20)
        assert np.allclose(rv.dropna(), 0.0)

    def test_indexed_by_date_and_sorted(self):
        df = frame([100.0, 101.0, 102.0, 101.0, 103.0]).iloc[::-1]  # reversed input
        rv = trailing_realized_vol(df, window=2)
        assert list(rv.index) == business_days(5)
        assert np.isnan(rv.iloc[0]) and np.isnan(rv.iloc[1])
        assert not np.isnan(rv.iloc[2])


class TestForward:
    def test_matured_dates_only(self):
        df = frame(np.linspace(100.0, 120.0, 60))
        fwd = forward_realized_vol(df, horizon_days=30)
        assert fwd.name == "fwd_rv_30d"
        last = df["date"].iloc[-1]
        for date, val in fwd.items():
            matured = (date + dt.timedelta(days=30)) <= last
            assert (not np.isnan(val)) == matured, date

    def test_value_matches_hand_computation(self):
        rng = np.random.default_rng(7)
        prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 80)))
        df = frame(prices)
        fwd = forward_realized_vol(df, horizon_days=30, annualization_days=252)
        t = df["date"].iloc[5]
        end = t + dt.timedelta(days=30)
        sub = df[(df["date"] > t) & (df["date"] <= end)]
        # returns *into* those dates: include the close on t as the base
        base = df[df["date"] == t]["adjusted_close"].iloc[0]
        series = np.concatenate([[base], sub["adjusted_close"].to_numpy()])
        rets = np.diff(np.log(series))
        expected = np.std(rets, ddof=1) * np.sqrt(252)
        assert fwd.loc[t] == pytest.approx(expected, rel=1e-9)

    def test_too_few_returns_is_nan(self):
        # 3-day horizon holds at most 3 returns < MIN_FORWARD_RETURNS
        assert MIN_FORWARD_RETURNS > 3
        fwd = forward_realized_vol(frame([100.0 + i for i in range(40)]), horizon_days=3)
        assert fwd.isna().all()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_realized_vol.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.realized_vol'`

- [ ] **Step 4: Implement**

`src/models/realized_vol.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_realized_vol.py -q`
Expected: PASS (all 8). Then `.venv/bin/pytest -q` — everything green (the config change adds keys only).

- [ ] **Step 6: Commit**

```bash
git add config.yaml src/models/realized_vol.py tests/test_realized_vol.py
git commit -m "feat: hand-built realized vol — trailing and forward, on adjusted close"
```

---

### Task 2: `daily_metrics` analytics + parquet storage

**Files:**
- Create: `src/analytics/daily_metrics.py`
- Modify: `src/data/storage.py` (add `read_underlying`, `daily_metrics_path`, `read_daily_metrics`, `write_daily_metrics`)
- Test: `tests/test_daily_metrics.py`

**Interfaces:**
- Consumes: `compute_term_structure` output (`TERM_COLUMNS = ["expiry", "dte", "atm_iv"]`), `compute_chain_iv` stats dict, Task 1 functions.
- Produces (`src/analytics/daily_metrics.py`):
  - `IV_COLUMNS = ["date", "spot", "source", "atm_iv_30d", "atm_iv_30d_dte", "iv_convergence"]`
  - `rv_columns(cfg) -> list[str]` = `[f"rv_{w}d" for w in windows] + [f"fwd_rv_{h}d"]` → with the shipped config `["rv_20d", "rv_60d", "fwd_rv_30d"]`
  - `metric_columns(cfg) -> list[str]` = `IV_COLUMNS + rv_columns(cfg)`
  - `interp_atm_iv(term: pd.DataFrame, target_dte: int) -> tuple[float, float]` — `(atm_iv, effective_dte)`; linear in DTE when bracketed, nearest expiry when within `ATM_NEAREST_TOLERANCE_DAYS = 10`, else `(nan, nan)`.
  - `session_metrics_row(session_date, spot, source, term, iv_stats, cfg) -> dict` with keys `IV_COLUMNS`.
  - `upsert_session(metrics: pd.DataFrame, row: dict, cfg) -> pd.DataFrame` — dedupe on `date`, newest wins, sorted by date, columns `metric_columns(cfg)` (RV columns NaN for the new row until refreshed).
  - `refresh_rv_columns(metrics: pd.DataFrame, underlying: pd.DataFrame, cfg) -> pd.DataFrame` — recomputes every RV column for every row from the underlying.
- Produces (`src/data/storage.py`):
  - `read_underlying(root) -> pd.DataFrame`
  - `daily_metrics_path(root) -> Path` = `root/data/daily_metrics.parquet`
  - `read_daily_metrics(root, columns: list[str]) -> pd.DataFrame` — empty frame with `columns` if the file is missing; if present, missing columns are added as NaN (schema forward-compat for Phase 5's skew column).
  - `write_daily_metrics(df, root) -> Path` — atomic; raises `ValueError` if `"date"` is not a column or has duplicates.

- [ ] **Step 1: Write the failing tests**

`tests/test_daily_metrics.py`:

```python
"""daily_metrics: ATM-IV-in-DTE interpolation, session rows, RV refresh, upsert."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analytics.daily_metrics import (
    IV_COLUMNS, interp_atm_iv, metric_columns, refresh_rv_columns,
    rv_columns, session_metrics_row, upsert_session,
)
from src.data import storage


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def term(rows):
    return pd.DataFrame(rows, columns=["expiry", "dte", "atm_iv"])


def underlying(n=80, start=dt.date(2026, 5, 1)):
    dates = list(pd.bdate_range(start, periods=n).date)
    rng = np.random.default_rng(3)
    px = 700.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"date": dates, "close": px, "adjusted_close": px,
                         "volume": [1] * n})


class TestColumns:
    def test_shipped_config_names(self):
        assert rv_columns(cfg()) == ["rv_20d", "rv_60d", "fwd_rv_30d"]
        assert metric_columns(cfg()) == IV_COLUMNS + ["rv_20d", "rv_60d", "fwd_rv_30d"]


class TestInterpAtmIv:
    def test_bracketed_is_linear_in_dte(self):
        t = term([(dt.date(2026, 9, 18), 20, 0.10), (dt.date(2026, 10, 16), 40, 0.20)])
        iv, eff = interp_atm_iv(t, 30)
        assert iv == pytest.approx(0.15) and eff == 30

    def test_exact_expiry_returns_it(self):
        t = term([(dt.date(2026, 9, 18), 30, 0.13)])
        assert interp_atm_iv(t, 30) == (pytest.approx(0.13), 30)

    def test_nearest_within_tolerance(self):
        t = term([(dt.date(2026, 9, 18), 22, 0.11), (dt.date(2026, 11, 20), 85, 0.14)])
        iv, eff = interp_atm_iv(t, 30)     # bracketed -> interpolate, not nearest
        assert 0.11 < iv < 0.14 and eff == 30
        t2 = term([(dt.date(2026, 9, 25), 38, 0.12)])
        assert interp_atm_iv(t2, 30) == (pytest.approx(0.12), 38)

    def test_out_of_range_is_nan(self):
        t = term([(dt.date(2026, 11, 20), 85, 0.14)])
        iv, eff = interp_atm_iv(t, 30)
        assert np.isnan(iv) and np.isnan(eff)
        iv, eff = interp_atm_iv(term([]), 30)
        assert np.isnan(iv) and np.isnan(eff)

    def test_nan_atm_iv_rows_ignored(self):
        t = term([(dt.date(2026, 9, 18), 20, np.nan), (dt.date(2026, 10, 16), 40, 0.20)])
        iv, eff = interp_atm_iv(t, 30)
        assert iv == pytest.approx(0.20) and eff == 40


class TestSessionRow:
    def test_row_keys_and_values(self):
        t = term([(dt.date(2026, 9, 18), 20, 0.10), (dt.date(2026, 10, 16), 40, 0.20)])
        row = session_metrics_row(dt.date(2026, 8, 28), 771.1, "yfinance", t,
                                  {"convergence": 0.993}, cfg())
        assert list(row) == IV_COLUMNS
        assert row["date"] == dt.date(2026, 8, 28)
        assert row["atm_iv_30d"] == pytest.approx(0.15)
        assert row["atm_iv_30d_dte"] == 30
        assert row["iv_convergence"] == 0.993 and row["source"] == "yfinance"


class TestUpsert:
    def test_new_row_appends_with_nan_rv(self):
        m = pd.DataFrame(columns=metric_columns(cfg()))
        row = session_metrics_row(dt.date(2026, 8, 28), 771.1, "yfinance",
                                  term([]), {"convergence": 0.0}, cfg())
        out = upsert_session(m, row, cfg())
        assert list(out.columns) == metric_columns(cfg()) and len(out) == 1
        assert out["rv_20d"].isna().all()

    def test_same_date_replaces_and_sorts(self):
        m = pd.DataFrame(columns=metric_columns(cfg()))
        for d, s in ((dt.date(2026, 8, 28), 771.1), (dt.date(2026, 8, 27), 766.0),
                     (dt.date(2026, 8, 28), 772.2)):
            m = upsert_session(m, session_metrics_row(d, s, "yfinance", term([]),
                                                      {"convergence": 1.0}, cfg()), cfg())
        assert list(m["date"]) == [dt.date(2026, 8, 27), dt.date(2026, 8, 28)]
        assert m["spot"].iloc[-1] == 772.2


class TestRefreshRv:
    def test_fills_rv_for_every_row_and_matures_forward(self):
        u = underlying(n=80)
        m = pd.DataFrame(columns=metric_columns(cfg()))
        early, late = u["date"].iloc[30], u["date"].iloc[-1]
        for d in (early, late):
            m = upsert_session(m, session_metrics_row(d, 700.0, "yfinance", term([]),
                                                      {"convergence": 1.0}, cfg()), cfg())
        out = refresh_rv_columns(m, u, cfg())
        assert list(out.columns) == metric_columns(cfg())
        assert out["rv_20d"].notna().all() and out["rv_60d"].isna().iloc[0]  # 30 < 60 returns
        assert out["fwd_rv_30d"].notna().iloc[0]      # early row has matured
        assert out["fwd_rv_30d"].isna().iloc[1]       # last row cannot have

    def test_refresh_is_idempotent_and_overwrites_stale(self):
        u = underlying(n=80)
        m = pd.DataFrame(columns=metric_columns(cfg()))
        m = upsert_session(m, session_metrics_row(u["date"].iloc[30], 700.0, "yfinance",
                                                  term([]), {"convergence": 1.0}, cfg()), cfg())
        m.loc[0, "rv_20d"] = 9.99                      # stale/wrong value
        once = refresh_rv_columns(m, u, cfg())
        twice = refresh_rv_columns(once, u, cfg())
        assert once["rv_20d"].iloc[0] != 9.99
        pd.testing.assert_frame_equal(once, twice)


class TestStorage:
    def test_read_missing_gives_empty_with_columns(self, tmp_path):
        df = storage.read_daily_metrics(tmp_path, metric_columns(cfg()))
        assert df.empty and list(df.columns) == metric_columns(cfg())

    def test_write_read_roundtrip_adds_missing_columns(self, tmp_path):
        m = pd.DataFrame({"date": [dt.date(2026, 8, 28)], "spot": [771.1]})
        p = storage.write_daily_metrics(m, tmp_path)
        assert p == tmp_path / "data" / "daily_metrics.parquet"
        back = storage.read_daily_metrics(tmp_path, metric_columns(cfg()))
        assert list(back.columns) == metric_columns(cfg())
        assert back["date"].iloc[0] == dt.date(2026, 8, 28)
        assert back["rv_20d"].isna().all()

    def test_write_rejects_duplicate_dates(self, tmp_path):
        m = pd.DataFrame({"date": [dt.date(2026, 8, 28)] * 2, "spot": [1.0, 2.0]})
        with pytest.raises(ValueError):
            storage.write_daily_metrics(m, tmp_path)

    def test_read_underlying(self, tmp_path):
        storage.upsert_underlying(underlying(n=5), tmp_path)
        assert len(storage.read_underlying(tmp_path)) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_daily_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.analytics.daily_metrics'`

- [ ] **Step 3: Implement the analytics module**

`src/analytics/daily_metrics.py`:

```python
"""daily_metrics: one row per session (SPEC 2.2 storage, 3 P5). Pure.

IV-side columns are facts about that session's chain, computed once.
RV-side columns are recomputed for EVERY row from the full underlying
history on every run, so forward RV back-fills as days mature. Parquet
I/O lives in src.data.storage; replay of stored chains in
src.data.metrics_backfill.
"""
import datetime as dt

import numpy as np
import pandas as pd

from src.models.realized_vol import forward_realized_vol, trailing_realized_vol

IV_COLUMNS = ["date", "spot", "source", "atm_iv_30d", "atm_iv_30d_dte", "iv_convergence"]
ATM_NEAREST_TOLERANCE_DAYS = 10


def rv_columns(cfg: dict) -> list[str]:
    rv = cfg["realized_vol"]
    return [f"rv_{w}d" for w in rv["windows"]] + [f"fwd_rv_{rv['forward_horizon_days']}d"]


def metric_columns(cfg: dict) -> list[str]:
    return IV_COLUMNS + rv_columns(cfg)


def interp_atm_iv(term: pd.DataFrame, target_dte: int) -> tuple[float, float]:
    """ATM IV at `target_dte`: linear in DTE when bracketed by stored
    expiries, the nearest expiry when within tolerance, else NaN. Returns
    (atm_iv, effective_dte) so callers can label what they actually got."""
    t = term.dropna(subset=["atm_iv"]).sort_values("dte") if len(term) else term
    if t is None or t.empty:
        return (np.nan, np.nan)
    dtes = t["dte"].to_numpy(dtype=float)
    ivs = t["atm_iv"].to_numpy(dtype=float)
    if dtes.min() <= target_dte <= dtes.max():
        return float(np.interp(target_dte, dtes, ivs)), float(target_dte)
    j = int(np.argmin(np.abs(dtes - target_dte)))
    if abs(dtes[j] - target_dte) <= ATM_NEAREST_TOLERANCE_DAYS:
        return float(ivs[j]), float(dtes[j])
    return (np.nan, np.nan)


def session_metrics_row(session_date: dt.date, spot: float, source: str,
                        term: pd.DataFrame, iv_stats: dict, cfg: dict) -> dict:
    atm_iv, eff_dte = interp_atm_iv(term, int(cfg["target_dte"]["atm_panel"]))
    return {
        "date": session_date, "spot": float(spot), "source": source,
        "atm_iv_30d": atm_iv, "atm_iv_30d_dte": eff_dte,
        "iv_convergence": float(iv_stats["convergence"]),
    }


def upsert_session(metrics: pd.DataFrame, row: dict, cfg: dict) -> pd.DataFrame:
    cols = metric_columns(cfg)
    new = pd.DataFrame([row]).reindex(columns=cols)
    base = metrics.reindex(columns=cols)
    merged = pd.concat([base, new], ignore_index=True) if len(base) else new
    return (merged.drop_duplicates("date", keep="last")
                  .sort_values("date").reset_index(drop=True)[cols])


def refresh_rv_columns(metrics: pd.DataFrame, underlying: pd.DataFrame,
                       cfg: dict) -> pd.DataFrame:
    rv_cfg = cfg["realized_vol"]
    ann = rv_cfg["annualization_days"]
    series = [trailing_realized_vol(underlying, w, ann) for w in rv_cfg["windows"]]
    series.append(forward_realized_vol(underlying, rv_cfg["forward_horizon_days"], ann))
    rv = pd.concat(series, axis=1).reset_index()          # date + rv columns
    out = metrics.drop(columns=rv_columns(cfg), errors="ignore")
    out = out.merge(rv, on="date", how="left")
    return out[metric_columns(cfg)].sort_values("date").reset_index(drop=True)
```

- [ ] **Step 4: Add the storage functions**

Append to `src/data/storage.py` (after `upsert_underlying`):

```python
def read_underlying(root: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "data" / "underlying.parquet")


def daily_metrics_path(root: Path) -> Path:
    return Path(root) / "data" / "daily_metrics.parquet"


def read_daily_metrics(root: Path, columns: list[str]) -> pd.DataFrame:
    path = daily_metrics_path(root)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path).reindex(columns=columns)


def write_daily_metrics(df: pd.DataFrame, root: Path) -> Path:
    if "date" not in df.columns:
        raise ValueError("daily_metrics needs a 'date' column")
    if df["date"].duplicated().any():
        raise ValueError("daily_metrics has duplicate dates")
    return _atomic_parquet(df.sort_values("date").reset_index(drop=True),
                           daily_metrics_path(root))
```

Note: `reindex(columns=...)` on the read path is what makes Phase 5's future `skew_25d` column a non-event — old files gain NaN columns, never a schema error.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_daily_metrics.py -q` — PASS (14). Then `.venv/bin/pytest -q` — green.

If `refresh_rv_columns`'s merge fails on dtype (`date` as `object` of `datetime.date` on one side vs `datetime64` on the other), coerce both sides with `pd.to_datetime(...).dt.date` before the merge — the underlying parquet stores `datetime.date` objects (verified in `data/underlying.parquet`), and `pd.Index(df["date"])` preserves them, so this should not be needed; do not convert the stored `date` column to `datetime64`.

- [ ] **Step 6: Commit**

```bash
git add src/analytics/daily_metrics.py src/data/storage.py tests/test_daily_metrics.py
git commit -m "feat: daily_metrics — 30-DTE ATM IV row per session, RV columns refreshed from history"
```

---

### Task 3: Rebuild `daily_metrics` from stored chains (replay)

**Files:**
- Create: `src/data/metrics_backfill.py`, `scripts/rebuild_daily_metrics.py`
- Test: `tests/test_metrics_backfill.py`

**Interfaces:**
- Consumes: `storage.chain_path`/`read_underlying`/`write_daily_metrics`; `compute_chain_iv(chain, r, q)`; `compute_term_structure(chain_iv)`; Task 2 functions.
- Produces: `rebuild_daily_metrics(root: Path, r: float, q: float, cfg: dict, write: bool = True) -> pd.DataFrame` — replays every `data/chains/*.parquet` (sorted by filename = session date) into a fresh metrics frame, refreshes RV from `data/underlying.parquet`, writes it when `write=True`, returns it.
- Source labeling rule: a chain's `source` is `"yfinance"` if every row is yfinance, otherwise the modal source string (backfill days are uniform anyway).

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics_backfill.py`:

```python
"""Replay stored chains -> daily_metrics.parquet (stateless rebuild)."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import storage
from src.data.base import CHAIN_COLUMNS
from src.data.metrics_backfill import rebuild_daily_metrics
from src.models.black_scholes import bs_price

R, Q = 0.0415, 0.0098


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def chain(session, spot, sigma, source, expiries=((30, 0), (60, 0))):
    rows = []
    for dte, _ in expiries:
        expiry = session + dt.timedelta(days=dte)
        for strike in (spot - 20.0, spot - 10.0, spot, spot + 10.0, spot + 20.0):
            for kind in ("call", "put"):
                px = float(bs_price(spot, strike, dte / 365.0, R, sigma, Q, kind))
                live = source == "yfinance"
                rows.append({
                    "snapshot_date": session, "spot": spot, "expiry": expiry, "dte": dte,
                    "strike": strike, "kind": kind,
                    "bid": px - 0.05 if live else np.nan, "ask": px + 0.05 if live else np.nan,
                    "mid": px if live else np.nan, "close": px,
                    "volume": 10, "open_interest": 100.0, "vendor_iv": np.nan,
                    "source": source})
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


def seed(root):
    dates = list(pd.bdate_range(dt.date(2026, 6, 1), periods=70).date)
    px = np.linspace(740.0, 770.0, 70)
    storage.upsert_underlying(pd.DataFrame({"date": dates, "close": px,
                                            "adjusted_close": px, "volume": [1] * 70}), root)
    s1, s2 = dates[10], dates[-1]
    storage.write_chain(chain(s1, float(px[10]), 0.18, "massive-backfill"), s1, root)
    storage.write_chain(chain(s2, float(px[-1]), 0.24, "yfinance"), s2, root)
    return s1, s2


class TestRebuild:
    def test_one_row_per_chain_with_recovered_atm_iv(self, tmp_path):
        s1, s2 = seed(tmp_path)
        out = rebuild_daily_metrics(tmp_path, R, Q, cfg())
        assert list(out["date"]) == [s1, s2]
        assert out["atm_iv_30d"].to_numpy() == pytest.approx([0.18, 0.24], abs=1e-5)
        assert list(out["source"]) == ["massive-backfill", "yfinance"]
        assert out["iv_convergence"].to_numpy() == pytest.approx([1.0, 1.0])
        assert out["rv_20d"].notna().iloc[1]
        assert out["fwd_rv_30d"].notna().iloc[0] and out["fwd_rv_30d"].isna().iloc[1]
        assert storage.daily_metrics_path(tmp_path).exists()

    def test_write_false_leaves_disk_untouched(self, tmp_path):
        seed(tmp_path)
        rebuild_daily_metrics(tmp_path, R, Q, cfg(), write=False)
        assert not storage.daily_metrics_path(tmp_path).exists()

    def test_rerun_is_idempotent(self, tmp_path):
        seed(tmp_path)
        a = rebuild_daily_metrics(tmp_path, R, Q, cfg())
        b = rebuild_daily_metrics(tmp_path, R, Q, cfg())
        pd.testing.assert_frame_equal(a, b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_metrics_backfill.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.metrics_backfill'`

- [ ] **Step 3: Implement**

`src/data/metrics_backfill.py`:

```python
"""Rebuild data/daily_metrics.parquet by replaying every stored chain.

Stateless by design (same philosophy as SPEC 3 P8): the metrics file is a
pure function of data/chains/ + data/underlying.parquet + (r, q, cfg), so
any analytics fix heals the whole history on the next rebuild. r and q
are the caller's current values -- a documented approximation.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.daily_metrics import (
    metric_columns, refresh_rv_columns, session_metrics_row, upsert_session,
)
from src.analytics.term_structure import compute_term_structure
from src.data import storage


def chain_source_label(chain: pd.DataFrame) -> str:
    if (chain["source"] == "yfinance").all():
        return "yfinance"
    return str(chain["source"].mode().iloc[0])


def rebuild_daily_metrics(root: Path, r: float, q: float, cfg: dict,
                          write: bool = True) -> pd.DataFrame:
    root = Path(root)
    metrics = pd.DataFrame(columns=metric_columns(cfg))
    for path in sorted((root / "data" / "chains").glob("*.parquet")):
        session = dt.date.fromisoformat(path.stem)
        chain = pd.read_parquet(path)
        chain_iv, stats = compute_chain_iv(chain, r, q)
        term = compute_term_structure(chain_iv)
        row = session_metrics_row(session, float(chain["spot"].iloc[0]),
                                  chain_source_label(chain), term, stats, cfg)
        metrics = upsert_session(metrics, row, cfg)
    metrics = refresh_rv_columns(metrics, storage.read_underlying(root), cfg)
    if write:
        storage.write_daily_metrics(metrics, root)
    return metrics
```

`scripts/rebuild_daily_metrics.py`:

```python
"""Rebuild data/daily_metrics.parquet from stored chains. No network.

r and q come from docs/status.json (the last real run) when present, else
config.yaml fallbacks. Usage: .venv/bin/python scripts/rebuild_daily_metrics.py
"""
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.metrics_backfill import rebuild_daily_metrics  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    r, q = cfg["rates"]["risk_free_fallback"], cfg["rates"]["dividend_yield_fallback"]
    status_path = root / "docs" / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        r, q = status.get("risk_free_rate", r), status.get("dividend_yield", q)
    out = rebuild_daily_metrics(root, r, q, cfg)
    print(f"rebuilt {len(out)} rows -> data/daily_metrics.parquet (r={r:.4%}, q={q:.4%})")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_metrics_backfill.py -q` — PASS (3). Then `.venv/bin/pytest -q` — green.

- [ ] **Step 5: Commit**

```bash
git add src/data/metrics_backfill.py scripts/rebuild_daily_metrics.py tests/test_metrics_backfill.py
git commit -m "feat: stateless rebuild of daily_metrics by replaying stored chains"
```

---

### Task 4: P1 input sensitivity — analytics + tornado figure

**Files:**
- Create: `src/analytics/sensitivity.py`
- Modify: `src/render/figures.py` (add `build_sensitivity_figure`)
- Test: `tests/test_sensitivity.py`, `tests/test_render.py` (extend)

**Interfaces:**
- Consumes: `bs_price`.
- Produces:
  - `SENSITIVITY_COLUMNS = ["input", "label", "group", "base_value", "down_pct", "up_pct", "span"]`
  - `compute_sensitivity(spot, sigma, dte, r, q, bump) -> pd.DataFrame` — ~30-DTE ATM call (`K = spot`), one row per input in `("sigma", "S", "r", "q", "T")`; `down_pct`/`up_pct` = fractional price change under `x·(1−bump)` / `x·(1+bump)`; `group` is `"argued"` for σ/r/q and `"observed"` for S/T; sorted by `span = |up_pct − down_pct|` descending within the frame.
  - `build_sensitivity_figure(sens: pd.DataFrame) -> go.Figure` — two-subplot tornado (argued | observed), horizontal bars, `−bump` and `+bump` traces; empty/NaN input → annotation figure like `build_smile_figure`'s empty case.

- [ ] **Step 1: Write the failing tests**

`tests/test_sensitivity.py`:

```python
"""P1: ±bump sensitivity of a ~30-DTE ATM call to each BS input."""
import numpy as np
import pytest

from src.analytics.sensitivity import SENSITIVITY_COLUMNS, compute_sensitivity
from src.models.black_scholes import bs_price

SPOT, SIGMA, DTE, R, Q = 770.0, 0.12, 30, 0.0415, 0.0098


class TestSensitivity:
    def test_shape_and_groups(self):
        s = compute_sensitivity(SPOT, SIGMA, DTE, R, Q, bump=0.20)
        assert list(s.columns) == SENSITIVITY_COLUMNS
        assert set(s["input"]) == {"sigma", "S", "r", "q", "T"}
        assert set(s[s["group"] == "argued"]["input"]) == {"sigma", "r", "q"}
        assert set(s[s["group"] == "observed"]["input"]) == {"S", "T"}
        assert s["span"].is_monotonic_decreasing

    def test_values_are_relative_to_base_price(self):
        s = compute_sensitivity(SPOT, SIGMA, DTE, R, Q, bump=0.20).set_index("input")
        base = bs_price(SPOT, SPOT, DTE / 365.0, R, SIGMA, Q, "call")
        up = bs_price(SPOT, SPOT, DTE / 365.0, R, SIGMA * 1.2, Q, "call")
        assert s.loc["sigma", "up_pct"] == pytest.approx((up - base) / base)
        assert s.loc["sigma", "base_value"] == SIGMA

    def test_sigma_dominates_the_argued_group_and_rate_is_tiny(self):
        s = compute_sensitivity(SPOT, SIGMA, DTE, R, Q, bump=0.20).set_index("input")
        assert s.loc["sigma", "up_pct"] == pytest.approx(0.20, abs=0.03)   # ~linear ATM
        assert s.loc["sigma", "span"] > 5 * s.loc["r", "span"]
        assert s.loc["sigma", "span"] > 5 * s.loc["q", "span"]
        assert s.loc["r", "span"] < 0.05

    def test_spot_is_the_observed_giant(self):
        s = compute_sensitivity(SPOT, SIGMA, DTE, R, Q, bump=0.20).set_index("input")
        assert s.loc["S", "span"] > s.loc["sigma", "span"]          # why the groups exist
        assert s.loc["S", "down_pct"] == pytest.approx(-1.0, abs=0.02)  # 20% crash: worthless

    def test_nan_sigma_gives_nan_rows_not_raise(self):
        s = compute_sensitivity(SPOT, float("nan"), DTE, R, Q, bump=0.20)
        assert len(s) == 5 and s["up_pct"].isna().all()
```

Append to `tests/test_render.py`:

```python
class TestSensitivityFigure:
    def test_two_subplots_and_two_bar_traces_each(self):
        from src.analytics.sensitivity import compute_sensitivity
        from src.render.figures import build_sensitivity_figure
        fig = build_sensitivity_figure(compute_sensitivity(770.0, 0.12, 30, 0.04, 0.01, 0.20))
        bars = [t for t in fig.data if t.type == "bar"]
        assert len(bars) == 4                        # (down, up) x (argued, observed)
        assert {t.xaxis for t in bars} == {"x", "x2"}
        assert "Volatility" in "".join(str(t.y) for t in bars)

    def test_nan_input_renders_annotation(self):
        from src.analytics.sensitivity import compute_sensitivity
        from src.render.figures import build_sensitivity_figure
        fig = build_sensitivity_figure(compute_sensitivity(770.0, float("nan"), 30, 0.04, 0.01, 0.2))
        assert not [t for t in fig.data if t.type == "bar"]
        assert fig.layout.annotations and "No" in fig.layout.annotations[0].text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_sensitivity.py tests/test_render.py -q`
Expected: FAIL — `ModuleNotFoundError` for `src.analytics.sensitivity`, `ImportError` for `build_sensitivity_figure`.

- [ ] **Step 3: Implement the analytics**

`src/analytics/sensitivity.py`:

```python
"""P1: input sensitivity of the ~30-DTE ATM call (SPEC 3 P1). Pure.

Every input is bumped by the same ±bump so the comparison is honest. The
`group` column is the panel's real message: sigma, r and q are ESTIMATED
(the market can disagree about them); S and T are OBSERVED (a 20% spot
move is a crash, not a disagreement) -- the render layer gives each group
its own axis so sigma's dominance within the argued group is visible.
"""
import numpy as np
import pandas as pd

from src.models.black_scholes import bs_price

SENSITIVITY_COLUMNS = ["input", "label", "group", "base_value", "down_pct", "up_pct", "span"]
_INPUTS = [
    ("sigma", "Volatility σ", "argued"),
    ("r", "Risk-free rate r", "argued"),
    ("q", "Dividend yield q", "argued"),
    ("S", "Spot S", "observed"),
    ("T", "Time to expiry T", "observed"),
]


def _call(S, sigma, T, r, q, strike):
    return float(bs_price(S=S, K=strike, T=T, r=r, sigma=sigma, q=q, kind="call"))


def compute_sensitivity(spot: float, sigma: float, dte: float, r: float, q: float,
                        bump: float) -> pd.DataFrame:
    base_in = {"S": float(spot), "sigma": float(sigma), "T": float(dte) / 365.0,
               "r": float(r), "q": float(q)}
    base = _call(base_in["S"], base_in["sigma"], base_in["T"], base_in["r"], base_in["q"], spot)
    rows = []
    for name, label, group in _INPUTS:
        pct = {}
        for tag, f in (("down", 1.0 - bump), ("up", 1.0 + bump)):
            kw = dict(base_in)
            kw[name] = base_in[name] * f
            px = _call(kw["S"], kw["sigma"], kw["T"], kw["r"], kw["q"], spot)
            pct[tag] = (px - base) / base if base and np.isfinite(base) else np.nan
        rows.append({"input": name, "label": label, "group": group,
                     "base_value": base_in[name],
                     "down_pct": pct["down"], "up_pct": pct["up"],
                     "span": abs(pct["up"] - pct["down"])})
    df = pd.DataFrame(rows, columns=SENSITIVITY_COLUMNS)
    return df.sort_values("span", ascending=False, na_position="last").reset_index(drop=True)
```

- [ ] **Step 4: Implement the figure**

Add to `src/render/figures.py` (import `from plotly.subplots import make_subplots` at the top):

```python
def _empty_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16))
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def build_sensitivity_figure(sens: pd.DataFrame) -> go.Figure:
    title = "Which input moves the price? ±20% on each, ~30-DTE ATM call"
    if sens.empty or sens["up_pct"].isna().all():
        return _empty_figure(title, "No ATM implied vol for this session")
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.14,
        subplot_titles=("Inputs the market argues about", "Inputs everyone observes"))
    for col, group in ((1, "argued"), (2, "observed")):
        g = sens[sens["group"] == group].sort_values("span")   # biggest bar on top
        for tag, color, name in (("down_pct", "#c44e52", "input −20%"),
                                 ("up_pct", "#4c72b0", "input +20%")):
            fig.add_trace(go.Bar(
                y=g["label"], x=g[tag], orientation="h", name=name,
                marker_color=color, showlegend=(col == 1),
                hovertemplate="%{y}: %{x:+.1%}<extra>" + name + "</extra>"),
                row=1, col=col)
    fig.update_layout(title=title, barmode="overlay", **_LAYOUT)
    fig.update_xaxes(title_text="Price change", tickformat="+.0%")
    return fig
```

`barmode="overlay"` with one negative and one positive bar per label draws the classic tornado.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_sensitivity.py tests/test_render.py -q` — PASS. Then `.venv/bin/pytest -q` — green.

- [ ] **Step 6: Commit**

```bash
git add src/analytics/sensitivity.py src/render/figures.py tests/test_sensitivity.py tests/test_render.py
git commit -m "feat: P1 input-sensitivity analytics and two-group tornado figure"
```

---

### Task 5: P4 Greeks panel — tiles, curves, display scaling

**Files:**
- Create: `src/analytics/greeks_panel.py`, `src/render/tiles.py`
- Modify: `src/render/figures.py` (add `build_greeks_curves_figure`)
- Test: `tests/test_greeks_panel.py`, `tests/test_render.py` (extend)

**Interfaces:**
- Consumes: `compute_chain_iv` output frame (columns `CHAIN_COLUMNS + ["price_used", "iv"]`), `greeks(S, K, T, r, sigma, q, kind)`.
- Produces (`src/analytics/greeks_panel.py`):
  - `TILE_COLUMNS = ["kind", "expiry", "dte", "strike", "iv", "delta", "gamma", "vega", "theta", "rho"]` (raw derivatives)
  - `CURVE_COLUMNS = ["kind", "strike", "moneyness", "delta", "gamma"]`
  - `compute_greeks_panel(chain_iv, r, q, cfg) -> tuple[pd.DataFrame, pd.DataFrame]` = `(tiles, curves)` for the expiry nearest `cfg["target_dte"]["atm_panel"]`; tiles = the converged strike nearest spot per kind (≤ 2 rows); curves = every converged row of that expiry, each at its **own** market IV. Empty chain → two empty frames with the right columns.
- Produces (`src/render/tiles.py`):
  - `greek_tiles_html(tiles, tiles_prev, prev_label) -> str` — `<div class='tiles'>` block: per kind, a tile per Greek with the **display-scaled** value (theta/365, vega/100, rho/100; delta & gamma unscaled) and, when `tiles_prev` has that kind, a `Δ` line vs the previous session. Returns a `<p class='placeholder'>` when tiles are empty.
- Produces (`src/render/figures.py`): `build_greeks_curves_figure(curves, spot) -> go.Figure` — 1×2 subplots (delta vs strike, gamma vs strike), one trace per kind, dotted ATM line at `spot`.

- [ ] **Step 1: Write the failing tests**

`tests/test_greeks_panel.py`:

```python
"""P4: ATM Greeks tiles + delta/gamma curves at each strike's own IV."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.greeks_panel import CURVE_COLUMNS, TILE_COLUMNS, compute_greeks_panel
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price, greeks

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def make_chain(expiries=((dt.date(2026, 9, 25), 28), (dt.date(2026, 11, 20), 84)),
               strikes=tuple(np.arange(700.0, 841.0, 10.0)), sigma=0.22):
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                px = float(bs_price(SPOT, strike, dte / 365.0, R, sigma, Q, kind))
                rows.append({"snapshot_date": TODAY, "spot": SPOT, "expiry": expiry,
                             "dte": dte, "strike": strike, "kind": kind,
                             "bid": px * 0.99, "ask": px * 1.01, "mid": px, "close": px,
                             "volume": 10, "open_interest": 100.0, "vendor_iv": np.nan,
                             "source": "yfinance"})
    out, _ = compute_chain_iv(pd.DataFrame(rows, columns=CHAIN_COLUMNS), R, Q)
    return out


class TestTiles:
    def test_picks_nearest_expiry_and_atm_strike_per_kind(self):
        tiles, _ = compute_greeks_panel(make_chain(), R, Q, cfg())
        assert list(tiles.columns) == TILE_COLUMNS
        assert sorted(tiles["kind"]) == ["call", "put"]
        assert (tiles["dte"] == 28).all() and (tiles["strike"] == SPOT).all()

    def test_values_match_closed_form_and_call_delta_near_half(self):
        tiles, _ = compute_greeks_panel(make_chain(), R, Q, cfg())
        call = tiles[tiles["kind"] == "call"].iloc[0]
        ref = greeks(SPOT, SPOT, 28 / 365.0, R, call["iv"], Q, "call")
        for g in ("delta", "gamma", "vega", "theta", "rho"):
            assert call[g] == pytest.approx(ref[g], rel=1e-9)
        assert call["delta"] == pytest.approx(0.5, abs=0.05)

    def test_put_call_delta_parity(self):
        tiles, _ = compute_greeks_panel(make_chain(), R, Q, cfg())
        t = tiles.set_index("kind")
        assert t.loc["call", "delta"] - t.loc["put", "delta"] == pytest.approx(
            np.exp(-Q * 28 / 365.0), abs=1e-9)
        assert t.loc["call", "gamma"] == pytest.approx(t.loc["put", "gamma"], rel=1e-9)

    def test_missing_kind_yields_one_tile(self):
        chain = make_chain()
        chain = chain[chain["kind"] == "call"]
        tiles, curves = compute_greeks_panel(chain, R, Q, cfg())
        assert list(tiles["kind"]) == ["call"] and (curves["kind"] == "call").all()


class TestCurves:
    def test_gamma_peaks_near_spot_and_delta_is_monotone(self):
        _, curves = compute_greeks_panel(make_chain(), R, Q, cfg())
        assert list(curves.columns) == CURVE_COLUMNS
        calls = curves[curves["kind"] == "call"].sort_values("strike")
        peak = calls.loc[calls["gamma"].idxmax(), "strike"]
        assert abs(peak / SPOT - 1) < 0.02
        assert calls["delta"].is_monotonic_decreasing
        puts = curves[curves["kind"] == "put"].sort_values("strike")
        assert (puts["delta"] <= 0).all() and puts["delta"].is_monotonic_decreasing

    def test_curves_use_each_strikes_own_iv(self):
        chain = make_chain()
        # poison one strike's IV; its delta must move, neighbours must not
        chain.loc[(chain["strike"] == 800.0) & (chain["kind"] == "call") & (chain["dte"] == 28),
                  "iv"] = 0.50
        _, curves = compute_greeks_panel(chain, R, Q, cfg())
        calls = curves[curves["kind"] == "call"].set_index("strike")
        ref_800 = greeks(SPOT, 800.0, 28 / 365.0, R, 0.50, Q, "call")["delta"]
        ref_790 = greeks(SPOT, 790.0, 28 / 365.0, R, 0.22, Q, "call")["delta"]
        assert calls.loc[800.0, "delta"] == pytest.approx(ref_800, rel=1e-6)
        assert calls.loc[790.0, "delta"] == pytest.approx(ref_790, rel=1e-6)

    def test_empty_chain(self):
        tiles, curves = compute_greeks_panel(make_chain().iloc[0:0], R, Q, cfg())
        assert tiles.empty and curves.empty
        assert list(tiles.columns) == TILE_COLUMNS and list(curves.columns) == CURVE_COLUMNS
```

Append to `tests/test_render.py`:

```python
class TestGreekTiles:
    def _tiles(self, theta=-36.5, vega=95.0, rho=31.0, delta=0.52):
        import datetime as dt
        return pd.DataFrame([{
            "kind": "call", "expiry": dt.date(2026, 9, 25), "dte": 28, "strike": 770.0,
            "iv": 0.12, "delta": delta, "gamma": 0.021, "vega": vega, "theta": theta, "rho": rho,
        }])

    def test_display_scaling(self):
        from src.render.tiles import greek_tiles_html
        html = greek_tiles_html(self._tiles(), None, "previous session")
        assert "-0.100" in html          # theta/365 = -0.1 per day
        assert "0.950" in html           # vega/100 per vol point
        assert "0.310" in html           # rho/100 per 1%
        assert "0.520" in html and "per day" in html and "per vol pt" in html

    def test_one_day_change_line(self):
        from src.render.tiles import greek_tiles_html
        html = greek_tiles_html(self._tiles(delta=0.52), self._tiles(delta=0.50), "prev session 2026-08-27")
        assert "+0.020" in html and "prev session 2026-08-27" in html

    def test_empty_is_placeholder(self):
        from src.render.tiles import greek_tiles_html
        from src.analytics.greeks_panel import TILE_COLUMNS
        html = greek_tiles_html(pd.DataFrame(columns=TILE_COLUMNS), None, "x")
        assert "placeholder" in html


class TestGreeksCurvesFigure:
    def test_two_subplots_per_kind(self):
        from src.render.figures import build_greeks_curves_figure
        curves = pd.DataFrame({
            "kind": ["call", "call", "put", "put"], "strike": [760.0, 780.0, 760.0, 780.0],
            "moneyness": [0.987, 1.013, 0.987, 1.013],
            "delta": [0.6, 0.4, -0.4, -0.6], "gamma": [0.02, 0.02, 0.02, 0.02]})
        fig = build_greeks_curves_figure(curves, 770.0)
        scatters = [t for t in fig.data if t.type == "scatter" and t.name in ("call", "put")]
        assert len(scatters) == 4 and {t.xaxis for t in scatters} == {"x", "x2"}
```

(`tests/test_render.py` already imports `pandas as pd` at module level — verify; if not, add it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_greeks_panel.py tests/test_render.py -q`
Expected: FAIL — `ModuleNotFoundError` for `src.analytics.greeks_panel` and `src.render.tiles`.

- [ ] **Step 3: Implement the analytics**

`src/analytics/greeks_panel.py`:

```python
"""P4: Greeks for the ~30-DTE expiry (SPEC 3 P4). Pure, raw derivatives.

Tiles: the converged strike nearest spot per kind. Curves: every converged
strike of that expiry, each evaluated at ITS OWN market IV -- so the
curves are what a desk would actually see, not a flat-vol textbook plot.
Display scaling (theta/day, vega/vol-pt, rho/1%) belongs to src.render.
"""
import numpy as np
import pandas as pd

from src.models.black_scholes import greeks

TILE_COLUMNS = ["kind", "expiry", "dte", "strike", "iv", "delta", "gamma", "vega", "theta", "rho"]
CURVE_COLUMNS = ["kind", "strike", "moneyness", "delta", "gamma"]


def _empty():
    return pd.DataFrame(columns=TILE_COLUMNS), pd.DataFrame(columns=CURVE_COLUMNS)


def compute_greeks_panel(chain_iv: pd.DataFrame, r: float, q: float,
                         cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    solved = chain_iv[chain_iv["iv"].notna()]
    if solved.empty:
        return _empty()
    spot = float(solved["spot"].iloc[0])
    target = int(cfg["target_dte"]["atm_panel"])
    available = solved[["expiry", "dte"]].drop_duplicates().reset_index(drop=True)
    pick = available.iloc[(available["dte"] - target).abs().idxmin()]
    g = solved[solved["expiry"] == pick["expiry"]].copy()
    T = float(pick["dte"]) / 365.0

    frames = []
    for kind in ("call", "put"):
        k = g[g["kind"] == kind].sort_values("strike")
        if k.empty:
            continue
        gr = greeks(S=spot, K=k["strike"].to_numpy(dtype=float), T=T, r=r,
                    sigma=k["iv"].to_numpy(dtype=float), q=q, kind=kind)
        frames.append(pd.DataFrame({
            "kind": kind, "expiry": pick["expiry"], "dte": int(pick["dte"]),
            "strike": k["strike"].to_numpy(dtype=float), "iv": k["iv"].to_numpy(dtype=float),
            "moneyness": k["strike"].to_numpy(dtype=float) / spot,
            **{name: np.atleast_1d(gr[name]) for name in ("delta", "gamma", "vega", "theta", "rho")},
        }))
    full = pd.concat(frames, ignore_index=True)
    nearest = full.loc[full.groupby("kind")["strike"]
                           .transform(lambda s: (s - spot).abs()).groupby(full["kind"]).idxmin()]
    tiles = nearest[TILE_COLUMNS].sort_values("kind").reset_index(drop=True)
    curves = full[CURVE_COLUMNS].sort_values(["kind", "strike"]).reset_index(drop=True)
    return tiles, curves
```

If the `nearest` one-liner reads badly to you, this is the equivalent explicit form — use either:

```python
    idx = [grp.index[(grp["strike"] - spot).abs().argmin()] for _, grp in full.groupby("kind")]
    nearest = full.loc[idx]
```

- [ ] **Step 4: Implement tiles HTML and the curves figure**

`src/render/tiles.py`:

```python
"""Greek stat tiles (SPEC 3 P4). This is where SPEC 2.3's display scaling
lives: theta per DAY (/365), vega per VOL POINT (/100), rho per 1% (/100).
delta and gamma are shown raw."""
import pandas as pd

_GREEKS = [  # (column, label, unit, divisor)
    ("delta", "Delta", "per $1 spot", 1.0),
    ("gamma", "Gamma", "delta per $1", 1.0),
    ("vega", "Vega", "per vol pt", 100.0),
    ("theta", "Theta", "per day", 365.0),
    ("rho", "Rho", "per 1% rate", 100.0),
]

TILES_CSS = """
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
         gap: 10px; margin: 8px 0 16px; }
.tile { border: 1px solid #e5e5ee; border-radius: 8px; padding: 10px 12px; }
.tile .k { font-size: 0.8rem; color: #555; } .tile .v { font-size: 1.25rem; font-weight: 600; }
.tile .u, .tile .d { font-size: 0.78rem; color: #777; }
.tiles-head { margin: 14px 0 4px; font-weight: 600; }
"""


def greek_tiles_html(tiles: pd.DataFrame, tiles_prev: pd.DataFrame | None,
                     prev_label: str) -> str:
    if tiles is None or tiles.empty:
        return "<p class='placeholder'>No converged ATM quotes to compute Greeks for.</p>"
    parts = []
    prev = tiles_prev.set_index("kind") if tiles_prev is not None and len(tiles_prev) else None
    for _, row in tiles.iterrows():
        parts.append(
            f"<div class='tiles-head'>ATM {row['kind']} · K {row['strike']:.0f} · "
            f"{row['expiry'].isoformat()} ({int(row['dte'])}d) · IV {row['iv']:.1%}</div>"
            "<div class='tiles'>")
        for col, label, unit, div in _GREEKS:
            val = row[col] / div
            change = ""
            if prev is not None and row["kind"] in prev.index:
                d = val - prev.loc[row["kind"], col] / div
                change = f"<div class='d'>{d:+.3f} vs {prev_label}</div>"
            parts.append(f"<div class='tile'><div class='k'>{label}</div>"
                         f"<div class='v'>{val:.3f}</div><div class='u'>{unit}</div>{change}</div>")
        parts.append("</div>")
    return "".join(parts)
```

Add to `src/render/figures.py`:

```python
def build_greeks_curves_figure(curves: pd.DataFrame, spot: float) -> go.Figure:
    title = "Delta and gamma across strikes — ~30-DTE expiry, each at its own market IV"
    if curves.empty:
        return _empty_figure(title, "No converged quotes for the ~30-DTE expiry")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Delta vs strike", "Gamma vs strike"))
    colors = {"call": "#4c72b0", "put": "#dd8452"}
    for kind, g in curves.groupby("kind"):
        g = g.sort_values("strike")
        for col, y in ((1, "delta"), (2, "gamma")):
            fig.add_trace(go.Scatter(
                x=g["strike"], y=g[y], mode="lines+markers", name=kind,
                legendgroup=kind, showlegend=(col == 1), marker=dict(size=4),
                line=dict(color=colors.get(kind)),
                hovertemplate="K %{x:.0f}: %{y:.4f}<extra>" + kind + "</extra>"),
                row=1, col=col)
    for col in (1, 2):
        fig.add_vline(x=spot, line=dict(dash="dot", color="grey", width=1), row=1, col=col)
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_xaxes(title_text="Strike")
    return fig
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_greeks_panel.py tests/test_render.py -q` — PASS. Then `.venv/bin/pytest -q` — green. If `add_vline` raises under warnings-as-errors for the subplot form, replace it with a `go.Scatter` dotted line trace at `x=[spot, spot]` spanning `[g[y].min(), g[y].max()]` per subplot with `hoverinfo="skip"`, `showlegend=False` — same look, no dependency on annotation helpers.

- [ ] **Step 6: Commit**

```bash
git add src/analytics/greeks_panel.py src/render/tiles.py src/render/figures.py tests/test_greeks_panel.py tests/test_render.py
git commit -m "feat: P4 Greeks panel — ATM tiles with 1-day change, delta/gamma curves at market IV"
```

---

### Task 6: P5 IV-vs-RV — series, summary stat, figure; page assembly for Q1/Q3

**Files:**
- Create: `src/analytics/iv_rv.py`
- Modify: `src/render/figures.py` (add `build_iv_rv_figure`), `src/render/page.py` (Q1/Q3 wiring, captions, `extras`, CSS, GitHub link)
- Test: `tests/test_iv_rv.py`, `tests/test_render.py` (extend)

**Interfaces:**
- Consumes: metrics frame with `metric_columns(cfg)`.
- Produces (`src/analytics/iv_rv.py`):
  - `IV_RV_COLUMNS = ["date", "atm_iv", "rv_trailing", "fwd_rv", "spread", "spread_running_mean"]`
  - `iv_rv_series(metrics, cfg) -> pd.DataFrame` — `atm_iv = atm_iv_30d`, `rv_trailing = rv_{windows[0]}d`, `fwd_rv = fwd_rv_{h}d`, `spread = atm_iv − rv_trailing`, `spread_running_mean` = expanding mean of `spread` over rows where it is finite; rows where `atm_iv` is NaN are dropped; sorted by date.
  - `iv_rv_summary(series) -> dict` — `{"history_since": date|None, "n_sessions": int, "evaluable_days": int, "share_iv_above_fwd_rv": float|nan, "mean_spread_trailing": float|nan}` where evaluable = rows with both `atm_iv` and `fwd_rv` finite.
  - `iv_rv_summary_html(summary) -> str` — one `<p class='stat'>` sentence for the page.
- Produces (`src/render/figures.py`): `build_iv_rv_figure(series, summary, cfg) -> go.Figure` — traces: `rv_trailing`, `atm_iv` (with `fill="tonexty"` shading between them), `fwd_rv` (dashed, plotted **at t**), `spread_running_mean` (dotted, secondary axis); "accumulating since <date>" annotation; empty → annotation figure.
- Produces (`src/render/page.py`): `render_page(figures: dict, status: dict, extras: dict | None = None) -> str` — `extras[pid]` is an HTML string inserted between the panel `<h3>` and its figure (P4 tiles, P5 stat line). `QUESTIONS` updated: Q1 → `["P1", "P4"]`, Q3 → `["P5"]`. `_PANEL_TITLES` and `CAPTIONS` gain P1/P4/P5. Footer links `https://github.com/TK-Chang239/BlackScholesVolDashboard`. `_CSS` gains `TILES_CSS` and `.stat { font-weight: 600; margin: 4px 0 8px; }`.

- [ ] **Step 1: Write the failing tests**

`tests/test_iv_rv.py`:

```python
"""P5: implied vs realized series and the forward-RV summary stat."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analytics.daily_metrics import metric_columns
from src.analytics.iv_rv import IV_RV_COLUMNS, iv_rv_series, iv_rv_summary, iv_rv_summary_html


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def metrics(rows):
    df = pd.DataFrame(rows)
    return df.reindex(columns=metric_columns(cfg()))


D = [dt.date(2026, 8, 20) + dt.timedelta(days=i) for i in range(5)]


class TestSeries:
    def test_columns_spread_and_running_mean(self):
        m = metrics([
            {"date": D[0], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": 0.11},
            {"date": D[1], "atm_iv_30d": 0.14, "rv_20d": 0.10, "fwd_rv_30d": np.nan},
            {"date": D[2], "atm_iv_30d": np.nan, "rv_20d": 0.10, "fwd_rv_30d": 0.09},
        ])
        s = iv_rv_series(m, cfg())
        assert list(s.columns) == IV_RV_COLUMNS
        assert list(s["date"]) == [D[0], D[1]]                # NaN-IV row dropped
        assert s["spread"].to_numpy() == pytest.approx([0.02, 0.04])
        assert s["spread_running_mean"].to_numpy() == pytest.approx([0.02, 0.03])

    def test_running_mean_skips_nan_rv(self):
        m = metrics([
            {"date": D[0], "atm_iv_30d": 0.12, "rv_20d": np.nan},
            {"date": D[1], "atm_iv_30d": 0.14, "rv_20d": 0.10},
        ])
        s = iv_rv_series(m, cfg())
        assert np.isnan(s["spread_running_mean"].iloc[0])
        assert s["spread_running_mean"].iloc[1] == pytest.approx(0.04)

    def test_empty(self):
        s = iv_rv_series(metrics([]), cfg())
        assert s.empty and list(s.columns) == IV_RV_COLUMNS


class TestSummary:
    def test_counts_and_share(self):
        m = metrics([
            {"date": D[0], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": 0.11},  # IV > fwd
            {"date": D[1], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": 0.13},  # IV < fwd
            {"date": D[2], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": 0.10},  # IV > fwd
            {"date": D[3], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": np.nan},
        ])
        summ = iv_rv_summary(iv_rv_series(m, cfg()))
        assert summ["history_since"] == D[0] and summ["n_sessions"] == 4
        assert summ["evaluable_days"] == 3
        assert summ["share_iv_above_fwd_rv"] == pytest.approx(2 / 3)
        assert summ["mean_spread_trailing"] == pytest.approx(0.02)
        html = iv_rv_summary_html(summ)
        assert "67%" in html and "3 evaluable" in html and D[0].isoformat() in html

    def test_no_evaluable_days(self):
        m = metrics([{"date": D[0], "atm_iv_30d": 0.12, "rv_20d": 0.10, "fwd_rv_30d": np.nan}])
        summ = iv_rv_summary(iv_rv_series(m, cfg()))
        assert summ["evaluable_days"] == 0 and np.isnan(summ["share_iv_above_fwd_rv"])
        assert "not yet" in iv_rv_summary_html(summ)

    def test_empty_summary(self):
        summ = iv_rv_summary(iv_rv_series(metrics([]), cfg()))
        assert summ["history_since"] is None and summ["n_sessions"] == 0
        assert "no history" in iv_rv_summary_html(summ).lower()
```

Append to `tests/test_render.py`:

```python
class TestIvRvFigure:
    def _series(self):
        import datetime as dt
        d = [dt.date(2026, 8, 20) + dt.timedelta(days=i) for i in range(3)]
        return pd.DataFrame({"date": d, "atm_iv": [0.12, 0.13, 0.14],
                             "rv_trailing": [0.10, 0.10, 0.11], "fwd_rv": [0.11, np.nan, np.nan],
                             "spread": [0.02, 0.03, 0.03],
                             "spread_running_mean": [0.02, 0.025, 0.0267]})

    def test_traces_and_since_annotation(self):
        import yaml
        from src.analytics.iv_rv import iv_rv_summary
        from src.render.figures import build_iv_rv_figure
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        s = self._series()
        fig = build_iv_rv_figure(s, iv_rv_summary(s), cfg)
        names = [t.name for t in fig.data]
        assert any("implied" in n.lower() for n in names)
        assert any("trailing" in n.lower() for n in names)
        assert any("forward" in n.lower() for n in names)
        assert any(t.fill == "tonexty" for t in fig.data)
        assert any("2026-08-20" in (a.text or "") for a in fig.layout.annotations)

    def test_empty_series(self):
        import yaml
        from src.analytics.iv_rv import IV_RV_COLUMNS, iv_rv_summary
        from src.render.figures import build_iv_rv_figure
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        s = pd.DataFrame(columns=IV_RV_COLUMNS)
        fig = build_iv_rv_figure(s, iv_rv_summary(s), cfg)
        assert not fig.data and fig.layout.annotations


class TestPagePhase4:
    def _figs(self):
        import plotly.graph_objects as go
        return {p: go.Figure() for p in ("P1", "P2", "P3", "P4", "P5")}

    def _status(self):
        return {"spot": 771.1, "snapshot_date": "2026-08-27", "source": "yfinance",
                "iv_convergence": 0.99, "last_success_utc": "2026-08-28T14:38:34+00:00",
                "rows_stored": 1459}

    def test_q1_and_q3_render_panels_not_placeholders(self):
        from src.render.page import render_page
        html = render_page(self._figs(), self._status())
        assert "Input-sensitivity panel arrives" not in html
        assert "Implied-vs-realized panel arrives" not in html
        q1 = html.index("Q1."); q2 = html.index("Q2."); q3 = html.index("Q3."); q4 = html.index("Q4.")
        assert q1 < html.index("Which input moves") < q2 or q1 < html.index("Input sensitivity") < q2
        assert q3 < html.index("Implied vs realized") < q4

    def test_extras_are_inserted_under_their_panel(self):
        from src.render.page import render_page
        html = render_page(self._figs(), self._status(),
                           extras={"P4": "<div class='tiles'>TILES-HERE</div>",
                                   "P5": "<p class='stat'>STAT-HERE</p>"})
        assert html.index("Greeks") < html.index("TILES-HERE")
        assert html.index("Implied vs realized") < html.index("STAT-HERE")

    def test_footer_links_repo(self):
        from src.render.page import render_page
        assert "https://github.com/TK-Chang239/BlackScholesVolDashboard" in \
            render_page(self._figs(), self._status())

    def test_every_rendered_panel_has_a_caption(self):
        from src.render.page import CAPTIONS, render_page
        html = render_page(self._figs(), self._status())
        for pid in ("P1", "P2", "P3", "P4", "P5"):
            assert CAPTIONS[pid] in html
```

(`tests/test_render.py` needs `import numpy as np` and `import pandas as pd` at module level for these; add if missing.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_iv_rv.py tests/test_render.py -q`
Expected: FAIL — `ModuleNotFoundError: src.analytics.iv_rv`; `ImportError: build_iv_rv_figure`; `TypeError: render_page() got an unexpected keyword argument 'extras'`; placeholder assertions fail.

- [ ] **Step 3: Implement `iv_rv`**

`src/analytics/iv_rv.py`:

```python
"""P5: implied vs realized vol (SPEC 3 P5). Pure.

Trailing RV is the live visual; FORWARD RV is the honest forecast target:
the vol that actually happened over the ~30 days after each IV quote. The
summary stat is computed against forward RV only.
"""
import numpy as np
import pandas as pd

IV_RV_COLUMNS = ["date", "atm_iv", "rv_trailing", "fwd_rv", "spread", "spread_running_mean"]


def iv_rv_series(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rv_cfg = cfg["realized_vol"]
    trailing = f"rv_{rv_cfg['windows'][0]}d"
    forward = f"fwd_rv_{rv_cfg['forward_horizon_days']}d"
    if metrics.empty:
        return pd.DataFrame(columns=IV_RV_COLUMNS)
    s = pd.DataFrame({
        "date": metrics["date"],
        "atm_iv": metrics["atm_iv_30d"].astype(float),
        "rv_trailing": metrics[trailing].astype(float),
        "fwd_rv": metrics[forward].astype(float),
    })
    s = s[s["atm_iv"].notna()].sort_values("date").reset_index(drop=True)
    s["spread"] = s["atm_iv"] - s["rv_trailing"]
    s["spread_running_mean"] = s["spread"].expanding().mean()   # NaNs skipped by pandas
    return s[IV_RV_COLUMNS]


def iv_rv_summary(series: pd.DataFrame) -> dict:
    if series.empty:
        return {"history_since": None, "n_sessions": 0, "evaluable_days": 0,
                "share_iv_above_fwd_rv": np.nan, "mean_spread_trailing": np.nan}
    ev = series[series["atm_iv"].notna() & series["fwd_rv"].notna()]
    return {
        "history_since": series["date"].iloc[0],
        "n_sessions": int(len(series)),
        "evaluable_days": int(len(ev)),
        "share_iv_above_fwd_rv": float((ev["atm_iv"] > ev["fwd_rv"]).mean()) if len(ev) else np.nan,
        "mean_spread_trailing": float(series["spread"].mean()) if series["spread"].notna().any() else np.nan,
    }


def iv_rv_summary_html(summary: dict) -> str:
    if summary["n_sessions"] == 0:
        return "<p class='stat'>No history yet — the series starts with the first stored session.</p>"
    since = summary["history_since"].isoformat()
    if summary["evaluable_days"] == 0:
        return (f"<p class='stat'>History since {since} ({summary['n_sessions']} sessions); "
                "no day is old enough for its forward realized vol to be known yet — "
                "the forecast scorecard is not yet evaluable.</p>")
    return (f"<p class='stat'>Implied vol exceeded subsequently-realized vol on "
            f"{summary['share_iv_above_fwd_rv']:.0%} of {summary['evaluable_days']} evaluable "
            f"days (history since {since}, {summary['n_sessions']} sessions).</p>")
```

- [ ] **Step 4: Implement the figure**

Add to `src/render/figures.py`:

```python
def build_iv_rv_figure(series: pd.DataFrame, summary: dict, cfg: dict) -> go.Figure:
    rv_cfg = cfg["realized_vol"]
    title = "Implied vs realized volatility — ~30-DTE ATM IV against what SPY actually did"
    if series.empty:
        return _empty_figure(title, "No sessions with a 30-DTE ATM implied vol yet")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=series["date"], y=series["rv_trailing"], mode="lines+markers",
        name=f"Realized vol, trailing {rv_cfg['windows'][0]}d", line=dict(color="#55a868"),
        marker=dict(size=4), hovertemplate="%{x}: %{y:.1%}<extra>trailing RV</extra>"))
    fig.add_trace(go.Scatter(
        x=series["date"], y=series["atm_iv"], mode="lines+markers",
        name="Implied vol, ~30-DTE ATM", line=dict(color="#4c72b0"), marker=dict(size=4),
        fill="tonexty", fillcolor="rgba(76,114,176,0.12)",
        hovertemplate="%{x}: %{y:.1%}<extra>implied</extra>"))
    fig.add_trace(go.Scatter(
        x=series["date"], y=series["fwd_rv"], mode="lines+markers",
        name=f"Realized vol, forward {rv_cfg['forward_horizon_days']}d (plotted at quote date)",
        line=dict(color="#dd8452", dash="dash"), marker=dict(size=4), connectgaps=False,
        hovertemplate="%{x}: %{y:.1%}<extra>forward RV</extra>"))
    fig.add_trace(go.Scatter(
        x=series["date"], y=series["spread_running_mean"], mode="lines",
        name="Running mean of IV − trailing RV", line=dict(color="grey", dash="dot"),
        hovertemplate="%{x}: %{y:+.1%}<extra>mean spread</extra>"), secondary_y=True)
    since = summary["history_since"].isoformat() if summary["history_since"] else "n/a"
    fig.add_annotation(text=f"accumulating since {since} · {summary['n_sessions']} sessions",
                       xref="paper", yref="paper", x=0.0, y=1.08, showarrow=False,
                       font=dict(size=11, color="#555"), xanchor="left")
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_yaxes(title_text="Annualized vol", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Spread", tickformat="+.0%", secondary_y=True, showgrid=False)
    return fig
```

- [ ] **Step 5: Update `page.py`**

In `src/render/page.py`:

1. Import: `from src.render.tiles import TILES_CSS`.
2. Add to `CAPTIONS` (verbatim):

```python
    "P1": (
        "Five inputs, one formula. Bumping each by ±20% shows that among the "
        "inputs a trader must estimate, volatility is the only one worth arguing "
        "about — the rate and dividend bars barely register. Spot and time are "
        "shown separately because nobody disagrees about them: a 20% move in "
        "spot is a crash, not a difference of opinion. This is why option desks "
        "quote volatility, not price."
    ),
    "P4": (
        "The Greeks are the model's partial derivatives — how the price responds "
        "to spot (delta), how delta itself responds (gamma), and how the price "
        "bleeds with time (theta) or breathes with volatility (vega). Each tile "
        "is computed by this project's own closed-form code for the at-the-money "
        "~30-day option; the curves evaluate every strike at its own market "
        "implied vol, which is what a hedging desk actually uses. Gamma peaking "
        "at the money is the engine behind the hedged-P&L story in Q4."
    ),
    "P5": (
        "Implied vol is the market's price for the next month of movement; "
        "realized vol is what actually happened. The shaded gap is implied minus "
        "trailing realized — the live view — but the honest scorecard is the "
        "dashed forward series: the vol that subsequently occurred after each "
        "quote. When implied sits persistently above forward realized, option "
        "sellers are being paid a premium for bearing variance risk; that "
        "premium, not forecasting skill, is what the gap usually measures."
    ),
```

3. `QUESTIONS`: Q1 → `["P1", "P4"]` with coming text `""`; Q3 → `["P5"]` with coming text `""`. (Q2/Q4/Q5 unchanged.)
4. `_PANEL_TITLES` → `{"P1": "Input sensitivity (tornado)", "P2": "IV smile / skew by expiry", "P3": "ATM IV term structure", "P4": "Greeks — ATM ~30-DTE call & put", "P5": "Implied vs realized volatility"}`.
5. `_CSS`: append `TILES_CSS` and `.stat { font-weight: 600; margin: 4px 0 8px; }` — i.e. `_CSS = """...""" + TILES_CSS + ".stat { font-weight: 600; margin: 4px 0 8px; }\n"`.
6. Signature `def render_page(figures: dict, status: dict, extras: dict | None = None) -> str:` and inside the panel loop, after the `<h3>` (before `<div class='figure'>`), insert `extras.get(pid, "")`:

```python
        for pid in panel_ids:
            if pid in figures:
                parts.append(f"<h3>{_PANEL_TITLES[pid]}</h3>")
                parts.append((extras or {}).get(pid, ""))
                parts.append("<div class='figure'>")
                parts.append(pio.to_html(
                    figures[pid], full_html=False, include_plotlyjs=False,
                    config={"responsive": True, "displaylogo": False}))
                parts.append(f"</div><p class='caption'>{CAPTIONS[pid]}</p>")
                rendered_any = True
```

7. Footer: replace `"Source code: GitHub (link once public)."` with `"Source code: <a href='https://github.com/TK-Chang239/BlackScholesVolDashboard'>github.com/TK-Chang239/BlackScholesVolDashboard</a>."`

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_iv_rv.py tests/test_render.py -q` — PASS. Then `.venv/bin/pytest -q` — expect exactly one pre-existing failure: `tests/test_pipeline.py::TestRenderStage::test_run_renders_page_and_reports_convergence` (`panels_rendered == ["P2","P3"]`) is still true at this point and passes; if anything else fails, fix it here before committing. (Task 7 changes that assertion.)

- [ ] **Step 7: Commit**

```bash
git add src/analytics/iv_rv.py src/render/figures.py src/render/page.py tests/test_iv_rv.py tests/test_render.py
git commit -m "feat: P5 IV-vs-RV series, forward-RV scorecard, page wiring for Q1/Q3 with captions"
```

---

### Task 7: Wire metrics + P1/P4/P5 into `run_daily`

**Files:**
- Modify: `src/run_daily.py`
- Test: `tests/test_pipeline.py` (extend)

**Interfaces:**
- Consumes everything from Tasks 2–6.
- Produces: `run(...)` status dict gains `"atm_iv_30d"` (float or None), `"daily_metrics_rows"` (int), `"history_since"` (ISO date or None); `"panels_rendered"` becomes `["P1", "P2", "P3", "P4", "P5"]`. Write order: chain → daily_metrics → index.html → status.json.

- [ ] **Step 1: Extend the pipeline tests**

In `tests/test_pipeline.py`, change the assertion in `test_run_renders_page_and_reports_convergence` to `assert status["panels_rendered"] == ["P1", "P2", "P3", "P4", "P5"]`, and add:

```python
class TestDailyMetricsStage:
    def test_first_run_writes_one_metrics_row(self, tmp_path):
        status = run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        m = pd.read_parquet(storage.daily_metrics_path(tmp_path))
        assert len(m) == 1 and m["date"].iloc[0] == TODAY
        assert m["source"].iloc[0] == "yfinance"
        assert "rv_20d" in m.columns and "fwd_rv_30d" in m.columns
        # single-strike fixture cannot bracket spot -> no ATM IV, and that is recorded honestly
        assert np.isnan(m["atm_iv_30d"].iloc[0]) and status["atm_iv_30d"] is None
        assert status["daily_metrics_rows"] == 1
        assert status["history_since"] == TODAY.isoformat()

    def test_second_session_appends_and_rerun_replaces(self, tmp_path):
        run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        class NextDayEODHD(FakeEODHD):
            def get_underlying_history(self, symbol, start=None):
                df = underlying_frame()
                extra = pd.DataFrame({"date": [TODAY + dt.timedelta(days=1)],
                                      "close": [772.0], "adjusted_close": [772.0],
                                      "volume": [1]})
                return pd.concat([df, extra], ignore_index=True)
        s2 = run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path,
                 today=TODAY + dt.timedelta(days=1))
        s3 = run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path,
                 today=TODAY + dt.timedelta(days=1))
        m = pd.read_parquet(storage.daily_metrics_path(tmp_path))
        assert list(m["date"]) == [TODAY, TODAY + dt.timedelta(days=1)]
        assert s2["daily_metrics_rows"] == 2 and s3["daily_metrics_rows"] == 2

    def test_bracketing_chain_records_atm_iv_and_renders_tiles(self, tmp_path):
        from src.models.black_scholes import bs_price

        def bracketing_chain(spot, expiry, dte):
            T = dte / 365.0
            rows = []
            for strike in (spot - 20.0, spot, spot + 20.0):
                for kind in ("call", "put"):
                    price = float(bs_price(spot, strike, T, 0.0415, 0.20, 0.0098, kind))
                    rows.append({
                        "expiry": expiry, "strike": strike, "kind": kind,
                        "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                        "close": price, "volume": 100, "open_interest": 500,
                        "vendor_iv": 0.20, "source": "yfinance",
                    })
            return pd.DataFrame(rows)

        class BracketingLive:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return bracketing_chain(spot, dt.date(2026, 9, 25), 28)

        status = run(FakeEODHD(), BracketingLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert status["atm_iv_30d"] == pytest.approx(0.20, abs=1e-4)
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "class='tiles'" in page and "Delta" in page
        assert "Volatility σ" in page                    # tornado rendered with data
        assert "accumulating since" in page

    def test_render_failure_leaves_metrics_untouched(self, tmp_path, monkeypatch):
        run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        before = storage.daily_metrics_path(tmp_path).read_bytes()
        import src.run_daily as rd
        monkeypatch.setattr(rd, "render_page",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        class NextDayEODHD(FakeEODHD):
            def get_underlying_history(self, symbol, start=None):
                df = underlying_frame()
                extra = pd.DataFrame({"date": [TODAY + dt.timedelta(days=1)],
                                      "close": [772.0], "adjusted_close": [772.0],
                                      "volume": [1]})
                return pd.concat([df, extra], ignore_index=True)
        with pytest.raises(RuntimeError):
            run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path,
                today=TODAY + dt.timedelta(days=1))
        assert storage.daily_metrics_path(tmp_path).read_bytes() == before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -q`
Expected: FAIL — `panels_rendered` mismatch; `daily_metrics.parquet` missing; `KeyError: 'atm_iv_30d'`.

- [ ] **Step 3: Modify `run_daily.py`**

Add imports:

```python
from src.analytics.daily_metrics import (
    metric_columns, refresh_rv_columns, session_metrics_row, upsert_session,
)
from src.analytics.greeks_panel import compute_greeks_panel
from src.analytics.iv_rv import iv_rv_series, iv_rv_summary, iv_rv_summary_html
from src.analytics.sensitivity import compute_sensitivity
from src.render.figures import (
    build_greeks_curves_figure, build_iv_rv_figure, build_sensitivity_figure,
    build_smile_figure, build_term_structure_figure,
)
from src.render.tiles import greek_tiles_html
```

Replace the block from `figures = {...}` through the write stage with:

```python
    # ---- P4 Greeks (today + previous session for the 1-day change) ----
    tiles, curves = compute_greeks_panel(chain_iv, risk_free_rate, dividend_yield, cfg)
    tiles_prev = None
    if prior_dates:
        tiles_prev, _ = compute_greeks_panel(prev_iv, risk_free_rate, dividend_yield, cfg)

    # ---- daily_metrics: this session's row + RV refresh over the whole history ----
    metrics = storage.read_daily_metrics(root, metric_columns(cfg))
    metrics = upsert_session(
        metrics, session_metrics_row(session_date, spot, source, ts_today, iv_stats, cfg), cfg)
    metrics = refresh_rv_columns(metrics, storage.read_underlying(root), cfg)
    this_row = metrics[metrics["date"] == session_date].iloc[0]
    atm_iv_30d = float(this_row["atm_iv_30d"])
    atm_dte = float(this_row["atm_iv_30d_dte"])

    # ---- P1 sensitivity: prefer the 30-DTE ATM IV; else the chain's median IV ----
    sigma_p1 = atm_iv_30d if np.isfinite(atm_iv_30d) else float(np.nanmedian(chain_iv["iv"])) \
        if chain_iv["iv"].notna().any() else float("nan")
    dte_p1 = atm_dte if np.isfinite(atm_dte) else float(cfg["target_dte"]["atm_panel"])
    sens = compute_sensitivity(spot, sigma_p1, dte_p1, risk_free_rate, dividend_yield,
                               cfg["sensitivity"]["bump"])

    # ---- P5 ----
    series = iv_rv_series(metrics, cfg)
    summary = iv_rv_summary(series)

    figures = {
        "P1": build_sensitivity_figure(sens),
        "P2": build_smile_figure(smile, spot),
        "P3": build_term_structure_figure(ts_today, ts_prev, previous_label=prev_label),
        "P4": build_greeks_curves_figure(curves, spot),
        "P5": build_iv_rv_figure(series, summary, cfg),
    }
    extras = {"P4": greek_tiles_html(tiles, tiles_prev, prev_label),
              "P5": iv_rv_summary_html(summary)}

    status = {
        "last_success_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": session_date.isoformat(),
        "source": source,
        "rows_stored": int(len(chain)),
        "spot": spot,
        "risk_free_rate": risk_free_rate,
        "dividend_yield": dividend_yield,
        "iv_convergence": round(iv_stats["convergence"], 4),
        "atm_iv_30d": round(atm_iv_30d, 6) if np.isfinite(atm_iv_30d) else None,
        "daily_metrics_rows": int(len(metrics)),
        "history_since": (summary["history_since"].isoformat()
                          if summary["history_since"] else None),
        "panels_rendered": sorted(figures),
    }
    html = render_page(figures, status, extras=extras)

    # ---- write stage: chain -> daily_metrics -> page -> status, each atomic ----
    storage.write_chain(chain, session_date, root)
    storage.write_daily_metrics(metrics, root)
    storage.write_text(html, Path(root) / "docs" / "index.html")
    storage.write_status(status, root)
    return status
```

Add `import numpy as np` at the top. Note `prev_iv` must be hoisted: it is currently assigned inside `if prior_dates:` — keep it there (it is only read under the same condition).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -q` — PASS. Then `.venv/bin/pytest -q` — green, no warnings (warnings are errors).

Watch for: `history_since` in `test_first_run_writes_one_metrics_row` requires `iv_rv_series` to keep the row — but that fixture's `atm_iv_30d` is NaN and `iv_rv_series` drops NaN-IV rows, so `summary["history_since"]` is `None` there. **Resolve by making `status["history_since"]` come from the metrics frame, not the series:** `metrics["date"].iloc[0].isoformat() if len(metrics) else None`. Update the code above accordingly (the summary's own `history_since` still drives the figure annotation and stat line, which is correct — those describe the *plotted* series).

- [ ] **Step 5: Commit**

```bash
git add src/run_daily.py tests/test_pipeline.py
git commit -m "feat: daily run writes daily_metrics and renders P1/P4/P5 — chain, metrics, page, status"
```

---

### Task 8: Real run — rebuild history, render, verify, teach (Phase 4 exit)

**Files:**
- Modify: `LEARNING_LOG.md`; commits include `data/daily_metrics.parquet`, `docs/index.html`, `docs/status.json`, any new `data/chains/*.parquet`.

**Note for the executor:** the daily run needs network + `.env` (sandbox off for that one command). Never print secrets. The rebuild script is offline.

- [ ] **Step 1: Rebuild `daily_metrics` from the 11 stored chains**

Run: `.venv/bin/python scripts/rebuild_daily_metrics.py`
Expected: `rebuilt 11 rows`, one per file in `data/chains/` (2024-09-03, 2026-08-14 … 2026-08-27), `atm_iv_30d` populated for every row whose chain brackets spot (expect all; the term structures have a 22–50 DTE bracket around 30), `rv_20d`/`rv_60d` populated, `fwd_rv_30d` populated **only** for 2024-09-03 (the only session more than 30 days old). Record the table.

- [ ] **Step 2: Real daily run**

Run: `.venv/bin/python -m src.run_daily`
Expected: status prints `panels_rendered: ["P1","P2","P3","P4","P5"]`, `atm_iv_30d` a number near the P3 curve's 30-day level (~12%), `daily_metrics_rows` 11 or 12 (12 if the session rolled to 2026-08-28), `history_since: "2024-09-03"`.

- [ ] **Step 3: Verify the page**

```bash
.venv/bin/python - <<'EOF'
from pathlib import Path
from src.render.page import _BUNDLE
html = Path("docs/index.html").read_text()
own = html.replace(_BUNDLE.read_text(), "")
size = len(html.encode())
assert size < 5_000_000, size
assert "https://cdn.plot.ly" not in own and "<script src=" not in own
for needle in ("Input sensitivity", "Greeks", "Implied vs realized", "class='tiles'",
               "accumulating since", "github.com/TK-Chang239/BlackScholesVolDashboard"):
    assert needle in html, needle
assert "arrives in Phase 4" not in html
print(f"page OK: {size/1e6:.2f} MB, self-contained, Q1+Q3 populated")
EOF
```

Open `docs/index.html` in a browser (or `open docs/index.html`) and eyeball: tornado has σ as the longest bar in the left subplot and S dwarfing everything in the right; tiles show call delta ≈ 0.5; gamma curve peaks at spot; P5 shows a cluster of ~10 points in Aug 2026 and one lone 2024 point.

- [ ] **Step 4: Learning-log entries (teaching document, SPEC §5 learning rule)**

Append under `## Entries`, dated 2026-08-28, four entries — each 2–4 paragraphs, formulas explained, real numbers from Steps 1–3 quoted:

1. **"Realized volatility: measuring what happened"** — why log returns (time-additive), why *adjusted* close (a dividend is not a down-move; cite `test_uses_adjusted_close_not_close`), why `ddof=1`, why √252, what trailing vs forward RV each answer and why forward is the honest forecast target for a 30-DTE IV. Quote today's `rv_20d`, `rv_60d`, and the single matured `fwd_rv_30d` (2024-09-03) from the rebuilt table.
2. **"Sensitivity: which inputs matter"** — the tornado numbers for all five inputs at ±20%; the near-linearity of price in σ at the money (`price ≈ 0.4·S·σ·√T`, so a 20% σ bump ≈ 20% price move); why r and q barely register at 30 days; the honest statement that S dominates everything in absolute terms and why the panel groups it separately (this is the plan's documented deviation from SPEC §3 P1's literal "σ dominates").
3. **"Greeks in practice"** — the display-scaling convention and *why* the model layer refuses to do it; the ATM call delta ≈ e^{-qT}N(d1) ≈ 0.5+; put–call delta parity `Δc − Δp = e^{-qT}` (cite `test_put_call_delta_parity`); gamma peaking near the forward, not exactly spot; what the 1-day change tiles are comparing (today's session vs the prior stored session, which may be close-based — label). Quote today's tiles.
4. **"Phase 4: IV vs RV on real history"** — the state of P5: how many sessions, how many evaluable days (expect 1 — the 2024 probe), the current IV − trailing-RV spread and its sign, the variance-risk-premium framing, and the honest limits: (a) 11 sessions is not a time series yet; (b) the 2024-09-03 point sits alone two years left of the cluster and stretches the x-axis — a rendering ugliness accepted for now (a "recent window" toggle is a possible follow-up); (c) r/q replay approximation for historical rows; (d) the SPEC's "≥ 3 months of history" acceptance for P5 is explicitly *not* yet met and will be met by accumulation, not by code.

- [ ] **Step 5: Full suite, then commit**

```bash
.venv/bin/pytest -q
git add data/daily_metrics.parquet data/chains data/underlying.parquet docs/ LEARNING_LOG.md
git commit -m "feat: Phase 4 — daily_metrics history, P1 sensitivity, P4 Greeks, P5 IV-vs-RV from real data"
```

- [ ] **Step 6: Report the hand-off**

Report to the controller: Phase 4 exit ("panels render from real stored history") is met locally and committed. Remaining user actions (unchanged from Phase 3): enable GitHub Pages on `main` → `/docs`; after that, `git push` publishes. Mention that `scripts/rebuild_daily_metrics.py` is safe to re-run any time (stateless) — e.g. after the wide backfill lands more chains.

---

## Self-Review Notes (already applied)

- **Spec coverage.** §2.3 `realized_vol.py` (T1: adjusted close, log returns, √252, 20d/60d). §2.2 `daily_metrics.parquet` one-row-per-day with ATM IV, trailing RV, forward RV back-filled (T2 refresh + T3 rebuild + T7 daily write). §3 P1 compute/output (T4), acceptance handled via the documented two-group deviation in Global Constraints. §3 P4 tiles + 1-day change + delta/gamma-vs-strike (T5), acceptance "delta ≈ 0.5 ATM, gamma peaks ATM" tested. §3 P5 IV, trailing RV, forward RV, shaded spread, running mean, forward-based summary stat (T6). §2.5 captions/mobile/self-contained (T6 + T8 check). §2.1 failure policy extended to `daily_metrics` (T7 test). Phase 4 exit criterion (T8). P6's skew column is Phase 5 — `read_daily_metrics`'s `reindex` makes adding it schema-safe.
- **Deliberately deferred (YAGNI):** bid/ask IV band toggle (P2, still optional), a "recent window" zoom for P5's lone 2024 point, per-panel mobile tuning beyond the responsive config (Phase 7's mobile pass).
- **Type consistency.** `compute_greeks_panel(chain_iv, r, q, cfg) -> (tiles, curves)` used identically in T5 tests and T7. `interp_atm_iv` returns `(atm_iv, effective_dte)`; `session_metrics_row` stores them as `atm_iv_30d`/`atm_iv_30d_dte`; T7 reads those names. `iv_rv_series(metrics, cfg)` → `iv_rv_summary(series)` → `build_iv_rv_figure(series, summary, cfg)` / `iv_rv_summary_html(summary)` chain matches across T6/T7. `render_page(figures, status, extras=None)` matches T6 tests and T7 call. Storage names (`read_underlying`, `daily_metrics_path`, `read_daily_metrics`, `write_daily_metrics`) match across T2/T3/T7.
- **Known fixture limitation.** The shared `chain_frame()` pipeline fixture has one strike (700), so ATM IV is NaN there; T7's tests assert that honestly and use a bracketing chain where a real value is required — same pattern Phase 3 used for the overlay test.
- **Placeholder scan:** no TBD/TODO; every code step carries its code; the one "use either" alternative in T5 is a complete second implementation, not a gap.
