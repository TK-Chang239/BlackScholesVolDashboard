# vol-lens Phase 3 — IV Panels & First Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run this project's own IV solver over stored chains and publish the first dashboard — the P2 smile and P3 ATM term-structure panels in one self-contained `docs/index.html`.

**Architecture:** A shared analytics step (`chain_iv`) routes price by source (mid for live, close otherwise), inverts it with the Phase 1 solver, and reports convergence. Panel modules (`smile`, `term_structure`) are pure `(chain_iv_df, cfg) → tidy DataFrame` functions per SPEC §2.4. The render layer builds Plotly figures and assembles one static page around a vendored, pinned plotly.js cartesian bundle (self-contained, no CDN at view time). `run_daily` gains compute→render stages with a strict compute-everything-then-write-everything ordering so a failure anywhere leaves yesterday's page and status untouched. A small backfill-hardening task (per-day error isolation) clears the path for the user's wide backfill run.

**Tech Stack:** plotly (Python, pinned) · vendored plotly.js cartesian bundle · pandas · existing model + data layers · pytest

**Spec:** `SPEC.md` — §2.4, §2.5, §3 P2/P3 (binding), §2.1 failure policy, §5 Phase 3 exit criterion: "Convergence ≥ 95% on liquid quotes; first `index.html` published to Pages." Publishing needs a GitHub remote — a user action; this plan's exit is a verified local `docs/index.html` plus hand-off instructions.

## Global Constraints

- Vendor IV (`vendor_iv`) is never used in computations — all IVs come from `src/models/black_scholes.implied_vol` (SPEC §2.2).
- Price routing: `mid` for `source == "yfinance"` rows, `close` otherwise (SPEC §2.2).
- Analytics functions are pure: `(DataFrame, cfg) → tidy DataFrame`, no I/O, individually unit-testable (SPEC §2.4).
- One self-contained `docs/index.html` < 5 MB; no CDN/external requests at view time; mobile-readable (viewport meta, responsive figures) (SPEC §2.5).
- Page layout per SPEC §2.5: header (title, underlying, spot, last-updated) → the five guiding questions as section headings → panels under their question (P2/P3 under Q2; other sections carry an honest "arrives in Phase N" line) → footer (status, disclaimer, GitHub link). Every panel gets a 2–3 sentence caption (text specified in Task 4 — use verbatim).
- Failure policy (SPEC §2.1): ALL computation (chain, IVs, panels, HTML string, status dict) completes before ANY write; write order chain → index.html → status.json, each atomic. A failure anywhere leaves the previous page and status live.
- T = `dte / 365.0` (calendar-day year fraction) everywhere in analytics.
- `r` and `q` for IV inversion are the run's current values (from EODHD with config fallbacks). The yesterday-overlay reuses today's r/q — documented approximation, immaterial at these tenors (the sensitivity panel will demonstrate exactly this).
- pytest offline, warnings-as-errors; TDD per task; commit per task. `.venv/bin/pytest`, `.venv/bin/python`.
- Model-layer note: if you touch `src/models/black_scholes.py`, rename `_d1_d2`'s argument order to `(S, K, T, r, sigma, q)` and update its three call sites + docstring (deferred Phase 1 minor). No task below requires touching it.

---

## File Structure

```
src/analytics/
├── __init__.py            # Task 1 (empty)
├── chain_iv.py            # Task 1 — price routing + IV solve + convergence stats
├── smile.py               # Task 2 — P2 tidy frame (3 expiries, OTM convention)
└── term_structure.py      # Task 3 — P3 tidy frame (interpolated ATM IV per expiry)
src/render/
├── __init__.py            # Task 4 (empty)
├── assets/plotly-cartesian-2.35.2.min.js   # Task 4 — vendored, pinned
├── figures.py             # Task 4 — Plotly figure builders
└── page.py                # Task 4 — page assembly (captions live here)
src/run_daily.py           # Task 5 — modify: compute+render stages, iv_convergence in status
src/data/storage.py        # Task 5 — modify: add atomic write_text
src/data/backfill.py       # Task 6 — modify: per-day error isolation (dates_failed)
tests/test_chain_iv.py     # Task 1
tests/test_smile.py        # Task 2
tests/test_term_structure.py  # Task 3
tests/test_render.py       # Task 4
tests/test_pipeline.py     # Task 5 — extend
tests/test_backfill.py     # Task 6 — extend
```

Shared test helper: Tasks 1–3 build synthetic chains through `bs_price` with *known* vols, so the solver's recovery is the assertion — no magic expected-IV constants. Each test file defines its own `synthetic_chain` (deliberate duplication; test files must stay independently runnable).

---

### Task 1: `chain_iv` — price routing, IV solve, convergence stats

**Files:**
- Create: `src/analytics/__init__.py` (empty), `src/analytics/chain_iv.py`
- Test: `tests/test_chain_iv.py`

**Interfaces:**
- Consumes: `src.models.black_scholes.implied_vol(price, S, K, T, r, q, kind)`; `src.data.base.LIVE_SOURCES`; chain frames with `CHAIN_COLUMNS`.
- Produces: `compute_chain_iv(chain: pd.DataFrame, r: float, q: float) -> tuple[pd.DataFrame, dict]` — returns a copy of the chain with two added columns, `price_used` (float, NaN where no usable price) and `iv` (float, NaN where unsolvable), plus stats `{"n_priced": int, "n_converged": int, "convergence": float}` where convergence = n_converged / n_priced (0.0 when n_priced == 0). Tasks 2–3 consume the returned frame; Task 5 consumes stats.

- [ ] **Step 1: Write the failing tests**

`tests/test_chain_iv.py`:

```python
"""chain_iv: source-routed pricing + IV recovery of known synthetic vols."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def synthetic_chain(source="yfinance", sigma_fn=lambda k, t: 0.22):
    """Chain whose prices come from bs_price with known per-row vols."""
    rows = []
    for expiry, dte in ((dt.date(2026, 9, 18), 21), (dt.date(2026, 11, 20), 84)):
        for strike in (700.0, 740.0, 770.0, 800.0, 840.0):
            for kind in ("call", "put"):
                sigma = sigma_fn(strike, dte)
                price = bs_price(S=SPOT, K=strike, T=dte / 365.0, r=R,
                                 sigma=sigma, q=Q, kind=kind)
                is_live = source == "yfinance"
                rows.append({
                    "snapshot_date": TODAY, "spot": SPOT, "expiry": expiry,
                    "dte": dte, "strike": strike, "kind": kind,
                    "bid": price * 0.99 if is_live else np.nan,
                    "ask": price * 1.01 if is_live else np.nan,
                    "mid": price if is_live else np.nan,
                    "close": price * 1.02 if is_live else price,
                    "volume": 10, "open_interest": 100,
                    "vendor_iv": 0.99,  # poison: must never be read
                    "source": source,
                })
    return pd.DataFrame(rows, columns=CHAIN_COLUMNS)


class TestPriceRouting:
    def test_live_rows_use_mid_not_close(self):
        df, _ = compute_chain_iv(synthetic_chain("yfinance"), R, Q)
        # close was poisoned at 1.02x; recovery to the exact vol proves mid was used
        assert np.allclose(df["iv"], 0.22, atol=1e-6)
        assert np.allclose(df["price_used"], df["mid"])

    def test_backfill_rows_use_close(self):
        df, _ = compute_chain_iv(synthetic_chain("massive-backfill"), R, Q)
        assert np.allclose(df["iv"], 0.22, atol=1e-6)
        assert np.allclose(df["price_used"], df["close"])


class TestRecovery:
    def test_recovers_a_skewed_smile(self):
        skew = lambda k, t: 0.20 + 0.30 * max(0.0, (770.0 - k) / 770.0)
        df, stats = compute_chain_iv(synthetic_chain(sigma_fn=skew), R, Q)
        for _, row in df.iterrows():
            assert row["iv"] == pytest.approx(skew(row["strike"], row["dte"]), abs=1e-6)
        assert stats["convergence"] == 1.0

    def test_vendor_iv_never_consulted(self):
        df, _ = compute_chain_iv(synthetic_chain(), R, Q)
        assert not np.allclose(df["iv"], df["vendor_iv"])


class TestStats:
    def test_unpriceable_row_counts(self):
        chain = synthetic_chain()
        chain.loc[0, "mid"] = np.nan          # live row with no mid -> no usable price
        chain.loc[1, "mid"] = 1e9             # absurd price -> no-arb NaN from solver
        df, stats = compute_chain_iv(chain, R, Q)
        assert np.isnan(df.loc[0, "iv"]) and np.isnan(df.loc[1, "iv"])
        assert stats["n_priced"] == len(chain) - 1        # NaN price excluded
        assert stats["n_converged"] == len(chain) - 2     # absurd price counted, failed
        assert stats["convergence"] == pytest.approx((len(chain) - 2) / (len(chain) - 1))

    def test_empty_chain(self):
        df, stats = compute_chain_iv(synthetic_chain().iloc[0:0], R, Q)
        assert df.empty and stats == {"n_priced": 0, "n_converged": 0, "convergence": 0.0}

    def test_input_not_mutated(self):
        chain = synthetic_chain()
        before = chain.copy()
        compute_chain_iv(chain, R, Q)
        pd.testing.assert_frame_equal(chain, before)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_chain_iv.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

`src/analytics/chain_iv.py`:

```python
"""Shared analytics step: invert stored chain prices into implied vols.

Price routing per SPEC 2.2: mid for live (yfinance) rows, close otherwise.
All IVs come from our own solver -- vendor_iv is never read. Convergence
stats feed status.json and the Phase 3 exit criterion (>= 95% on liquid
quotes).
"""
import numpy as np
import pandas as pd

from src.data.base import LIVE_SOURCES
from src.models.black_scholes import implied_vol


def compute_chain_iv(chain: pd.DataFrame, r: float, q: float) -> tuple[pd.DataFrame, dict]:
    df = chain.copy()
    live = df["source"].isin(LIVE_SOURCES)
    df["price_used"] = np.where(live, df["mid"], df["close"])
    df["iv"] = np.nan
    if not df.empty:
        t_years = df["dte"].to_numpy(dtype=float) / 365.0
        for kind in ("call", "put"):
            m = (df["kind"] == kind).to_numpy()
            if not m.any():
                continue
            df.loc[m, "iv"] = implied_vol(
                df.loc[m, "price_used"].to_numpy(),
                S=df.loc[m, "spot"].to_numpy(dtype=float),
                K=df.loc[m, "strike"].to_numpy(dtype=float),
                T=t_years[m], r=r, q=q, kind=kind,
            )
    n_priced = int(df["price_used"].notna().sum())
    n_converged = int(df["iv"].notna().sum())
    stats = {
        "n_priced": n_priced,
        "n_converged": n_converged,
        "convergence": (n_converged / n_priced) if n_priced else 0.0,
    }
    return df, stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_chain_iv.py -v`, then full suite (91 prior + new).
Expected: all pass, no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/analytics/ tests/test_chain_iv.py
git commit -m "feat: chain IV analytics — source-routed pricing, own-solver inversion, convergence stats"
```

---

### Task 2: P2 smile analytics

**Files:**
- Create: `src/analytics/smile.py`
- Test: `tests/test_smile.py`

**Interfaces:**
- Consumes: the `compute_chain_iv` output frame (has `iv`, plus all `CHAIN_COLUMNS`); `cfg["target_dte"]["smile_expiries"]` = [30, 90, 180].
- Produces: `compute_smile(chain_iv: pd.DataFrame, cfg: dict) -> pd.DataFrame` with columns exactly `["expiry", "dte", "strike", "moneyness", "kind", "iv"]`, sorted by (expiry, strike), fresh index — OTM convention (puts below spot, calls at/above spot), converged rows only, at most 3 expiries (each the nearest stored expiry to one target DTE, deduplicated).

- [ ] **Step 1: Write the failing tests**

`tests/test_smile.py`:

```python
"""P2 smile frame: expiry targeting, OTM convention, tidy output."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.smile import compute_smile
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098
CFG = {"target_dte": {"smile_expiries": [30, 90, 180]}}

EXPIRIES = [  # (expiry, dte) — six monthlies like a real stored chain
    (dt.date(2026, 9, 18), 21), (dt.date(2026, 10, 16), 49),
    (dt.date(2026, 11, 20), 84), (dt.date(2026, 12, 18), 112),
    (dt.date(2027, 1, 15), 140), (dt.date(2027, 2, 19), 175),
]


def chain_with_skew():
    rows = []
    for expiry, dte in EXPIRIES:
        for strike in (700.0, 740.0, 770.0, 800.0, 840.0):
            for kind in ("call", "put"):
                sigma = 0.20 + 0.25 * max(0.0, (SPOT - strike) / SPOT)
                price = bs_price(S=SPOT, K=strike, T=dte / 365.0, r=R,
                                 sigma=sigma, q=Q, kind=kind)
                rows.append({
                    "snapshot_date": TODAY, "spot": SPOT, "expiry": expiry,
                    "dte": dte, "strike": strike, "kind": kind,
                    "bid": price * 0.99, "ask": price * 1.01, "mid": price,
                    "close": price, "volume": 10, "open_interest": 100,
                    "vendor_iv": np.nan, "source": "yfinance",
                })
    df = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    out, _ = compute_chain_iv(df, R, Q)
    return out


class TestSmile:
    def test_selects_nearest_expiry_per_target(self):
        got = compute_smile(chain_with_skew(), CFG)
        # nearest to 30 -> 21d (9/18); to 90 -> 84d (11/20); to 180 -> 175d (2/19)
        assert sorted(got["dte"].unique()) == [21, 84, 175]

    def test_otm_convention(self):
        got = compute_smile(chain_with_skew(), CFG)
        puts, calls = got[got["kind"] == "put"], got[got["kind"] == "call"]
        assert (puts["strike"] < SPOT).all()
        assert (calls["strike"] >= SPOT).all()
        # exactly one row per strike per expiry after OTM split
        assert not got.duplicated(["expiry", "strike"]).any()

    def test_downward_put_skew_visible(self):
        got = compute_smile(chain_with_skew(), CFG)
        one = got[got["dte"] == 21].sort_values("strike")
        assert one["iv"].iloc[0] > one["iv"].iloc[-1]  # 700-strike IV > 840-strike IV

    def test_output_schema_and_moneyness(self):
        got = compute_smile(chain_with_skew(), CFG)
        assert list(got.columns) == ["expiry", "dte", "strike", "moneyness", "kind", "iv"]
        assert got["moneyness"].between(0.70, 1.30).all()
        assert got.index.tolist() == list(range(len(got)))
        assert got["iv"].notna().all()

    def test_duplicate_targets_collapse(self):
        # only two expiries stored: targets 30/90/180 must not repeat one
        df = chain_with_skew()
        df = df[df["dte"].isin([21, 84])]
        got = compute_smile(df, CFG)
        assert sorted(got["dte"].unique()) == [21, 84]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_smile.py -v` — ImportError expected.

- [ ] **Step 3: Implement**

`src/analytics/smile.py`:

```python
"""P2: IV smile by expiry (SPEC 3 P2). Pure function, tidy output.

OTM convention: puts carry the smile below spot, calls at/above spot --
one continuous line per expiry built from the options people actually
trade at each strike.
"""
import pandas as pd

SMILE_COLUMNS = ["expiry", "dte", "strike", "moneyness", "kind", "iv"]


def compute_smile(chain_iv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    spot = float(chain_iv["spot"].iloc[0]) if len(chain_iv) else float("nan")
    available = chain_iv[["expiry", "dte"]].drop_duplicates().reset_index(drop=True)
    chosen: list = []
    for target in cfg["target_dte"]["smile_expiries"]:
        if available.empty:
            break
        nearest = available.iloc[(available["dte"] - target).abs().idxmin()]["expiry"]
        if nearest not in chosen:
            chosen.append(nearest)
    df = chain_iv[chain_iv["expiry"].isin(chosen)].copy()
    otm = ((df["kind"] == "put") & (df["strike"] < spot)) | (
        (df["kind"] == "call") & (df["strike"] >= spot)
    )
    df = df[otm & df["iv"].notna()].copy()
    df["moneyness"] = df["strike"] / spot
    return (df[SMILE_COLUMNS]
            .sort_values(["expiry", "strike"])
            .reset_index(drop=True))
```

Note: `idxmin` on the abs-difference works because `available` has a fresh 0..n index (`reset_index` above) — keep that line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_smile.py -v`, then full suite. All green, no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/analytics/smile.py tests/test_smile.py
git commit -m "feat: P2 smile analytics — expiry targeting and OTM convention"
```

---

### Task 3: P3 ATM term-structure analytics

**Files:**
- Create: `src/analytics/term_structure.py`
- Test: `tests/test_term_structure.py`

**Interfaces:**
- Consumes: `compute_chain_iv` output frame.
- Produces: `compute_term_structure(chain_iv: pd.DataFrame) -> pd.DataFrame` with columns exactly `["expiry", "dte", "atm_iv"]`, sorted by dte, fresh index. ATM IV per expiry = linear interpolation of IV at K = spot, done separately for calls and puts (each needs a converged strike on both sides of spot), then averaged over the kinds that interpolated. Expiries where neither kind brackets spot are dropped.

- [ ] **Step 1: Write the failing tests**

`tests/test_term_structure.py`:

```python
"""P3 term structure: strike-space ATM interpolation per expiry."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.term_structure import compute_term_structure
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def make_chain(expiries, strikes=(740.0, 760.0, 780.0, 800.0), sigma_by_dte=None):
    sigma_by_dte = sigma_by_dte or {}
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                sigma = sigma_by_dte.get(dte, 0.22)
                price = bs_price(S=SPOT, K=strike, T=dte / 365.0, r=R,
                                 sigma=sigma, q=Q, kind=kind)
                rows.append({
                    "snapshot_date": TODAY, "spot": SPOT, "expiry": expiry,
                    "dte": dte, "strike": strike, "kind": kind,
                    "bid": price * 0.99, "ask": price * 1.01, "mid": price,
                    "close": price, "volume": 10, "open_interest": 100,
                    "vendor_iv": np.nan, "source": "yfinance",
                })
    df = pd.DataFrame(rows, columns=CHAIN_COLUMNS)
    out, _ = compute_chain_iv(df, R, Q)
    return out


class TestTermStructure:
    def test_flat_vol_interpolates_exactly(self):
        chain = make_chain([(dt.date(2026, 9, 18), 21), (dt.date(2026, 11, 20), 84)],
                           sigma_by_dte={21: 0.18, 84: 0.24})
        got = compute_term_structure(chain)
        assert got["dte"].tolist() == [21, 84]
        assert got["atm_iv"].iloc[0] == pytest.approx(0.18, abs=1e-5)
        assert got["atm_iv"].iloc[1] == pytest.approx(0.24, abs=1e-5)

    def test_interpolation_is_linear_between_bracketing_strikes(self):
        # give the two strikes around spot different known vols; ATM must be
        # the strike-linear blend at spot=770 between K=760 and K=780
        chain = make_chain([(dt.date(2026, 9, 18), 21)], strikes=(760.0, 780.0))
        # overwrite IVs directly with a synthetic wedge: iv = a + b*K
        chain["iv"] = 0.10 + (chain["strike"] - 760.0) * 0.001  # 760->0.10, 780->0.12
        got = compute_term_structure(chain)
        assert got["atm_iv"].iloc[0] == pytest.approx(0.11, abs=1e-12)

    def test_unbracketed_expiry_dropped(self):
        chain = make_chain([(dt.date(2026, 9, 18), 21)], strikes=(780.0, 800.0, 820.0))
        got = compute_term_structure(chain)  # all strikes above spot: no bracket
        assert got.empty

    def test_one_kind_bracketing_suffices(self):
        chain = make_chain([(dt.date(2026, 9, 18), 21)], strikes=(760.0, 780.0))
        chain.loc[chain["kind"] == "put", "iv"] = np.nan  # puts unusable
        got = compute_term_structure(chain)
        assert len(got) == 1 and got["atm_iv"].notna().all()

    def test_schema_and_sorting(self):
        chain = make_chain([(dt.date(2026, 11, 20), 84), (dt.date(2026, 9, 18), 21)])
        got = compute_term_structure(chain)
        assert list(got.columns) == ["expiry", "dte", "atm_iv"]
        assert got["dte"].tolist() == sorted(got["dte"].tolist())
        assert got.index.tolist() == list(range(len(got)))

    def test_empty_input(self):
        chain = make_chain([(dt.date(2026, 9, 18), 21)]).iloc[0:0]
        got = compute_term_structure(chain)
        assert got.empty and list(got.columns) == ["expiry", "dte", "atm_iv"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_term_structure.py -v` — ImportError expected.

- [ ] **Step 3: Implement**

`src/analytics/term_structure.py`:

```python
"""P3: ATM IV term structure (SPEC 3 P3). Pure function, tidy output.

ATM IV per expiry: linear interpolation in strike at K = spot, computed
per kind (needs a converged IV strictly below-or-at and strictly above
spot), then averaged across the kinds that could interpolate -- parity
says call and put ATM IVs should agree; averaging smooths quote noise.
"""
import numpy as np
import pandas as pd

TERM_COLUMNS = ["expiry", "dte", "atm_iv"]


def _interp_at(spot: float, g: pd.DataFrame) -> float | None:
    g = g[g["iv"].notna()].sort_values("strike")
    below = g[g["strike"] <= spot].tail(1)
    above = g[g["strike"] > spot].head(1)
    if below.empty or above.empty:
        return None
    k0, v0 = float(below["strike"].iloc[0]), float(below["iv"].iloc[0])
    k1, v1 = float(above["strike"].iloc[0]), float(above["iv"].iloc[0])
    if k1 == k0:
        return v0
    return v0 + (v1 - v0) * (spot - k0) / (k1 - k0)


def compute_term_structure(chain_iv: pd.DataFrame) -> pd.DataFrame:
    if chain_iv.empty:
        return pd.DataFrame(columns=TERM_COLUMNS)
    spot = float(chain_iv["spot"].iloc[0])
    rows = []
    for (expiry, dte), g in chain_iv.groupby(["expiry", "dte"], sort=False):
        kind_ivs = [
            v for kind in ("call", "put")
            if (v := _interp_at(spot, g[g["kind"] == kind])) is not None
        ]
        if kind_ivs:
            rows.append({"expiry": expiry, "dte": int(dte),
                         "atm_iv": float(np.mean(kind_ivs))})
    return (pd.DataFrame(rows, columns=TERM_COLUMNS)
            .sort_values("dte").reset_index(drop=True))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_term_structure.py -v`, then full suite. All green.

- [ ] **Step 5: Commit**

```bash
git add src/analytics/term_structure.py tests/test_term_structure.py
git commit -m "feat: P3 term-structure analytics — strike-space ATM interpolation"
```

---

### Task 4: Render layer — vendored plotly, figures, page assembly

**Files:**
- Create: `src/render/__init__.py` (empty), `src/render/assets/plotly-cartesian-2.35.2.min.js` (downloaded), `src/render/figures.py`, `src/render/page.py`
- Modify: `requirements.txt`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: the smile and term-structure tidy frames; a status dict shaped like `run_daily`'s.
- Produces:
  - `figures.build_smile_figure(smile: pd.DataFrame, spot: float) -> plotly.graph_objects.Figure`
  - `figures.build_term_structure_figure(today: pd.DataFrame, previous: pd.DataFrame | None) -> Figure`
  - `page.render_page(figures: dict[str, "Figure"], status: dict) -> str` — the complete HTML document, self-contained, `figures` keyed `"P2"`/`"P3"` (a missing key renders that panel's placeholder).

**Network note for the executor:** Step 1 downloads one file from cdn.plot.ly (sandbox off for that command). Everything else is offline.

- [ ] **Step 1: Pin plotly + vendor the JS bundle**

Append to `requirements.txt`:

```
plotly==5.24.1
```

`.venv/bin/pip install -r requirements.txt`

Download the cartesian partial bundle matching plotly 5.24.1's embedded plotly.js (2.35.2):

```bash
curl -fsSL https://cdn.plot.ly/plotly-cartesian-2.35.2.min.js -o src/render/assets/plotly-cartesian-2.35.2.min.js
ls -la src/render/assets/  # expect roughly 1.0–1.4 MB
```

If the CDN download fails, fall back to copying the full bundle from the installed package (`.venv/bin/python -c "import plotly, pathlib, shutil; src = pathlib.Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'; shutil.copy(src, 'src/render/assets/plotly-cartesian-2.35.2.min.js')"`) — larger (~3.5 MB) but still under the 5 MB page budget; note which one you shipped in your report.

- [ ] **Step 2: Write the failing tests**

`tests/test_render.py`:

```python
"""Render layer: figures carry the right traces; page is self-contained."""
import datetime as dt

import pandas as pd
import pytest

from src.render.figures import build_smile_figure, build_term_structure_figure
from src.render.page import render_page

SPOT = 770.0


def smile_frame():
    return pd.DataFrame({
        "expiry": [dt.date(2026, 9, 18)] * 4 + [dt.date(2026, 11, 20)] * 4,
        "dte": [21] * 4 + [84] * 4,
        "strike": [700.0, 740.0, 770.0, 800.0] * 2,
        "moneyness": [k / SPOT for k in (700.0, 740.0, 770.0, 800.0)] * 2,
        "kind": ["put", "put", "call", "call"] * 2,
        "iv": [0.28, 0.24, 0.21, 0.20, 0.26, 0.23, 0.215, 0.205],
    })


def term_frame(shift=0.0):
    return pd.DataFrame({
        "expiry": [dt.date(2026, 9, 18), dt.date(2026, 11, 20)],
        "dte": [21, 84],
        "atm_iv": [0.21 + shift, 0.23 + shift],
    })


def fake_status():
    return {
        "last_success_utc": "2026-08-28T21:35:00+00:00",
        "snapshot_date": "2026-08-28", "source": "yfinance",
        "rows_stored": 400, "spot": SPOT,
        "risk_free_rate": 0.0415, "dividend_yield": 0.0098,
        "iv_convergence": 0.985,
    }


class TestFigures:
    def test_smile_one_trace_per_expiry_plus_atm_marker(self):
        fig = build_smile_figure(smile_frame(), SPOT)
        # expiry traces are lines+markers; the ATM reference is lines-only
        expiry_traces = [t for t in fig.data if t.mode and "markers" in t.mode]
        assert len(expiry_traces) == 2  # one per expiry
        assert any("ATM" in (t.name or "") for t in fig.data)

    def test_term_structure_overlay_present_and_absent(self):
        fig2 = build_term_structure_figure(term_frame(), term_frame(shift=0.01))
        assert len(fig2.data) == 2
        fig1 = build_term_structure_figure(term_frame(), None)
        assert len(fig1.data) == 1

    def test_term_overlay_is_muted(self):
        fig = build_term_structure_figure(term_frame(), term_frame(shift=0.01))
        prev = [t for t in fig.data if "yesterday" in (t.name or "").lower()]
        assert len(prev) == 1


class TestPage:
    def page(self, **kw):
        figs = {"P2": build_smile_figure(smile_frame(), SPOT),
                "P3": build_term_structure_figure(term_frame(), None)}
        figs.update(kw.pop("figures", {}))
        return render_page(figs, fake_status())

    def test_self_contained_no_external_refs(self):
        html = self.page()
        assert "https://cdn.plot.ly" not in html
        assert "<script src=" not in html          # all scripts inline
        assert html.count("<html") == 1

    def test_bundle_inlined(self):
        html = self.page()
        assert len(html) > 500_000                  # the vendored bundle is in there
        assert len(html.encode()) < 5_000_000       # SPEC 2.5 budget

    def test_structure_header_questions_footer(self):
        html = self.page()
        assert "viewport" in html
        for must in ("SPY", "770", "2026-08-28",     # header: underlying, spot, date
                     "Q1", "Q2", "Q3", "Q4", "Q5",   # five question sections
                     "Educational project",           # footer disclaimer
                     "yfinance"):                     # data-source transparency
            assert must in html, must

    def test_missing_panel_renders_placeholder(self):
        html = render_page({}, fake_status())
        assert "Phase" in html                       # 'arrives in Phase N' notes
        assert "<html" in html
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_render.py -v` — ImportError expected.

- [ ] **Step 4: Implement**

`src/render/figures.py`:

```python
"""Plotly figure builders for the dashboard panels. Pure: frames in, Figure out."""
import pandas as pd
import plotly.graph_objects as go

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=50, r=20, t=40, b=45),
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    font=dict(size=13),
)


def build_smile_figure(smile: pd.DataFrame, spot: float) -> go.Figure:
    fig = go.Figure()
    for (expiry, dte), g in smile.groupby(["expiry", "dte"], sort=True):
        g = g.sort_values("strike")
        fig.add_trace(go.Scatter(
            x=g["moneyness"], y=g["iv"], mode="lines+markers",
            name=f"{expiry.isoformat()} ({dte}d)",
            hovertemplate="K/S %{x:.3f}<br>IV %{y:.1%}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[1.0, 1.0], y=[smile["iv"].min(), smile["iv"].max()],
        mode="lines", name="ATM (K = S)",
        line=dict(dash="dot", color="grey", width=1),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="Implied volatility smile — solved from market quotes",
        xaxis_title="Moneyness K/S", yaxis_title="Implied vol",
        yaxis_tickformat=".0%", **_LAYOUT)
    return fig


def build_term_structure_figure(today: pd.DataFrame,
                                previous: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()
    if previous is not None and len(previous):
        fig.add_trace(go.Scatter(
            x=previous["dte"], y=previous["atm_iv"], mode="lines+markers",
            name="yesterday", line=dict(color="lightgrey"),
            marker=dict(color="lightgrey"),
            hovertemplate="%{x}d: %{y:.1%}<extra>yesterday</extra>"))
    fig.add_trace(go.Scatter(
        x=today["dte"], y=today["atm_iv"], mode="lines+markers",
        name="today",
        hovertemplate="%{x}d: %{y:.1%}<extra>today</extra>"))
    fig.update_layout(
        title="ATM implied vol term structure",
        xaxis_title="Days to expiry", yaxis_title="ATM implied vol",
        yaxis_tickformat=".0%", **_LAYOUT)
    return fig
```

`src/render/page.py`:

```python
"""Assemble the one static dashboard page (SPEC 2.5).

Self-contained: the vendored plotly.js bundle is inlined; no external
requests at view time. Captions are part of the deliverable -- they carry
the interview-prep story (SPEC 2.5) -- keep them verbatim.
"""
from pathlib import Path

import plotly.io as pio

_BUNDLE = Path(__file__).parent / "assets" / "plotly-cartesian-2.35.2.min.js"

CAPTIONS = {
    "P2": (
        "Black-Scholes assumes one volatility should price every strike. The "
        "market disagrees: implied vol rises steadily for lower strikes — the "
        "skew — because crash protection commands a premium the model does not "
        "anticipate. Each line is one expiry, and every point was solved from a "
        "real quote by this project's own Newton–Brent inverter."
    ),
    "P3": (
        "Each point is the at-the-money implied vol for one expiry — the "
        "market's volatility price over that horizon. An upward slope reads as "
        "calm now, more risk priced later; kinks flag scheduled events such as "
        "Fed meetings. Yesterday's curve in grey shows the day-over-day shift."
    ),
}

QUESTIONS = [
    ("Q1", "Everyone has the same formula. Why does anyone disagree about prices?",
     [], "Input-sensitivity panel arrives in Phase 4."),
    ("Q2", "Where does the market ignore the model, and why?",
     ["P2", "P3"], "Model-vs-market heatmap arrives in Phase 5."),
    ("Q3", "Is implied volatility a prediction, or a price?",
     [], "Implied-vs-realized panel arrives in Phase 4."),
    ("Q4", "The model claims you can hedge an option perfectly. Can you?",
     [], "Delta-hedged P&L simulation arrives in Phase 6."),
    ("Q5", "Given all these flaws, why does every desk still use Black-Scholes?",
     [], "Put-call parity checker arrives in Phase 5."),
]

_PANEL_TITLES = {"P2": "IV smile / skew by expiry", "P3": "ATM IV term structure"}

_CSS = """
body { font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
       max-width: 960px; margin: 0 auto; padding: 0 16px 48px; color: #1a1a2e; }
h1 { margin-top: 28px; } h2 { margin-top: 40px; border-bottom: 2px solid #eee;
     padding-bottom: 6px; } .meta, .caption, .placeholder, footer
     { color: #555; font-size: 0.92rem; } .caption { margin: 4px 0 24px; }
.placeholder { font-style: italic; } footer { margin-top: 48px;
     border-top: 1px solid #eee; padding-top: 12px; }
.figure { width: 100%; overflow-x: auto; }
"""


def render_page(figures: dict, status: dict) -> str:
    bundle = _BUNDLE.read_text()
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>vol-lens — Black-Scholes vs the market, daily</title>",
        f"<style>{_CSS}</style>",
        f"<script>{bundle}</script>",
        "</head><body>",
        "<h1>vol-lens</h1>",
        "<p class='meta'>A daily reality-check of the Black-Scholes model "
        "against the live US options market. Underlying: <b>SPY</b> · "
        f"spot <b>{status['spot']:.2f}</b> · session "
        f"<b>{status['snapshot_date']}</b> · chain source "
        f"<b>{status['source']}</b> · IV solver convergence "
        f"<b>{status.get('iv_convergence', float('nan')):.1%}</b></p>",
    ]
    for qid, question, panel_ids, coming in QUESTIONS:
        parts.append(f"<h2>{qid}. {question}</h2>")
        rendered_any = False
        for pid in panel_ids:
            if pid in figures:
                parts.append(f"<h3>{_PANEL_TITLES[pid]}</h3><div class='figure'>")
                parts.append(pio.to_html(
                    figures[pid], full_html=False, include_plotlyjs=False,
                    config={"responsive": True, "displaylogo": False}))
                parts.append(f"</div><p class='caption'>{CAPTIONS[pid]}</p>")
                rendered_any = True
        if not rendered_any:
            parts.append(f"<p class='placeholder'>{coming}</p>")
    parts.append(
        "<footer>Educational project — simulated analytics on end-of-day "
        f"data; nothing here is investment advice. Last updated "
        f"{status['last_success_utc']} · {status['rows_stored']} contracts "
        "stored · SPY options are American-style, priced here with a "
        "European model (a measured approximation, discussed in the "
        "repository). Source code: GitHub (link once public).</footer>",
        "</body></html>")
    return "".join(parts)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_render.py -v`, then full suite. All green, no warnings.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/render/ tests/test_render.py
git commit -m "feat: render layer — vendored plotly bundle, P2/P3 figures, self-contained page"
```

---

### Task 5: Wire compute + render into `run_daily`

**Files:**
- Modify: `src/run_daily.py`, `src/data/storage.py`
- Test: `tests/test_pipeline.py` (extend)

**Interfaces:**
- Consumes: everything above.
- Produces: `storage.write_text(text: str, path: Path) -> Path` (atomic, same tmp+rename pattern); `run(...)` unchanged signature, now also renders `docs/index.html` and adds `"iv_convergence"` (float) and `"panels_rendered"` (list[str]) to status. Compute-then-write ordering: ALL computation — chain, IVs, panels, yesterday's overlay, HTML string, status dict — completes before ANY write; then chain → index.html → status.json.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pipeline.py`)

```python
class TestRenderStage:
    def test_run_renders_page_and_reports_convergence(self, tmp_path):
        status = run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "<html" in page and "vol-lens" in page
        assert "https://cdn.plot.ly" not in page
        assert 0.0 <= status["iv_convergence"] <= 1.0
        assert status["panels_rendered"] == ["P2", "P3"]
        # status still written last and consistent with the page
        on_disk = json.loads((tmp_path / "docs" / "status.json").read_text())
        assert on_disk == status

    def test_yesterday_overlay_uses_prior_chain_when_present(self, tmp_path):
        run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        # second run for the "next" session: underlying now ends a day later
        class NextDayEODHD(FakeEODHD):
            def get_underlying_history(self, symbol, start=None):
                df = underlying_frame()
                extra = pd.DataFrame({"date": [TODAY + dt.timedelta(days=1)],
                                      "close": [772.0], "adjusted_close": [772.0],
                                      "volume": [1]})
                return pd.concat([df, extra], ignore_index=True)
        status = run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(),
                     tmp_path, today=TODAY + dt.timedelta(days=1))
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "yesterday" in page
        assert status["snapshot_date"] == (TODAY + dt.timedelta(days=1)).isoformat()

    def test_render_failure_leaves_page_and_status_untouched(self, tmp_path, monkeypatch):
        run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        page_before = (tmp_path / "docs" / "index.html").read_text()
        status_before = (tmp_path / "docs" / "status.json").read_text()
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
            run(NextDayEODHD(), FakeLive(), FakeFallback(), real_cfg(),
                tmp_path, today=TODAY + dt.timedelta(days=1))
        assert (tmp_path / "docs" / "index.html").read_text() == page_before
        assert (tmp_path / "docs" / "status.json").read_text() == status_before
        # and no new chain file landed for the failed session either
        assert not storage.chain_exists(TODAY + dt.timedelta(days=1), tmp_path)
```

Add `import datetime as dt` to the test file's imports if not present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py -v` — new tests fail (no render stage), old ones pass.

- [ ] **Step 3: Implement**

`src/data/storage.py` — append:

```python
def write_text(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path
```

`src/run_daily.py` — extend `run()` (keeping the session-date, fallback, and no-partial-write logic exactly as is). New top-level imports allowed: `from src.analytics.chain_iv import compute_chain_iv`, `from src.analytics.smile import compute_smile`, `from src.analytics.term_structure import compute_term_structure`, `from src.render.figures import build_smile_figure, build_term_structure_figure`, `from src.render.page import render_page`, `import pandas as pd` — none import vendor libraries, so the fixture pipeline test still imports cleanly. After the (existing) chain/fallback logic and rate fetches, and BEFORE any write:

```python
    # ---- compute stage: everything derived before anything is written ----
    chain_iv, iv_stats = compute_chain_iv(chain, risk_free_rate, dividend_yield)
    smile = compute_smile(chain_iv, cfg)
    ts_today = compute_term_structure(chain_iv)

    ts_prev = None
    prior_dates = sorted(
        d for d in sorted_underlying["date"] if d < session_date
        and storage.chain_exists(d, root)
    )
    if prior_dates:
        prev_chain = pd.read_parquet(storage.chain_path(prior_dates[-1], root))
        prev_iv, _ = compute_chain_iv(prev_chain, risk_free_rate, dividend_yield)
        ts_prev = compute_term_structure(prev_iv)

    figures = {"P2": build_smile_figure(smile, spot),
               "P3": build_term_structure_figure(ts_today, ts_prev)}

    status = { ... existing keys ...,
        "iv_convergence": round(iv_stats["convergence"], 4),
        "panels_rendered": ["P2", "P3"],
    }
    html = render_page(figures, status)

    # ---- write stage: chain -> page -> status, each atomic ----
    storage.write_chain(chain, session_date, root)
    storage.write_text(html, Path(root) / "docs" / "index.html")
    storage.write_status(status, root)
    return status
```

(Adapt the existing function body — the sketch shows ordering and names; keep the existing empty-chain/fallback/staleness code intact. `status` must be fully assembled before `render_page` so the page shows the same numbers status.json records.)

Update the module docstring's stage list to "fetch → filter → compute → render → store".

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py -v`, then the full suite. All green, no warnings.

- [ ] **Step 5: Commit**

```bash
git add src/run_daily.py src/data/storage.py tests/test_pipeline.py
git commit -m "feat: compute+render stages in daily run — IV panels, convergence in status, atomic page write"
```

---

### Task 6: Backfill per-day error isolation

**Files:**
- Modify: `src/data/backfill.py`
- Test: `tests/test_backfill.py` (extend)

**Interfaces:**
- Consumes/produces: `backfill(...)` summary dict gains `"dates_failed": int`; a day whose fetch/filter/write raises (any `Exception`) is logged (`log(f"{day}: FAILED — {exc!r}")`) and skipped, the loop continues. KeyboardInterrupt is NOT caught (a user abort stays an abort).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_backfill.py`)

```python
class FailingMassive(FakeMassive):
    def __init__(self, fail_on):
        super().__init__()
        self.fail_on = fail_on

    def get_historical_chain(self, symbol, snapshot_date, spot, cfg):
        if snapshot_date in self.fail_on:
            self.requested.append((snapshot_date, spot))
            raise RuntimeError("vendor exploded")
        return super().get_historical_chain(symbol, snapshot_date, spot, cfg)


class TestErrorIsolation:
    def test_one_failing_day_does_not_abort_the_run(self, tmp_path):
        m = FailingMassive(fail_on={DATES[1]})
        logs = []
        summary = backfill(m, FakeEODHD(), cfg(), tmp_path,
                           start=DATES[0], end=DATES[-1], log=logs.append)
        assert summary["dates_written"] == 2
        assert summary["dates_failed"] == 1
        assert storage.chain_exists(DATES[0], tmp_path)
        assert not storage.chain_exists(DATES[1], tmp_path)
        assert storage.chain_exists(DATES[2], tmp_path)
        assert any("FAILED" in line for line in logs)

    def test_failed_day_is_retriable_on_next_run(self, tmp_path):
        backfill(FailingMassive(fail_on={DATES[1]}), FakeEODHD(), cfg(), tmp_path,
                 start=DATES[0], end=DATES[-1], log=lambda *_: None)
        summary = backfill(FakeMassive(), FakeEODHD(), cfg(), tmp_path,
                           start=DATES[0], end=DATES[-1], log=lambda *_: None)
        assert summary["dates_skipped"] == 2 and summary["dates_written"] == 1

    def test_keyboard_interrupt_propagates(self, tmp_path):
        class AbortingMassive(FakeMassive):
            def get_historical_chain(self, *a, **k):
                raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            backfill(AbortingMassive(), FakeEODHD(), cfg(), tmp_path,
                     start=DATES[0], end=DATES[0], log=lambda *_: None)
```

Add `import pytest` to the test file's imports if not present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_backfill.py -v` — new tests fail (`dates_failed` missing / abort on first failure).

- [ ] **Step 3: Implement** — in `src/data/backfill.py`, initialize `"dates_failed": 0` in the summary and wrap the per-day body (spot lookup → fetch → filter → write → counters) in:

```python
        try:
            ...existing per-day body...
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            summary["dates_failed"] += 1
            log(f"{day}: FAILED — {exc!r}")
            continue
```

Update the module docstring: failed days are counted, logged, and left absent on disk, so the next invocation retries exactly those days.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_backfill.py -v`, then full suite. All green.

- [ ] **Step 5: Commit**

```bash
git add src/data/backfill.py tests/test_backfill.py
git commit -m "feat: per-day error isolation in backfill — failed days logged, counted, retriable"
```

---

### Task 7: REAL render run + hand-off (Phase 3 exit)

**Files:**
- Modify: `LEARNING_LOG.md`; commits include `docs/index.html`, `docs/status.json`, new `data/` files

**Note for the executor:** needs network + `.env` (sandbox off for the run command). Never print secrets.

- [ ] **Step 1: Real daily run**

Run: `.venv/bin/python -m src.run_daily`
Expected: status printed with `iv_convergence` and `panels_rendered: ["P2","P3"]`; `docs/index.html` exists. If a chain for the current session already exists from a previous run today, that's fine — `write_chain` overwrites atomically with the fresh fetch.

- [ ] **Step 2: Verify the page**

```bash
.venv/bin/python - <<'EOF'
from pathlib import Path
html = Path("docs/index.html").read_text()
size = len(html.encode())
assert size < 5_000_000, size
assert "https://cdn.plot.ly" not in html
assert "plotly" in html.lower() and "Q2" in html
print(f"page OK: {size/1e6:.2f} MB, self-contained")
EOF
```

Also record the convergence number vs the ≥95% exit criterion. If convergence < 95% on this run, do NOT tune anything — record the number, the source (off-hours yfinance quotes produce junk mids on sparse books), and flag it for the controller; the criterion is evaluated on liquid quotes from a proper post-close run.

- [ ] **Step 3: Learning-log entry**

Append a dated entry **"Phase 3: the first page"** under `## Entries`: convergence achieved (number, source, row count), how the smile actually looks on real data (is the put skew visible?), the term-structure shape, page size in MB, and one honest paragraph on anything ugly (sparse smile lines from a thin chain, gaps, overlay quirks).

- [ ] **Step 4: Commit**

```bash
git add docs/ data/ LEARNING_LOG.md
git commit -m "feat: first rendered dashboard — P2 smile and P3 term structure from real data"
```

- [ ] **Step 5: Report the publish hand-off**

Report to the controller: Phase 3's "published to Pages" half requires the user to (1) create a GitHub repository and push, (2) enable Pages serving `/docs` on main, (3) then the page is live. The local page is done and verified; the README's dashboard link placeholder can be filled once the URL exists. Do not create any remote yourself.

---

## Self-Review Notes (already applied)

- SPEC coverage: §2.4 pure panel functions (T1–T3), §2.5 page requirements including captions/mobile/self-containment/budget (T4), §3 P2 acceptance (skew visible — tested synthetically T2; ≥95% convergence — measured T7), §3 P3 acceptance (overlay — T4/T5), §2.1 failure policy extended to the page (T5's render-failure test), Phase 3 exit criterion split honestly: local page verified (T7), Pages publish user-gated.
- P2's optional bid/ask IV band toggle (SPEC "optional") is deliberately deferred — YAGNI until a real page exists to judge.
- Type consistency: `compute_chain_iv` returns (frame, stats) consumed by name in T2/T3/T5; figure builders' signatures match T5's calls; `write_text` matches T5's usage. `sorted_underlying`/`session_date`/`risk_free_rate`/`dividend_yield` names refer to the existing Phase 2 `run()` locals (verified against current `src/run_daily.py`).
- The yesterday-overlay approximation (today's r/q on yesterday's chain) is a Global Constraint with rationale, not an accident.
