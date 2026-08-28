# vol-lens Phase 0–1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold the vol-lens repository, verify EODHD options-data access for SPY, and build the fully unit-tested hand-built Black-Scholes model layer (pricer, Greeks, implied-vol solver).

**Architecture:** Phase 0 creates the repo skeleton (config, learning log, dependency setup) and a manual verification script that probes EODHD's REST API and documents what the plan tier actually provides. Phase 1 builds `src/models/black_scholes.py` — pure, NumPy-broadcast functions with a NaN-never-raise contract — test-first against published textbook values, internal identities (put-call parity, dividend-shift), finite-difference Greek checks, and IV round-trips. A GitHub Actions CI workflow runs the suite on every push.

**Tech Stack:** Python 3.12 · NumPy · SciPy (`stats.norm`, `optimize.brentq`) · PyYAML · requests · pytest · GitHub Actions

**Spec:** `SPEC.md` (repo root; reviewed and committed 2026-08-27). This plan implements SPEC §2.3 (model layer), §4 (unit tests), and §5 Phases 0–1.

## Global Constraints

- Python 3.12 (CI pins 3.12; local ≥ 3.11 acceptable, flag to user if lower).
- **No pricing libraries** — model layer uses only NumPy, `scipy.stats.norm`, `scipy.optimize.brentq` (SPEC §7: QuantLib/py_vollib banned, even as dev deps).
- Model functions are **pure and vectorized** via NumPy broadcasting — no Python loops over option chains (sole exception: the scalar Brent fallback loop over Newton's stragglers, SPEC §2.3).
- Greeks are **raw calculus derivatives** (theta per year, vega per unit vol, rho per unit rate); display scaling belongs to the render layer (SPEC §2.3).
- **NaN-never-raise** for numeric junk: invalid numeric inputs yield NaN, never exceptions (SPEC §2.3 edge contract). Programming errors (e.g. `kind="cal"`) DO raise `ValueError`.
- Secrets never committed: `EODHD_API_TOKEN` lives in the environment / GitHub Actions secrets only; `.env` is gitignored (SPEC §2.1).
- `docs/` is reserved as the GitHub Pages root (SPEC §6) — no internal docs go there.
- TDD throughout; commit at the end of every task.
- `LEARNING_LOG.md` entries are written **in the same task** as the code they explain (SPEC §5 learning rule).

---

## File Structure

```
vol-lens/  (= /Users/tkchang/Projects/BlackScholes)
├── config.yaml                    # Task 1 — all tunable parameters (SPEC §6)
├── requirements.txt               # Task 1 — runtime + test deps for Phases 0–1
├── pyproject.toml                 # Task 1 — pytest config only (pythonpath)
├── .gitignore                     # Task 1
├── LEARNING_LOG.md                # Task 1 skeleton; entries appended Tasks 2–5
├── plans/                         # this plan
├── scripts/
│   └── verify_eodhd.py            # Task 2 — Phase 0 data-access probe (manual)
├── src/
│   ├── __init__.py                # Task 1 (empty)
│   └── models/
│       ├── __init__.py            # Task 1 (empty)
│       └── black_scholes.py       # Tasks 3–5 — pricer, greeks, implied_vol
├── tests/
│   ├── test_env.py                # Task 1 — import smoke test
│   ├── test_pricing.py            # Task 3
│   ├── test_greeks.py             # Task 4
│   └── test_implied_vol.py        # Task 5
└── .github/workflows/
    └── ci.yml                     # Task 6 — pytest on every push
```

Module responsibilities: `black_scholes.py` owns all three model functions plus the private `_d1_d2` helper — they share formulas and constants and change together (SPEC §6 puts them in one file). Tests split by function so failures localize. `realized_vol.py` is **not** in this plan — it isn't needed until Phase 4 (YAGNI).

---

### Task 1: Repository scaffold

**Files:**
- Create: `.gitignore`, `requirements.txt`, `pyproject.toml`, `config.yaml`, `LEARNING_LOG.md`, `src/__init__.py`, `src/models/__init__.py`, `tests/test_env.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a working venv at `.venv/` with all deps; `pytest` runnable from repo root with `src.` imports resolving; `config.yaml` keys used by later phases (`symbol`, `chain_filter.*`, `rates.risk_free_fallback`, `rates.dividend_yield_fallback`); `LEARNING_LOG.md` with a `## Phase 0 findings — EODHD data access` section that Task 2 fills in and an `## Entries` section that Tasks 3–5 append to.

- [ ] **Step 1: Check Python version**

Run: `python3 --version`
Expected: 3.12.x (3.11 acceptable — if lower, stop and ask the user how they want to install 3.12).

- [ ] **Step 2: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.env
.DS_Store
```

- [ ] **Step 3: Write `requirements.txt`**

Only what Phases 0–1 need (pandas/plotly/pyarrow arrive with Phase 2+):

```
numpy>=1.26
scipy>=1.11
pyyaml>=6.0
requests>=2.31
pytest>=8.0
```

- [ ] **Step 4: Write `pyproject.toml`** (pytest config only — packaging metadata is YAGNI for a non-installable project)

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

- [ ] **Step 5: Write `config.yaml`** (values verbatim from SPEC §2.2/§3)

```yaml
# vol-lens configuration — single source of tunable parameters (SPEC §6)
symbol: SPY
exchange_suffix: ".US"          # EODHD ticker convention, e.g. SPY.US

chain_filter:                   # SPEC §2.2
  n_monthly_expiries: 6
  dte_min: 7
  dte_max: 365
  moneyness_min: 0.70           # K/S
  moneyness_max: 1.30

target_dte:                     # SPEC §3
  atm_panel: 30                 # P1/P4/P5/P6/P8 reference tenor
  smile_expiries: [30, 90, 180] # P2

realized_vol:                   # SPEC §2.3
  windows: [20, 60]
  annualization_days: 252

rates:                          # SPEC §2.2 fallbacks (primary source is EODHD)
  risk_free_fallback: 0.04      # 3m T-bill, update if EODHD source unavailable
  dividend_yield_fallback: 0.013  # SPY trailing-12m yield fallback

hedge_sim:                      # SPEC §3 P8
  structure: short_straddle
  entry_dte: 30
  hedge_frequency: daily
  transaction_cost_bps: 0       # v1; v2 adds a toggle
```

- [ ] **Step 6: Write `LEARNING_LOG.md`** skeleton

```markdown
# LEARNING_LOG — vol-lens

Written by the AI pair as a teaching document (SPEC §1/§5): every formula
explained, every numerical failure documented, so the author can study it to
meet the SPEC §9 interview bar. Answered in full during Phase 8.

## The ten questions

Questions 1–5 are the public guiding questions (see README). 6–10 are the
private list.

1. Everyone has the same formula. Why does anyone disagree about prices?
2. Where does the market ignore the model, and why?
3. Is implied volatility a prediction, or a price?
4. The model claims you can hedge an option perfectly. Can you?
5. Given all these flaws, why does every trading desk still use Black-Scholes?
6. Why is vega largest at-the-money and near zero deep ITM/OTM — and what
   does that geometry do to anything that tries to invert price into vol?
7. What exactly breaks in Newton–Raphson for deep-OTM options, and how does
   the Brent safeguard rescue it? (Answered with the actual failure we hit.)
8. Does implied vol differ when solved from bid vs mid vs ask, and by how
   much across the chain? (SPEC §2.2 stores all three for this.)
9. How large is the error from pricing American-style SPY options with a
   European model (and continuous-yield q instead of discrete dividends),
   and where in the chain does it concentrate?
10. Why does discrete daily delta hedging leave P&L noise, and what does
    daily hedged P&L ≈ ½ Γ S² (σ²_realized − σ²_implied) dt actually say?

## Phase 0 findings — EODHD data access

*(Filled in by the Phase 0 verification run: endpoints that work, fields
available, history depth, rate limits, EOD settlement time vs the cron hour.)*

## Entries

*(Dated entries appended as each piece of the model layer is built.)*
```

- [ ] **Step 7: Create package markers and smoke test**

Create empty `src/__init__.py` and `src/models/__init__.py`.

`tests/test_env.py`:

```python
"""Environment smoke test: deps importable, src package resolvable."""


def test_imports():
    import numpy
    import requests  # noqa: F401
    import scipy.optimize
    import scipy.stats
    import yaml  # noqa: F401

    import src.models  # noqa: F401

    assert numpy.asarray([1.0]).dtype == numpy.float64
    assert callable(scipy.stats.norm.cdf)
    assert callable(scipy.optimize.brentq)


def test_config_loads():
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["symbol"] == "SPY"
    assert cfg["chain_filter"]["moneyness_min"] == 0.70
    assert cfg["rates"]["dividend_yield_fallback"] > 0
```

- [ ] **Step 8: Create venv, install, run the suite**

Run:
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -v
```
Expected: 2 passed.

- [ ] **Step 9: Commit**

```bash
git add .gitignore requirements.txt pyproject.toml config.yaml LEARNING_LOG.md src/ tests/ plans/
git commit -m "chore: Phase 0 scaffold — config, learning log, deps, smoke tests"
```

---

### Task 2: EODHD data-access verification (Phase 0 gate)

**Files:**
- Create: `scripts/verify_eodhd.py`
- Modify: `LEARNING_LOG.md` (fill the "Phase 0 findings" section)

**Interfaces:**
- Consumes: `EODHD_API_TOKEN` environment variable (from the user).
- Produces: a written record in `LEARNING_LOG.md` of which options endpoint works, available fields, history depth, and rate limits. Phase 2's `EODHDProvider` will be designed against this record. No code interfaces.

**Note for the executor:** this task needs (a) the user's EODHD REST API token and (b) network access. If `EODHD_API_TOKEN` is unset, STOP and ask the user for it — that is user-only input. The Bash sandbox blocks network; run the script with sandbox disabled or ask the user to run `! .venv/bin/python scripts/verify_eodhd.py` themselves.

- [ ] **Step 1: Write `scripts/verify_eodhd.py`**

The script probes candidate endpoints (EODHD has migrated options data between `/api/options/` and the UnicornBay marketplace endpoints; discovering which one this account can reach is the point of the task) and prints a report. It is a manual diagnostic, not pipeline code — no tests.

```python
"""Phase 0 gate: verify EODHD data access for vol-lens (SPEC §5 Phase 0).

Probes every data source the pipeline needs and prints a report to paste
into LEARNING_LOG.md. Read-only; makes a handful of requests.

Usage: EODHD_API_TOKEN=... python scripts/verify_eodhd.py
"""
import datetime as dt
import json
import os
import sys

import requests

BASE = "https://eodhd.com/api"
TIMEOUT = 30


def token() -> str:
    tok = os.environ.get("EODHD_API_TOKEN", "").strip()
    if not tok:
        sys.exit("EODHD_API_TOKEN is not set. Export it and re-run.")
    return tok


def get(path: str, **params) -> tuple[int, object]:
    params.setdefault("api_token", token())
    params.setdefault("fmt", "json")
    r = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
    try:
        body = r.json()
    except ValueError:
        body = r.text[:500]
    return r.status_code, body


def show(title: str, status: int, body: object, n: int = 1) -> None:
    print(f"\n=== {title} — HTTP {status} ===")
    if isinstance(body, list):
        print(f"rows: {len(body)}")
        for row in body[:n]:
            print(json.dumps(row, indent=2)[:1500])
    elif isinstance(body, dict):
        print(json.dumps(body, indent=2)[:2000])
    else:
        print(str(body)[:500])


def main() -> None:
    today = dt.date.today()

    # 1. Account/plan details (also confirms the token is valid)
    status, body = get("user")
    show("Account details (plan, rate limits, credits)", status, body)

    # 2. Underlying EOD history — needed for spot + realized vol
    status, body = get("eod/SPY.US", **{"from": str(today - dt.timedelta(days=30))})
    show("SPY underlying EOD (last 30d)", status, body, n=2)

    # 3. Dividends — needed for trailing-12m dividend yield q (SPEC §2.2)
    status, body = get("div/SPY.US", **{"from": str(today - dt.timedelta(days=400))})
    show("SPY dividends (trailing ~13 months)", status, body, n=3)

    # 4. Options chain — try the current marketplace endpoint first,
    #    then the legacy endpoint. Whichever answers 200 wins.
    status, body = get(
        "mp/unicornbay/options/eod",
        **{"filter[underlying_symbol]": "SPY", "page[limit]": 3},
    )
    show("Options EOD — marketplace (mp/unicornbay/options/eod)", status, body)
    if status != 200:
        status, body = get("options/SPY.US")
        show("Options EOD — legacy (options/SPY.US)", status, body)

    # 5. Options history depth — same endpoint, ~6 months back (SPEC §2.2 backfill)
    past = today - dt.timedelta(days=180)
    status, body = get(
        "mp/unicornbay/options/eod",
        **{
            "filter[underlying_symbol]": "SPY",
            "filter[date_from]": str(past),
            "filter[date_to]": str(past + dt.timedelta(days=4)),
            "page[limit]": 3,
        },
    )
    show(f"Options history probe ({past})", status, body)

    # 6. Risk-free rate candidates (3m T-bill; fallback lives in config.yaml)
    for sym in ("US3M.INDX", "US3M.GBOND"):
        status, body = get(f"eod/{sym}", **{"from": str(today - dt.timedelta(days=14))})
        show(f"Risk-free candidate {sym}", status, body, n=1)
        if status == 200:
            break

    print(
        "\nDone. Record in LEARNING_LOG.md: which options endpoint answered 200,"
        "\nthe field names present (bid/ask/last/volume/open_interest/expiry/"
        "\nstrike/type, any vendor IV/greeks), history depth, rate limits from"
        "\nthe account payload, and roughly when today's options rows appeared"
        "\n(check the data's own date field vs the cron hour, SPEC §2.1)."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `EODHD_API_TOKEN=<from user> .venv/bin/python scripts/verify_eodhd.py` (sandbox disabled, or user-run).
Expected: HTTP 200 on account, underlying, and dividends; at least one options endpoint returning contract rows with bid/ask fields. If **no** options endpoint answers 200, STOP — report to the user: the plan tier lacks options data, and per SPEC §8 the `DataProvider` fallback decision (yfinance) must be made before Phase 2. Do not proceed to mark Phase 0 complete.

- [ ] **Step 3: Record findings in `LEARNING_LOG.md`**

Replace the placeholder paragraph under `## Phase 0 findings — EODHD data access` with the actual results, covering every item the script's closing message lists: working endpoint + example request, field inventory (noting whether vendor IV/Greeks are present — stored later for cross-checks, never used in computations, SPEC §2.2), history depth found, rate limits/credits, and observed data settlement timing vs the `30 1 * * 2-6` cron.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_eodhd.py LEARNING_LOG.md
git commit -m "feat: Phase 0 EODHD verification script + documented findings"
```

---

### Task 3: `bs_price` — the pricer

**Files:**
- Create: `src/models/black_scholes.py`
- Test: `tests/test_pricing.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure function).
- Produces: `bs_price(S, K, T, r, sigma, q=0.0, kind="call") -> float | np.ndarray` — scalars in, `float` out; arrays in, broadcast `np.ndarray` out. `kind` is the scalar string `"call"` or `"put"` (chains price calls and puts in two calls). Also the private helper `_d1_d2(S, K, T, sigma, r, q) -> (d1, d2)` and module constants `_MIN_VOL = 1e-4`, `_MAX_VOL = 5.0`, `_EPS = 1e-12`, reused by Tasks 4–5.

- [ ] **Step 1: Write the failing tests**

`tests/test_pricing.py`:

```python
"""bs_price tests: textbook anchors, identities, boundaries, NaN contract.

Textbook anchor: Hull's standard example — S=42, K=40, r=10%, sigma=20%,
T=0.5, q=0: call = 4.7594, put = 0.8086 (4 dp).
"""
import numpy as np
import pytest

from src.models.black_scholes import bs_price


HULL = dict(S=42.0, K=40.0, T=0.5, r=0.10, sigma=0.20)

# Shared grid for identity tests: realistic SPY-ish ranges (SPEC §2.2 filter)
S0 = 100.0
GRID_K = np.linspace(70.0, 130.0, 13)
GRID_T = np.array([7 / 365, 30 / 365, 0.25, 0.5, 1.0, 2.0])[:, None]
GRID = dict(S=S0, K=GRID_K, T=GRID_T, r=0.04, sigma=0.22, q=0.013)


class TestTextbookValues:
    def test_hull_call(self):
        assert bs_price(**HULL, kind="call") == pytest.approx(4.7594, abs=1e-4)

    def test_hull_put(self):
        assert bs_price(**HULL, kind="put") == pytest.approx(0.8086, abs=1e-4)


class TestIdentities:
    def test_put_call_parity(self):
        """C - P == S e^{-qT} - K e^{-rT}, an internal identity (machine eps)."""
        c = bs_price(**GRID, kind="call")
        p = bs_price(**GRID, kind="put")
        fwd = S0 * np.exp(-GRID["q"] * GRID_T) - GRID_K * np.exp(-0.04 * GRID_T)
        np.testing.assert_allclose(c - p, np.broadcast_to(fwd, c.shape), atol=1e-10)

    def test_dividend_yield_is_spot_shift(self):
        """BSM(S,...,q) == BS(S e^{-qT},...,q=0) exactly — validates q handling."""
        q, T = 0.013, GRID_T
        with_q = bs_price(S=S0, K=GRID_K, T=T, r=0.04, sigma=0.22, q=q, kind="call")
        shifted = bs_price(S=S0 * np.exp(-q * T), K=GRID_K, T=T, r=0.04, sigma=0.22, q=0.0, kind="call")
        np.testing.assert_allclose(with_q, shifted, atol=1e-12)


class TestBoundaries:
    def test_sigma_zero_gives_discounted_forward_intrinsic(self):
        got = bs_price(S=S0, K=GRID_K, T=0.5, r=0.04, sigma=0.0, q=0.013, kind="call")
        want = np.maximum(S0 * np.exp(-0.013 * 0.5) - GRID_K * np.exp(-0.04 * 0.5), 0.0)
        np.testing.assert_allclose(got, want, atol=1e-10)

    def test_t_zero_gives_intrinsic(self):
        assert bs_price(S=105.0, K=100.0, T=0.0, r=0.04, sigma=0.2, kind="call") == pytest.approx(5.0)
        assert bs_price(S=95.0, K=100.0, T=0.0, r=0.04, sigma=0.2, kind="call") == pytest.approx(0.0)
        assert bs_price(S=100.0, K=100.0, T=0.0, r=0.04, sigma=0.2, kind="call") == pytest.approx(0.0)

    def test_no_arbitrage_bounds(self):
        c = bs_price(**GRID, kind="call")
        lo = np.maximum(S0 * np.exp(-GRID["q"] * GRID_T) - GRID_K * np.exp(-0.04 * GRID_T), 0.0)
        hi = S0 * np.exp(-GRID["q"] * GRID_T)
        assert np.all(c >= np.broadcast_to(lo, c.shape) - 1e-12)
        assert np.all(c <= np.broadcast_to(hi, c.shape) + 1e-12)

    def test_monotone_increasing_in_vol(self):
        sig = np.linspace(0.05, 1.0, 20)
        prices = bs_price(S=100.0, K=110.0, T=0.25, r=0.04, sigma=sig, kind="call")
        assert np.all(np.diff(prices) > 0)


class TestContract:
    def test_invalid_numeric_inputs_yield_nan_not_raise(self):
        bad = [
            dict(S=-1.0), dict(S=0.0), dict(K=-5.0), dict(K=0.0),
            dict(T=-0.1), dict(sigma=-0.2), dict(S=np.nan), dict(r=np.inf),
        ]
        base = dict(S=100.0, K=100.0, T=0.5, r=0.04, sigma=0.2)
        for override in bad:
            out = bs_price(**{**base, **override}, kind="call")
            assert np.isnan(out), f"expected NaN for {override}"

    def test_nan_is_elementwise(self):
        out = bs_price(S=100.0, K=np.array([100.0, -1.0, 110.0]), T=0.5, r=0.04, sigma=0.2, kind="call")
        assert not np.isnan(out[0]) and np.isnan(out[1]) and not np.isnan(out[2])

    def test_bad_kind_raises(self):
        with pytest.raises(ValueError):
            bs_price(S=100.0, K=100.0, T=0.5, r=0.04, sigma=0.2, kind="cal")

    def test_scalar_in_float_out(self):
        out = bs_price(S=100.0, K=100.0, T=0.5, r=0.04, sigma=0.2, kind="call")
        assert isinstance(out, float)

    def test_broadcast_shape(self):
        out = bs_price(S=100.0, K=GRID_K, T=GRID_T, r=0.04, sigma=0.2, kind="call")
        assert out.shape == (GRID_T.size, GRID_K.size)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pricing.py -v`
Expected: collection error / ImportError — `bs_price` doesn't exist yet.

- [ ] **Step 3: Implement `bs_price`**

`src/models/black_scholes.py`:

```python
"""Hand-built Black-Scholes-Merton model layer (SPEC §2.3).

No pricing libraries: NumPy + scipy.stats.norm + scipy.optimize.brentq only.
All functions are pure and vectorized via NumPy broadcasting.

Contracts:
- NaN-never-raise: invalid *numeric* inputs (negative T or sigma, nonpositive
  S or K, non-finite values) yield NaN elementwise; callers count NaNs into
  status.json. Programming errors (bad `kind`) raise ValueError.
- Edge behavior: as T→0 or sigma→0 the price converges to discounted
  intrinsic on the forward, max(S e^{-qT} - K e^{-rT}, 0) for calls; this
  emerges from the clamped d1/d2 limit rather than special-case branches.
- Greeks (Task 4) are raw calculus derivatives: theta per year, vega per
  unit vol, rho per unit rate. Display scaling is the render layer's job.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_MIN_VOL = 1e-4
_MAX_VOL = 5.0
_EPS = 1e-12


def _validate_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _as_arrays(*vals):
    return [np.asarray(v, dtype=np.float64) for v in vals]


def _invalid_mask(S, K, T, r, sigma, q):
    return (
        (S <= 0) | (K <= 0) | (T < 0) | (sigma < 0)
        | ~np.isfinite(S + K + T + r + sigma + q)
    )


def _d1_d2(S, K, T, sigma, r, q):
    sqrt_t = np.sqrt(T)
    denom = np.maximum(sigma * sqrt_t, _EPS)  # sigma*sqrt(T) underflow guard
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / denom
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """European option price with continuous dividend yield q.

    Scalars in -> float out; arrays in -> broadcast ndarray out.
    """
    _validate_kind(kind)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    with np.errstate(all="ignore"):
        invalid = _invalid_mask(S, K, T, r, sigma, q)
        d1, d2 = _d1_d2(S, K, T, sigma, r, q)
        disc_s = S * np.exp(-q * T)
        disc_k = K * np.exp(-r * T)
        if kind == "call":
            price = disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
        else:
            price = disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
        price = np.where(invalid, np.nan, price)
    return price if price.ndim else float(price)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pricing.py -v`
Expected: all pass. If the Hull anchors fail at 1e-4, check `norm.cdf` argument order and the sign of `q` in d1 before touching tolerances — the identities passing while anchors fail means a formula transcription slip.

- [ ] **Step 5: Append the learning-log entry**

Append to `LEARNING_LOG.md` under `## Entries` a dated entry titled **"The pricing formula"** covering, in prose: what d1 and d2 each mean (d2 = risk-neutral probability the option finishes ITM; d1 = d2 plus the σ√T shift from measuring under the stock numeraire); why the dividend yield enters as a spot discount (`test_dividend_yield_is_spot_shift` is the executable statement of this); why put-call parity holds to machine epsilon *by construction* here while real markets only hold it to the bid-ask spread (the P7 distinction); and why the σ→0 limit is *forward* intrinsic, not plain intrinsic. Cite the two test classes as evidence.

- [ ] **Step 6: Commit**

```bash
git add src/models/black_scholes.py tests/test_pricing.py LEARNING_LOG.md
git commit -m "feat: hand-built Black-Scholes pricer with textbook + identity tests"
```

---

### Task 4: `greeks`

**Files:**
- Modify: `src/models/black_scholes.py` (append function)
- Test: `tests/test_greeks.py`

**Interfaces:**
- Consumes: `_d1_d2`, `_as_arrays`, `_invalid_mask`, `_validate_kind`, `_EPS`, `bs_price` (for finite-difference tests) from Task 3.
- Produces: `greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict[str, float | np.ndarray]` with keys `"delta"`, `"gamma"`, `"vega"`, `"theta"`, `"rho"` — raw derivatives: theta = ∂P/∂t per **year** (negative for long options), vega per **unit** vol, rho per **unit** rate. Same scalar/broadcast semantics and NaN contract as `bs_price`.

- [ ] **Step 1: Write the failing tests**

`tests/test_greeks.py`:

```python
"""greeks tests: Hull's published table + finite differences of our own pricer.

Hull anchor (Ch. 19 example): S=49, K=50, r=5%, sigma=20%, T=20/52, q=0:
delta=0.522, gamma=0.066, vega=12.1, theta=-4.31/yr, rho=8.91 (as published).
Finite differences against bs_price are the tight check.
"""
import numpy as np
import pytest

from src.models.black_scholes import bs_price, greeks

HULL = dict(S=49.0, K=50.0, T=20 / 52, r=0.05, sigma=0.20)
BASE = dict(S=100.0, K=105.0, T=0.4, r=0.04, sigma=0.25, q=0.013)


class TestHullTable:
    def test_call_greeks_match_published(self):
        g = greeks(**HULL, kind="call")
        # Tolerances = Hull's published rounding precision; the tight
        # correctness check is TestFiniteDifferences below.
        assert g["delta"] == pytest.approx(0.522, abs=1e-3)
        assert g["gamma"] == pytest.approx(0.066, abs=1e-3)
        assert g["vega"] == pytest.approx(12.1, abs=5e-2)
        assert g["theta"] == pytest.approx(-4.31, abs=1e-2)
        assert g["rho"] == pytest.approx(8.91, abs=1e-2)


class TestFiniteDifferences:
    """Central differences of bs_price, both kinds, with q != 0."""

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_delta(self, kind):
        h = BASE["S"] * 1e-5
        up = bs_price(**{**BASE, "S": BASE["S"] + h}, kind=kind)
        dn = bs_price(**{**BASE, "S": BASE["S"] - h}, kind=kind)
        assert greeks(**BASE, kind=kind)["delta"] == pytest.approx((up - dn) / (2 * h), rel=1e-6)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_gamma(self, kind):
        h = BASE["S"] * 1e-4
        up = bs_price(**{**BASE, "S": BASE["S"] + h}, kind=kind)
        mid = bs_price(**BASE, kind=kind)
        dn = bs_price(**{**BASE, "S": BASE["S"] - h}, kind=kind)
        fd = (up - 2 * mid + dn) / h**2
        assert greeks(**BASE, kind=kind)["gamma"] == pytest.approx(fd, rel=1e-4)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_vega(self, kind):
        h = 1e-6
        up = bs_price(**{**BASE, "sigma": BASE["sigma"] + h}, kind=kind)
        dn = bs_price(**{**BASE, "sigma": BASE["sigma"] - h}, kind=kind)
        assert greeks(**BASE, kind=kind)["vega"] == pytest.approx((up - dn) / (2 * h), rel=1e-6)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_theta_is_calendar_derivative(self, kind):
        """theta = dP/dt = -dP/dT: price at shorter expiry minus longer."""
        h = 1e-6
        shorter = bs_price(**{**BASE, "T": BASE["T"] - h}, kind=kind)
        longer = bs_price(**{**BASE, "T": BASE["T"] + h}, kind=kind)
        assert greeks(**BASE, kind=kind)["theta"] == pytest.approx((shorter - longer) / (2 * h), rel=1e-5)

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_rho(self, kind):
        h = 1e-6
        up = bs_price(**{**BASE, "r": BASE["r"] + h}, kind=kind)
        dn = bs_price(**{**BASE, "r": BASE["r"] - h}, kind=kind)
        assert greeks(**BASE, kind=kind)["rho"] == pytest.approx((up - dn) / (2 * h), rel=1e-5)


class TestShapes:
    def test_atm_call_delta_near_half_and_gamma_peaks_atm(self):
        strikes = np.linspace(70.0, 130.0, 61)
        g = greeks(S=100.0, K=strikes, T=30 / 365, r=0.04, sigma=0.2, q=0.013, kind="call")
        atm = np.argmin(np.abs(strikes - 100.0))
        assert 0.45 < g["delta"][atm] < 0.60          # ~N(d1), slightly above 0.5
        assert np.argmax(g["gamma"]) in range(atm - 2, atm + 3)  # gamma peaks ~ATM
        assert np.all(np.diff(g["delta"]) < 0)        # call delta falls as K rises

    def test_put_call_delta_relation(self):
        """delta_call - delta_put == e^{-qT} (parity differentiated in S)."""
        c = greeks(**BASE, kind="call")["delta"]
        p = greeks(**BASE, kind="put")["delta"]
        assert c - p == pytest.approx(np.exp(-BASE["q"] * BASE["T"]), abs=1e-12)

    def test_invalid_inputs_yield_nan(self):
        g = greeks(S=100.0, K=100.0, T=-0.5, r=0.04, sigma=0.2, kind="call")
        assert all(np.isnan(v) for v in g.values())

    def test_vectorized_output_shape(self):
        g = greeks(S=100.0, K=np.linspace(70, 130, 13), T=0.25, r=0.04, sigma=0.2, kind="put")
        assert g["delta"].shape == (13,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_greeks.py -v`
Expected: ImportError — `greeks` not defined.

- [ ] **Step 3: Implement `greeks`** (append to `src/models/black_scholes.py`)

```python
def greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Closed-form BSM Greeks as raw calculus derivatives.

    theta is per YEAR, vega per UNIT vol, rho per UNIT rate — display
    scaling (per-day, per-vol-point, per-1%) belongs to the render layer.
    """
    _validate_kind(kind)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    with np.errstate(all="ignore"):
        invalid = _invalid_mask(S, K, T, r, sigma, q)
        d1, d2 = _d1_d2(S, K, T, sigma, r, q)
        sqrt_t = np.sqrt(T)
        pdf1 = norm.pdf(d1)
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)
        gamma = disc_q * pdf1 / np.maximum(S * sigma * sqrt_t, _EPS)
        vega = S * disc_q * pdf1 * sqrt_t
        common_theta = -S * disc_q * pdf1 * sigma / np.maximum(2 * sqrt_t, _EPS)
        if kind == "call":
            delta = disc_q * norm.cdf(d1)
            theta = common_theta - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
            rho = K * T * disc_r * norm.cdf(d2)
        else:
            delta = disc_q * (norm.cdf(d1) - 1.0)
            theta = common_theta + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
            rho = -K * T * disc_r * norm.cdf(-d2)
        out = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
        out = {k: np.where(invalid, np.nan, v) for k, v in out.items()}
    return {k: (v if v.ndim else float(v)) for k, v in out.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_greeks.py -v`
Expected: all pass. If Hull values are off by a scale factor (×365, ×100), the raw-derivative convention was violated — fix the formula, never the test.

- [ ] **Step 5: Append the learning-log entry**

Append a dated entry **"The Greeks"** covering: each Greek's meaning and formula; why ATM call delta sits slightly *above* 0.5 (the r−q+σ²/2 drift in d1 — a classic interview trap); why gamma and vega share the φ(d1) kernel and both peak near ATM (this seeds question 6); the theta sign convention chosen (∂P/∂t per year) and how the render layer will display per-day; and the delta-parity relation `delta_c − delta_p = e^{−qT}` as differentiated put-call parity.

- [ ] **Step 6: Commit**

```bash
git add src/models/black_scholes.py tests/test_greeks.py LEARNING_LOG.md
git commit -m "feat: closed-form Greeks with Hull-table and finite-difference tests"
```

---

### Task 5: `implied_vol`

**Files:**
- Modify: `src/models/black_scholes.py` (append function + import)
- Test: `tests/test_implied_vol.py`

**Interfaces:**
- Consumes: `bs_price`, `_d1_d2`, `_as_arrays`, `_validate_kind`, `_MIN_VOL`, `_MAX_VOL`, `_EPS` from Tasks 3–4; adds `from scipy.optimize import brentq` to module imports.
- Produces: `implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=20) -> float | np.ndarray` — σ per element, NaN where the price violates no-arbitrage bounds or carries no recoverable vol information. Phase 3's smile panel calls exactly this signature. (`tol` is in *price* space; σ-space accuracy ≈ tol/vega, which is why 1e-10 rather than the spec review's first guess of 1e-8 — SPEC §2.3 amended to match.)

- [ ] **Step 1: Write the failing tests**

`tests/test_implied_vol.py`:

```python
"""implied_vol tests: round-trip recovery, corner contract, no-arb NaNs.

SPEC §4 contract: exact recovery (1e-6) on the normal grid; near-degenerate
corners must be accurate-or-NaN — never silently wrong.
"""
import numpy as np
import pytest

from src.models.black_scholes import bs_price, implied_vol

R, Q = 0.04, 0.013
S0 = 100.0


def roundtrip(sigma, K, T, kind):
    price = bs_price(S=S0, K=K, T=T, r=R, sigma=sigma, q=Q, kind=kind)
    return implied_vol(price, S=S0, K=K, T=T, r=R, q=Q, kind=kind)


class TestNormalGridExactRecovery:
    """Liquid regime: recovery to 1e-6, no NaNs.

    Grid bounds matter: sigma-space accuracy of a price-space Newton stop is
    ~ price_tol / vega, so the 1e-6 guarantee only holds where vega is
    healthy (roughly |d1| < 4). Short-dated far wings live in the corner
    contract below instead — same reason real chains' wing IVs are noisy.
    """

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_roundtrip_grid(self, kind):
        sigma = np.array([0.15, 0.25, 0.40, 0.60, 1.00])[:, None, None]
        K = np.linspace(85.0, 115.0, 7)[None, :, None]
        T = np.array([30 / 365, 90 / 365, 0.5, 1.0])[None, None, :]
        got = roundtrip(sigma, K, T, kind)
        assert not np.any(np.isnan(got)), "no NaNs allowed on the normal grid"
        np.testing.assert_allclose(got, np.broadcast_to(sigma, got.shape), atol=1e-6)


class TestCornerContract:
    """Degenerate corners: recover to 1e-6 OR return NaN — never wrong."""

    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_corners_accurate_or_nan(self, kind):
        sigma = np.array([0.05, 0.20, 0.90, 3.00])[:, None, None]
        K = np.array([70.0, 100.0, 130.0, 300.0])[None, :, None]
        T = np.array([1 / 365, 7 / 365, 2.0])[None, None, :]
        got = roundtrip(sigma, K, T, kind)
        want = np.broadcast_to(sigma, got.shape)
        ok = np.isnan(got) | (np.abs(got - want) < 1e-6)
        assert np.all(ok), f"silently-wrong IVs at {np.argwhere(~ok)}"

    def test_deep_otm_tiny_vega_does_not_return_garbage(self):
        """The naive-Newton killer: vega ~ 0, price ~ 1e-30."""
        price = bs_price(S=S0, K=300.0, T=0.05, r=R, sigma=0.30, q=Q, kind="call")
        got = implied_vol(price, S=S0, K=300.0, T=0.05, r=R, q=Q, kind="call")
        assert np.isnan(got) or abs(got - 0.30) < 1e-6


class TestNoArbitrageBounds:
    def test_price_above_upper_bound_is_nan(self):
        assert np.isnan(implied_vol(101.0, S=S0, K=100.0, T=0.5, r=R, q=Q, kind="call"))

    def test_price_below_intrinsic_is_nan(self):
        # discounted forward intrinsic of this ITM call is ~ 20.9; 15 violates it
        assert np.isnan(implied_vol(15.0, S=S0, K=80.0, T=0.5, r=R, q=Q, kind="call"))

    def test_nonpositive_or_nan_price_is_nan(self):
        for bad in (0.0, -1.0, np.nan):
            assert np.isnan(implied_vol(bad, S=S0, K=100.0, T=0.5, r=R, q=Q, kind="call"))

    def test_zero_t_is_nan(self):
        assert np.isnan(implied_vol(5.0, S=S0, K=100.0, T=0.0, r=R, q=Q, kind="call"))


class TestVectorization:
    def test_mixed_good_and_bad_elementwise(self):
        K = np.array([100.0, 110.0, 100.0])
        price = np.array(
            [bs_price(S=S0, K=100.0, T=0.5, r=R, sigma=0.2, q=Q, kind="call"), 200.0, -1.0]
        )
        got = implied_vol(price, S=S0, K=K, T=0.5, r=R, q=Q, kind="call")
        assert abs(got[0] - 0.2) < 1e-6
        assert np.isnan(got[1]) and np.isnan(got[2])

    def test_scalar_in_float_out(self):
        price = bs_price(S=S0, K=105.0, T=0.5, r=R, sigma=0.25, q=Q, kind="put")
        out = implied_vol(price, S=S0, K=105.0, T=0.5, r=R, q=Q, kind="put")
        assert isinstance(out, float) and abs(out - 0.25) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_implied_vol.py -v`
Expected: ImportError — `implied_vol` not defined.

- [ ] **Step 3: Implement `implied_vol`**

Add `from scipy.optimize import brentq` to the module imports, then append:

```python
def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=20):
    """Invert bs_price for sigma. SPEC §2.3 algorithm:

    1. No-arbitrage pre-check; violations -> NaN, no iteration.
    2. Brenner-Subrahmanyam initial guess, clamped to [_MIN_VOL, _MAX_VOL].
    3. Vectorized Newton-Raphson on our own vega (tol in price, max_iter).
    4. Scalar brentq fallback for Newton's failures (tiny vega, out of
       bounds, no convergence). No sign change on the bracket -> NaN.

    tol is in price space: resulting sigma accuracy is ~ tol/vega, so a
    tight price tolerance is what buys 1e-6 sigma recovery off-ATM.
    """
    _validate_kind(kind)
    arrs = np.broadcast_arrays(*_as_arrays(price, S, K, T, r, q))
    price, S, K, T, r, q = (a.copy() for a in arrs)
    out = np.full(price.shape, np.nan)

    with np.errstate(all="ignore"):
        disc_s = S * np.exp(-q * T)
        disc_k = K * np.exp(-r * T)
        if kind == "call":
            lower, upper = np.maximum(disc_s - disc_k, 0.0), disc_s
        else:
            lower, upper = np.maximum(disc_k - disc_s, 0.0), disc_k
        solvable = (
            np.isfinite(price) & (price > 0) & (price >= lower) & (price <= upper)
            & (S > 0) & (K > 0) & (T > 0) & np.isfinite(S + K + T + r + q)
        )

        # Brenner-Subrahmanyam: near-exact ATM, decent start elsewhere.
        sigma = np.clip(
            np.sqrt(2.0 * np.pi / np.maximum(T, _EPS)) * price / S, _MIN_VOL, _MAX_VOL
        )
        active = solvable.copy()
        for _ in range(max_iter):
            if not active.any():
                break
            diff = bs_price(S, K, T, r, sigma, q, kind) - price
            d1, _ = _d1_d2(S, K, T, sigma, r, q)
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            converged = active & (np.abs(diff) < tol)
            out = np.where(converged, sigma, out)
            active &= ~converged

            # Tiny vega or non-finite step: Newton would explode -> Brent.
            active &= ~(active & ((vega < 1e-10) | ~np.isfinite(diff)))

            step = np.where(active, diff / np.maximum(vega, 1e-300), 0.0)
            sigma = sigma - step
            # Stepped out of bounds: hand off to Brent, don't clamp-and-lie.
            active &= ~((sigma < _MIN_VOL) | (sigma > _MAX_VOL) | ~np.isfinite(sigma))

        need_brent = solvable & np.isnan(out)

    it = np.nditer(need_brent, flags=["multi_index"])
    for flag in it:
        if not flag:
            continue
        idx = it.multi_index

        def f(s, idx=idx):
            return bs_price(S[idx], K[idx], T[idx], r[idx], s, q[idx], kind) - price[idx]

        try:
            out[idx] = brentq(f, _MIN_VOL, _MAX_VOL, xtol=1e-10)
        except ValueError:
            pass  # no sign change on [MIN_VOL, MAX_VOL]: price unreachable -> NaN

    return out if out.ndim else float(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_implied_vol.py -v`
Expected: all pass. Likely first-run failures and their meanings: normal-grid NaNs → the `active` bookkeeping is dropping converged elements (check mask order); corner garbage → the out-of-bounds handoff is clamping instead of releasing to Brent. Debug the masks, never widen tolerances.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -v`
Expected: every test from Tasks 1–5 passes.

- [ ] **Step 6: Append the learning-log entry**

Append a dated entry **"Inverting the formula: implied vol"** covering: why inversion is well-posed (price strictly increasing in σ — the monotonicity test from Task 3 is the proof); where the Brenner–Subrahmanyam √(2π/T)·(price/S) guess comes from (ATM price ≈ 0.4·S·σ√T linearization); exactly what breaks Newton deep OTM — vega in the denominator underflows and the step explodes (this **answers question 7**; record any actual failure observed while getting Step 4 green); why a *price-space* stopping rule gives σ-space error ≈ tol/vega, which forced tol=1e-10 and still can't rescue the far wings (found during plan self-review — the first grid design failed exactly this way on paper); why Brent can't fail where a root exists (bracketing); and why bounds violations must be NaN, not a clamped number (a fabricated vol would silently poison the smile panel). Also note for question 6: the vega geometry seen in Task 4 is *why* deep-wing IVs are numerically fragile everywhere, not just in our code.

- [ ] **Step 7: Commit**

```bash
git add src/models/black_scholes.py tests/test_implied_vol.py LEARNING_LOG.md
git commit -m "feat: Newton+Brent implied-vol solver with round-trip and corner-contract tests"
```

---

### Task 6: CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `requirements.txt` and the test suite from Tasks 1–5.
- Produces: tests running on every push/PR (SPEC §4). The Phase 7 daily workflow will be a separate file (`daily.yml`) — this one stays lightweight.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: ci

on: [push, pull_request]

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest -v
```

- [ ] **Step 2: Validate locally**

Run: `.venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && .venv/bin/pytest -q`
Expected: no YAML error; full suite passes.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run test suite on every push"
```

- [ ] **Step 4: Report phase exit status to the user**

Phase 1's exit criterion (SPEC §5) is "all §4 model tests green" — verified in Task 5 Step 5 and here. Phase 0's is "one manual script pulls and prints today's SPY chain" — verified in Task 2 (or explicitly blocked on the options-data finding). Report both statuses, plus a reminder that CI only actually runs once the user creates the GitHub remote and pushes (user action — repo name, visibility, and account are theirs to choose).
