# Phase 6 — P8 Delta-Hedged P&L Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build P8 — a stateless delta-hedged short-straddle simulation replayed from `data/chains/`, rendering three panels (cumulative P&L, P&L-vs-edge scatter, daily P&L histogram) under Q4 of the dashboard.

**Architecture:** A pure engine (`src/analytics/hedge_sim.py`) simulates one monthly trade at a time from in-memory frames; a thin replay driver (`src/data/hedge_replay.py`) globs the stored chains, IV-solves each with the existing `compute_chain_iv`, and feeds the engine — the same shape as `src/data/metrics_backfill.py`. Nothing is persisted: the whole P&L history is recomputed from stored chains on every run, so a later bug fix retroactively heals it (SPEC §3 P8 "Design"). Rendering lives in a new `src/render/hedge_figures.py` so `figures.py` does not grow further.

**Tech Stack:** Python 3.13, NumPy, pandas, `scipy.stats.norm` (via our own `src/models/black_scholes.py` only), Plotly graph_objects, pytest. No new dependencies.

**Spec:** `SPEC.md` — read §3 P8 (the panel), §2.1 (failure policy), §5 (phase table, row 6), §8 (sparse-history note).

## Global Constraints

- **No pricing libraries.** Every formula is implemented from scratch in this repo (`SPEC.md:25`). Prices, Greeks and IVs come only from `src.models.black_scholes`. Never read the chain's `vendor_iv` column.
- **Failure policy (SPEC §2.1).** In `src/run_daily.py`, *all* computation happens before *any* write. The write order is chain → `daily_metrics` → `docs/index.html` → `docs/status.json`, each atomic. Adding P8 must not move a write earlier or introduce one.
- **Stateless replay (SPEC §3 P8 "Design").** The simulation never accrues marks day by day and never writes sim state to disk. It is a pure function of `data/chains/` + `data/underlying.parquet` + (r, q, config).
- **Self-contained page (SPEC:212).** `docs/index.html` inlines the vendored Plotly bundle. Never add an external `<script src=…>`, CDN link, or web font.
- **Labeled as a simulation (SPEC §3 P8 "Note").** The panel is a *simulation for learning*, explicitly labeled as such on the page. It is not tracking real positions and is not advice.
- **Tests run offline and deterministically.** No network, no wall-clock dependence, no randomness. `pytest.ini` sets `filterwarnings = ["error"]`, so any pandas/NumPy warning raised by new code fails the suite — in particular, never index a DataFrame in a way that emits a `SettingWithCopyWarning`, and never let a NumPy operation emit `RuntimeWarning: invalid value`.
- **Config is the single source of tunables (SPEC §6).** The `hedge_sim:` block already exists in `config.yaml` with `structure: short_straddle`, `entry_dte: 30`, `hedge_frequency: daily`, `transaction_cost_bps: 0`. Read these; do not hard-code them.
- **Test command:** `.venv/bin/python -m pytest -q` from the repo root. Every task ends with the full suite green (285 tests pass on `main` before this plan starts).
- **Commit style:** conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`), imperative mood, no trailing period.

---

## Domain background (read once — every task assumes it)

You are simulating the textbook Black-Scholes hedging argument on real SPY data.

**The trade.** On the first stored session of each calendar month, *sell* one at-the-money straddle (one call + one put at the same strike and expiry) on the expiry nearest 30 days out. Quantity is **one option on one share** — no 100× contract multiplier — so every dollar figure on the page is per share. Hold it to expiry, re-hedging to delta-neutral once a day.

**Why the P&L looks the way it does.** A short straddle that is continuously delta-hedged earns, per unit time, approximately `½ Γ S² (σ²_implied − σ²_realized) dt`. Sell vol at 20 and the world realizes 12, and the position makes money; sell at 12 and the world realizes 20, and it loses. That is the entire point of the panel: the scatter of per-trade P&L against `(entry IV − realized vol over the trade's life)` should slope upward, and the regression line through it is the deliverable.

**The three sources of data.**
- `data/chains/YYYY-MM-DD.parquet` — one stored option chain per session. Columns are in `src/data/base.py:CHAIN_COLUMNS`. `compute_chain_iv` adds `price_used` (mid for `source == "yfinance"`, close otherwise) and `iv` (our own solver; NaN when it did not converge).
- `data/underlying.parquet` — columns `date`, `close`, `adjusted_close`, `volume`, one row per trading day back to 1993. This is the **complete** trading-day calendar; the chain archive is sparse by comparison.
- `config.yaml` — the `hedge_sim`, `chain_filter` and `rates` blocks.

**Verified fact you may rely on:** for every stored chain, `chain["spot"].iloc[0]` equals `underlying.close` on that date exactly. So the simulation uses `underlying.close` as its single spot series with no reconciliation needed. Use `close` for spot/hedging (the actual traded price) and `adjusted_close` for realized-vol returns (the existing convention in `src/models/realized_vol.py`).

**Two structural gaps you must handle, not paper over.**
1. `config.yaml: chain_filter.dte_min = 7` — a stored chain only contains expiries at least 7 days out. So in a trade's final week, its own expiry is **absent from every stored chain by design**. This is expected, not missing data.
2. The chain archive has holes (days that failed to backfill). Those are genuine data gaps.

The engine handles both the same way — mark the straddle with our own BS price at the last observed implied vols, and *count it* — but the counting is what keeps the panel honest: a trade whose marks are mostly model-generated is reported as `sparse` and kept off the scatter.

**Sign conventions, fixed for the whole plan.**
- The position is **short** the straddle. `V_t` is the straddle's market value (a positive number); the position is worth `−V_t`.
- `H_t` is the number of **shares held long**. Short-straddle delta is `−(Δ_call + Δ_put)`, so delta-neutrality means `H_t = Δ_call + Δ_put`. Near the money that is a small positive number (e.g. `0.53 + (−0.47) = 0.06`).
- Portfolio value `PV_t = cash_t + H_t·S_t − V_t`, and the cash account is seeded so that `PV_0 = 0` exactly. Cumulative P&L therefore *is* `PV_t`.

---

## File structure

| File | Responsibility | Task |
|---|---|---|
| `src/models/realized_vol.py` (modify) | add `realized_vol_between` — annualized vol over an explicit date window | 2 |
| `src/analytics/hedge_sim.py` (create) | pure engine: entry selection, straddle selection, one trade's daily marks, all trades, portfolio aggregation, regression fit, summary | 1, 2, 3 |
| `src/data/hedge_replay.py` (create) | reads `data/chains/` + `data/underlying.parquet`, IV-solves, calls the engine | 4 |
| `src/render/base.py` (create) | `LAYOUT` and `empty_figure` shared by both figure modules | 5 |
| `src/render/figures.py` (modify) | import the two shared helpers from `base.py`; no other change | 5 |
| `src/render/hedge_figures.py` (create) | `build_hedge_pnl_figure`, `build_hedge_histogram_figure`, `build_hedge_scatter_figure` | 5, 6 |
| `src/render/stats.py` (modify) | `hedge_summary_html` — the stat sentence and the simulation label | 6 |
| `src/render/page.py` (modify) | P8a/P8b/P8c titles + captions, Q4 wiring | 7 |
| `src/run_daily.py` (modify) | call the replay, add three figures + one extra, add status keys | 7 |
| `tests/test_hedge_sim.py` (create) | engine unit tests | 1, 2, 3 |
| `tests/test_hedge_replay.py` (create) | replay driver against fixture parquet files | 4 |
| `tests/test_render.py` (modify) | figure + caption + page tests | 5, 6, 7 |
| `tests/test_pipeline.py` (modify) | status keys and end-to-end wiring | 7 |
| `LEARNING_LOG.md`, `README.md` (modify) | Phase 6 teaching entry with real numbers; phase checkbox | 8 |

---

### Task 1: Entry dates and straddle selection

**Files:**
- Create: `src/analytics/hedge_sim.py`
- Test: `tests/test_hedge_sim.py`

**Interfaces:**
- Consumes: `config.yaml` keys `hedge_sim.entry_dte`, `hedge_sim.structure`, `hedge_sim.hedge_frequency`; chain frames already run through `src.analytics.chain_iv.compute_chain_iv` (so they carry `price_used` and `iv`).
- Produces:
  - `MAX_ENTRY_DTE_ERROR = 15`
  - `MIN_MARKET_MARK_SHARE = 0.7`
  - `TRADE_COLUMNS: list[str]`
  - `DAILY_COLUMNS: list[str]`
  - `entry_sessions(session_dates: list[datetime.date]) -> list[datetime.date]`
  - `select_straddle(chain_iv: pd.DataFrame, cfg: dict) -> dict | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hedge_sim.py`:

```python
"""P8 hedge-simulation engine tests (SPEC 3 P8). Offline, deterministic."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.hedge_sim import (
    MAX_ENTRY_DTE_ERROR, entry_sessions, select_straddle,
)

CFG = {
    "symbol": "SPY",
    "chain_filter": {"dte_min": 7, "dte_max": 365},
    "hedge_sim": {"structure": "short_straddle", "entry_dte": 30,
                  "hedge_frequency": "daily", "transaction_cost_bps": 0},
    "rates": {"risk_free_fallback": 0.04, "dividend_yield_fallback": 0.013},
}


def make_chain(session: dt.date, spot: float, expiries: dict[dt.date, float],
               strikes=(95.0, 100.0, 105.0), source="massive-backfill") -> pd.DataFrame:
    """A minimal IV-solved chain: `expiries` maps expiry -> flat IV for that expiry."""
    rows = []
    for expiry, iv in expiries.items():
        for strike in strikes:
            for kind in ("call", "put"):
                intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
                price = intrinsic + 2.0
                rows.append({
                    "snapshot_date": session, "spot": spot, "expiry": expiry,
                    "dte": (expiry - session).days, "strike": strike, "kind": kind,
                    "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                    "close": price, "volume": 10.0, "open_interest": 100.0,
                    "vendor_iv": np.nan, "source": source,
                    "price_used": price, "iv": iv,
                })
    return pd.DataFrame(rows)


class TestEntrySessions:
    def test_one_entry_per_calendar_month(self):
        dates = [dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 30),
                 dt.date(2026, 7, 6), dt.date(2026, 7, 7)]
        assert entry_sessions(dates) == [dt.date(2026, 6, 1), dt.date(2026, 7, 6)]

    def test_first_stored_session_wins_even_when_it_is_not_the_first_trading_day(self):
        # SPEC says "first trading day each month"; the archive may not hold it.
        dates = [dt.date(2026, 8, 14), dt.date(2026, 8, 17)]
        assert entry_sessions(dates) == [dt.date(2026, 8, 14)]

    def test_unsorted_input_is_sorted_first(self):
        dates = [dt.date(2026, 7, 20), dt.date(2026, 7, 2), dt.date(2026, 8, 3)]
        assert entry_sessions(dates) == [dt.date(2026, 7, 2), dt.date(2026, 8, 3)]

    def test_empty(self):
        assert entry_sessions([]) == []


class TestSelectStraddle:
    def test_picks_expiry_nearest_target_dte(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 100.0, {
            dt.date(2026, 8, 21): 0.20,   # 7 dte
            dt.date(2026, 9, 18): 0.18,   # 35 dte -> nearest to 30
            dt.date(2026, 10, 16): 0.19,  # 63 dte
        })
        sel = select_straddle(chain, CFG)
        assert sel["expiry"] == dt.date(2026, 9, 18)
        assert sel["dte"] == 35

    def test_picks_strike_nearest_spot(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 101.0, {dt.date(2026, 9, 18): 0.18})
        sel = select_straddle(chain, CFG)
        assert sel["strike"] == 100.0

    def test_straddle_is_the_sum_of_both_legs_and_iv_is_their_mean(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 100.0, {dt.date(2026, 9, 18): 0.18})
        chain.loc[(chain["strike"] == 100.0) & (chain["kind"] == "put"), "iv"] = 0.22
        sel = select_straddle(chain, CFG)
        assert sel["straddle"] == pytest.approx(sel["call_price"] + sel["put_price"])
        assert sel["call_iv"] == pytest.approx(0.18)
        assert sel["put_iv"] == pytest.approx(0.22)
        assert sel["iv"] == pytest.approx(0.20)

    def test_strike_needs_both_legs_converged(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 100.0, {dt.date(2026, 9, 18): 0.18})
        chain.loc[(chain["strike"] == 100.0) & (chain["kind"] == "put"), "iv"] = np.nan
        sel = select_straddle(chain, CFG)
        assert sel["strike"] in (95.0, 105.0)   # 100 is unusable, a neighbour wins

    def test_returns_none_when_nearest_expiry_is_too_far_from_target(self):
        session = dt.date(2026, 8, 14)
        far = session + dt.timedelta(days=30 + MAX_ENTRY_DTE_ERROR + 1)
        chain = make_chain(session, 100.0, {far: 0.18})
        assert select_straddle(chain, CFG) is None

    def test_returns_none_on_empty_chain(self):
        empty = make_chain(dt.date(2026, 8, 14), 100.0, {dt.date(2026, 9, 18): 0.18}).iloc[0:0]
        assert select_straddle(empty, CFG) is None

    def test_unsupported_structure_raises(self):
        session = dt.date(2026, 8, 14)
        chain = make_chain(session, 100.0, {dt.date(2026, 9, 18): 0.18})
        cfg = {**CFG, "hedge_sim": {**CFG["hedge_sim"], "structure": "iron_condor"}}
        with pytest.raises(ValueError, match="iron_condor"):
            select_straddle(chain, cfg)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hedge_sim.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.analytics.hedge_sim'`

- [ ] **Step 3: Write the implementation**

Create `src/analytics/hedge_sim.py`:

```python
"""P8: delta-hedged short-straddle simulation (SPEC 3 P8).

Stateless by construction: nothing here reads or writes sim state. Every
number is recomputed from stored chains + the underlying close series on
every run, so a later fix to this module heals the whole P&L history.

Position convention: SHORT one straddle on ONE share (no 100x contract
multiplier), so every dollar figure is per share. `H` is shares held long;
delta-neutrality against a short straddle means H = delta_call + delta_put.
"""
import datetime as dt

import numpy as np
import pandas as pd

# A monthly ladder puts the nearest expiry within ~15 days of any target, so a
# larger error means the chain is degenerate -- skip the month rather than
# quietly entering a 7-day trade and calling it a 30-day one.
MAX_ENTRY_DTE_ERROR = 15

# Below this share of market-quoted marks a trade is reported as `sparse` and
# kept off the scatter: its P&L is then mostly our own model talking to itself.
MIN_MARKET_MARK_SHARE = 0.7

TRADE_COLUMNS = [
    "entry_date", "expiry", "strike", "dte_at_entry", "entry_iv", "entry_straddle",
    "exit_date", "exit_value", "pnl", "n_days", "n_market_marks", "n_model_marks",
    "market_mark_share", "lifetime_rv", "edge", "status",
]

DAILY_COLUMNS = [
    "date", "entry_date", "expiry", "strike", "spot", "straddle", "call_iv", "put_iv",
    "hedge_shares", "cash", "pv", "pnl_day", "pnl_option", "pnl_hedge",
    "pnl_interest", "pnl_dividend", "pnl_cost", "mark_source",
]


def entry_sessions(session_dates) -> list[dt.date]:
    """The first stored session of each calendar month.

    SPEC says "the first trading day each month". The chain archive does not
    necessarily hold that day, so the rule is the first session we actually
    stored in the month; the page caption states this.
    """
    out: list[dt.date] = []
    seen: set[tuple[int, int]] = set()
    for d in sorted(session_dates):
        key = (d.year, d.month)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def select_straddle(chain_iv: pd.DataFrame, cfg: dict) -> dict | None:
    """The ATM straddle to sell on this session, or None if the chain cannot carry one."""
    structure = cfg["hedge_sim"]["structure"]
    if structure != "short_straddle":
        raise ValueError(f"unsupported hedge_sim.structure: {structure}")
    usable = chain_iv[chain_iv["iv"].notna() & chain_iv["price_used"].notna()]
    if usable.empty:
        return None
    target = int(cfg["hedge_sim"]["entry_dte"])
    available = usable[["expiry", "dte"]].drop_duplicates().sort_values("dte")
    pick = available.iloc[(available["dte"] - target).abs().to_numpy().argmin()]
    if abs(int(pick["dte"]) - target) > MAX_ENTRY_DTE_ERROR:
        return None

    g = usable[usable["expiry"] == pick["expiry"]]
    spot = float(g["spot"].iloc[0])
    legs = g.pivot_table(index="strike", columns="kind",
                         values=["price_used", "iv"], aggfunc="first")
    both = legs.dropna()
    if both.empty or "call" not in legs["price_used"].columns \
            or "put" not in legs["price_used"].columns:
        return None
    strikes = both.index.to_numpy(dtype=float)
    strike = float(strikes[np.abs(strikes - spot).argmin()])
    row = both.loc[strike]
    call_price, put_price = float(row[("price_used", "call")]), float(row[("price_used", "put")])
    call_iv, put_iv = float(row[("iv", "call")]), float(row[("iv", "put")])
    return {
        "expiry": pick["expiry"], "dte": int(pick["dte"]), "strike": strike,
        "call_price": call_price, "put_price": put_price,
        "straddle": call_price + put_price,
        "call_iv": call_iv, "put_iv": put_iv, "iv": 0.5 * (call_iv + put_iv),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hedge_sim.py -q`
Expected: PASS (11 tests)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS — 285 existing + 11 new

- [ ] **Step 6: Commit**

```bash
git add src/analytics/hedge_sim.py tests/test_hedge_sim.py
git commit -m "feat: P8 entry-session and ATM straddle selection"
```

---

### Task 2: One trade, marked and hedged daily

**Files:**
- Modify: `src/models/realized_vol.py` (append one function)
- Modify: `src/analytics/hedge_sim.py`
- Test: `tests/test_hedge_sim.py` (append), `tests/test_realized_vol.py` (append)

**Interfaces:**
- Consumes: `select_straddle(...) -> dict` and the module constants from Task 1; `src.models.black_scholes.bs_price(S, K, T, r, sigma, q, kind)` and `greeks(S, K, T, r, sigma, q, kind) -> dict` with keys `delta, gamma, vega, theta, rho` (scalars in → floats out).
- Produces:
  - `realized_vol_between(underlying: pd.DataFrame, start: dt.date, end: dt.date, annualization_days: int = 252) -> float`
  - `hedge_shares(spot, strike, T, r, q, call_iv, put_iv) -> float`
  - `simulate_trade(entry_date, sel, chains, underlying, r, q, cfg) -> tuple[pd.DataFrame, dict]`

**Semantics — implement exactly this recursion.** `Δ` is *calendar* days between consecutive trading days (so a weekend accrues three days of interest).

```
day 0 (entry):  S0 = close[entry_date]
                V0 = sel["straddle"];  H0 = hedge_shares(...)
                cash0 = V0 - H0*S0        # selling the straddle raises V0, buying H0 shares spends H0*S0
                pv0   = cash0 + H0*S0 - V0 = 0     (exactly, by construction)

day t:          interest = cash_{t-1} * (exp(r*Δ/365) - 1)
                dividend = H_{t-1} * S_{t-1} * (exp(q*Δ/365) - 1)
                hedge    = H_{t-1} * (S_t - S_{t-1})
                option   = -(V_t - V_{t-1})
                cost     = |H_t - H_{t-1}| * S_t * transaction_cost_bps / 1e4
                cash_t   = cash_{t-1} + interest + dividend - (H_t - H_{t-1})*S_t - cost
                pv_t     = cash_t + H_t*S_t - V_t
                pnl_day  = interest + dividend + hedge + option - cost
```

`pv_t - pv_{t-1} == pnl_day` is an algebraic identity, not an approximation. Test it.

**How `V_t`, `call_iv`, `put_iv` and `H_t` are obtained each day:**
- `d == settle_date` (the last trading day ≤ expiry, once that day exists): `V_t = abs(S_t - strike)`, `H_t = 0.0` (unwind), IVs carried forward, `mark_source = "settlement"`.
- Otherwise, if `chains[d]` exists and holds a row for this `(expiry, strike)` in **both** kinds with `iv` and `price_used` non-null: `V_t = call_price + put_price`, IVs updated from those rows, `mark_source = "market"`.
- Otherwise: `V_t = bs_price(call at last call_iv) + bs_price(put at last put_iv)`, IVs carried forward, `mark_source = "model"`.
- `H_t = hedge_shares(S_t, strike, dte/365, r, q, call_iv, put_iv)` with `dte = (expiry - d).days`, using whichever IVs are current after the step above (SPEC: "our **own** BS deltas at current market IV").

**Status:** `"settled"` if the trade reached `settle_date` **and** `market_mark_share >= MIN_MARKET_MARK_SHARE`; `"sparse"` if it reached settlement with a lower share; `"open"` if the underlying series ends before the expiry. `market_mark_share = n_market_marks / n_days` where `n_days` counts every row **after** day 0 (day 0 is the entry, always a market price).

- [ ] **Step 1: Write the failing test for `realized_vol_between`**

Append to `tests/test_realized_vol.py`:

```python
class TestRealizedVolBetween:
    def test_matches_hand_computation_over_an_explicit_window(self):
        import datetime as dt
        import numpy as np
        import pandas as pd
        from src.models.realized_vol import realized_vol_between

        dates = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(6)]
        closes = [100.0, 101.0, 100.5, 102.0, 101.5, 103.0]
        u = pd.DataFrame({"date": dates, "close": closes, "adjusted_close": closes,
                          "volume": [1.0] * 6})
        got = realized_vol_between(u, dates[1], dates[4])
        rets = np.diff(np.log(closes[1:5]))
        assert got == pytest.approx(float(np.std(rets, ddof=1) * np.sqrt(252)))

    def test_window_is_inclusive_of_both_endpoints_as_prices(self):
        import datetime as dt
        import pandas as pd
        from src.models.realized_vol import realized_vol_between

        dates = [dt.date(2026, 1, 5) + dt.timedelta(days=i) for i in range(4)]
        u = pd.DataFrame({"date": dates, "close": [100.0, 100.0, 100.0, 100.0],
                          "adjusted_close": [100.0, 100.0, 100.0, 100.0],
                          "volume": [1.0] * 4})
        assert realized_vol_between(u, dates[0], dates[3]) == pytest.approx(0.0)

    def test_too_few_returns_is_nan(self):
        import datetime as dt
        import numpy as np
        import pandas as pd
        from src.models.realized_vol import realized_vol_between

        dates = [dt.date(2026, 1, 5), dt.date(2026, 1, 6)]
        u = pd.DataFrame({"date": dates, "close": [100.0, 101.0],
                          "adjusted_close": [100.0, 101.0], "volume": [1.0, 1.0]})
        assert np.isnan(realized_vol_between(u, dates[0], dates[1]))
```

Note: `tests/test_realized_vol.py` already imports `pytest`, `numpy as np` and `pandas as pd` at module level; the local imports above are deliberate so the class is self-contained — keep them.

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_realized_vol.py -q`
Expected: FAIL — `ImportError: cannot import name 'realized_vol_between'`

- [ ] **Step 3: Implement `realized_vol_between`**

Append to `src/models/realized_vol.py`:

```python
def realized_vol_between(underlying: pd.DataFrame, start: dt.date, end: dt.date,
                         annualization_days: int = 252) -> float:
    """Annualized close-to-close realized vol from the returns inside [start, end].

    Same estimator as `trailing_realized_vol` (log returns on ADJUSTED close,
    sample std with ddof=1, sqrt-time annualization) but over an explicit
    window rather than a rolling one -- P8 needs the vol a trade actually
    lived through. Both endpoints are included as PRICES, so a window of n
    sessions yields n-1 returns. NaN below 2 returns.
    """
    u = _sorted(underlying)
    window = u[(u["date"] >= start) & (u["date"] <= end)]
    rets = log_returns(window["adjusted_close"]).dropna()
    if len(rets) < 2:
        return float("nan")
    return float(rets.std(ddof=1) * np.sqrt(annualization_days))
```

Add `import datetime as dt` to the module's imports if it is not already there.

- [ ] **Step 4: Verify it passes**

Run: `.venv/bin/python -m pytest tests/test_realized_vol.py -q`
Expected: PASS

- [ ] **Step 5: Write the failing tests for the trade engine**

Append to `tests/test_hedge_sim.py` (the `make_chain` helper and `CFG` from Task 1 are in scope):

```python
from src.analytics.hedge_sim import MIN_MARKET_MARK_SHARE, hedge_shares, simulate_trade


def make_underlying(start: dt.date, closes: list[float]) -> pd.DataFrame:
    """Consecutive weekday sessions starting at `start` (no weekend logic needed:
    the engine only ever reads the dates present in this frame)."""
    dates = [start + dt.timedelta(days=i) for i in range(len(closes))]
    return pd.DataFrame({"date": dates, "close": closes,
                         "adjusted_close": closes, "volume": [1.0] * len(closes)})


class TestHedgeShares:
    def test_atm_straddle_hedge_is_small_and_positive(self):
        h = hedge_shares(100.0, 100.0, 30 / 365, 0.04, 0.013, 0.20, 0.20)
        assert 0.0 < h < 0.15

    def test_deep_itm_call_dominates(self):
        h = hedge_shares(130.0, 100.0, 30 / 365, 0.04, 0.013, 0.20, 0.20)
        assert h == pytest.approx(1.0, abs=0.02)


class TestSimulateTrade:
    HORIZON = 30      # entry_dte in CFG; select_straddle rejects anything >15 days off it

    def _setup(self, closes, n_chain_days=None, spot0=100.0):
        """A 31-session archive whose single expiry is exactly `HORIZON` days out.

        `closes` is padded by HOLDING ITS LAST VALUE to 31 sessions, so a short
        list describes the opening moves and then a flat tail -- the settlement
        spot is `closes[-1]`.
        """
        entry = dt.date(2026, 6, 1)
        expiry = entry + dt.timedelta(days=self.HORIZON)
        n = self.HORIZON + 1
        series = list(closes) + [closes[-1]] * (n - len(closes))
        u = make_underlying(entry, series)
        n_chain_days = n if n_chain_days is None else n_chain_days
        chains = {}
        for i in range(n_chain_days):
            d = entry + dt.timedelta(days=i)
            chains[d] = make_chain(d, series[i], {expiry: 0.20}, strikes=(95.0, 100.0, 105.0))
        sel = select_straddle(chains[entry], CFG)
        assert sel is not None
        return entry, expiry, sel, chains, u

    def test_day_zero_pv_is_exactly_zero(self):
        entry, expiry, sel, chains, u = self._setup([100.0] * 6)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert daily["pv"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert daily["pnl_day"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert daily["mark_source"].iloc[0] == "market"

    def test_pnl_components_sum_to_daily_pnl_and_to_pv_change(self):
        entry, expiry, sel, chains, u = self._setup(
            [100.0, 101.5, 99.0, 103.0, 98.5, 100.0])
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        parts = (daily["pnl_option"] + daily["pnl_hedge"] + daily["pnl_interest"]
                 + daily["pnl_dividend"] - daily["pnl_cost"])
        assert np.allclose(parts.to_numpy(), daily["pnl_day"].to_numpy(), atol=1e-10)
        assert np.allclose(daily["pv"].diff().dropna().to_numpy(),
                           daily["pnl_day"].iloc[1:].to_numpy(), atol=1e-10)

    def test_settles_at_intrinsic_with_hedge_unwound(self):
        entry, expiry, sel, chains, u = self._setup([100.0, 100.0, 100.0, 107.0])
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        last = daily.iloc[-1]
        assert last["date"] == expiry
        assert last["mark_source"] == "settlement"
        assert last["straddle"] == pytest.approx(abs(107.0 - sel["strike"]))
        assert last["hedge_shares"] == pytest.approx(0.0)
        assert trade["status"] == "settled"
        assert trade["pnl"] == pytest.approx(last["pv"])
        assert trade["exit_date"] == expiry

    def test_trade_is_open_when_underlying_stops_before_expiry(self):
        entry = dt.date(2026, 6, 1)
        expiry = entry + dt.timedelta(days=30)
        u = make_underlying(entry, [100.0, 101.0, 102.0])
        chains = {entry + dt.timedelta(days=i): make_chain(
            entry + dt.timedelta(days=i), 100.0 + i, {expiry: 0.20}) for i in range(3)}
        sel = select_straddle(chains[entry], CFG)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert trade["status"] == "open"
        assert trade["exit_date"] == entry + dt.timedelta(days=2)
        assert not np.isnan(trade["edge"])   # edge is measured to the last mark

    def test_missing_chain_day_is_model_marked_and_counted(self):
        entry, expiry, sel, chains, u = self._setup([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        del chains[entry + dt.timedelta(days=2)]
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        row = daily[daily["date"] == entry + dt.timedelta(days=2)].iloc[0]
        assert row["mark_source"] == "model"
        assert row["straddle"] > 0
        assert trade["n_model_marks"] == 1
        assert trade["n_market_marks"] == trade["n_days"] - 1

    def test_mostly_model_marked_trade_is_reported_sparse(self):
        entry, expiry, sel, chains, u = self._setup([100.0] * 8, n_chain_days=2)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert trade["market_mark_share"] < MIN_MARKET_MARK_SHARE
        assert trade["status"] == "sparse"

    def test_transaction_costs_reduce_pnl(self):
        closes = [100.0, 103.0, 97.0, 104.0, 99.0]
        entry, expiry, sel, chains, u = self._setup(closes)
        free, t_free = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        cfg = {**CFG, "hedge_sim": {**CFG["hedge_sim"], "transaction_cost_bps": 50}}
        costed, t_costed = simulate_trade(entry, sel, chains, u, 0.04, 0.013, cfg)
        assert costed["pnl_cost"].sum() > 0
        assert t_costed["pnl"] < t_free["pnl"]

    def test_short_straddle_makes_money_when_realized_vol_is_far_below_implied(self):
        # Sold at 20 vol, the world does not move at all: the seller keeps the premium.
        entry, expiry, sel, chains, u = self._setup([100.0] * 8)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert trade["pnl"] > 0
        assert trade["lifetime_rv"] == pytest.approx(0.0, abs=1e-9)
        assert trade["edge"] == pytest.approx(trade["entry_iv"] - trade["lifetime_rv"])

    def test_unsupported_hedge_frequency_raises(self):
        entry, expiry, sel, chains, u = self._setup([100.0] * 4)
        cfg = {**CFG, "hedge_sim": {**CFG["hedge_sim"], "hedge_frequency": "weekly"}}
        with pytest.raises(ValueError, match="weekly"):
            simulate_trade(entry, sel, chains, u, 0.04, 0.013, cfg)

    def test_daily_frame_has_the_declared_columns(self):
        from src.analytics.hedge_sim import DAILY_COLUMNS
        entry, expiry, sel, chains, u = self._setup([100.0] * 5)
        daily, trade = simulate_trade(entry, sel, chains, u, 0.04, 0.013, CFG)
        assert list(daily.columns) == DAILY_COLUMNS
```

Delete the stray `assert daily["pnl_cum"] ...` line from `test_pnl_components_sum_to_daily_pnl_and_to_pv_change` if you prefer — `pnl_cum` is **not** a column of the per-trade daily frame (it is produced by `portfolio_daily` in Task 3). Keep the first two assertions of that test.

- [ ] **Step 6: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hedge_sim.py -q`
Expected: FAIL — `ImportError: cannot import name 'hedge_shares'`

- [ ] **Step 7: Implement the trade engine**

Append to `src/analytics/hedge_sim.py` (add `from src.models.black_scholes import bs_price, greeks` and `from src.models.realized_vol import realized_vol_between` to the imports):

```python
def hedge_shares(spot: float, strike: float, T: float, r: float, q: float,
                 call_iv: float, put_iv: float) -> float:
    """Shares to hold long so the SHORT straddle is delta-neutral.

    Each leg is evaluated at ITS OWN market implied vol -- the same rule P4
    uses for the Greeks curves -- so the hedge is what a desk reading these
    quotes would actually put on, not a flat-vol textbook number.
    """
    if T <= 0:
        return 0.0
    dc = greeks(S=spot, K=strike, T=T, r=r, sigma=call_iv, q=q, kind="call")["delta"]
    dp = greeks(S=spot, K=strike, T=T, r=r, sigma=put_iv, q=q, kind="put")["delta"]
    return float(dc + dp)


def _quote(chain_iv: pd.DataFrame | None, expiry, strike: float) -> tuple[float, float, float] | None:
    """(straddle value, call_iv, put_iv) for this leg pair, or None if not quoted."""
    if chain_iv is None or chain_iv.empty:
        return None
    g = chain_iv[(chain_iv["expiry"] == expiry) & (chain_iv["strike"] == strike)
                 & chain_iv["iv"].notna() & chain_iv["price_used"].notna()]
    legs = {kind: g[g["kind"] == kind] for kind in ("call", "put")}
    if any(v.empty for v in legs.values()):
        return None
    call, put = legs["call"].iloc[0], legs["put"].iloc[0]
    return (float(call["price_used"]) + float(put["price_used"]),
            float(call["iv"]), float(put["iv"]))


def simulate_trade(entry_date: dt.date, sel: dict, chains: dict, underlying: pd.DataFrame,
                   r: float, q: float, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Replay one short straddle from entry to settlement (or to the last stored session)."""
    frequency = cfg["hedge_sim"]["hedge_frequency"]
    if frequency != "daily":
        raise ValueError(f"unsupported hedge_sim.hedge_frequency: {frequency}")
    bps = float(cfg["hedge_sim"]["transaction_cost_bps"])
    expiry, strike = sel["expiry"], sel["strike"]

    u = underlying.sort_values("date")
    closes = dict(zip(u["date"], u["close"].astype(float)))
    days = [d for d in sorted(closes) if entry_date <= d <= expiry]
    # A holiday expiry never appears in the calendar, so settlement is the last
    # session at or before it -- and only once the calendar has actually reached it.
    settle_date = days[-1] if days and max(closes) >= expiry else None

    spot = closes[entry_date]
    call_iv, put_iv = sel["call_iv"], sel["put_iv"]
    value = float(sel["straddle"])
    shares = hedge_shares(spot, strike, (expiry - entry_date).days / 365.0, r, q, call_iv, put_iv)
    cash = value - shares * spot
    rows = [{
        "date": entry_date, "entry_date": entry_date, "expiry": expiry, "strike": strike,
        "spot": spot, "straddle": value, "call_iv": call_iv, "put_iv": put_iv,
        "hedge_shares": shares, "cash": cash, "pv": cash + shares * spot - value,
        "pnl_day": 0.0, "pnl_option": 0.0, "pnl_hedge": 0.0, "pnl_interest": 0.0,
        "pnl_dividend": 0.0, "pnl_cost": 0.0, "mark_source": "market",
    }]
    n_market = n_model = 0

    for day in days[1:]:
        gap = (day - rows[-1]["date"]).days
        prev = rows[-1]
        new_spot = closes[day]
        if day == settle_date:
            new_value, source, new_shares = abs(new_spot - strike), "settlement", 0.0
        else:
            quoted = _quote(chains.get(day), expiry, strike)
            if quoted is not None:
                new_value, call_iv, put_iv = quoted
                source = "market"
                n_market += 1
            else:
                T = (expiry - day).days / 365.0
                new_value = float(
                    bs_price(new_spot, strike, T, r, call_iv, q, "call")
                    + bs_price(new_spot, strike, T, r, put_iv, q, "put"))
                source = "model"
                n_model += 1
            new_shares = hedge_shares(new_spot, strike, (expiry - day).days / 365.0,
                                      r, q, call_iv, put_iv)

        interest = prev["cash"] * (np.exp(r * gap / 365.0) - 1.0)
        dividend = prev["hedge_shares"] * prev["spot"] * (np.exp(q * gap / 365.0) - 1.0)
        hedge = prev["hedge_shares"] * (new_spot - prev["spot"])
        option = -(new_value - prev["straddle"])
        cost = abs(new_shares - prev["hedge_shares"]) * new_spot * bps / 1e4
        cash = prev["cash"] + interest + dividend - (new_shares - prev["hedge_shares"]) * new_spot - cost
        rows.append({
            "date": day, "entry_date": entry_date, "expiry": expiry, "strike": strike,
            "spot": new_spot, "straddle": new_value, "call_iv": call_iv, "put_iv": put_iv,
            "hedge_shares": new_shares, "cash": cash,
            "pv": cash + new_shares * new_spot - new_value,
            "pnl_day": interest + dividend + hedge + option - cost,
            "pnl_option": option, "pnl_hedge": hedge, "pnl_interest": interest,
            "pnl_dividend": dividend, "pnl_cost": cost, "mark_source": source,
        })
        if day == settle_date:
            n_market += 1     # settlement is a market fact, not a model mark

    daily = pd.DataFrame(rows, columns=DAILY_COLUMNS)
    last = daily.iloc[-1]
    n_days = len(daily) - 1
    share = (n_market / n_days) if n_days else 1.0
    settled = settle_date is not None and last["date"] == settle_date
    status = "open" if not settled else ("settled" if share >= MIN_MARKET_MARK_SHARE else "sparse")
    lifetime_rv = realized_vol_between(underlying, entry_date, last["date"])
    trade = {
        "entry_date": entry_date, "expiry": expiry, "strike": strike,
        "dte_at_entry": int(sel["dte"]), "entry_iv": float(sel["iv"]),
        "entry_straddle": float(sel["straddle"]), "exit_date": last["date"],
        "exit_value": float(last["straddle"]), "pnl": float(last["pv"]),
        "n_days": int(n_days), "n_market_marks": int(n_market), "n_model_marks": int(n_model),
        "market_mark_share": float(share), "lifetime_rv": lifetime_rv,
        "edge": float(sel["iv"]) - lifetime_rv, "status": status,
    }
    return daily, trade
```

- [ ] **Step 8: Verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hedge_sim.py tests/test_realized_vol.py -q`
Expected: PASS

- [ ] **Step 9: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add src/analytics/hedge_sim.py src/models/realized_vol.py tests/test_hedge_sim.py tests/test_realized_vol.py
git commit -m "feat: P8 daily mark-to-market and delta-hedge recursion for one trade"
```

---

### Task 3: All trades, portfolio aggregation, regression, summary

**Files:**
- Modify: `src/analytics/hedge_sim.py`
- Test: `tests/test_hedge_sim.py` (append)

**Interfaces:**
- Consumes: `entry_sessions`, `select_straddle`, `simulate_trade`, `TRADE_COLUMNS`, `MIN_MARKET_MARK_SHARE`.
- Produces:
  - `simulate(chains: dict[dt.date, pd.DataFrame], underlying: pd.DataFrame, r: float, q: float, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]` → `(trades, daily)`; `daily` is every trade's rows concatenated (trades overlap in time — a 35-day trade entered in June is still live when July's is opened).
  - `portfolio_daily(daily: pd.DataFrame) -> pd.DataFrame` with columns `["date", "pnl_day", "pnl_cum", "n_open"]`.
  - `fit_pnl_vs_edge(trades: pd.DataFrame) -> dict` with keys `n, slope, intercept, r2` (`slope` in **dollars per vol point**, i.e. computed against `edge * 100`).
  - `hedge_summary(trades: pd.DataFrame, port: pd.DataFrame, fit: dict) -> dict` with keys `n_trades, n_settled, n_open, n_sparse, first_entry, last_mark, cum_pnl, n_days, mean_daily_pnl, sd_daily_pnl, market_mark_share, next_settlement, slope, r2`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hedge_sim.py`:

```python
from src.analytics.hedge_sim import (
    TRADE_COLUMNS, fit_pnl_vs_edge, hedge_summary, portfolio_daily, simulate,
)


class TestSimulate:
    def _archive(self, months, closes_per_month=8):
        """One entry per month; each trade's expiry is 5 sessions after entry."""
        chains, closes, dates = {}, [], []
        day = dt.date(2026, 6, 1)
        for _ in range(months * closes_per_month):
            dates.append(day)
            closes.append(100.0)
            day += dt.timedelta(days=4)      # ~4-day steps walk through several months
        u = pd.DataFrame({"date": dates, "close": closes, "adjusted_close": closes,
                          "volume": [1.0] * len(dates)})
        for i, d in enumerate(dates):
            expiry = dates[min(i + 5, len(dates) - 1)]
            chains[d] = make_chain(d, 100.0, {expiry: 0.20})
        return chains, u

    def test_one_trade_per_month_with_declared_columns(self):
        chains, u = self._archive(months=2)
        trades, daily = simulate(chains, u, 0.04, 0.013, CFG)
        assert list(trades.columns) == TRADE_COLUMNS
        assert len(trades) == trades["entry_date"].nunique()
        months = {(d.year, d.month) for d in trades["entry_date"]}
        assert len(months) == len(trades)

    def test_empty_archive_returns_empty_frames_with_columns(self):
        from src.analytics.hedge_sim import DAILY_COLUMNS
        u = pd.DataFrame({"date": [], "close": [], "adjusted_close": [], "volume": []})
        trades, daily = simulate({}, u, 0.04, 0.013, CFG)
        assert list(trades.columns) == TRADE_COLUMNS and trades.empty
        assert list(daily.columns) == DAILY_COLUMNS and daily.empty


class TestPortfolioDaily:
    def test_sums_overlapping_trades_and_accumulates(self):
        daily = pd.DataFrame({
            "date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 2)],
            "entry_date": [dt.date(2026, 6, 1), dt.date(2026, 6, 1), dt.date(2026, 6, 2)],
            "pnl_day": [0.0, 1.5, 0.0],
        })
        port = portfolio_daily(daily)
        assert list(port.columns) == ["date", "pnl_day", "pnl_cum", "n_open"]
        assert port["pnl_day"].tolist() == [0.0, 1.5]
        assert port["pnl_cum"].tolist() == [0.0, 1.5]
        assert port["n_open"].tolist() == [1, 2]

    def test_empty(self):
        port = portfolio_daily(pd.DataFrame(columns=["date", "entry_date", "pnl_day"]))
        assert port.empty and list(port.columns) == ["date", "pnl_day", "pnl_cum", "n_open"]


class TestFit:
    def test_recovers_a_known_line(self):
        trades = pd.DataFrame({
            "edge": [0.01, 0.02, 0.03, 0.04],           # 1..4 vol points
            "pnl": [2.0, 4.0, 6.0, 8.0],                 # slope 2 $/vol point
            "status": ["settled"] * 4,
        })
        fit = fit_pnl_vs_edge(trades)
        assert fit["n"] == 4
        assert fit["slope"] == pytest.approx(2.0)
        assert fit["intercept"] == pytest.approx(0.0, abs=1e-9)
        assert fit["r2"] == pytest.approx(1.0)

    def test_ignores_open_and_sparse_trades(self):
        trades = pd.DataFrame({
            "edge": [0.01, 0.02, 0.03],
            "pnl": [2.0, 4.0, 999.0],
            "status": ["settled", "settled", "open"],
        })
        assert fit_pnl_vs_edge(trades)["n"] == 2

    def test_fewer_than_two_points_gives_no_line(self):
        trades = pd.DataFrame({"edge": [0.01], "pnl": [2.0], "status": ["settled"]})
        fit = fit_pnl_vs_edge(trades)
        assert fit["n"] == 1
        assert np.isnan(fit["slope"]) and np.isnan(fit["r2"])

    def test_nan_edge_is_dropped(self):
        trades = pd.DataFrame({
            "edge": [0.01, np.nan, 0.03], "pnl": [2.0, 4.0, 6.0],
            "status": ["settled"] * 3,
        })
        assert fit_pnl_vs_edge(trades)["n"] == 2


class TestHedgeSummary:
    def test_counts_statuses_and_reports_the_next_settlement(self):
        trades = pd.DataFrame({
            "entry_date": [dt.date(2026, 6, 1), dt.date(2026, 7, 1)],
            "expiry": [dt.date(2026, 7, 17), dt.date(2026, 8, 21)],
            "exit_date": [dt.date(2026, 7, 17), dt.date(2026, 8, 12)],
            "pnl": [1.0, -0.5], "edge": [0.01, 0.02],
            "n_days": [30, 20], "n_market_marks": [27, 20], "n_model_marks": [3, 0],
            "market_mark_share": [0.9, 1.0], "status": ["settled", "open"],
        })
        port = pd.DataFrame({"date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2)],
                             "pnl_day": [0.0, 1.0], "pnl_cum": [0.0, 1.0], "n_open": [1, 1]})
        s = hedge_summary(trades, port, {"n": 1, "slope": np.nan,
                                         "intercept": np.nan, "r2": np.nan})
        assert (s["n_trades"], s["n_settled"], s["n_open"], s["n_sparse"]) == (2, 1, 1, 0)
        assert s["next_settlement"] == dt.date(2026, 8, 21)
        assert s["cum_pnl"] == pytest.approx(1.0)
        assert s["first_entry"] == dt.date(2026, 6, 1)
        assert s["market_mark_share"] == pytest.approx(47 / 50)

    def test_empty_summary_is_all_zeros_and_nones(self):
        s = hedge_summary(pd.DataFrame(columns=TRADE_COLUMNS),
                          pd.DataFrame(columns=["date", "pnl_day", "pnl_cum", "n_open"]),
                          {"n": 0, "slope": np.nan, "intercept": np.nan, "r2": np.nan})
        assert s["n_trades"] == 0 and s["first_entry"] is None
        assert s["next_settlement"] is None
        assert np.isnan(s["cum_pnl"])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_hedge_sim.py -q`
Expected: FAIL — `ImportError: cannot import name 'simulate'`

- [ ] **Step 3: Implement**

Append to `src/analytics/hedge_sim.py`:

```python
def simulate(chains: dict, underlying: pd.DataFrame, r: float, q: float,
             cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay every monthly trade the stored archive supports."""
    trades, frames = [], []
    for entry in entry_sessions(list(chains)):
        sel = select_straddle(chains[entry], cfg)
        if sel is None:
            continue
        daily, trade = simulate_trade(entry, sel, chains, underlying, r, q, cfg)
        frames.append(daily)
        trades.append(trade)
    daily_all = (pd.concat(frames, ignore_index=True) if frames
                 else pd.DataFrame(columns=DAILY_COLUMNS))
    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return trades_df.sort_values("entry_date").reset_index(drop=True), daily_all


def portfolio_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Total P&L per calendar day across all live trades, and its running sum."""
    cols = ["date", "pnl_day", "pnl_cum", "n_open"]
    if daily.empty:
        return pd.DataFrame(columns=cols)
    g = (daily.groupby("date", as_index=False)
         .agg(pnl_day=("pnl_day", "sum"), n_open=("entry_date", "nunique"))
         .sort_values("date").reset_index(drop=True))
    g["pnl_cum"] = g["pnl_day"].cumsum()
    return g[cols]


def fit_pnl_vs_edge(trades: pd.DataFrame) -> dict:
    """Least-squares line through settled trades: P&L ($) vs edge (VOL POINTS)."""
    out = {"n": 0, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    if trades.empty:
        return out
    s = trades[(trades["status"] == "settled") & trades["edge"].notna()
               & trades["pnl"].notna()]
    out["n"] = int(len(s))
    if len(s) < 2:
        return out
    x = s["edge"].to_numpy(dtype=float) * 100.0
    y = s["pnl"].to_numpy(dtype=float)
    if np.ptp(x) == 0:            # a vertical cloud has no line through it
        return out
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    out.update(slope=float(slope), intercept=float(intercept),
               r2=(1.0 - float((resid ** 2).sum()) / ss_tot) if ss_tot > 0 else float("nan"))
    return out


def hedge_summary(trades: pd.DataFrame, port: pd.DataFrame, fit: dict) -> dict:
    """Everything the page's stat line and status.json need, in one dict."""
    counts = trades["status"].value_counts().to_dict() if not trades.empty else {}
    open_trades = trades[trades["status"] == "open"] if not trades.empty else trades
    marks = int(trades["n_days"].sum()) if not trades.empty else 0
    return {
        "n_trades": int(len(trades)),
        "n_settled": int(counts.get("settled", 0)),
        "n_open": int(counts.get("open", 0)),
        "n_sparse": int(counts.get("sparse", 0)),
        "first_entry": trades["entry_date"].min() if not trades.empty else None,
        "last_mark": port["date"].max() if not port.empty else None,
        "cum_pnl": float(port["pnl_cum"].iloc[-1]) if not port.empty else float("nan"),
        "n_days": int(len(port)),
        "mean_daily_pnl": (float(port["pnl_day"].iloc[1:].mean())
                           if len(port) > 1 else float("nan")),
        "sd_daily_pnl": (float(port["pnl_day"].iloc[1:].std(ddof=1))
                         if len(port) > 2 else float("nan")),
        "market_mark_share": (float(trades["n_market_marks"].sum() / marks)
                              if marks else float("nan")),
        "next_settlement": (open_trades["expiry"].min() if len(open_trades) else None),
        "slope": fit["slope"], "r2": fit["r2"],
    }
```

- [ ] **Step 4: Verify they pass**

Run: `.venv/bin/python -m pytest tests/test_hedge_sim.py -q`
Expected: PASS

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/analytics/hedge_sim.py tests/test_hedge_sim.py
git commit -m "feat: P8 multi-trade replay, portfolio aggregation and edge regression"
```

---

### Task 4: Replay driver over the stored archive

**Files:**
- Create: `src/data/hedge_replay.py`
- Test: `tests/test_hedge_replay.py`

**Interfaces:**
- Consumes: `src.analytics.chain_iv.compute_chain_iv(chain, r, q) -> (df, stats)`; `src.data.storage.read_underlying(root) -> pd.DataFrame`; `simulate`, `portfolio_daily`, `fit_pnl_vs_edge`, `hedge_summary`.
- Produces: `replay_hedge_sim(root, r, q, cfg, extra_chains=None, underlying=None) -> dict` with keys `trades`, `daily`, `portfolio`, `fit`, `summary`.

`extra_chains` exists because `src/run_daily.py` computes everything *before* writing today's chain file (SPEC §2.1) — today's session would otherwise be invisible to the sim until tomorrow. It maps `date -> already-IV-solved chain frame` and takes precedence over anything on disk for that date.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hedge_replay.py`:

```python
"""P8 replay driver: stored parquet in, simulation out. Offline."""
import datetime as dt

import numpy as np
import pandas as pd

from src.data.hedge_replay import replay_hedge_sim

CFG = {
    "symbol": "SPY",
    "chain_filter": {"dte_min": 7, "dte_max": 365},
    "hedge_sim": {"structure": "short_straddle", "entry_dte": 30,
                  "hedge_frequency": "daily", "transaction_cost_bps": 0},
}


def write_archive(root, dates, expiry, spot=100.0):
    (root / "data" / "chains").mkdir(parents=True)
    for d in dates:
        rows = []
        for strike in (95.0, 100.0, 105.0):
            for kind in ("call", "put"):
                intrinsic = max(spot - strike, 0.0) if kind == "call" else max(strike - spot, 0.0)
                price = intrinsic + 3.0
                rows.append({
                    "snapshot_date": d, "spot": spot, "expiry": expiry,
                    "dte": (expiry - d).days, "strike": strike, "kind": kind,
                    "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                    "close": price, "volume": 10.0, "open_interest": 100.0,
                    "vendor_iv": np.nan, "source": "massive-backfill",
                })
        pd.DataFrame(rows).to_parquet(root / "data" / "chains" / f"{d.isoformat()}.parquet")
    u = pd.DataFrame({"date": dates, "close": [spot] * len(dates),
                      "adjusted_close": [spot] * len(dates), "volume": [1.0] * len(dates)})
    u.to_parquet(root / "data" / "underlying.parquet")


def test_replays_the_archive_and_returns_every_key(tmp_path):
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(6)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 7, 1))
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    assert set(out) == {"trades", "daily", "portfolio", "fit", "summary"}
    assert len(out["trades"]) == 1
    assert out["trades"]["entry_date"].iloc[0] == dates[0]
    assert out["summary"]["n_open"] == 1
    assert len(out["portfolio"]) == 6


def test_ignores_non_date_filenames(tmp_path):
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(4)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 7, 1))
    (tmp_path / "data" / "chains" / "notes.parquet").write_bytes(b"not a parquet")
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    assert len(out["trades"]) == 1


def test_extra_chains_are_visible_before_the_file_is_written(tmp_path):
    dates = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(4)]
    write_archive(tmp_path, dates, expiry=dt.date(2026, 7, 1))
    today = dates[-1] + dt.timedelta(days=1)
    stored = pd.read_parquet(tmp_path / "data" / "chains" / f"{dates[-1].isoformat()}.parquet")
    fresh = stored.assign(snapshot_date=today, dte=lambda d: (dt.date(2026, 7, 1) - today).days)
    from src.analytics.chain_iv import compute_chain_iv
    solved, _ = compute_chain_iv(fresh, 0.04, 0.013)
    u = pd.read_parquet(tmp_path / "data" / "underlying.parquet")
    u = pd.concat([u, u.tail(1).assign(date=today)], ignore_index=True)
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG,
                           extra_chains={today: solved}, underlying=u)
    assert out["portfolio"]["date"].max() == today


def test_empty_archive_is_not_an_error(tmp_path):
    (tmp_path / "data" / "chains").mkdir(parents=True)
    pd.DataFrame({"date": [], "close": [], "adjusted_close": [], "volume": []}) \
        .to_parquet(tmp_path / "data" / "underlying.parquet")
    out = replay_hedge_sim(tmp_path, 0.04, 0.013, CFG)
    assert out["trades"].empty
    assert out["summary"]["n_trades"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_hedge_replay.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.hedge_replay'`

- [ ] **Step 3: Implement**

Create `src/data/hedge_replay.py`:

```python
"""Replay the P8 hedge simulation over the stored chain archive (SPEC 3 P8).

Same shape as src/data/metrics_backfill.py: read every stored chain, solve
its IVs with the shared solver, hand the result to a pure analytics engine.
Nothing is persisted -- the whole P&L history is a function of data/chains/
+ data/underlying.parquet + (r, q, cfg), recomputed on every run.
"""
import datetime as dt
from pathlib import Path

import pandas as pd

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.hedge_sim import (
    fit_pnl_vs_edge, hedge_summary, portfolio_daily, simulate,
)
from src.data import storage

_CHAIN_GLOB = "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].parquet"


def replay_hedge_sim(root: Path, r: float, q: float, cfg: dict,
                     extra_chains: dict | None = None,
                     underlying: pd.DataFrame | None = None) -> dict:
    root = Path(root)
    chains: dict[dt.date, pd.DataFrame] = {}
    for path in sorted((root / "data" / "chains").glob(_CHAIN_GLOB)):
        session = dt.date.fromisoformat(path.stem)
        solved, _ = compute_chain_iv(pd.read_parquet(path), r, q)
        chains[session] = solved
    # Today's chain is computed before it is written (SPEC 2.1 failure policy),
    # so the caller passes it in rather than the sim missing the current session.
    chains.update(extra_chains or {})

    if underlying is None:
        underlying = storage.read_underlying(root)
    trades, daily = simulate(chains, underlying, r, q, cfg)
    port = portfolio_daily(daily)
    fit = fit_pnl_vs_edge(trades)
    return {"trades": trades, "daily": daily, "portfolio": port, "fit": fit,
            "summary": hedge_summary(trades, port, fit)}
```

- [ ] **Step 4: Verify it passes**

Run: `.venv/bin/python -m pytest tests/test_hedge_replay.py -q`
Expected: PASS

- [ ] **Step 5: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/data/hedge_replay.py tests/test_hedge_replay.py
git commit -m "feat: stateless P8 replay driver over the stored chain archive"
```

---

### Task 5: Shared figure base + cumulative-P&L and histogram figures

**Files:**
- Create: `src/render/base.py`
- Modify: `src/render/figures.py` (imports only)
- Create: `src/render/hedge_figures.py`
- Test: `tests/test_render.py` (append)

**Interfaces:**
- Consumes: `portfolio_daily` output columns `["date", "pnl_day", "pnl_cum", "n_open"]`; `TRADE_COLUMNS`.
- Produces:
  - `src/render/base.py`: `LAYOUT: dict`, `empty_figure(title: str, message: str, **layout) -> go.Figure`
  - `src/render/hedge_figures.py`: `build_hedge_pnl_figure(port, trades, daily) -> go.Figure`, `build_hedge_histogram_figure(port) -> go.Figure`

**Why `base.py`:** `figures.py` is already 400+ lines with eight builders, and the Phase 5 review recorded "split figures.py" as a deferred item. Extracting only the two genuinely shared helpers keeps the diff small and stops `hedge_figures.py` from importing another module's privates.

- [ ] **Step 1: Create `src/render/base.py` and rewire `figures.py`**

Create `src/render/base.py` by moving the existing definitions verbatim out of `src/render/figures.py`:

```python
"""Layout defaults and the empty-state figure, shared by every figure module."""
import plotly.graph_objects as go

LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=50, r=20, t=100, b=45),
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.12),
    font=dict(size=13),
    title_y=0.97,
    title_yanchor="top",
)


def empty_figure(title: str, message: str, **layout) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, **{**LAYOUT, **layout})
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(size=14, color="#666"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
```

Before writing it, open `src/render/figures.py` and copy the **current** bodies of `_LAYOUT` and `_empty_figure` exactly — if they differ from the above in any detail, the file on disk wins. Then in `figures.py`, delete both definitions and add near the top:

```python
from src.render.base import LAYOUT as _LAYOUT, empty_figure as _empty_figure
```

`_PARITY_LAYOUT` stays in `figures.py`. No other line of `figures.py` changes.

- [ ] **Step 2: Verify the refactor changed nothing**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, same count as before (285 + everything added so far). If any render test fails, the move was not verbatim — fix it before continuing.

- [ ] **Step 3: Commit the refactor on its own**

```bash
git add src/render/base.py src/render/figures.py
git commit -m "refactor: extract shared LAYOUT and empty_figure into src/render/base.py"
```

- [ ] **Step 4: Write the failing figure tests**

Append to `tests/test_render.py` (match the file's existing import style and test-class conventions):

```python
class TestHedgeFigures:
    def _port(self):
        import datetime as dt
        return pd.DataFrame({
            "date": [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(5)],
            "pnl_day": [0.0, 1.0, -0.5, 2.0, -1.0],
            "pnl_cum": [0.0, 1.0, 0.5, 2.5, 1.5],
            "n_open": [1, 1, 1, 2, 2],
        })

    def _trades(self):
        import datetime as dt
        return pd.DataFrame({
            "entry_date": [dt.date(2026, 6, 1)], "expiry": [dt.date(2026, 7, 17)],
            "strike": [100.0], "dte_at_entry": [30], "entry_iv": [0.20],
            "entry_straddle": [6.0], "exit_date": [dt.date(2026, 6, 5)],
            "exit_value": [4.0], "pnl": [1.5], "n_days": [4],
            "n_market_marks": [4], "n_model_marks": [0], "market_mark_share": [1.0],
            "lifetime_rv": [0.12], "edge": [0.08], "status": ["open"],
        })

    def test_pnl_figure_plots_the_cumulative_line(self):
        from src.render.hedge_figures import build_hedge_pnl_figure
        port = self._port()
        fig = build_hedge_pnl_figure(port, self._trades(), pd.DataFrame())
        cum = [t for t in fig.data if "cumulative" in (t.name or "").lower()]
        assert len(cum) == 1
        assert list(cum[0].y) == port["pnl_cum"].tolist()
        assert "$" in fig.layout.yaxis.title.text

    def test_pnl_figure_is_empty_state_with_no_data(self):
        from src.render.hedge_figures import build_hedge_pnl_figure
        fig = build_hedge_pnl_figure(
            pd.DataFrame(columns=["date", "pnl_day", "pnl_cum", "n_open"]),
            pd.DataFrame(), pd.DataFrame())
        assert len(fig.data) == 0
        assert fig.layout.annotations[0].text

    def test_histogram_excludes_the_zero_entry_day_and_marks_the_mean(self):
        from src.render.hedge_figures import build_hedge_histogram_figure
        fig = build_hedge_histogram_figure(self._port())
        bars = [t for t in fig.data if t.type == "histogram"]
        assert len(bars) == 1
        assert list(bars[0].x) == [1.0, -0.5, 2.0, -1.0]
        assert any("mean" in (a.text or "").lower() for a in fig.layout.annotations)

    def test_histogram_is_empty_state_with_one_day(self):
        from src.render.hedge_figures import build_hedge_histogram_figure
        import datetime as dt
        one = pd.DataFrame({"date": [dt.date(2026, 6, 1)], "pnl_day": [0.0],
                            "pnl_cum": [0.0], "n_open": [1]})
        fig = build_hedge_histogram_figure(one)
        assert len(fig.data) == 0
```

- [ ] **Step 5: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_render.py -q -k Hedge`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.render.hedge_figures'`

- [ ] **Step 6: Implement the two figures**

Create `src/render/hedge_figures.py`:

```python
"""P8 figures: cumulative P&L, the P&L-vs-edge scatter, the daily P&L histogram.

Kept out of figures.py so neither file has to hold every panel on the page.
All money is per share: the simulation sells ONE option on ONE share.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.render.base import LAYOUT, empty_figure

_PNL_AXIS = "P&L ($ per share)"


def build_hedge_pnl_figure(port: pd.DataFrame, trades: pd.DataFrame,
                           daily: pd.DataFrame) -> go.Figure:
    """Portfolio cumulative P&L, with each trade's own running P&L behind it."""
    if port.empty:
        return empty_figure("Cumulative P&L — all simulated trades",
                            "No simulated trade yet — the first opens on the "
                            "first stored session of a month.")
    fig = go.Figure()
    if not daily.empty:
        for entry, g in daily.groupby("entry_date", sort=True):
            g = g.sort_values("date")
            fig.add_trace(go.Scatter(
                x=g["date"], y=g["pv"], mode="lines", name=f"trade {entry.isoformat()}",
                line=dict(width=1, color="#b0b7c3"), legendgroup="trades",
                showlegend=False, hovertemplate="%{x}<br>trade P&L $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=port["date"], y=port["pnl_cum"], mode="lines", name="Cumulative (all trades)",
        line=dict(width=2.5, color="#2b6cb0"),
        hovertemplate="%{x}<br>cumulative $%{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(width=1, color="#999", dash="dot"))
    fig.update_layout(title="Cumulative P&L — all simulated trades", **LAYOUT)
    fig.update_yaxes(title_text=_PNL_AXIS)
    fig.update_xaxes(title_text="Session")
    return fig


def build_hedge_histogram_figure(port: pd.DataFrame) -> go.Figure:
    """Distribution of daily hedged P&L.

    The archive's FIRST session has no prior mark to difference against, so it
    is dropped. Later trades' own opening days are not dropped: they are folded
    into a portfolio total that already carries the older trades' moves.
    """
    daily_pnl = port["pnl_day"].iloc[1:] if len(port) > 1 else pd.Series(dtype=float)
    daily_pnl = daily_pnl.dropna()
    if len(daily_pnl) < 1:
        return empty_figure("Daily hedged P&L — distribution",
                            "Not enough sessions yet to show a distribution.")
    fig = go.Figure(go.Histogram(x=daily_pnl.to_numpy(dtype=float), nbinsx=25,
                                 marker=dict(color="#2b6cb0", line=dict(width=0.5, color="white")),
                                 name="Daily P&L",
                                 hovertemplate="$%{x}<br>%{y} sessions<extra></extra>"))
    mean = float(daily_pnl.mean())
    fig.add_vline(x=0, line=dict(width=1, color="#999", dash="dot"))
    fig.add_vline(x=mean, line=dict(width=2, color="#c05621"),
                  annotation_text=f"mean ${mean:.2f}", annotation_position="top right")
    fig.update_layout(title="Daily hedged P&L — distribution", showlegend=False, **LAYOUT)
    fig.update_xaxes(title_text=_PNL_AXIS)
    fig.update_yaxes(title_text="Sessions")
    return fig
```

- [ ] **Step 7: Verify they pass**

Run: `.venv/bin/python -m pytest tests/test_render.py -q`
Expected: PASS

- [ ] **Step 8: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/render/hedge_figures.py tests/test_render.py
git commit -m "feat: P8 cumulative-P&L and daily-P&L-distribution figures"
```

---

### Task 6: The money chart — P&L vs edge scatter, and the stat line

**Files:**
- Modify: `src/render/hedge_figures.py`
- Modify: `src/render/stats.py`
- Test: `tests/test_render.py` (append)

**Interfaces:**
- Consumes: `trades` frame (`TRADE_COLUMNS`), `fit` dict (`n, slope, intercept, r2`), `summary` dict from `hedge_summary`.
- Produces:
  - `build_hedge_scatter_figure(trades: pd.DataFrame, fit: dict, next_settlement=None) -> go.Figure`
  - `hedge_summary_html(summary: dict) -> str`

The x-axis is **vol points**: `edge * 100`, matching P6's and P9's convention (the Phase 5 review caught a 100× scale error here — do not repeat it). The y-axis is dollars per share.

`hedge_summary_html` must carry the SPEC §3 P8 "Note" label — the panel is a *simulation for learning*, not a real position and not advice — and must not describe a slope it does not have. Branch on `n_settled` exactly like `iv_rv_summary_html` does on `evaluable_days`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render.py`:

```python
class TestHedgeScatterAndStat:
    def _settled(self, n=4):
        import datetime as dt
        return pd.DataFrame({
            "entry_date": [dt.date(2026, m, 1) for m in range(3, 3 + n)],
            "expiry": [dt.date(2026, m + 1, 17) for m in range(3, 3 + n)],
            "strike": [100.0] * n, "dte_at_entry": [30] * n,
            "entry_iv": [0.20] * n, "entry_straddle": [6.0] * n,
            "exit_date": [dt.date(2026, m + 1, 17) for m in range(3, 3 + n)],
            "exit_value": [4.0] * n, "pnl": [1.0, 2.0, 3.0, 4.0][:n],
            "n_days": [30] * n, "n_market_marks": [27] * n, "n_model_marks": [3] * n,
            "market_mark_share": [0.9] * n,
            "lifetime_rv": [0.19, 0.18, 0.17, 0.16][:n],
            "edge": [0.01, 0.02, 0.03, 0.04][:n], "status": ["settled"] * n,
        })

    def test_scatter_is_in_vol_points_and_draws_the_fit(self):
        from src.analytics.hedge_sim import fit_pnl_vs_edge
        from src.render.hedge_figures import build_hedge_scatter_figure
        trades = self._settled()
        fit = fit_pnl_vs_edge(trades)
        fig = build_hedge_scatter_figure(trades, fit)
        pts = [t for t in fig.data if t.mode and "markers" in t.mode]
        assert len(pts) == 1
        assert list(pts[0].x) == pytest.approx([1.0, 2.0, 3.0, 4.0])   # vol POINTS
        assert list(pts[0].y) == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert any(t.mode == "lines" for t in fig.data)     # the regression line
        text = " ".join(a.text for a in fig.layout.annotations)
        assert "slope" in text.lower() and "vol point" in text.lower()
        assert "vol point" in fig.layout.xaxis.title.text.lower()

    def test_scatter_omits_open_trades(self):
        import datetime as dt
        from src.render.hedge_figures import build_hedge_scatter_figure
        trades = self._settled()
        trades.loc[3, "status"] = "open"
        fig = build_hedge_scatter_figure(trades, {"n": 3, "slope": 1.0,
                                                  "intercept": 0.0, "r2": 1.0})
        pts = [t for t in fig.data if t.mode and "markers" in t.mode][0]
        assert len(pts.x) == 3

    def test_scatter_empty_state_names_the_next_settlement(self):
        import datetime as dt
        from src.render.hedge_figures import build_hedge_scatter_figure
        fig = build_hedge_scatter_figure(
            pd.DataFrame(columns=["edge", "pnl", "status", "entry_date"]),
            {"n": 0, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")},
            next_settlement=dt.date(2026, 9, 18))
        assert len(fig.data) == 0
        assert "2026-09-18" in fig.layout.annotations[0].text

    def test_scatter_with_one_point_draws_no_line(self):
        from src.render.hedge_figures import build_hedge_scatter_figure
        trades = self._settled(n=1)
        fig = build_hedge_scatter_figure(trades, {"n": 1, "slope": float("nan"),
                                                  "intercept": float("nan"),
                                                  "r2": float("nan")})
        assert not any(t.mode == "lines" for t in fig.data)


class TestHedgeSummaryHtml:
    def _summary(self, **over):
        import datetime as dt
        base = {"n_trades": 1, "n_settled": 0, "n_open": 1, "n_sparse": 0,
                "first_entry": dt.date(2026, 8, 14), "last_mark": dt.date(2026, 8, 28),
                "cum_pnl": 1.25, "n_days": 11, "mean_daily_pnl": 0.12,
                "sd_daily_pnl": 0.4, "market_mark_share": 1.0,
                "next_settlement": dt.date(2026, 9, 18),
                "slope": float("nan"), "r2": float("nan")}
        return {**base, **over}

    def test_labels_the_panel_as_a_simulation(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary())
        assert "simulation" in html.lower()
        assert "class='stat'" in html

    def test_no_settled_trade_says_so_and_names_the_date(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary())
        assert "2026-09-18" in html
        assert "slope" not in html.lower()

    def test_one_settled_trade_is_not_called_a_relationship(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_settled=1, n_open=0, slope=3.1, r2=1.0))
        assert "one" in html.lower() or "1 " in html
        assert "slope" not in html.lower()

    def test_enough_trades_reports_the_slope(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=6, n_settled=6, n_open=0,
                                                slope=2.4, r2=0.71))
        assert "2.4" in html and "vol point" in html.lower()
        assert "0.71" in html or "71" in html

    def test_no_trades_at_all(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=0, n_open=0, n_days=0,
                                                first_entry=None, next_settlement=None,
                                                cum_pnl=float("nan")))
        assert "no simulated trade" in html.lower()

    def test_model_marked_share_is_disclosed_when_material(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(market_mark_share=0.82))
        assert "18%" in html or "82%" in html
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_render.py -q -k "HedgeScatter or HedgeSummaryHtml"`
Expected: FAIL — `ImportError: cannot import name 'build_hedge_scatter_figure'`

- [ ] **Step 3: Implement the scatter**

Append to `src/render/hedge_figures.py`:

```python
def build_hedge_scatter_figure(trades: pd.DataFrame, fit: dict,
                               next_settlement=None) -> go.Figure:
    """The money chart: round-trip P&L against the edge the trade was sold at.

    x is in VOL POINTS (edge * 100) to match P6 and P9; y is dollars per share.
    Only settled trades appear -- an open trade has no round-trip P&L and no
    lifetime realized vol yet.
    """
    title = "Per-trade P&L vs (entry IV − realized vol)"
    settled = (trades[(trades["status"] == "settled") & trades["edge"].notna()]
               if not trades.empty else trades)
    if settled.empty:
        when = (f" The first settles {next_settlement.isoformat()}."
                if next_settlement is not None else "")
        return empty_figure(title, "No simulated trade has reached expiry yet." + when)

    x = settled["edge"].to_numpy(dtype=float) * 100.0
    y = settled["pnl"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text", name="Settled trades",
        text=[d.strftime("%b %Y") for d in settled["entry_date"]],
        textposition="top center", textfont=dict(size=10, color="#666"),
        marker=dict(size=11, color="#2b6cb0", line=dict(width=1, color="white")),
        customdata=np.column_stack([settled["entry_iv"] * 100, settled["lifetime_rv"] * 100]),
        hovertemplate=("%{text}<br>entry IV %{customdata[0]:.1f} vs realized "
                       "%{customdata[1]:.1f} vol pts<br>P&L $%{y:.2f}<extra></extra>")))
    if np.isfinite(fit.get("slope", np.nan)):
        xs = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(
            x=xs, y=fit["slope"] * xs + fit["intercept"], mode="lines",
            name="Least-squares fit", line=dict(width=2, color="#c05621", dash="dash"),
            hoverinfo="skip"))
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left", yanchor="top",
            showarrow=False, align="left", font=dict(size=12, color="#444"),
            text=(f"slope ${fit['slope']:.2f} per vol point · R² {fit['r2']:.2f} "
                  f"· {fit['n']} trades"))
    fig.add_hline(y=0, line=dict(width=1, color="#999", dash="dot"))
    fig.add_vline(x=0, line=dict(width=1, color="#999", dash="dot"))
    fig.update_layout(title=title, **LAYOUT)
    fig.update_xaxes(title_text="Entry IV − realized vol over the trade's life (vol points)")
    fig.update_yaxes(title_text=_PNL_AXIS)
    return fig
```

- [ ] **Step 4: Implement `hedge_summary_html`**

Append to `src/render/stats.py` (keep the file's existing `<p class='stat'>` convention):

```python
_SIM_LABEL = ("Simulation for learning — a hypothetical short straddle sold at the "
              "mid, delta-hedged once a day on one share. Not a real position, not advice.")


def hedge_summary_html(summary: dict) -> str:
    """The P8 stat line. Never describes a slope the sample cannot support."""
    if summary["n_trades"] == 0:
        return (f"<p class='stat'>No simulated trade yet — the first opens on the first "
                f"stored session of a month.</p><p class='caption'>{_SIM_LABEL}</p>")

    since = summary["first_entry"].isoformat()
    money = f"${summary['cum_pnl']:,.2f}" if np.isfinite(summary["cum_pnl"]) else "n/a"
    head = (f"{summary['n_trades']} simulated trade"
            f"{'s' if summary['n_trades'] != 1 else ''} since {since}: "
            f"cumulative P&L {money} over {summary['n_days']} sessions")

    n = summary["n_settled"]
    if n == 0:
        when = summary["next_settlement"]
        tail = ("; none has reached expiry yet, so the P&L-vs-edge scatter is still empty"
                + (f" — the first settles {when.isoformat()}" if when is not None else ""))
    elif n == 1:
        tail = "; exactly one trade has settled — one round trip, not a relationship"
    elif n < 5:
        tail = (f"; {n} trades have settled — too few to read a line through, "
                "so the fit is drawn but not claimed")
    else:
        tail = (f"; across {n} settled trades the fit is ${summary['slope']:.2f} of P&L "
                f"per vol point of edge (R² {summary['r2']:.2f})")

    share = summary.get("market_mark_share", float("nan"))
    if np.isfinite(share) and share < 0.999:
        tail += (f". {1 - share:.0%} of daily marks are our own model, not a quote — "
                 "an expiry drops out of the stored chain in its final week")
    if summary["n_sparse"]:
        tail += (f". {summary['n_sparse']} trade(s) are excluded from the scatter for "
                 "having too few market marks")
    return f"<p class='stat'>{head}{tail}.</p><p class='caption'>{_SIM_LABEL}</p>"
```

Add `import numpy as np` to `src/render/stats.py` if it is not already imported.

- [ ] **Step 5: Verify they pass**

Run: `.venv/bin/python -m pytest tests/test_render.py -q`
Expected: PASS

- [ ] **Step 6: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/render/hedge_figures.py src/render/stats.py tests/test_render.py
git commit -m "feat: P8 P&L-vs-edge scatter with least-squares fit and honest stat line"
```

---

### Task 7: Wire P8 into the page and the daily pipeline

**Files:**
- Modify: `src/render/page.py`
- Modify: `src/run_daily.py`
- Test: `tests/test_render.py` (append), `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: `replay_hedge_sim(root, r, q, cfg, extra_chains=None, underlying=None) -> dict`; `build_hedge_pnl_figure(port, trades, daily)`, `build_hedge_scatter_figure(trades, fit, next_settlement=None)`, `build_hedge_histogram_figure(port)`, `hedge_summary_html(summary)`.
- Produces: panel ids `"P8a"`, `"P8b"`, `"P8c"` in `figures`; the `"P8a"` key in `extras`; nine new `status.json` keys.

- [ ] **Step 1: Write the failing page tests**

Append to `tests/test_render.py`:

```python
class TestP8Page:
    def test_q4_lists_the_three_p8_panels(self):
        from src.render.page import QUESTIONS
        q4 = [q for q in QUESTIONS if q[0] == "Q4"][0]
        assert q4[2] == ["P8a", "P8b", "P8c"]

    def test_every_p8_panel_has_a_title_and_a_caption(self):
        from src.render.page import CAPTIONS, _PANEL_TITLES
        for pid in ("P8a", "P8b", "P8c"):
            assert _PANEL_TITLES[pid]
            assert CAPTIONS[pid]

    def test_scatter_caption_states_the_gamma_pnl_result_and_the_hedging_error(self):
        from src.render.page import CAPTIONS
        cap = CAPTIONS["P8b"].lower()
        assert "gamma" in cap or "Γ" in CAPTIONS["P8b"]
        assert "discrete" in cap and "hedg" in cap

    def test_page_renders_the_p8_panels_and_the_simulation_label(self):
        from src.render.page import _PANEL_TITLES, render_page
        from src.render.base import empty_figure
        figures = {pid: empty_figure(pid, "x") for pid in ("P8a", "P8b", "P8c")}
        status = {"spot": 100.0, "snapshot_date": "2026-08-28", "source": "yfinance",
                  "iv_convergence": 0.99, "last_success_utc": "2026-08-29T00:00:00+00:00",
                  "rows_stored": 10}
        html = render_page(figures, status, extras={"P8a": "<p class='stat'>SIMLABEL</p>"})
        assert "SIMLABEL" in html
        assert "Phase 6" not in html          # the Q4 placeholder is gone
        for pid in ("P8a", "P8b", "P8c"):
            assert _PANEL_TITLES[pid] in html
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_render.py -q -k P8Page`
Expected: FAIL — `KeyError: 'P8a'` / the Q4 assertion fails

- [ ] **Step 3: Edit `src/render/page.py`**

Add three entries to `_PANEL_TITLES`, in this order after `"P7"`:

```python
    "P8a": "Cumulative P&L — simulated delta-hedged straddles",
    "P8b": "Per-trade P&L vs (entry IV − realized vol)",
    "P8c": "Daily hedged P&L — distribution",
```

Replace the Q4 tuple in `QUESTIONS` with:

```python
    ("Q4", "The model claims you can hedge an option perfectly. Can you?",
     ["P8a", "P8b", "P8c"], ""),
```

Add three captions to `CAPTIONS`:

```python
    "P8a": (
        "On the first stored session of each month this simulation sells one "
        "at-the-money straddle about 30 days out and delta-hedges it once a day "
        "with this project's own Black-Scholes deltas, at each leg's own market "
        "implied vol. Cash earns the risk-free rate and the hedge earns the "
        "dividend yield, so the line below is the whole position: option marks, "
        "share hedge, carry. The thin grey lines are the individual trades; the "
        "blue line is their sum."
    ),
    "P8b": (
        "The money chart. Each dot is one round trip: what the straddle was sold "
        "at, minus what the underlying actually went on to realize, against the "
        "dollars the hedged position made or lost. Black-Scholes predicts the "
        "relationship — a delta-hedged short option earns about "
        "½ Γ S² (σ²_implied − σ²_realized) dt per day, so selling vol above what "
        "arrives should pay and selling it below should not. The slope is that "
        "prediction measured on real quotes. The scatter around the line is not "
        "noise to be explained away: it is discrete-hedging error, the price of "
        "re-hedging once a day instead of continuously, plus the vol "
        "mark-to-market the model assumes away."
    ),
    "P8c": (
        "The same P&L, one session at a time. Black-Scholes says a perfectly "
        "hedged option position has no risk left; a histogram with visible width "
        "is the direct measurement of how wrong that is at a daily hedging "
        "frequency. The spread is gamma and vol moves; the centre is the premium "
        "being earned or bled."
    ),
```

- [ ] **Step 4: Write the failing pipeline test**

Append to `tests/test_pipeline.py` (follow the file's existing fake-provider fixtures — reuse them, do not invent new ones):

```python
class TestP8Status:
    def test_status_carries_the_hedge_sim_keys(self, tmp_path):
        # READ tests/test_pipeline.py FIRST. It already has an end-to-end test that
        # builds fake providers and calls `run(...)`, returning a status dict. Reuse
        # that machinery verbatim -- fixture, fakes, helper, whatever it is -- and
        # bind its status dict to `status` here. Do NOT invent a second pipeline
        # fixture, and do NOT invent a helper named `_run_pipeline`.
        status = ...                              # <- the existing end-to-end run
        for key in ("hedge_trades", "hedge_trades_settled", "hedge_trades_open",
                    "hedge_trades_sparse", "hedge_cum_pnl", "hedge_sessions",
                    "hedge_market_mark_share", "hedge_slope_per_vol_point", "hedge_r2"):
            assert key in status
        assert status["panels_rendered"] == sorted(
            ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8a", "P8b", "P8c", "P9"])
```

If `tests/test_pipeline.py` has no such helper, adapt the assertion into the existing end-to-end test that already inspects `status` — do not duplicate a whole pipeline fixture.

- [ ] **Step 5: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: FAIL — missing keys / `panels_rendered` mismatch

- [ ] **Step 6: Edit `src/run_daily.py`**

Add imports:

```python
from src.data.hedge_replay import replay_hedge_sim
from src.render.hedge_figures import (
    build_hedge_histogram_figure, build_hedge_pnl_figure, build_hedge_scatter_figure,
)
from src.render.stats import hedge_summary_html, parity_summary_html
```

In the compute stage, immediately after the P9 block and before the annotations block, add:

```python
    # ---- P8 hedge simulation: a full stateless replay of the stored archive.
    # Today's chain is not on disk yet (SPEC 2.1: compute before write), so it is
    # handed in directly -- otherwise the current session would be invisible to
    # the sim until tomorrow.
    hedge = replay_hedge_sim(root, risk_free_rate, dividend_yield, cfg,
                             extra_chains={session_date: chain_iv},
                             underlying=sorted_underlying)
    hsum = hedge["summary"]
```

Add to the `figures` dict:

```python
        "P8a": build_hedge_pnl_figure(hedge["portfolio"], hedge["trades"], hedge["daily"]),
        "P8b": build_hedge_scatter_figure(hedge["trades"], hedge["fit"],
                                          next_settlement=hsum["next_settlement"]),
        "P8c": build_hedge_histogram_figure(hedge["portfolio"]),
```

Add to `extras`:

```python
              "P8a": hedge_summary_html(hsum),
```

Add to `status`, immediately after the `implied_carry_statistic` entry:

```python
        "hedge_trades": hsum["n_trades"],
        "hedge_trades_settled": hsum["n_settled"],
        "hedge_trades_open": hsum["n_open"],
        "hedge_trades_sparse": hsum["n_sparse"],
        "hedge_sessions": hsum["n_days"],
        "hedge_cum_pnl": _round_or_none(hsum["cum_pnl"], 4),
        "hedge_market_mark_share": _round_or_none(hsum["market_mark_share"], 4),
        "hedge_slope_per_vol_point": _round_or_none(hsum["slope"], 4),
        "hedge_r2": _round_or_none(hsum["r2"], 4),
        "hedge_next_settlement": (hsum["next_settlement"].isoformat()
                                  if hsum["next_settlement"] is not None else None),
```

`_round_or_none` already returns `None` for non-finite input — check its signature before use; it is `_round_or_none(value, digits=6)`.

- [ ] **Step 7: Verify**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/render/page.py src/run_daily.py tests/test_render.py tests/test_pipeline.py
git commit -m "feat: publish P8 under Q4 and report the hedge sim in status.json"
```

---

### Task 8: Real run, verification, and the Phase 6 learning-log entry

**Files:**
- Modify: `LEARNING_LOG.md`, `README.md`
- Produces: regenerated `docs/index.html`, `docs/status.json`, and the session's chain parquet.

This task is where the phase is judged against SPEC §5's exit criterion for Phase 6: **"First simulated trade running mark-to-market daily; scatter renders (sparse is fine)."**

- [ ] **Step 1: Inspect the simulation offline, before any network run**

Write a throwaway script under `$TMPDIR` (do **not** commit it) that calls `replay_hedge_sim` against the repo's real archive and prints the trades frame, the summary dict and the fit:

```python
import json, sys, yaml
from pathlib import Path
root = Path(".").resolve()
sys.path.insert(0, str(root))
from src.data.hedge_replay import replay_hedge_sim
cfg = yaml.safe_load((root / "config.yaml").read_text())
status = json.loads((root / "docs" / "status.json").read_text())
out = replay_hedge_sim(root, status["risk_free_rate"], status["dividend_yield"], cfg)
print(out["trades"].to_string(index=False))
print(out["fit"]); print(out["summary"])
print(out["portfolio"].to_string(index=False))
```

Run it and read the output carefully. Record the real numbers — you need them for the learning log. Sanity-check at least: every trade's `pnl` equals its daily frame's last `pv`; `market_mark_share` is plausible; no trade shows an absurd P&L (a straddle on a ~$770 underlying is worth roughly $20–40, so a per-trade P&L outside ±$50 means something is wrong — investigate before proceeding).

- [ ] **Step 2: Run the pipeline for real**

Network is required, and the sandbox blocks it — run with the sandbox disabled:

`.venv/bin/python -m src.run_daily`

Expected: exits 0, prints the status dict, rewrites `docs/index.html` and `docs/status.json`.

- [ ] **Step 3: Verify the rendered page**

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
html = Path("docs/index.html").read_text()
assert "<script src=" not in html, "page must stay self-contained"
for needle in ("Cumulative P&L", "Per-trade P&L vs", "Daily hedged P&L",
               "Simulation for learning"):
    assert needle in html, needle
print(f"ok — {len(html)/1e6:.2f} MB")
PY
```

Also confirm `docs/status.json` carries the ten `hedge_*` keys with sensible values.

- [ ] **Step 4: Write the Phase 6 learning-log entry**

Append a Phase 6 section to `LEARNING_LOG.md` in the voice and depth of the existing Phase 4 and Phase 5 entries — a teaching document, with **real numbers from Step 1 and Step 2**, not placeholders. It must cover:

1. **The hedging argument, derived.** Why a delta-hedged short option's P&L over a day is approximately `½ Γ S² (σ²_implied − σ²_realized) dt`: the delta term is hedged away by construction, so what is left is the gamma term against the actual squared move, less the theta the model charges for it. Say plainly which term the panel measures and which it cannot.
2. **The P&L identity, and why it is a test rather than a comment.** `pv_t − pv_{t−1} == interest + dividend + hedge + option − cost` holds exactly; state what a failure of that assertion would have meant.
3. **What the archive actually supports.** How many trades, how many settled, how many open, how many sparse; the market-mark share and exactly why the final week of every trade is model-marked (`chain_filter.dte_min = 7`). Be explicit that a scatter with fewer than five settled trades is not a result, using the same honesty the P5 entry uses about "100% of 1".
4. **The entry-rule substitution.** SPEC says the first *trading* day of each month; the archive gave us the first *stored* session. Name the dates it actually picked and what the substitution costs.
5. **Anything that surprised you** in the real numbers — a trade whose P&L is dominated by one day, a hedge that barely moved, a settlement that landed far from the strike.

- [ ] **Step 5: Tick the phase box in `README.md`**

Change the Phase 6 line in the phase checklist to `- [x] Phase 6 — P8 delta-hedged P&L simulation` (match the file's existing wording for the phase, and follow how Phases 2–5 are written).

- [ ] **Step 6: Full suite one more time**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add LEARNING_LOG.md README.md docs/index.html docs/status.json data/
git commit -m "feat: first real P8 run — hedge simulation live on the dashboard"
```

---

## Self-review

**Spec coverage (SPEC §3 P8, clause by clause):**

| Spec clause | Task |
|---|---|
| "first trading day each month, sell one ~30-DTE ATM straddle at mid; record entry IV" | 1 (`entry_sessions`, `select_straddle`; `price_used` is mid for live rows, close for backfilled ones per SPEC §2.2) |
| "re-hedge to delta-neutral using **our own** BS deltas at current market IV" | 2 (`hedge_shares`, each leg at its own market IV) |
| "accrue hedge position P&L from underlying moves" | 2 (`pnl_hedge`) |
| "mark the short straddle to market (mid)" | 2 (`mark_source == "market"`; model marks counted and disclosed) |
| "Cash earns r" | 2 (`pnl_interest`; dividends on the share hedge accrue at q for internal consistency with the deltas) |
| "No transaction costs in v1 (v2 toggle: X bps per hedge trade)" | 2 (`transaction_cost_bps`, default 0, tested at 50) |
| "At expiry: settle, record round-trip P&L and entry IV vs realized vol over its life" | 2 (settlement at intrinsic; `lifetime_rv`, `edge`) |
| Output (a) cumulative P&L across all monthly trades | 5 (`build_hedge_pnl_figure`) |
| Output (b) scatter of per-trade P&L vs (entry IV − lifetime RV), with the regression line | 6 (`build_hedge_scatter_figure`, `fit_pnl_vs_edge`) |
| Output (c) daily P&L distribution histogram | 5 (`build_hedge_histogram_figure`) |
| Accept: caption states the gamma-P&L result and flags discrete-hedging error | 7 (`CAPTIONS["P8b"]`, asserted by a test) |
| Design: stateless, replayed from `data/chains/` on every run, no sim state | 4 (`replay_hedge_sim`) |
| Note: labeled a simulation for learning, not advice | 6 (`_SIM_LABEL`), 7 (rendered above P8a) |
| §5 Phase 6 exit criterion | 8 |

**Accept clause deliberately not claimed:** "scatter shows clear positive relationship between P&L and (IV − RV)". With the archive as it stands the scatter has too few settled trades to show any relationship, and the plan's stat line and captions say so rather than dressing up a small sample. This is the SPEC's own anticipated case ("sparse is fine", §5; §8's stateless-replay note). The panel heals itself as chain history accumulates — no code change needed.

**Placeholder scan:** every code step carries the actual code; every test step carries the actual assertions; no "TBD", no "handle edge cases", no "similar to Task N".

**Type consistency:** `select_straddle` returns the dict consumed by `simulate_trade` (keys `expiry, dte, strike, call_price, put_price, straddle, call_iv, put_iv, iv`). `simulate` returns `(trades, daily)`; `portfolio_daily` consumes `daily` and produces `["date","pnl_day","pnl_cum","n_open"]`, which `build_hedge_pnl_figure` and `build_hedge_histogram_figure` consume. `fit_pnl_vs_edge` returns `{n, slope, intercept, r2}`, consumed by `build_hedge_scatter_figure` and `hedge_summary`. `hedge_summary` returns the keys `hedge_summary_html` and `run_daily`'s status block read. `replay_hedge_sim` returns exactly `{trades, daily, portfolio, fit, summary}`, matching Task 7's use.
