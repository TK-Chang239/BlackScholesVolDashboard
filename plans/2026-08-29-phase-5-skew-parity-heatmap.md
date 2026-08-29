# vol-lens Phase 5 — Skew Series, Parity Checker & Model-vs-Market Heatmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the three Phase 5 panels — P6 25-delta skew time series (appended to `daily_metrics`, with a manual annotations file), P7 put-call parity checker with tradeable-violation count and ATM implied carry, P9 model-vs-market price heatmap — so Q2 gains its price-space evidence and Q5 gets its model-free argument.

**Architecture:** Three new pure analytics modules over the existing `chain_iv` frame (`skew`, `parity`, `model_vs_market`), plus a tiny `annotations` loader. `daily_metrics` gains two skew columns (schema forward-compat via the existing `reindex` read path; history filled by the existing stateless rebuild). Render gains three figure builders sharing one "muted within bid–ask spread" heatmap convention (faint full layer + opaque violation layer), and P9 carries a Plotly `updatemenus` toggle between %-of-price and vol-point deviations. `run_daily` computes everything before the unchanged write order (chain → daily_metrics → index.html → status.json).

**Tech Stack:** NumPy/pandas · `src.models.black_scholes` (`bs_price`, `greeks`) · plotly (`go.Heatmap`, `make_subplots`, `updatemenus`) · PyYAML · pytest (offline, warnings-as-errors) · `.venv/bin/python`, `.venv/bin/pytest`

**Spec:** `SPEC.md` — §3 P6/P7/P9 (binding), §2.2 (`daily_metrics` one row per day incl. skew metric; every stored row records `source`), §2.4 (pure analytics), §2.5 (page/captions/mobile), §2.1 failure policy, §5 Phase 5 exit criterion: **"Daily metrics appending correctly; tradeable-violation count ≈ 0."**

## Global Constraints

- No pricing libraries. Deltas for the skew interpolation come from `src.models.black_scholes.greeks` at each row's **own** market IV; model prices for P9 from `bs_price`. Vendor IV is never read.
- `T = dte / 365.0` everywhere. Price used per row is the existing `price_used` column from `compute_chain_iv` (mid for live rows, close otherwise).
- **P6 (SPEC §3 P6):** skew = 25-delta **put** IV − 25-delta **call** IV at the expiry nearest `cfg["target_dte"]["atm_panel"]` (=30), interpolated **in delta space** (puts on the OTM side `K < spot` at δ = −0.25; calls `K ≥ spot` at δ = +0.25); NaN when either side is not bracketed. Stored as a fraction (e.g. 0.031); the render layer shows vol points (×100). Annotations come from `data/annotations.yaml` (`annotations: [{date, note}, …]`); a missing/empty file means no annotations, never an error.
- **P7 (SPEC §3 P7):** for each (expiry, strike) with both kinds priced: `lhs = C − P`, `rhs = S·e^{−qT} − K·e^{−rT}`, `deviation = lhs − rhs` (dollars). `spread = (call_ask − call_bid) + (put_ask − put_bid)`, NaN for close-based rows. **Tradeable violation ⇔ |deviation| > spread AND spread is finite.** Close-based sessions therefore report zero tradeable violations *and* say so explicitly on the page ("spreads unknown"). Implied carry (stretch, included): per expiry, `C − P` interpolated linearly in strike to `K = spot` (exact — `C − P` is linear in K), `F = spot + (C − P)_ATM · e^{rT}`, `implied_carry = ln(F / spot) / T`, compared with our `r − q`.
- **P9 (SPEC §3 P9):** flat vol = today's `atm_iv_30d` (same fallback chain as P1: chain median IV, else NaN → empty figure). `deviation_pct = (market − model) / market` (positive = market richer than the flat-vol model); `deviation_vol = iv − flat_vol`. `within_spread ⇔ |market − model| ≤ (ask − bid) / 2` on quoted rows; unquoted rows are never muted.
- **Shared muting convention (P7 and P9):** two `go.Heatmap` layers per panel — every cell at `opacity=0.3` with no colorbar, then only the non-muted cells at full opacity with the colorbar. Diverging `RdBu` scale centred at zero; `hoverongaps=False`.
- **daily_metrics schema:** `IV_COLUMNS` gains `["skew_25d", "skew_25d_dte"]` (after `iv_convergence`). `session_metrics_row` gains an optional `skew: dict | None` argument; old parquet files read back with NaN skew columns and are repopulated by `scripts/rebuild_daily_metrics.py`.
- Section placement (SPEC §2.5): **Q2 → P2, P3, P6, P9** (in that order); **Q5 → P7**; Q4 keeps its "arrives in Phase 6" line. Captions in Task 6 are verbatim. P7's caption must attribute deep-ITM put gaps to American early exercise, not arbitrage.
- Failure policy unchanged: compute everything, then chain → daily_metrics → index.html → status.json, each atomic.
- One self-contained page < 5 MB; no external requests; mobile-readable.
- pytest offline, warnings-as-errors; TDD per task; run `.venv/bin/pytest -q` before each commit; commit per task.

---

## File Structure

```
data/annotations.yaml                # Task 2 — manual P6 annotations (seed: empty list + commented example)
src/analytics/skew.py                # Task 1 — 25-delta skew at ~30 DTE (delta-space interpolation)
src/analytics/daily_metrics.py       # Task 1 — modify: skew columns, session_metrics_row(skew=)
src/data/metrics_backfill.py         # Task 1 — modify: compute skew during replay
src/analytics/annotations.py         # Task 2 — load_annotations(path)
src/analytics/parity.py              # Task 3 — parity table, summary, implied carry
src/analytics/model_vs_market.py     # Task 5 — flat-vol model vs market table
src/render/figures.py                # Tasks 2,4,5 — modify: _apply_recent_range, _muted_heatmap_layers, P6/P7/P9 builders
src/render/stats.py                  # Task 4 — parity_summary_html
src/render/page.py                   # Task 6 — modify: Q2/Q5 wiring, titles, captions
src/run_daily.py                     # Task 6 — modify: P6/P7/P9 compute + status keys
tests/test_skew.py                   # Task 1
tests/test_daily_metrics.py          # Task 1 — extend
tests/test_metrics_backfill.py       # Task 1 — extend
tests/test_annotations.py            # Task 2
tests/test_parity.py                 # Task 3
tests/test_model_vs_market.py        # Task 5
tests/test_render.py                 # Tasks 2,4,5 — extend
tests/test_pipeline.py               # Task 6 — extend
LEARNING_LOG.md                      # Task 7
```

Test convention: synthetic chains built through `bs_price` with known vols; each test file owns its fixture helper.

---

### Task 1: P6 skew analytics + `daily_metrics` skew columns + replay

**Files:**
- Create: `src/analytics/skew.py`
- Modify: `src/analytics/daily_metrics.py` (`IV_COLUMNS`, `session_metrics_row`), `src/data/metrics_backfill.py` (compute skew in the loop)
- Test: `tests/test_skew.py`; extend `tests/test_daily_metrics.py`, `tests/test_metrics_backfill.py`

**Interfaces:**
- Consumes: `compute_chain_iv` frame (`CHAIN_COLUMNS + ["price_used", "iv"]`), `greeks(S, K, T, r, sigma, q, kind)`.
- Produces: `SKEW_KEYS = ["skew_expiry", "skew_dte", "put_iv_25d", "call_iv_25d", "skew_25d"]`; `compute_skew_25d(chain_iv, r, q, cfg) -> dict` with those keys (`skew_expiry` is a `datetime.date` or `None`; the rest floats, NaN when unavailable). `IV_COLUMNS` becomes `["date","spot","source","atm_iv_30d","atm_iv_30d_dte","iv_convergence","skew_25d","skew_25d_dte"]`. `session_metrics_row(session_date, spot, source, term, iv_stats, cfg, skew=None)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_skew.py`:

```python
"""P6: 25-delta skew, interpolated in delta space from each strike's own IV."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.skew import SKEW_KEYS, compute_skew_25d
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price, greeks

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def make_chain(sigma_fn=lambda k: 0.22,
               strikes=tuple(np.arange(640.0, 901.0, 10.0)),
               expiries=((dt.date(2026, 9, 25), 28), (dt.date(2026, 11, 20), 84))):
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                px = float(bs_price(SPOT, strike, dte / 365.0, R, sigma_fn(strike), Q, kind))
                rows.append({"snapshot_date": TODAY, "spot": SPOT, "expiry": expiry, "dte": dte,
                             "strike": strike, "kind": kind, "bid": px * 0.99, "ask": px * 1.01,
                             "mid": px, "close": px, "volume": 10, "open_interest": 100.0,
                             "vendor_iv": np.nan, "source": "yfinance"})
    out, _ = compute_chain_iv(pd.DataFrame(rows, columns=CHAIN_COLUMNS), R, Q)
    return out


class TestSkew:
    def test_flat_vol_has_zero_skew_at_nearest_expiry(self):
        s = compute_skew_25d(make_chain(), R, Q, cfg())
        assert list(s) == SKEW_KEYS
        assert s["skew_expiry"] == dt.date(2026, 9, 25) and s["skew_dte"] == 28
        assert s["put_iv_25d"] == pytest.approx(0.22, abs=1e-6)
        assert s["call_iv_25d"] == pytest.approx(0.22, abs=1e-6)
        assert abs(s["skew_25d"]) < 1e-9

    def test_put_skew_is_positive_and_bracketed_by_neighbours(self):
        skew_fn = lambda k: 0.20 + 0.30 * max(0.0, (SPOT - k) / SPOT)
        s = compute_skew_25d(make_chain(sigma_fn=skew_fn), R, Q, cfg())
        assert s["skew_25d"] > 0.01
        assert s["call_iv_25d"] == pytest.approx(0.20, abs=1e-6)     # flat on the call side
        # the 25-delta put sits somewhere below spot: its IV must lie between the
        # smallest and largest OTM-put vols in the chain
        assert 0.20 < s["put_iv_25d"] < skew_fn(640.0)

    def test_interpolates_at_delta_not_strike(self):
        # Two chains with the same strikes but different vol levels have different
        # 25-delta strikes; the interpolated IV must follow the skew function at the
        # strike whose own-IV delta is -0.25, to first order.
        skew_fn = lambda k: 0.15 + 0.40 * max(0.0, (SPOT - k) / SPOT)
        chain = make_chain(sigma_fn=skew_fn)
        s = compute_skew_25d(chain, R, Q, cfg())
        g = chain[(chain["dte"] == 28) & (chain["kind"] == "put") & (chain["strike"] < SPOT)]
        deltas = greeks(SPOT, g["strike"].to_numpy(float), 28 / 365.0, R,
                        g["iv"].to_numpy(float), Q, "put")["delta"]
        k_star = float(np.interp(-0.25, np.sort(deltas), g["strike"].to_numpy(float)[np.argsort(deltas)]))
        assert s["put_iv_25d"] == pytest.approx(skew_fn(k_star), abs=0.003)

    def test_unbracketed_side_is_nan(self):
        chain = make_chain(strikes=(760.0, 770.0, 780.0))   # OTM puts: only 760 -> no bracket
        s = compute_skew_25d(chain, R, Q, cfg())
        assert np.isnan(s["put_iv_25d"]) and np.isnan(s["skew_25d"])

    def test_empty_chain(self):
        s = compute_skew_25d(make_chain().iloc[0:0], R, Q, cfg())
        assert s["skew_expiry"] is None and np.isnan(s["skew_25d"])
```

Extend `tests/test_daily_metrics.py`:

```python
class TestSkewColumns:
    def test_columns_include_skew(self):
        assert IV_COLUMNS[-2:] == ["skew_25d", "skew_25d_dte"]
        assert metric_columns(cfg())[:8] == IV_COLUMNS

    def test_row_without_skew_is_nan(self):
        row = session_metrics_row(dt.date(2026, 8, 28), 771.1, "yfinance", term([]),
                                  {"convergence": 1.0}, cfg())
        assert list(row) == IV_COLUMNS
        assert np.isnan(row["skew_25d"]) and np.isnan(row["skew_25d_dte"])

    def test_row_with_skew(self):
        skew = {"skew_expiry": dt.date(2026, 9, 25), "skew_dte": 28,
                "put_iv_25d": 0.25, "call_iv_25d": 0.20, "skew_25d": 0.05}
        row = session_metrics_row(dt.date(2026, 8, 28), 771.1, "yfinance", term([]),
                                  {"convergence": 1.0}, cfg(), skew=skew)
        assert row["skew_25d"] == pytest.approx(0.05) and row["skew_25d_dte"] == 28
```

Also update the existing `TestColumns.test_shipped_config_names` expectation: `metric_columns(cfg()) == IV_COLUMNS + ["rv_20d", "rv_60d", "fwd_rv_30d"]` still holds by construction — leave it; and `TestSessionRow.test_row_keys_and_values` asserts `list(row) == IV_COLUMNS` — still holds.

Extend `tests/test_metrics_backfill.py` (inside `TestRebuild`):

```python
    def test_rebuild_fills_skew_columns(self, tmp_path):
        s1, s2 = seed(tmp_path)
        out = rebuild_daily_metrics(tmp_path, R, Q, cfg())
        # seed() chains have 5 strikes (spot-20..spot+20) at flat vol: OTM puts are
        # spot-20 and spot-10 -> two points; skew brackets only if -0.25 lies between
        # their deltas, so assert the column exists and is finite-or-NaN, not a value
        assert "skew_25d" in out.columns and "skew_25d_dte" in out.columns
        assert out["skew_25d_dte"].isna().all() or (out["skew_25d_dte"] == 30).all()
```

Then replace the last assertion with a real one by widening `chain()`'s strikes in that test file to `(spot - 60.0, spot - 40.0, spot - 20.0, spot, spot + 20.0, spot + 40.0, spot + 60.0)` (the existing tests only assert row counts, ATM IV recovery, sources, RV, and idempotence — all unaffected by extra strikes), and assert instead:

```python
        assert out["skew_25d"].abs().max() < 1e-6          # flat vol -> zero skew on both sessions
        assert (out["skew_25d_dte"] == 30).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_skew.py tests/test_daily_metrics.py tests/test_metrics_backfill.py -q`
Expected: FAIL — `ModuleNotFoundError: src.analytics.skew`; `IV_COLUMNS[-2:]` mismatch; `session_metrics_row() got an unexpected keyword argument 'skew'`.

- [ ] **Step 3: Implement `skew.py`**

```python
"""P6: 25-delta skew at the ~30-DTE expiry (SPEC 3 P6). Pure.

Interpolate implied vol in DELTA space, not strike space: the 25-delta
put is "the put that pays off on a ~25%-probability down-move", which is
the same contract regardless of spot level or vol regime. Deltas come
from our own greeks() at each row's own market IV. skew = put − call;
Black-Scholes says it should be zero.
"""
import numpy as np
import pandas as pd

from src.models.black_scholes import greeks

SKEW_KEYS = ["skew_expiry", "skew_dte", "put_iv_25d", "call_iv_25d", "skew_25d"]
TARGET_DELTA = 0.25


def _empty() -> dict:
    out = {k: np.nan for k in SKEW_KEYS}
    out["skew_expiry"] = None
    return out


def _interp_iv_at_delta(g: pd.DataFrame, target: float) -> float:
    g = g.dropna(subset=["iv", "delta"]).sort_values("delta")
    if len(g) < 2 or not (g["delta"].iloc[0] <= target <= g["delta"].iloc[-1]):
        return float("nan")
    return float(np.interp(target, g["delta"].to_numpy(dtype=float),
                           g["iv"].to_numpy(dtype=float)))


def compute_skew_25d(chain_iv: pd.DataFrame, r: float, q: float, cfg: dict) -> dict:
    solved = chain_iv[chain_iv["iv"].notna()]
    if solved.empty:
        return _empty()
    spot = float(solved["spot"].iloc[0])
    target_dte = int(cfg["target_dte"]["atm_panel"])
    available = solved[["expiry", "dte"]].drop_duplicates().reset_index(drop=True)
    pick = available.iloc[(available["dte"] - target_dte).abs().idxmin()]
    g = solved[solved["expiry"] == pick["expiry"]].copy()
    T = float(pick["dte"]) / 365.0
    g["delta"] = np.nan
    for kind in ("call", "put"):
        m = (g["kind"] == kind).to_numpy()
        if m.any():
            g.loc[m, "delta"] = np.atleast_1d(greeks(
                S=spot, K=g.loc[m, "strike"].to_numpy(dtype=float), T=T, r=r,
                sigma=g.loc[m, "iv"].to_numpy(dtype=float), q=q, kind=kind)["delta"])
    puts = g[(g["kind"] == "put") & (g["strike"] < spot)]
    calls = g[(g["kind"] == "call") & (g["strike"] >= spot)]
    put_iv = _interp_iv_at_delta(puts, -TARGET_DELTA)
    call_iv = _interp_iv_at_delta(calls, TARGET_DELTA)
    return {"skew_expiry": pick["expiry"], "skew_dte": int(pick["dte"]),
            "put_iv_25d": put_iv, "call_iv_25d": call_iv, "skew_25d": put_iv - call_iv}
```

- [ ] **Step 4: Extend `daily_metrics.py` and `metrics_backfill.py`**

In `src/analytics/daily_metrics.py`:

```python
IV_COLUMNS = ["date", "spot", "source", "atm_iv_30d", "atm_iv_30d_dte", "iv_convergence",
              "skew_25d", "skew_25d_dte"]
```

```python
def session_metrics_row(session_date: dt.date, spot: float, source: str,
                        term: pd.DataFrame, iv_stats: dict, cfg: dict,
                        skew: dict | None = None) -> dict:
    atm_iv, eff_dte = interp_atm_iv(term, int(cfg["target_dte"]["atm_panel"]))
    skew = skew or {}
    return {
        "date": session_date, "spot": float(spot), "source": source,
        "atm_iv_30d": atm_iv, "atm_iv_30d_dte": eff_dte,
        "iv_convergence": float(iv_stats["convergence"]),
        "skew_25d": float(skew.get("skew_25d", np.nan)),
        "skew_25d_dte": float(skew.get("skew_dte", np.nan)),
    }
```

In `src/data/metrics_backfill.py`, import `from src.analytics.skew import compute_skew_25d` and inside the loop, after `term = compute_term_structure(chain_iv)`:

```python
        skew = compute_skew_25d(chain_iv, r, q, cfg)
        row = session_metrics_row(session, float(chain["spot"].iloc[0]),
                                  chain_source_label(chain), term, stats, cfg, skew=skew)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_skew.py tests/test_daily_metrics.py tests/test_metrics_backfill.py -q` — PASS. Then `.venv/bin/pytest -q` — green (pipeline tests read metrics via `reindex`, so the two new columns appear as NaN; `test_first_run_writes_one_metrics_row` asserts specific columns exist, which still holds).

- [ ] **Step 6: Commit**

```bash
git add src/analytics/skew.py src/analytics/daily_metrics.py src/data/metrics_backfill.py tests/test_skew.py tests/test_daily_metrics.py tests/test_metrics_backfill.py
git commit -m "feat: P6 25-delta skew in delta space; skew columns in daily_metrics and replay"
```

---

### Task 2: Annotations file + P6 figure

**Files:**
- Create: `data/annotations.yaml`, `src/analytics/annotations.py`
- Modify: `src/render/figures.py` (add `_apply_recent_range`, `build_skew_figure`; refactor `build_iv_rv_figure` to use `_apply_recent_range`)
- Test: `tests/test_annotations.py`; extend `tests/test_render.py`

**Interfaces:**
- `load_annotations(path: Path) -> pd.DataFrame` with columns `["date", "note"]` (`date` as `datetime.date`), empty frame (same columns) when the file is missing or `annotations` is empty/null; raises `ValueError` on an entry missing `date` or `note`.
- `_apply_recent_range(fig, dates: pd.Series, recent_days: int = 180, span_days: int = 365) -> None` — sets `xaxis.range` to the last `recent_days` when the span exceeds `span_days` (extracted verbatim from `build_iv_rv_figure`'s last three lines).
- `build_skew_figure(metrics: pd.DataFrame, annotations: pd.DataFrame) -> go.Figure` — y in vol points (`skew_25d * 100`), one `lines+markers` trace named "25-delta skew (put − call)", `customdata=skew_25d_dte`, gap-broken via `_break_gaps`, annotation arrows for annotation dates that match a plotted session exactly, `_apply_recent_range`, zero line; empty/all-NaN → `_empty_figure`.

- [ ] **Step 1: Write the failing tests**

`tests/test_annotations.py`:

```python
"""P6 annotations: a 1-line-commit YAML file of dated notes."""
import datetime as dt

import pytest

from src.analytics.annotations import load_annotations


def test_missing_file_is_empty(tmp_path):
    df = load_annotations(tmp_path / "annotations.yaml")
    assert df.empty and list(df.columns) == ["date", "note"]


def test_empty_list_is_empty(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("annotations: []\n")
    assert load_annotations(p).empty


def test_entries_parse_to_dates(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("annotations:\n  - {date: 2026-09-17, note: FOMC}\n  - date: '2026-10-02'\n    note: payrolls\n")
    df = load_annotations(p)
    assert list(df["date"]) == [dt.date(2026, 9, 17), dt.date(2026, 10, 2)]
    assert list(df["note"]) == ["FOMC", "payrolls"]


def test_bad_entry_raises(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("annotations:\n  - {date: 2026-09-17}\n")
    with pytest.raises(ValueError):
        load_annotations(p)


def test_shipped_file_loads():
    from pathlib import Path
    df = load_annotations(Path("data/annotations.yaml"))
    assert list(df.columns) == ["date", "note"]
```

Append to `tests/test_render.py`:

```python
class TestSkewFigure:
    def _metrics(self, dates, skews):
        return pd.DataFrame({"date": dates, "skew_25d": skews,
                             "skew_25d_dte": [30.0] * len(dates)})

    def test_vol_points_and_annotation_match(self):
        import datetime as dt
        from src.render.figures import build_skew_figure
        d = [dt.date(2026, 8, 24) + dt.timedelta(days=i) for i in range(4)]
        m = self._metrics(d, [0.030, 0.032, 0.045, 0.031])
        ann = pd.DataFrame({"date": [d[2], dt.date(2026, 1, 1)], "note": ["risk-off", "absent"]})
        fig = build_skew_figure(m, ann)
        trace = [t for t in fig.data if "skew" in (t.name or "").lower()][0]
        assert max(v for v in trace.y if v == v) == pytest.approx(4.5)
        texts = [a.text for a in fig.layout.annotations]
        assert "risk-off" in texts and "absent" not in texts

    def test_gap_breaks_and_recent_range(self):
        import datetime as dt
        from src.render.figures import build_skew_figure
        d = [dt.date(2024, 9, 3)] + [dt.date(2026, 8, 24) + dt.timedelta(days=i) for i in range(3)]
        fig = build_skew_figure(self._metrics(d, [0.02, 0.03, 0.03, 0.03]),
                                pd.DataFrame(columns=["date", "note"]))
        trace = [t for t in fig.data if "skew" in (t.name or "").lower()][0]
        assert any(v is None or v != v for v in trace.y)
        assert fig.layout.xaxis.range is not None

    def test_all_nan_is_empty_figure(self):
        import datetime as dt
        from src.render.figures import build_skew_figure
        m = self._metrics([dt.date(2026, 8, 24)], [float("nan")])
        fig = build_skew_figure(m, pd.DataFrame(columns=["date", "note"]))
        assert not fig.data and fig.layout.annotations
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_annotations.py tests/test_render.py -q`
Expected: FAIL — `ModuleNotFoundError: src.analytics.annotations`; `ImportError: build_skew_figure`.

- [ ] **Step 3: Create the annotations file and loader**

`data/annotations.yaml`:

```yaml
# P6 skew-series annotations (SPEC 3 P6). One entry per notable day; adding a
# note is a one-line commit. Dates are market sessions (YYYY-MM-DD).
#
#   - {date: 2026-09-17, note: FOMC decision}
#
annotations: []
```

`src/analytics/annotations.py`:

```python
"""P6 annotations: dated notes for the skew time series (SPEC 3 P6)."""
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

COLUMNS = ["date", "note"]


def load_annotations(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    doc = yaml.safe_load(path.read_text()) or {}
    entries = doc.get("annotations") or []
    rows = []
    for e in entries:
        if not isinstance(e, dict) or "date" not in e or "note" not in e:
            raise ValueError(f"annotation entry needs 'date' and 'note': {e!r}")
        d = e["date"]
        if isinstance(d, dt.datetime):
            d = d.date()
        elif not isinstance(d, dt.date):
            d = dt.date.fromisoformat(str(d))
        rows.append({"date": d, "note": str(e["note"])})
    return pd.DataFrame(rows, columns=COLUMNS)
```

- [ ] **Step 4: Figure + range helper**

In `src/render/figures.py`:

```python
def _apply_recent_range(fig: go.Figure, dates: pd.Series,
                        recent_days: int = 180, span_days: int = 365) -> None:
    """Default the x-axis to the recent window when the history is long
    (a lone depth-probe session years back would otherwise crush the daily
    series into a sliver). Double-click restores autorange."""
    if len(dates) == 0:
        return
    first, last = dates.iloc[0], dates.iloc[-1]
    if (last - first).days > span_days:
        fig.update_xaxes(range=[last - timedelta(days=recent_days), last + timedelta(days=3)])
```

Replace the last three lines of `build_iv_rv_figure` (`first, last = ...` through `fig.update_xaxes(...)`) with `_apply_recent_range(fig, series["date"])`.

```python
def build_skew_figure(metrics: pd.DataFrame, annotations: pd.DataFrame) -> go.Figure:
    title = "25-delta skew — put IV minus call IV at the ~30-DTE expiry"
    series = metrics[["date", "skew_25d", "skew_25d_dte"]].dropna(subset=["skew_25d"])
    series = series.sort_values("date").reset_index(drop=True)
    if series.empty:
        return _empty_figure(title, "No session with a bracketed 25-delta skew yet",
                             yaxis_title="Vol points")
    plotted = _break_gaps(series)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["skew_25d"] * 100.0, mode="lines+markers",
        name="25-delta skew (put − call)", line=dict(color="#8172b2"), marker=dict(size=5),
        customdata=plotted["skew_25d_dte"],
        hovertemplate="%{x}: %{y:+.1f} vol pts (%{customdata:.0f}d expiry)<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dot"))
    by_date = dict(zip(series["date"], series["skew_25d"] * 100.0))
    for _, a in annotations.iterrows():
        if a["date"] in by_date:
            fig.add_annotation(x=a["date"], y=by_date[a["date"]], text=str(a["note"]),
                               showarrow=True, arrowhead=2, ax=0, ay=-40,
                               font=dict(size=11))
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_yaxes(title_text="Vol points (put − call)", ticksuffix="")
    _apply_recent_range(fig, series["date"])
    return fig
```

If `add_hline` misbehaves under warnings-as-errors, use a `go.Scatter` dotted line at y=0 across `[plotted.date.min(), plotted.date.max()]` with `showlegend=False, hoverinfo="skip"`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_annotations.py tests/test_render.py -q` — PASS. Then `.venv/bin/pytest -q` — green (the existing P5 range tests must still pass through the helper).

- [ ] **Step 6: Commit**

```bash
git add data/annotations.yaml src/analytics/annotations.py src/render/figures.py tests/test_annotations.py tests/test_render.py
git commit -m "feat: P6 skew time-series figure with annotations file; shared recent-range helper"
```

---

### Task 3: P7 parity analytics + implied carry

**Files:**
- Create: `src/analytics/parity.py`
- Test: `tests/test_parity.py`

**Interfaces:**
- `PARITY_COLUMNS = ["expiry","dte","strike","moneyness","call_price","put_price","lhs","rhs","deviation","spread","tradeable_violation"]`
- `compute_parity(chain_iv, r, q) -> pd.DataFrame[PARITY_COLUMNS]` — one row per (expiry, strike) with both kinds priced; `tradeable_violation` is a plain `bool` column (False where `spread` is NaN).
- `parity_summary(parity) -> dict` = `{"n_pairs": int, "n_quoted": int, "n_tradeable_violations": int, "share_within_spread": float|nan, "max_abs_deviation": float|nan}` (share among quoted pairs).
- `implied_carry(parity, spot, r) -> pd.DataFrame[["expiry","dte","implied_carry"]]` — per expiry whose strikes bracket spot; rows sorted by dte.
- `carry_at_target(carry, target_dte) -> tuple[float, float]` — `(implied_carry, dte)` at the expiry nearest `target_dte`, `(nan, nan)` if empty.

- [ ] **Step 1: Write the failing tests**

`tests/test_parity.py`:

```python
"""P7: put-call parity deviations, tradeable violations, implied carry."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.parity import (
    PARITY_COLUMNS, carry_at_target, compute_parity, implied_carry, parity_summary,
)
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def make_chain(source="yfinance", spread_frac=0.01,
               strikes=(700.0, 740.0, 760.0, 780.0, 800.0, 840.0),
               expiries=((dt.date(2026, 9, 25), 28), (dt.date(2026, 11, 20), 84))):
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                px = float(bs_price(SPOT, strike, dte / 365.0, R, 0.22, Q, kind))
                live = source == "yfinance"
                rows.append({"snapshot_date": TODAY, "spot": SPOT, "expiry": expiry, "dte": dte,
                             "strike": strike, "kind": kind,
                             "bid": px * (1 - spread_frac) if live else np.nan,
                             "ask": px * (1 + spread_frac) if live else np.nan,
                             "mid": px if live else np.nan, "close": px,
                             "volume": 10, "open_interest": 100.0, "vendor_iv": np.nan,
                             "source": source})
    out, _ = compute_chain_iv(pd.DataFrame(rows, columns=CHAIN_COLUMNS), R, Q)
    return out


class TestParity:
    def test_synthetic_prices_satisfy_parity_to_machine_eps(self):
        p = compute_parity(make_chain(), R, Q)
        assert list(p.columns) == PARITY_COLUMNS
        assert len(p) == 12
        assert np.abs(p["deviation"]).max() < 1e-9
        assert p["spread"].notna().all() and not p["tradeable_violation"].any()
        s = parity_summary(p)
        assert s["n_pairs"] == 12 and s["n_quoted"] == 12
        assert s["n_tradeable_violations"] == 0 and s["share_within_spread"] == 1.0

    def test_violation_beyond_spread_is_flagged(self):
        chain = make_chain()
        m = (chain["kind"] == "put") & (chain["strike"] == 760.0) & (chain["dte"] == 28)
        chain.loc[m, ["mid", "price_used"]] = chain.loc[m, "price_used"] + 5.0   # way outside spread
        p = compute_parity(chain, R, Q)
        row = p[(p["strike"] == 760.0) & (p["dte"] == 28)].iloc[0]
        assert row["deviation"] == pytest.approx(-5.0, abs=1e-9)
        assert bool(row["tradeable_violation"]) is True
        assert parity_summary(p)["n_tradeable_violations"] == 1

    def test_deviation_inside_spread_is_not_tradeable(self):
        chain = make_chain(spread_frac=0.05)
        m = (chain["kind"] == "call") & (chain["strike"] == 780.0) & (chain["dte"] == 28)
        chain.loc[m, "price_used"] = chain.loc[m, "price_used"] + 0.10
        p = compute_parity(chain, R, Q)
        row = p[(p["strike"] == 780.0) & (p["dte"] == 28)].iloc[0]
        assert row["deviation"] == pytest.approx(0.10, abs=1e-9)
        assert bool(row["tradeable_violation"]) is False

    def test_close_based_rows_have_no_spread_and_never_trade(self):
        p = compute_parity(make_chain(source="massive-backfill"), R, Q)
        assert p["spread"].isna().all() and not p["tradeable_violation"].any()
        s = parity_summary(p)
        assert s["n_quoted"] == 0 and np.isnan(s["share_within_spread"])

    def test_unpaired_strikes_are_dropped(self):
        chain = make_chain()
        chain = chain[~((chain["kind"] == "put") & (chain["strike"] == 700.0))]
        p = compute_parity(chain, R, Q)
        assert 700.0 not in set(p["strike"])

    def test_empty(self):
        p = compute_parity(make_chain().iloc[0:0], R, Q)
        assert p.empty and list(p.columns) == PARITY_COLUMNS
        s = parity_summary(p)
        assert s["n_pairs"] == 0 and np.isnan(s["max_abs_deviation"])


class TestImpliedCarry:
    def test_recovers_r_minus_q(self):
        c = implied_carry(compute_parity(make_chain(), R, Q), SPOT, R)
        assert list(c.columns) == ["expiry", "dte", "implied_carry"]
        assert list(c["dte"]) == [28, 84]
        assert c["implied_carry"].to_numpy() == pytest.approx(R - Q, abs=1e-8)
        val, dte = carry_at_target(c, 30)
        assert val == pytest.approx(R - Q, abs=1e-8) and dte == 28

    def test_expiry_not_bracketing_spot_is_skipped(self):
        chain = make_chain(strikes=(800.0, 840.0))     # all above spot
        c = implied_carry(compute_parity(chain, R, Q), SPOT, R)
        assert c.empty
        assert np.isnan(carry_at_target(c, 30)[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_parity.py -q`
Expected: FAIL — `ModuleNotFoundError: src.analytics.parity`.

- [ ] **Step 3: Implement**

`src/analytics/parity.py`:

```python
"""P7: put-call parity checker (SPEC 3 P7). Pure, model-free.

C − P = S·e^{−qT} − K·e^{−rT} needs no volatility and no distribution:
it is enforced by arbitrage alone. `deviation` is the dollar gap; a
violation is TRADEABLE only if it exceeds the combined bid–ask spread of
the two legs, so close-based sessions (no quotes) can never flag one and
must say so. Implied carry backs out the r − q the market is using, from
the ATM-interpolated C − P (exactly linear in K, so the interpolation is
exact).
"""
import numpy as np
import pandas as pd

PARITY_COLUMNS = ["expiry", "dte", "strike", "moneyness", "call_price", "put_price",
                  "lhs", "rhs", "deviation", "spread", "tradeable_violation"]
CARRY_COLUMNS = ["expiry", "dte", "implied_carry"]


def compute_parity(chain_iv: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    if chain_iv.empty:
        return pd.DataFrame(columns=PARITY_COLUMNS)
    spot = float(chain_iv["spot"].iloc[0])
    keep = ["expiry", "dte", "strike", "price_used", "bid", "ask"]
    calls = chain_iv[chain_iv["kind"] == "call"][keep].rename(
        columns={"price_used": "call_price", "bid": "call_bid", "ask": "call_ask"})
    puts = chain_iv[chain_iv["kind"] == "put"][keep].rename(
        columns={"price_used": "put_price", "bid": "put_bid", "ask": "put_ask"})
    p = calls.merge(puts, on=["expiry", "dte", "strike"], how="inner")
    p = p.dropna(subset=["call_price", "put_price"]).copy()
    if p.empty:
        return pd.DataFrame(columns=PARITY_COLUMNS)
    T = p["dte"].to_numpy(dtype=float) / 365.0
    p["lhs"] = p["call_price"] - p["put_price"]
    p["rhs"] = spot * np.exp(-q * T) - p["strike"] * np.exp(-r * T)
    p["deviation"] = p["lhs"] - p["rhs"]
    p["spread"] = (p["call_ask"] - p["call_bid"]) + (p["put_ask"] - p["put_bid"])
    p["tradeable_violation"] = (p["deviation"].abs() > p["spread"]) & p["spread"].notna()
    p["tradeable_violation"] = p["tradeable_violation"].astype(bool)
    p["moneyness"] = p["strike"] / spot
    return (p[PARITY_COLUMNS].sort_values(["expiry", "strike"])
            .reset_index(drop=True))


def parity_summary(parity: pd.DataFrame) -> dict:
    n = int(len(parity))
    quoted = parity[parity["spread"].notna()] if n else parity
    n_quoted = int(len(quoted))
    n_viol = int(parity["tradeable_violation"].sum()) if n else 0
    return {
        "n_pairs": n,
        "n_quoted": n_quoted,
        "n_tradeable_violations": n_viol,
        "share_within_spread": (1.0 - n_viol / n_quoted) if n_quoted else np.nan,
        "max_abs_deviation": float(parity["deviation"].abs().max()) if n else np.nan,
    }


def implied_carry(parity: pd.DataFrame, spot: float, r: float) -> pd.DataFrame:
    rows = []
    for (expiry, dte), g in parity.groupby(["expiry", "dte"], sort=False):
        g = g.sort_values("strike")
        ks = g["strike"].to_numpy(dtype=float)
        if len(ks) < 2 or not (ks.min() <= spot <= ks.max()):
            continue
        T = float(dte) / 365.0
        lhs_atm = float(np.interp(spot, ks, g["lhs"].to_numpy(dtype=float)))
        forward = spot + lhs_atm * np.exp(r * T)
        rows.append({"expiry": expiry, "dte": int(dte),
                     "implied_carry": float(np.log(forward / spot) / T)})
    return (pd.DataFrame(rows, columns=CARRY_COLUMNS).sort_values("dte")
            .reset_index(drop=True))


def carry_at_target(carry: pd.DataFrame, target_dte: int) -> tuple[float, float]:
    if carry.empty:
        return (np.nan, np.nan)
    j = int((carry["dte"] - target_dte).abs().idxmin())
    return float(carry.loc[j, "implied_carry"]), float(carry.loc[j, "dte"])
```

Derivation check for the implementer: `C − P = S e^{−qT} − K e^{−rT}` ⇒ `(C − P) e^{rT} = S e^{(r−q)T} − K = F − K` ⇒ `F = K + (C − P) e^{rT}`; at `K = spot`, `F = spot + lhs_atm · e^{rT}` and `carry = ln(F/spot)/T = r − q`. Exactly what the test asserts.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_parity.py -q` — PASS (8). Then `.venv/bin/pytest -q` — green.

- [ ] **Step 5: Commit**

```bash
git add src/analytics/parity.py tests/test_parity.py
git commit -m "feat: P7 put-call parity table, tradeable-violation summary, implied carry"
```

---

### Task 4: P7 figure (muted heatmap) + stat line

**Files:**
- Create: `src/render/stats.py`
- Modify: `src/render/figures.py` (add `_expiry_label`, `_muted_heatmap_layers`, `build_parity_figure`)
- Test: extend `tests/test_render.py`

**Interfaces:**
- `_expiry_label(expiry, dte) -> str` = `f"{expiry.isoformat()} ({dte}d)"`.
- `_muted_heatmap_layers(z_all, z_hot, x, y, colorbar_title, zmin, zmax, hovertemplate, **subplot) -> list[go.Heatmap]` — layer 1: `z_all`, `opacity=0.3`, `showscale=False`; layer 2: `z_hot` (NaN where muted), full opacity, `showscale=True`; both `colorscale="RdBu", zmid=0, hoverongaps=False`.
- `build_parity_figure(parity, spot) -> go.Figure` — x = expiry labels, y = strikes, z = deviation ($); hot cells = `tradeable_violation`; a dotted horizontal line at `spot`; annotation "close-based session — spreads unknown, tradeability not assessable" when no pair is quoted; empty → `_empty_figure`.
- `parity_summary_html(summary, carry_value, carry_dte, r, q) -> str` (in `src/render/stats.py`) — one `<p class='stat'>`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render.py`:

```python
class TestParityFigure:
    def _parity(self, quoted=True):
        import datetime as dt
        e1, e2 = dt.date(2026, 9, 25), dt.date(2026, 11, 20)
        rows = []
        for expiry, dte in ((e1, 28), (e2, 84)):
            for k, dev, viol in ((740.0, 0.02, False), (770.0, -0.01, False), (800.0, 0.9, True)):
                rows.append({"expiry": expiry, "dte": dte, "strike": k, "moneyness": k / 770.0,
                             "call_price": 10.0, "put_price": 9.0, "lhs": 1.0, "rhs": 1.0 - dev,
                             "deviation": dev, "spread": 0.3 if quoted else float("nan"),
                             "tradeable_violation": viol if quoted else False})
        return pd.DataFrame(rows)

    def test_two_layers_and_hot_cells(self):
        from src.render.figures import build_parity_figure
        fig = build_parity_figure(self._parity(), 770.0)
        heat = [t for t in fig.data if t.type == "heatmap"]
        assert len(heat) == 2
        faint, hot = heat
        assert faint.opacity == 0.3 and faint.showscale is False and hot.showscale is True
        hot_vals = [v for row in hot.z for v in row if v is not None and v == v]
        assert hot_vals == pytest.approx([0.9, 0.9])
        assert list(faint.x) == ["2026-09-25 (28d)", "2026-11-20 (84d)"]
        assert not any("spreads unknown" in (a.text or "") for a in fig.layout.annotations)

    def test_close_based_is_annotated_and_nothing_hot(self):
        from src.render.figures import build_parity_figure
        fig = build_parity_figure(self._parity(quoted=False), 770.0)
        hot = [t for t in fig.data if t.type == "heatmap"][1]
        assert all(v is None or v != v for row in hot.z for v in row)
        assert any("spreads unknown" in (a.text or "") for a in fig.layout.annotations)

    def test_empty(self):
        from src.analytics.parity import PARITY_COLUMNS
        from src.render.figures import build_parity_figure
        fig = build_parity_figure(pd.DataFrame(columns=PARITY_COLUMNS), 770.0)
        assert not fig.data and fig.layout.annotations


class TestParityStat:
    def test_quoted_sentence(self):
        from src.render.stats import parity_summary_html
        s = {"n_pairs": 420, "n_quoted": 420, "n_tradeable_violations": 3,
             "share_within_spread": 0.9929, "max_abs_deviation": 1.42}
        html = parity_summary_html(s, 0.0281, 21.0, 0.0383, 0.0098)
        assert "420" in html and "3 tradeable" in html and "99.3%" in html
        assert "2.81%" in html and "2.85%" in html and "21" in html

    def test_close_based_sentence(self):
        from src.render.stats import parity_summary_html
        s = {"n_pairs": 400, "n_quoted": 0, "n_tradeable_violations": 0,
             "share_within_spread": float("nan"), "max_abs_deviation": 2.0}
        html = parity_summary_html(s, float("nan"), float("nan"), 0.0383, 0.0098)
        assert "spreads unknown" in html and "carry" not in html.lower()

    def test_no_pairs(self):
        from src.render.stats import parity_summary_html
        s = {"n_pairs": 0, "n_quoted": 0, "n_tradeable_violations": 0,
             "share_within_spread": float("nan"), "max_abs_deviation": float("nan")}
        assert "No strike" in parity_summary_html(s, float("nan"), float("nan"), 0.04, 0.01)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_render.py -q -k "Parity"`
Expected: FAIL — `ImportError: build_parity_figure`; `ModuleNotFoundError: src.render.stats`.

- [ ] **Step 3: Implement figure helpers and P7 figure**

Add to `src/render/figures.py`:

```python
def _expiry_label(expiry, dte) -> str:
    return f"{expiry.isoformat()} ({int(dte)}d)"


def _muted_heatmap_layers(z_all, z_hot, x, y, colorbar_title: str, zmin: float, zmax: float,
                          hovertemplate: str, **subplot) -> list[go.Heatmap]:
    """Two layers: every cell faint, then only the non-muted cells at full
    opacity with the colorbar — SPEC 3 P7/P9's 'within-spread region muted'."""
    common = dict(x=x, y=y, colorscale="RdBu", zmid=0, zmin=zmin, zmax=zmax,
                  hoverongaps=False, hovertemplate=hovertemplate)
    return [
        go.Heatmap(z=z_all, opacity=0.3, showscale=False, name="within spread", **common),
        go.Heatmap(z=z_hot, opacity=1.0, showscale=True, name="beyond spread",
                   colorbar=dict(title=colorbar_title, **subplot), **common),
    ]


def build_parity_figure(parity: pd.DataFrame, spot: float) -> go.Figure:
    title = "Put-call parity — C − P versus S·e⁻ᵠᵀ − K·e⁻ʳᵀ, in dollars"
    if parity.empty:
        return _empty_figure(title, "No strike/expiry pairs with both a call and a put priced")
    p = parity.copy()
    p["label"] = [_expiry_label(e, d) for e, d in zip(p["expiry"], p["dte"])]
    order = (p[["label", "dte"]].drop_duplicates().sort_values("dte")["label"].tolist())
    z_all = p.pivot(index="strike", columns="label", values="deviation").reindex(columns=order)
    hot = p["deviation"].where(p["tradeable_violation"])
    z_hot = p.assign(hot=hot).pivot(index="strike", columns="label", values="hot").reindex(columns=order)
    lim = float(np.nanmax(np.abs(z_all.to_numpy(dtype=float)))) if z_all.size else 1.0
    lim = max(lim, 0.05)
    fig = go.Figure()
    for layer in _muted_heatmap_layers(
            z_all.to_numpy(dtype=float), z_hot.to_numpy(dtype=float), order, z_all.index.tolist(),
            "C − P − parity, $", -lim, lim,
            "%{x}<br>K %{y}: %{z:+.2f} $<extra></extra>"):
        fig.add_trace(layer)
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=spot, y1=spot,
                  line=dict(color="grey", width=1, dash="dot"))
    if p["spread"].notna().sum() == 0:
        fig.add_annotation(text="close-based session — spreads unknown, tradeability not assessable",
                           xref="paper", yref="paper", x=0.0, y=1.02, showarrow=False,
                           xanchor="left", font=dict(size=11, color="#555"))
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_xaxes(title_text="Expiry")
    fig.update_yaxes(title_text="Strike")
    return fig
```

(`import numpy as np` at the top of `figures.py` if not already present.)

`src/render/stats.py`:

```python
"""Stat sentences that sit above a panel (the P5 one lives in analytics.iv_rv)."""
import math


def parity_summary_html(summary: dict, carry_value: float, carry_dte: float,
                        r: float, q: float) -> str:
    if summary["n_pairs"] == 0:
        return "<p class='stat'>No strike/expiry pairs with both legs priced this session.</p>"
    if summary["n_quoted"] == 0:
        return (f"<p class='stat'>{summary['n_pairs']} strike/expiry pairs — close-based session, "
                "spreads unknown, so tradeability cannot be assessed; the heatmap shows raw "
                "deviations only.</p>")
    text = (f"<p class='stat'>{summary['n_pairs']} strike/expiry pairs · "
            f"{summary['n_tradeable_violations']} tradeable violation"
            f"{'' if summary['n_tradeable_violations'] == 1 else 's'} "
            f"(|deviation| beyond the combined bid–ask spread) · "
            f"{summary['share_within_spread']:.1%} within spread.")
    if carry_value is not None and not math.isnan(carry_value):
        text += (f" ATM implied carry at the {carry_dte:.0f}-day expiry: "
                 f"{carry_value:.2%} vs our r − q = {r - q:.2%}.")
    return text + "</p>"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_render.py -q` — PASS. Then `.venv/bin/pytest -q` — green.

- [ ] **Step 5: Commit**

```bash
git add src/render/figures.py src/render/stats.py tests/test_render.py
git commit -m "feat: P7 parity heatmap with within-spread muting and implied-carry stat line"
```

---

### Task 5: P9 model-vs-market analytics + toggle heatmap

**Files:**
- Create: `src/analytics/model_vs_market.py`
- Modify: `src/render/figures.py` (add `build_model_vs_market_figure`)
- Test: `tests/test_model_vs_market.py`; extend `tests/test_render.py`

**Interfaces:**
- `MVM_COLUMNS = ["kind","expiry","dte","strike","moneyness","market","model","deviation_pct","deviation_vol","quoted","within_spread"]`
- `compute_model_vs_market(chain_iv, flat_vol, r, q) -> pd.DataFrame[MVM_COLUMNS]` — rows with `price_used > 0` (both kinds; unconverged IV rows keep `deviation_vol` NaN); `quoted` = bid & ask finite; `within_spread` = `quoted & |market − model| ≤ (ask − bid)/2`; empty or non-finite `flat_vol` → empty frame.
- `build_model_vs_market_figure(mvm, flat_vol) -> go.Figure` — 1×2 subplots (calls | puts), per subplot the two muted layers for `%` (visible) and the two for vol points (hidden) → 8 heatmaps; `updatemenus` with two buttons "% of market price" / "Vol points" toggling `visible`; colour range ±100% / ±20 vol pts.

- [ ] **Step 1: Write the failing tests**

`tests/test_model_vs_market.py`:

```python
"""P9: every contract priced at one flat vol vs the market."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.model_vs_market import MVM_COLUMNS, compute_model_vs_market
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def make_chain(sigma_fn=lambda k: 0.22, source="yfinance", half_spread=0.05,
               strikes=(660.0, 700.0, 740.0, 770.0, 800.0, 840.0, 880.0),
               expiries=((dt.date(2026, 9, 25), 28), (dt.date(2026, 11, 20), 84))):
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                px = float(bs_price(SPOT, strike, dte / 365.0, R, sigma_fn(strike), Q, kind))
                live = source == "yfinance"
                rows.append({"snapshot_date": TODAY, "spot": SPOT, "expiry": expiry, "dte": dte,
                             "strike": strike, "kind": kind,
                             "bid": px - half_spread if live else np.nan,
                             "ask": px + half_spread if live else np.nan,
                             "mid": px if live else np.nan, "close": px,
                             "volume": 10, "open_interest": 100.0, "vendor_iv": np.nan,
                             "source": source})
    out, _ = compute_chain_iv(pd.DataFrame(rows, columns=CHAIN_COLUMNS), R, Q)
    return out


class TestModelVsMarket:
    def test_flat_market_matches_flat_model(self):
        m = compute_model_vs_market(make_chain(), 0.22, R, Q)
        assert list(m.columns) == MVM_COLUMNS and len(m) == 28
        assert np.abs(m["deviation_pct"]).max() < 1e-9
        assert np.abs(m["deviation_vol"]).max() < 1e-6
        assert m["quoted"].all() and m["within_spread"].all()

    def test_skewed_market_is_rich_in_the_put_wing(self):
        skew_fn = lambda k: 0.20 + 0.30 * max(0.0, (SPOT - k) / SPOT)
        m = compute_model_vs_market(make_chain(sigma_fn=skew_fn), 0.20, R, Q)
        wing = m[(m["kind"] == "put") & (m["strike"] == 660.0) & (m["dte"] == 28)].iloc[0]
        assert wing["deviation_pct"] > 0.5            # market far richer than flat-vol model
        assert wing["deviation_vol"] == pytest.approx(skew_fn(660.0) - 0.20, abs=1e-6)
        assert bool(wing["within_spread"]) is False
        atm = m[(m["kind"] == "call") & (m["strike"] == 770.0) & (m["dte"] == 28)].iloc[0]
        assert abs(atm["deviation_pct"]) < 1e-9

    def test_small_deviation_inside_half_spread_is_muted(self):
        m = compute_model_vs_market(make_chain(half_spread=1.0), 0.2205, R, Q)
        assert m["within_spread"].all()

    def test_close_based_rows_are_unquoted_and_never_muted(self):
        m = compute_model_vs_market(make_chain(source="massive-backfill"), 0.22, R, Q)
        assert not m["quoted"].any() and not m["within_spread"].any()
        assert np.abs(m["deviation_pct"]).max() < 1e-9

    def test_nan_flat_vol_or_empty_gives_empty(self):
        assert compute_model_vs_market(make_chain(), float("nan"), R, Q).empty
        e = compute_model_vs_market(make_chain().iloc[0:0], 0.22, R, Q)
        assert e.empty and list(e.columns) == MVM_COLUMNS
```

Append to `tests/test_render.py`:

```python
class TestModelVsMarketFigure:
    def _mvm(self):
        import datetime as dt
        rows = []
        for kind in ("call", "put"):
            for k, pct, vol, muted in ((740.0, 0.02, 0.002, True), (800.0, 0.35, 0.05, False)):
                rows.append({"kind": kind, "expiry": dt.date(2026, 9, 25), "dte": 28, "strike": k,
                             "moneyness": k / 770.0, "market": 10.0, "model": 10.0 * (1 - pct),
                             "deviation_pct": pct, "deviation_vol": vol, "quoted": True,
                             "within_spread": muted})
        return pd.DataFrame(rows)

    def test_eight_layers_toggle_and_visibility(self):
        from src.render.figures import build_model_vs_market_figure
        fig = build_model_vs_market_figure(self._mvm(), 0.12)
        heat = [t for t in fig.data if t.type == "heatmap"]
        assert len(heat) == 8
        assert [t.visible for t in heat] == [True] * 4 + [False] * 4
        assert {t.xaxis for t in heat} == {"x", "x2"}
        buttons = fig.layout.updatemenus[0].buttons
        assert [b.label for b in buttons] == ["% of market price", "Vol points"]
        assert list(buttons[1].args[0]["visible"]) == [False] * 4 + [True] * 4
        hot_pct_calls = heat[1]
        vals = [v for row in hot_pct_calls.z for v in row if v is not None and v == v]
        assert vals == pytest.approx([0.35])

    def test_empty(self):
        from src.analytics.model_vs_market import MVM_COLUMNS
        from src.render.figures import build_model_vs_market_figure
        fig = build_model_vs_market_figure(pd.DataFrame(columns=MVM_COLUMNS), 0.12)
        assert not fig.data and fig.layout.annotations
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_model_vs_market.py tests/test_render.py -q -k "ModelVsMarket"`
Expected: FAIL — `ModuleNotFoundError: src.analytics.model_vs_market`; `ImportError: build_model_vs_market_figure`.

- [ ] **Step 3: Implement analytics**

`src/analytics/model_vs_market.py`:

```python
"""P9: flat-vol Black-Scholes price vs the market, every contract (SPEC 3 P9).

One vol for the whole chain -- today's ~30-DTE ATM IV -- is what the model
literally assumes. The deviation pattern is the smile of P2 re-expressed
in price space; muting within the bid–ask spread mirrors P7.
"""
import numpy as np
import pandas as pd

from src.models.black_scholes import bs_price

MVM_COLUMNS = ["kind", "expiry", "dte", "strike", "moneyness", "market", "model",
               "deviation_pct", "deviation_vol", "quoted", "within_spread"]


def compute_model_vs_market(chain_iv: pd.DataFrame, flat_vol: float,
                            r: float, q: float) -> pd.DataFrame:
    if chain_iv.empty or flat_vol is None or not np.isfinite(flat_vol):
        return pd.DataFrame(columns=MVM_COLUMNS)
    df = chain_iv[chain_iv["price_used"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=MVM_COLUMNS)
    spot = float(df["spot"].iloc[0])
    T = df["dte"].to_numpy(dtype=float) / 365.0
    df["model"] = np.nan
    for kind in ("call", "put"):
        m = (df["kind"] == kind).to_numpy()
        if m.any():
            df.loc[m, "model"] = np.atleast_1d(bs_price(
                S=spot, K=df.loc[m, "strike"].to_numpy(dtype=float), T=T[m], r=r,
                sigma=float(flat_vol), q=q, kind=kind))
    df["market"] = df["price_used"].astype(float)
    df["moneyness"] = df["strike"] / spot
    df["deviation_pct"] = (df["market"] - df["model"]) / df["market"]
    df["deviation_vol"] = df["iv"] - float(flat_vol)
    df["quoted"] = df["bid"].notna() & df["ask"].notna()
    half = (df["ask"] - df["bid"]) / 2.0
    df["within_spread"] = (df["quoted"] & ((df["market"] - df["model"]).abs() <= half)).astype(bool)
    return (df[MVM_COLUMNS].sort_values(["kind", "expiry", "strike"])
            .reset_index(drop=True))
```

- [ ] **Step 4: Implement the figure**

Add to `src/render/figures.py`:

```python
def build_model_vs_market_figure(mvm: pd.DataFrame, flat_vol: float) -> go.Figure:
    title = "Model vs market — every contract priced at one flat vol"
    if mvm.empty:
        return _empty_figure(title, "No flat ~30-DTE ATM vol to price the chain with")
    title += f" ({flat_vol:.1%})"
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("Calls", "Puts"))
    metrics = [("deviation_pct", "% of market price", 1.0, "%{z:+.0%}"),
               ("deviation_vol", "Vol points", 0.20, "%{z:+.1%}")]
    for mi, (col, cb_title, lim, fmt) in enumerate(metrics):
        for ci, kind in enumerate(("call", "put"), start=1):
            g = mvm[mvm["kind"] == kind].copy()
            g["label"] = [_expiry_label(e, d) for e, d in zip(g["expiry"], g["dte"])]
            order = g[["label", "dte"]].drop_duplicates().sort_values("dte")["label"].tolist()
            z_all = g.pivot(index="strike", columns="label", values=col).reindex(columns=order)
            z_hot = (g.assign(hot=g[col].where(~g["within_spread"]))
                       .pivot(index="strike", columns="label", values="hot").reindex(columns=order))
            layers = _muted_heatmap_layers(
                z_all.to_numpy(dtype=float), z_hot.to_numpy(dtype=float), order,
                z_all.index.tolist(), cb_title, -lim, lim,
                "%{x}<br>K %{y}: " + fmt + "<extra>" + kind + "</extra>",
                x=1.02 if ci == 2 else 0.46, len=0.9)
            for layer in layers:
                layer.visible = (mi == 0)
                fig.add_trace(layer, row=1, col=ci)
    vis_pct = [True] * 4 + [False] * 4
    vis_vol = [False] * 4 + [True] * 4
    fig.update_layout(
        title=title, **_LAYOUT,
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.14, xanchor="left",
                          yanchor="bottom", showactive=True,
                          buttons=[dict(label="% of market price", method="update",
                                        args=[{"visible": vis_pct}]),
                                   dict(label="Vol points", method="update",
                                        args=[{"visible": vis_vol}])])])
    fig.update_xaxes(title_text="Expiry")
    fig.update_yaxes(title_text="Strike", row=1, col=1)
    return fig
```

The `updatemenus` sits at y=1.14 on the left; `_LAYOUT`'s legend is irrelevant here (heatmaps have no legend entries), so the two do not collide.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_model_vs_market.py tests/test_render.py -q` — PASS. Then `.venv/bin/pytest -q` — green.

- [ ] **Step 6: Commit**

```bash
git add src/analytics/model_vs_market.py src/render/figures.py tests/test_model_vs_market.py tests/test_render.py
git commit -m "feat: P9 model-vs-market heatmap with %/vol-point toggle and within-spread muting"
```

---

### Task 6: Page wiring + `run_daily` + pipeline tests

**Files:**
- Modify: `src/render/page.py`, `src/run_daily.py`
- Test: extend `tests/test_pipeline.py`, `tests/test_render.py`

**Interfaces:**
- `QUESTIONS`: Q2 → `["P2", "P3", "P6", "P9"]` with coming `""`; Q5 → `["P7"]` with coming `""`. `_PANEL_TITLES` gains `"P6": "25-delta skew — time series"`, `"P7": "Put-call parity checker"`, `"P9": "Model-vs-market price heatmap"`. `CAPTIONS` gains P6/P7/P9 (verbatim below).
- `run()` status gains `"skew_25d"` (rounded float or None), `"parity_pairs"` (int), `"parity_tradeable_violations"` (int), `"implied_carry"` (rounded float or None); `panels_rendered == ["P1","P2","P3","P4","P5","P6","P7","P9"]`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_pipeline.py`: change the `panels_rendered` assertion in `TestRenderStage.test_run_renders_page_and_reports_convergence` to `["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9"]`, and add:

```python
class TestPhase5Stage:
    def test_status_and_page_carry_skew_parity_heatmap(self, tmp_path):
        status = run(FakeEODHD(), FakeLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        # single-strike fixture: one parity pair, no skew bracket, no flat vol -> P9 empty
        assert status["parity_pairs"] == 1
        assert status["parity_tradeable_violations"] in (0, 1)
        assert status["skew_25d"] is None
        m = pd.read_parquet(storage.daily_metrics_path(tmp_path))
        assert "skew_25d" in m.columns and np.isnan(m["skew_25d"].iloc[0])
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "Put-call parity checker" in page and "Model-vs-market" in page
        assert "25-delta skew" in page
        assert "arrives in Phase 5" not in page
        assert "arrives in Phase 6" in page              # Q4 still pending

    def test_wide_chain_records_skew_and_zero_violations(self, tmp_path):
        from src.models.black_scholes import bs_price

        def wide_chain(spot, expiry, dte):
            T = dte / 365.0
            rows = []
            for strike in [spot + d for d in range(-100, 101, 10)]:
                for kind in ("call", "put"):
                    price = float(bs_price(spot, strike, T, 0.0415, 0.20, 0.0098, kind))
                    rows.append({"expiry": expiry, "strike": float(strike), "kind": kind,
                                 "bid": price - 0.05, "ask": price + 0.05, "mid": price,
                                 "close": price, "volume": 100, "open_interest": 500,
                                 "vendor_iv": 0.20, "source": "yfinance"})
            return pd.DataFrame(rows)

        class WideLive:
            def get_option_chain(self, symbol, snapshot_date, spot, cfg):
                return wide_chain(spot, dt.date(2026, 9, 25), 28)

        status = run(FakeEODHD(), WideLive(), FakeFallback(), real_cfg(), tmp_path, today=TODAY)
        assert status["skew_25d"] == pytest.approx(0.0, abs=1e-6)   # flat vol -> no skew
        assert status["parity_pairs"] == 21 and status["parity_tradeable_violations"] == 0
        assert status["implied_carry"] == pytest.approx(0.0415 - 0.0098, abs=1e-6)
        m = pd.read_parquet(storage.daily_metrics_path(tmp_path))
        assert m["skew_25d_dte"].iloc[0] == 28
        page = (tmp_path / "docs" / "index.html").read_text()
        assert "0 tradeable violations" in page
        assert "Vol points" in page                          # P9 toggle rendered
```

In `tests/test_render.py::TestPagePhase4` add:

```python
    def test_phase5_panels_and_sections(self):
        import plotly.graph_objects as go
        from src.render.page import CAPTIONS, render_page
        figs = {p: go.Figure() for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9")}
        html = render_page(figs, self._status())
        q2, q3, q5 = html.index("<h2>Q2."), html.index("<h2>Q3."), html.index("<h2>Q5.")
        assert q2 < html.index("25-delta skew") < html.index("Model-vs-market") < q3
        assert q5 < html.index("Put-call parity checker")
        assert "arrives in Phase 5" not in html
        for pid in ("P6", "P7", "P9"):
            assert CAPTIONS[pid] in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_render.py -q -k "Phase5 or phase5 or renders_page"`
Expected: FAIL — `KeyError: 'parity_pairs'`, `panels_rendered` mismatch, `KeyError: 'P6'` in CAPTIONS.

- [ ] **Step 3: Update `page.py`**

`QUESTIONS`:

```python
QUESTIONS = [
    ("Q1", "Everyone has the same formula. Why does anyone disagree about prices?",
     ["P1", "P4"], ""),
    ("Q2", "Where does the market ignore the model, and why?",
     ["P2", "P3", "P6", "P9"], ""),
    ("Q3", "Is implied volatility a prediction, or a price?",
     ["P5"], ""),
    ("Q4", "The model claims you can hedge an option perfectly. Can you?",
     [], "Delta-hedged P&L simulation arrives in Phase 6."),
    ("Q5", "Given all these flaws, why does every desk still use Black-Scholes?",
     ["P7"], ""),
]
```

`_PANEL_TITLES` additions: `"P6": "25-delta skew — time series"`, `"P7": "Put-call parity checker"`, `"P9": "Model-vs-market price heatmap"`.

`CAPTIONS` additions (verbatim):

```python
    "P6": (
        "The 25-delta skew is the price of crash insurance in one number: the "
        "implied vol of a put that pays off on a roughly one-in-four down-move, "
        "minus that of the mirror-image call, at the ~30-day expiry — interpolated "
        "in delta space from this project's own Greeks. Black-Scholes says it "
        "should be zero. For equity indices it almost never is, and it jumps on "
        "risk-off days; the annotations mark the ones we have noted."
    ),
    "P7": (
        "Put-call parity is the one relationship here that needs no model at all: "
        "a call minus a put must equal a forward, or there is a free lunch. Cells "
        "show the dollar gap between the two sides; anything smaller than the "
        "combined bid–ask spread is muted because it cannot be traded. Deep "
        "in-the-money puts often show a residual gap that is not arbitrage — SPY "
        "options are American, and early exercise makes European parity only "
        "approximate there. This model-free check is why desks trust Black-Scholes "
        "as a quoting convention even when they distrust it as a forecast."
    ),
    "P9": (
        "Price every contract with one flat volatility — today's ~30-day "
        "at-the-money implied vol — and compare to the market. The error is not "
        "noise: the wings are systematically rich against the flat-vol model, on "
        "both sides, day after day. A structured error means the world has a "
        "feature the model lacks (fat tails, volatility that moves), which is "
        "exactly why Heston and SABR exist. This is the smile of the panel above, "
        "re-expressed in price."
    ),
```

- [ ] **Step 4: Update `run_daily.py`**

Imports to add:

```python
from src.analytics.annotations import load_annotations
from src.analytics.model_vs_market import compute_model_vs_market
from src.analytics.parity import carry_at_target, compute_parity, implied_carry, parity_summary
from src.analytics.skew import compute_skew_25d
from src.render.figures import (
    build_greeks_curves_figure, build_iv_rv_figure, build_model_vs_market_figure,
    build_parity_figure, build_sensitivity_figure, build_skew_figure,
    build_smile_figure, build_term_structure_figure,
)
from src.render.stats import parity_summary_html
```

Compute stage changes — skew must be computed **before** the metrics upsert (it goes into the row):

```python
    # ---- P6 skew (into this session's daily_metrics row) ----
    skew = compute_skew_25d(chain_iv, risk_free_rate, dividend_yield, cfg)

    # ---- daily_metrics: this session's row + RV refresh over the whole history ----
    metrics = storage.read_daily_metrics(root, metric_columns(cfg))
    metrics = upsert_session(
        metrics, session_metrics_row(session_date, spot, source, ts_today, iv_stats, cfg,
                                     skew=skew), cfg)
```

After the P5 block:

```python
    # ---- P7 parity + implied carry ----
    parity = compute_parity(chain_iv, risk_free_rate, dividend_yield)
    psum = parity_summary(parity)
    carry = implied_carry(parity, spot, risk_free_rate)
    carry_value, carry_dte = carry_at_target(carry, int(cfg["target_dte"]["atm_panel"]))

    # ---- P9 model vs market at the same flat vol P1 uses ----
    mvm = compute_model_vs_market(chain_iv, sigma_p1, risk_free_rate, dividend_yield)

    annotations = load_annotations(Path(root) / "data" / "annotations.yaml")
```

Figures/extras:

```python
    figures = {
        "P1": build_sensitivity_figure(sens, cfg["sensitivity"]["bump"]),
        "P2": build_smile_figure(smile, spot),
        "P3": build_term_structure_figure(ts_today, ts_prev, previous_label=prev_label),
        "P4": build_greeks_curves_figure(curves, spot),
        "P5": build_iv_rv_figure(series, summary, cfg),
        "P6": build_skew_figure(metrics, annotations),
        "P7": build_parity_figure(parity, spot),
        "P9": build_model_vs_market_figure(mvm, sigma_p1),
    }
    extras = {"P4": greek_tiles_html(tiles, tiles_prev, prev_label),
              "P5": iv_rv_summary_html(summary),
              "P7": parity_summary_html(psum, carry_value, carry_dte,
                                        risk_free_rate, dividend_yield)}
```

Status additions (before `"panels_rendered"`):

```python
        "skew_25d": (round(float(skew["skew_25d"]), 6)
                     if np.isfinite(skew["skew_25d"]) else None),
        "parity_pairs": psum["n_pairs"],
        "parity_tradeable_violations": psum["n_tradeable_violations"],
        "implied_carry": round(carry_value, 6) if np.isfinite(carry_value) else None,
```

Write stage unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_pipeline.py tests/test_render.py -q` — PASS. Then `.venv/bin/pytest -q` — green, no warnings. If the shared single-strike fixture's parity row flags a "tradeable violation" (the fake 71.2/2.0 prices are not parity-consistent), that is why the first test accepts `(0, 1)`; do not change the fixture.

- [ ] **Step 6: Commit**

```bash
git add src/render/page.py src/run_daily.py tests/test_pipeline.py tests/test_render.py
git commit -m "feat: daily run computes skew, parity, model-vs-market; Q2/Q5 sections complete"
```

---

### Task 7: Real run — rebuild history, render, verify, teach (Phase 5 exit)

**Files:**
- Modify: `LEARNING_LOG.md`; commit includes `data/daily_metrics.parquet`, `data/annotations.yaml` (already committed in Task 2), `docs/index.html`, `docs/status.json`, any new `data/chains/*.parquet`.

**Note for the executor:** the daily run needs network + `.env` (sandbox off for that command only). Never print secrets. The rebuild is offline.

- [ ] **Step 1: Rebuild `daily_metrics` so the skew column is filled for history**

Run: `.venv/bin/python scripts/rebuild_daily_metrics.py`
Expected: 12–13 rows; `skew_25d` finite for every session whose ~30-DTE expiry brackets ±0.25 delta on both sides (expect all — the stored chains have $1–$5 strike ladders), `skew_25d_dte` the actual expiry DTE (21–32). Record the table.

- [ ] **Step 2: Real daily run**

Run: `.venv/bin/python -m src.run_daily`
Expected: `panels_rendered` lists 8 panels; `skew_25d` a small positive number (SPY's front-month put skew — expect ~0.03–0.08 i.e. 3–8 vol points); `parity_pairs` several hundred (live yfinance chain); `parity_tradeable_violations` **≈ 0** (the Phase 5 exit criterion — record the exact count and, if nonzero, list the offending strikes: expect deep-ITM puts, the American-exercise caveat); `implied_carry` within ~1% of `r − q`.

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
for needle in ("25-delta skew", "Put-call parity checker", "Model-vs-market", "tradeable violation",
               "Vol points", "implied carry"):
    assert needle in html, needle
assert "arrives in Phase 5" not in html
print(f"page OK: {size/1e6:.2f} MB, self-contained, 8 panels")
EOF
```

Then a visual pass: serve `docs/` locally (`.venv/bin/python -m http.server 8765 --bind 127.0.0.1` in the background — sandbox off — and open in a browser, or ask the controller to screenshot) and check: P6 shows a short series with a zero line; P7 heatmap has the spot line, mostly faint cells, few or no opaque cells; P9's toggle switches between % and vol points and the put wing is visibly red/rich.

- [ ] **Step 4: Learning-log entries**

Append under `## Entries`, dated 2026-08-29, four entries, each 2–4 paragraphs with real numbers from Steps 1–3 and cited test names:

1. **"25-delta skew: why interpolate in delta"** — what a 25-delta option is and why it is the same contract across regimes; the delta-space interpolation from each strike's own IV (`test_interpolates_at_delta_not_strike`); today's put/call 25-delta IVs and the skew in vol points; the history table's range; why Black-Scholes predicts zero.
2. **"Put-call parity: the model-free check"** — derivation of `C − P = S e^{−qT} − K e^{−rT}` from a static portfolio, why no vol appears; the tradeable-violation definition and today's count vs the ≈0 criterion; which strikes (if any) violate and the American early-exercise attribution for deep-ITM puts; the implied carry back-out (derivation `F = K + (C−P)e^{rT}`) and today's implied `r − q` vs ours (`test_recovers_r_minus_q`).
3. **"Model vs market: the smile in price space"** — the flat-vol pricing exercise; today's largest rich/cheap deviations in % and vol points, which wing, whether structured across expiries; the within-spread muting logic and what fraction of contracts the flat model prices *inside* the spread (compute it from `mvm`); why structured error motivates stochastic-vol models.
4. **"Phase 5: real run"** — the exit criteria honestly: daily_metrics appending with the new columns (row count, how many sessions have skew), tradeable-violation count; honest limits: the annotations file is empty (no news notes yet — a human adds them), skew history is 12–13 sessions, close-based sessions can never show violations so the criterion is only meaningful on live days.

- [ ] **Step 5: Full suite, then commit**

```bash
.venv/bin/pytest -q
git add data/daily_metrics.parquet data/chains data/underlying.parquet docs/ LEARNING_LOG.md
git commit -m "feat: Phase 5 — skew history, parity checker, model-vs-market heatmap from real data"
```

- [ ] **Step 6: Report the hand-off**

Report: Phase 5 exit met locally (metrics appending with skew; tradeable-violation count = N). The page publishes on push (Pages already serves `/docs` on `main`). Suggest the user add a first real note to `data/annotations.yaml` when a risk-off day occurs.

---

## Self-Review Notes (already applied)

- **Spec coverage.** P6 compute (delta-space 25-delta put − call at ~30 DTE, one number/day appended to `daily_metrics`), output (time series + `data/annotations.yaml`) — T1/T2. P7 compute (parity, tradeable = beyond combined bid–ask, implied carry stretch), output (strike×expiry heatmap with muting, violation count), caption's American-exercise attribution — T3/T4/T6. P9 compute (flat vol = today's ~30-DTE ATM IV, % of mid or vol-point toggle, calls/puts separately), output (muted heatmap) — T5. §2.2 skew column in `daily_metrics` — T1. Exit criterion measured — T7.
- **Deliberately deferred (YAGNI):** per-expiry skew (only ~30 DTE is specified); annotation *auto-detection* of spike days (SPEC says manual file); mobile tuning (Phase 7).
- **Type consistency.** `compute_skew_25d` dict keys match `session_metrics_row(skew=)`'s `.get` names (`skew_25d`, `skew_dte`) and `status["skew_25d"]`. `compute_parity` → `parity_summary`/`implied_carry`/`carry_at_target`/`build_parity_figure`/`parity_summary_html` argument shapes match across T3/T4/T6. `compute_model_vs_market(chain_iv, flat_vol, r, q)` and `build_model_vs_market_figure(mvm, flat_vol)` match T5/T6. `_muted_heatmap_layers` is defined in T4 and reused in T5 (same task order). `_apply_recent_range` defined in T2 and used by P5/P6.
- **Fixture note.** The shared single-strike pipeline fixture yields one parity pair whose fake prices are not parity-consistent; T6's first test tolerates a violation count of 0 or 1 rather than changing the fixture other tests depend on.
- **Placeholder scan:** none; every code step carries its code.
