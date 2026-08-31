# Phase 7 — Automation, Pages Polish, Mobile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the dashboard run itself — a scheduled GitHub Actions job that fetches, tests, renders and publishes each trading day without a human — and make the published page correct on a phone and honest about its own staleness.

**Architecture:** One new workflow (`.github/workflows/daily.yml`) wraps the existing `python -m src.run_daily` entry point: it installs, runs the test suite, runs the pipeline, and commits `data/` and `docs/` back to `main` only when the pipeline succeeded and something actually changed. No pipeline code changes — the failure policy already guarantees no partial writes. The render layer gains a staleness banner and a narrow-screen relayout; neither touches analytics.

**Tech Stack:** GitHub Actions (`actions/checkout@v4`, `actions/setup-python@v5`), GitHub Pages, Python 3.12 on the runner, Plotly (already vendored). No new Python dependencies.

**Spec:** `SPEC.md` — §2.1 (automation, publishing, secrets, failure policy), §4 (CI: tests on every push; the daily job runs tests before rendering), §5 row 7, §6 (`.github/workflows/` layout).

## Global Constraints

- **Scheduler (SPEC §2.1):** cron `30 1 * * 2-6` UTC — ≈21:30 ET, after the US close and EOD settlement; Tue–Sat UTC covers Mon–Fri US sessions. Also `workflow_dispatch` for manual runs.
- **Publishing (SPEC §2.1):** the workflow commits updated `data/` and `docs/` back to `main`; GitHub Pages serves the site. No servers, no cost.
- **Secrets (SPEC §2.1):** `EODHD_API_TOKEN` and `MASSIVE_API_KEY` come from GitHub Actions secrets. **Never** in code, never in committed data, never echoed into a log. `src/data/envfile.py:get_secret` already prefers `os.environ` over the gitignored `.env`, so the workflow supplies them as env vars and nothing else changes.
- **Failure policy (SPEC §2.1):** if the fetch fails (API down, holiday, rate limit) the job exits *without* committing, leaving yesterday's dashboard live. `src/run_daily.py` already computes everything before any write and raises on failure, so a non-zero exit is the whole mechanism — do not add a `try/except` that swallows it.
- **CI (SPEC §4):** tests run on every push in a separate lightweight workflow; the daily job runs the tests *before* rendering — a broken model never publishes.
- **Self-contained page (SPEC:212):** `docs/index.html` inlines the vendored Plotly bundle. Never add an external `<script src=…>`, CDN link, or web font — including in anything you add for the mobile pass.
- **No pricing libraries (SPEC:25).** This phase adds no numerics at all.
- **Tests run offline and deterministically.** No network, no wall-clock dependence, no randomness. `filterwarnings = ["error"]` in `pyproject.toml` makes any pandas/NumPy/Plotly warning a hard failure.
- **Test command:** `.venv/bin/python -m pytest -q` from the repo root. Green at **406 passing** before this plan starts.
- **Commit style:** conventional commits, imperative mood, no trailing period.

---

## Context an implementer needs

**The pipeline entry point.** `python -m src.run_daily` calls `main()`, which reads `config.yaml`, builds the three providers with `get_secret(...)`, calls `run(...)`, and prints the status dict as JSON. It raises on any failure. `run()` writes, in order and atomically: the session's chain parquet → `data/daily_metrics.parquet` → `docs/index.html` → `docs/status.json`. It also upserts `data/underlying.parquet` early, before the compute stage.

**What a successful run changes on disk:** one new (or rewritten) `data/chains/YYYY-MM-DD.parquet`, `data/underlying.parquet`, `data/daily_metrics.parquet`, `docs/index.html`, `docs/status.json`.

**Two live defects this phase fixes, both found by looking at the deployed site:**

1. GitHub Pages publishes from the **repo root** of `main`, not from `/docs`. So `https://tk-chang239.github.io/BlackScholesVolDashboard/` renders `README.md` through Jekyll, and the dashboard lives at `…/docs/`. The README's own "Live dashboard" link points at the root — i.e. at the README itself. A visitor who clicks it goes nowhere.
2. The README already claims the dashboard is "Updated automatically after each US close via GitHub Actions." No `daily.yml` exists. The claim is false until this phase lands.

**Mobile, assessed from the CSS rather than guessed:** `body` is `max-width: 960px; padding: 0 16px` and the Greek tiles already use `grid-template-columns: repeat(auto-fit, minmax(150px, 1fr))`, so both reflow correctly. `.figure` is `width: 100%; overflow-x: auto` and every chart is rendered with Plotly's `responsive: true`. The real problem is the three figures built with `make_subplots(rows=1, cols=2, …)` — at a 358 px content width each panel gets ~170 px, which is not a chart. They are at `src/render/figures.py:69` (sensitivity tornado), `:98` (Greeks curves), and `:353` (model-vs-market); `build_term_structure_figure` builds a single-panel `go.Figure()` with no `make_subplots` call and is not one of them. Those three need to stack on a narrow screen.

---

## File structure

| File | Responsibility | Task |
|---|---|---|
| `.github/workflows/daily.yml` (create) | the scheduled run: install → test → pipeline → commit | 1 |
| `.github/workflows/ci.yml` (modify) | skip the duplicate run the bot's own push would trigger | 1 |
| `src/render/page.py` (modify) | staleness banner; the narrow-screen CSS and relayout hook | 2, 3 |
| `src/render/figures.py` (modify) | tag the two-column figures so the relayout can find them | 3 |
| `README.md` (modify) | fix the self-referential live link; phase tick | 4, 6 |
| `tests/test_render.py` (modify) | banner and relayout tests | 2, 3 |
| `tests/test_workflows.py` (create) | assert the workflow's binding properties as data | 1 |
| `LEARNING_LOG.md` (modify) | Phase 7 entry | 6 |

---

### Task 1: The daily workflow

**Files:**
- Create: `.github/workflows/daily.yml`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_workflows.py`

**Interfaces:**
- Consumes: `python -m src.run_daily`; the repository secrets `EODHD_API_TOKEN` and `MASSIVE_API_KEY`.
- Produces: a workflow whose properties Task 6's verification run exercises.

The workflow is YAML, so the test is a schema test: parse the file and assert the properties that matter, so a later edit that breaks the schedule or drops the test gate fails the suite instead of failing silently at 01:30 UTC.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflows.py`:

```python
"""The daily workflow's binding properties, asserted as data (SPEC 2.1, 4)."""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / ".github" / "workflows" / "daily.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def load(path: Path) -> dict:
    # PyYAML parses the unquoted key `on:` as the boolean True, so both spellings
    # are accepted here rather than being worked around in the workflow file.
    doc = yaml.safe_load(path.read_text())
    doc["on"] = doc.get("on", doc.get(True))
    return doc


class TestDailyWorkflow:
    def test_exists_and_parses(self):
        assert DAILY.exists()
        assert load(DAILY)["jobs"]

    def test_cron_is_the_spec_schedule(self):
        triggers = load(DAILY)["on"]
        assert triggers["schedule"] == [{"cron": "30 1 * * 2-6"}]

    def test_manual_dispatch_is_available(self):
        assert "workflow_dispatch" in load(DAILY)["on"]

    def test_tests_run_before_the_pipeline(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        runs = [s.get("run", "") for s in steps]
        test_at = next(i for i, r in enumerate(runs) if "pytest" in r)
        run_at = next(i for i, r in enumerate(runs) if "src.run_daily" in r)
        assert test_at < run_at, "a broken model must never publish"

    def test_secrets_arrive_as_env_and_are_never_inlined(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        step = next(s for s in steps if "src.run_daily" in s.get("run", ""))
        assert set(step["env"]) == {"EODHD_API_TOKEN", "MASSIVE_API_KEY"}
        for value in step["env"].values():
            assert value.startswith("${{ secrets."), value
        assert "${{ secrets." not in "".join(s.get("run", "") for s in steps)

    def test_job_can_write_and_will_not_overlap_itself(self):
        doc = load(DAILY)
        assert doc["jobs"]["daily"]["permissions"]["contents"] == "write"
        assert doc["concurrency"]["group"]
        assert doc["concurrency"]["cancel-in-progress"] is False

    def test_commit_is_conditional_on_a_real_change(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        commit = next(s for s in steps if "git commit" in s.get("run", ""))
        assert "git diff --quiet" in commit["run"] or "--staged --quiet" in commit["run"]

    def test_bot_commit_does_not_retrigger_ci(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        commit = next(s for s in steps if "git commit" in s.get("run", ""))
        assert "[skip ci]" in commit["run"]

    def test_only_data_and_docs_are_committed(self):
        steps = load(DAILY)["jobs"]["daily"]["steps"]
        commit = next(s for s in steps if "git commit" in s.get("run", ""))
        assert "git add data docs" in commit["run"]
        assert "git add ." not in commit["run"]
        assert "git add -A" not in commit["run"]


class TestCIWorkflow:
    def test_still_runs_tests_on_push(self):
        doc = load(CI)
        assert "push" in doc["on"]
        assert any("pytest" in s.get("run", "") for s in doc["jobs"]["tests"]["steps"])

    def test_python_version_matches_the_daily_job(self):
        def python_of(doc):
            for job in doc["jobs"].values():
                for step in job["steps"]:
                    if "setup-python" in str(step.get("uses", "")):
                        return str(step["with"]["python-version"])
            return None
        assert python_of(load(CI)) == python_of(load(DAILY))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_workflows.py -q`
Expected: FAIL — `assert DAILY.exists()` is False.

- [ ] **Step 3: Write the workflow**

Create `.github/workflows/daily.yml`:

```yaml
name: daily

# 30 1 * * 2-6 UTC ~= 21:30 ET Mon-Fri: after the US close and after EOD data
# settles. Tue-Sat in UTC covers Mon-Fri US sessions (SPEC 2.1).
on:
  schedule:
    - cron: "30 1 * * 2-6"
  workflow_dispatch:

# A run commits to main. Never let two runs race for the same push, and never
# cancel one midway -- a cancelled job could leave the repo half-updated.
concurrency:
  group: daily-publish
  cancel-in-progress: false

jobs:
  daily:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - run: pip install -r requirements.txt

      # SPEC 4: a broken model never publishes. This runs BEFORE the pipeline,
      # and a failure here stops the job with nothing written.
      - name: Run tests
        run: pytest -q

      # SPEC 2.1 failure policy: run_daily computes everything before it writes
      # anything and raises on failure, so a non-zero exit here simply leaves
      # yesterday's dashboard live. Do not wrap this in error handling.
      - name: Run the daily pipeline
        env:
          EODHD_API_TOKEN: ${{ secrets.EODHD_API_TOKEN }}
          MASSIVE_API_KEY: ${{ secrets.MASSIVE_API_KEY }}
        run: python -m src.run_daily

      - name: Commit the new snapshot
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add data docs
          if git diff --staged --quiet; then
            echo "nothing changed - not committing"
            exit 0
          fi
          git commit -m "chore: daily snapshot $(date -u +%Y-%m-%d) [skip ci]"
          git push
```

- [ ] **Step 4: Align `ci.yml`**

`ci.yml` currently runs `pytest -v` on `[push, pull_request]` with Python 3.12. Change only the test command to `pytest -q` for a readable log, and leave the triggers alone — the daily job's `[skip ci]` is what prevents the duplicate run. Confirm `python-version` stays `"3.12"` in both files so `test_python_version_matches_the_daily_job` passes.

- [ ] **Step 5: Verify**

Run: `.venv/bin/python -m pytest tests/test_workflows.py -q` — Expected: PASS
Run: `.venv/bin/python -m pytest -q` — Expected: PASS (406 + the new tests)

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/daily.yml .github/workflows/ci.yml tests/test_workflows.py
git commit -m "feat: publish the dashboard from a scheduled Actions run"
```

---

### Task 2: Make staleness visible

**Files:**
- Modify: `src/render/page.py`
- Test: `tests/test_render.py` (append)

**Interfaces:**
- Consumes: `status["snapshot_date"]` (ISO date string) and `status["last_success_utc"]`.
- Produces: `staleness_banner(status, today) -> str` in `src/render/page.py`, rendered directly beneath the `<h1>`.

SPEC §2.1 requires that staleness be "visible, not silent". The footer already carries `last_success_utc`, but a reader has to know what "fresh" looks like to notice. A banner states it outright, and only when there is something to say — a fresh page carries no chrome. `STALENESS_DAYS = 5` already exists in `src/run_daily.py` as the pipeline's own hard limit; the banner uses its own, lower threshold because *displaying* a warning and *refusing to run* are different decisions.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render.py`:

```python
class TestStalenessBanner:
    def _status(self, snapshot="2026-08-28"):
        return {"spot": 100.0, "snapshot_date": snapshot, "source": "yfinance",
                "iv_convergence": 0.99, "last_success_utc": "2026-08-29T00:00:00+00:00",
                "rows_stored": 10}

    def test_fresh_page_shows_no_banner(self):
        import datetime as dt
        from src.render.page import staleness_banner
        assert staleness_banner(self._status(), today=dt.date(2026, 8, 29)) == ""

    def test_weekend_gap_is_not_stale(self):
        import datetime as dt
        from src.render.page import staleness_banner
        # Friday session read on Monday: three calendar days, zero missed sessions.
        assert staleness_banner(self._status("2026-08-28"), today=dt.date(2026, 8, 31)) == ""

    def test_a_stale_page_says_so_with_the_date_and_the_gap(self):
        import datetime as dt
        from src.render.page import staleness_banner
        html = staleness_banner(self._status("2026-08-28"), today=dt.date(2026, 9, 8))
        assert "2026-08-28" in html
        assert "11 days" in html
        assert "class='stale'" in html

    def test_banner_is_rendered_into_the_page(self):
        import datetime as dt
        from src.render.page import render_page
        html = render_page({}, self._status("2026-08-28"), today=dt.date(2026, 9, 8))
        assert "class='stale'" in html
        assert html.index("class='stale'") < html.index("Q1.")

    def test_page_without_today_argument_still_renders(self):
        from src.render.page import render_page
        assert "<h1>vol-lens</h1>" in render_page({}, self._status())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_render.py -q -k Staleness`
Expected: FAIL — `cannot import name 'staleness_banner'`

- [ ] **Step 3: Implement**

In `src/render/page.py`, add the threshold and the function, and add `.stale` to `_CSS`:

```python
# Calendar days since the snapshot before the page admits it may be stale. Four
# clears a Friday-session page read on the following Tuesday; anything longer
# means a scheduled run has actually been missed. This is a DISPLAY threshold and
# is deliberately not run_daily's STALENESS_DAYS, which decides whether to run.
STALE_AFTER_DAYS = 4


def staleness_banner(status: dict, today: dt.date | None = None) -> str:
    """A visible warning when the page is older than it should be, else ''.

    SPEC 2.1 asks for staleness to be visible rather than silent. A fresh page
    says nothing -- a banner that is always present is one nobody reads.
    """
    today = today or dt.date.today()
    try:
        snapshot = dt.date.fromisoformat(status["snapshot_date"])
    except (KeyError, TypeError, ValueError):
        return ""
    days = (today - snapshot).days
    if days <= STALE_AFTER_DAYS:
        return ""
    return (f"<p class='stale'>This page has not updated in {days} days — the "
            f"most recent session it shows is <b>{snapshot.isoformat()}</b>. The "
            "daily job has not published since then, so every number below is "
            "that session's, not today's.</p>")
```

Add `import datetime as dt` to the module's imports if absent, and append to `_CSS`:

```python
".stale { background: #fff5f5; border: 1px solid #f0c0c0; border-radius: 6px;\n"
"         padding: 10px 12px; margin: 12px 0; color: #8a2b2b; font-size: 0.92rem; }\n"
```

Then change `render_page`'s signature to `render_page(figures, status, extras=None, today=None)` and insert the banner immediately after the `<h1>vol-lens</h1>` part:

```python
        "<h1>vol-lens</h1>",
        staleness_banner(status, today),
```

- [ ] **Step 4: Verify**

Run: `.venv/bin/python -m pytest tests/test_render.py -q` — Expected: PASS
Run: `.venv/bin/python -m pytest -q` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/render/page.py tests/test_render.py
git commit -m "feat: warn on the page when the daily job has stopped publishing"
```

---

### Task 3: Make the two-column charts readable on a phone

**Files:**
- Modify: `src/render/figures.py`, `src/render/page.py`
- Test: `tests/test_render.py` (append)

**Interfaces:**
- Consumes: the three figures built with `make_subplots(rows=1, cols=2, …)`.
- Produces: `meta=dict(stack_narrow=True)` on those three figures, a `figure-wide` container class emitted by `render_page` for them, and a `@media` rule in `_CSS`.

**Why CSS and not JavaScript.** The obvious fix is a script that restacks the grid on resize. It does not work: `make_subplots` does **not** emit a `grid` object — it writes explicit axis domains. Verified on the real sensitivity figure:

```
has 'grid' key: False
xaxis.domain  = [0.0, 0.43]      yaxis.domain  = [0.0, 1.0]
xaxis2.domain = [0.57, 1.0]      yaxis2.domain = [0.0, 1.0]
```

So a `Plotly.relayout(gd, {'grid.rows': 2})` would silently do nothing while a test that merely asserted the script's presence still passed. Swapping the domains by hand is possible but must also move the subplot-title annotations and any colorbar, and its failure mode is a silently squashed chart.

The CSS approach cannot fail silently: below the breakpoint the marked figures get a `min-width`, Plotly (already `responsive: true`) draws at that width, and the existing `.figure { overflow-x: auto }` turns the excess into a swipe. The chart stays at a readable size instead of being compressed into an unreadable one.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render.py`:

```python
class TestNarrowScreenFigures:
    def _status(self):
        return {"spot": 100.0, "snapshot_date": "2026-08-28", "source": "yfinance",
                "iv_convergence": 0.99, "last_success_utc": "2026-08-29T00:00:00+00:00",
                "rows_stored": 10}

    def test_two_column_figures_are_marked(self):
        from src.analytics.sensitivity import compute_sensitivity
        from src.render.figures import build_sensitivity_figure
        sens = compute_sensitivity(100.0, 0.2, 30.0, 0.04, 0.013, 0.2)
        fig = build_sensitivity_figure(sens, 0.2)
        assert (fig.layout.meta or {}).get("stack_narrow") is True

    def test_single_column_figures_are_not_marked(self):
        from src.render.figures import build_smile_figure
        fig = build_smile_figure(pd.DataFrame(), 100.0)
        assert not (fig.layout.meta or {}).get("stack_narrow")

    def test_marked_figures_get_the_wide_container_class(self):
        from src.analytics.sensitivity import compute_sensitivity
        from src.render.figures import build_sensitivity_figure
        from src.render.page import render_page
        sens = compute_sensitivity(100.0, 0.2, 30.0, 0.04, 0.013, 0.2)
        html = render_page({"P1": build_sensitivity_figure(sens, 0.2)}, self._status())
        assert "class='figure figure-wide'" in html

    def test_unmarked_figures_get_the_plain_container(self):
        from src.render.base import empty_figure
        from src.render.page import render_page
        html = render_page({"P2": empty_figure("P2", "x")}, self._status())
        assert "class='figure'" in html
        assert "figure-wide" not in html

    def test_the_media_rule_exists_and_needs_no_external_resource(self):
        from src.render.base import empty_figure
        from src.render.page import render_page
        html = render_page({"P2": empty_figure("P2", "x")}, self._status())
        assert "@media (max-width: 700px)" in html
        assert ".figure-wide" in html
        assert "min-width" in html
        assert "<script src=" not in html
        assert "cdn." not in html
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_render.py -q -k NarrowScreen`
Expected: FAIL — `fig.layout.meta` is None.

- [ ] **Step 3: Mark the three figures**

In `src/render/figures.py`, find the builders that call `make_subplots(rows=1, cols=2` — there are three: the sensitivity tornado, the Greeks curves figure and the model-vs-market heatmap. `build_term_structure_figure` is a single-panel `go.Figure()` with no `make_subplots` call and is not one of them. Add `meta=dict(stack_narrow=True)` to each one's existing `update_layout(...)` call. Change no other layout property, and do not touch the two `make_subplots` calls that are not two-column (`specs=[[{"secondary_y": True}]]` is a single cell).

- [ ] **Step 4: Emit the wide container and add the media rule**

In `src/render/page.py`, the render loop currently appends `"<div class='figure'>"` before each figure. Make that class conditional on the figure's marker:

```python
                wide = " figure-wide" if (figures[pid].layout.meta or {}).get("stack_narrow") else ""
                parts.append(f"<div class='figure{wide}'>")
```

and append to `_CSS`:

```python
"@media (max-width: 700px) {\n"
"  /* Two-column subplot figures compress to ~170px per panel on a phone, which\n"
"     is not a chart. Hold them at a readable width and let .figure scroll. */\n"
"  .figure-wide > div { min-width: 660px; }\n"
"}\n"
```

- [ ] **Step 5: Verify**

Run: `.venv/bin/python -m pytest -q` — Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/render/figures.py src/render/page.py tests/test_render.py
git commit -m "feat: hold two-column charts at a readable width on narrow screens"
```

---

### Task 4: Fix the landing page's broken promise

**Files:**
- Modify: `README.md`

Two defects, both visible on the deployed site:

1. The "Live dashboard" line links to `https://tk-chang239.github.io/BlackScholesVolDashboard/`, which is the README itself — GitHub Pages publishes from the repo root, so the dashboard is at `…/docs/`. Point the link at `https://tk-chang239.github.io/BlackScholesVolDashboard/docs/`.
2. The same line claims the page is "Updated automatically after each US close via GitHub Actions." That became true only with Task 1 — leave the wording, since it is now accurate, but check it reads correctly against what `daily.yml` actually does (weekdays after the US close, skipping days the market did not trade).

Do not restructure the README. These are two small corrections.

- [ ] **Step 1: Make both corrections, then verify the link resolves**

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://tk-chang239.github.io/BlackScholesVolDashboard/docs/
```
Expected: `200`

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "fix: point the live-dashboard link at the dashboard"
```

---

### Task 5: Caption and copy pass

**Files:**
- Modify: `src/render/page.py` (only if a caption is wrong)
- Test: `tests/test_render.py` (only if a caption changes)

SPEC §5 row 7 lists "captions" as part of this phase. All eleven panels already have one; this is an accuracy pass, not a writing exercise.

- [ ] **Step 1: Read every caption in `CAPTIONS` against the panel it describes**

For each of P1, P2, P3, P4, P5, P6, P7, P8a, P8b, P8c, P9, check three things and record the answer per panel in your report:
- Does the caption describe what the figure now actually plots, including any units the fix waves changed?
- Does it claim anything the current sample cannot support?
- Does it contradict the stat line above it, or another caption?

- [ ] **Step 2: Fix only what is wrong**

Change a caption only where step 1 found a defect, and update the caption's covering test when you do. If every caption is accurate, change nothing and say so — a pass that finds nothing is a valid result and is not a reason to reword good prose.

- [ ] **Step 3: Verify and commit (only if something changed)**

Run: `.venv/bin/python -m pytest -q` — Expected: PASS

```bash
git add src/render/page.py tests/test_render.py
git commit -m "fix: correct <panel> caption"
```

---

### Task 6: Real run, workflow verification, learning log

**Files:**
- Modify: `LEARNING_LOG.md`, `README.md`
- Produces: regenerated `docs/index.html`, `docs/status.json`, the session's chain parquet.

**This task has a dependency the repository cannot satisfy on its own:** `EODHD_API_TOKEN` and `MASSIVE_API_KEY` must exist as GitHub Actions repository secrets before a workflow run can succeed. If they are absent, the local run in Step 1 still works (it reads the gitignored `.env`), but Steps 3–4 will fail with a `KeyError` on the runner. Report that rather than working around it.

- [ ] **Step 1: Run the pipeline locally**

`python -m src.run_daily` needs the network, which the Bash sandbox blocks — run it with the sandbox disabled. Expected: exit 0, and a rewritten `docs/index.html` carrying the Task 3 script.

- [ ] **Step 2: Verify the page**

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
html = Path("docs/index.html").read_text()
assert "<script src=" not in html, "page must stay self-contained"
for needle in ("stack_narrow", "matchMedia", "Cumulative P&L", "Simulation for learning"):
    assert needle in html, needle
print(f"ok - {len(html)/1e6:.2f} MB")
PY
```

- [ ] **Step 3: Commit and push, then trigger the workflow manually**

Commit the run's output, push to `main`, then trigger `daily.yml` via `workflow_dispatch` (`gh workflow run daily.yml`) and watch it (`gh run watch`). A manual dispatch is the honest test of the schedule's machinery: it exercises install, tests, pipeline, and the commit step, all on the runner with the real secrets.

- [ ] **Step 4: Verify what the workflow published**

Confirm the run is green; that it either produced a `chore: daily snapshot …` commit on `main` or logged "nothing changed - not committing"; and that the live page at `…/docs/` reflects it. Record the run's URL.

- [ ] **Step 5: Write the Phase 7 learning-log entry**

Append a Phase 7 section to `LEARNING_LOG.md` in the voice and depth of the Phase 5 and Phase 6 entries, with real numbers from your own run. Cover:
1. **What "unattended" actually requires** — why the tests run before the pipeline, why the commit is conditional on a real diff, why the job must not cancel itself mid-run, and what each of those prevents concretely.
2. **The failure policy, end to end.** `run_daily` computes before it writes and raises on failure; the workflow adds nothing to that. Say what a reader would see on the site the morning after a failed fetch, and why that is the correct behaviour rather than a bug.
3. **The two live defects** the deployment survey found — Pages serving the README at the root while the dashboard sat at `/docs/`, and the README claiming automation that did not exist. Both had been shipped and unnoticed for phases; say what would have caught them earlier.
4. **The exit criterion is not met on the day this lands.** SPEC §5 row 7 asks for *five consecutive unattended green daily runs*. One manual dispatch is not that. State plainly what has been demonstrated and what has only been set up, in the same spirit as the Phase 6 entry's treatment of its unmet scatter criterion.
5. Anything that surprised you.

- [ ] **Step 6: Tick the Phase 7 checkbox in `README.md`**, matching how Phases 2–6 are written.

- [ ] **Step 7: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q` — Expected: PASS

```bash
git add LEARNING_LOG.md README.md docs/ data/
git commit -m "feat: first scheduled run publishes the dashboard"
```

---

## Self-review

**Spec coverage:**

| Spec clause | Task |
|---|---|
| §2.1 cron `30 1 * * 2-6` UTC + `workflow_dispatch` | 1 |
| §2.1 workflow commits `data/` and `docs/` back to `main`; Pages serves it | 1, 4 |
| §2.1 secrets from Actions, never in code or committed data | 1 |
| §2.1 failure policy: exit without committing, yesterday's dashboard stays live | 1 (no code change needed; asserted by test) |
| §2.1 `status.json` rendered so staleness is visible, not silent | 2 |
| §4 tests on every push, separate lightweight workflow | 1 (`ci.yml` retained) |
| §4 the daily job runs tests before rendering | 1 (asserted by `test_tests_run_before_the_pipeline`) |
| §5 row 7: automation + Pages polish + captions + mobile pass | 1, 4 / 5 / 3 |
| §5 row 7 exit criterion: five consecutive unattended green runs | 6, honestly reported as not yet met |
| §6 `.github/workflows/` layout | 1 |

**Criterion deliberately not claimed:** five consecutive unattended green daily runs cannot happen on the day this merges — it is five calendar weekdays of accumulation. Task 6 demonstrates the machinery with a manual dispatch and states the distinction rather than implying the criterion is satisfied.

**Placeholder scan:** every code step carries actual code; every test step carries actual assertions. Task 5 is deliberately conditional ("fix only what is wrong") rather than mandating a change — that is a real instruction, not a placeholder.

**Type consistency:** `staleness_banner(status, today)` is called by `render_page(figures, status, extras=None, today=None)` and by its own tests; the `meta=dict(stack_narrow=True)` marker set in `figures.py` is the same key the `page.py` script reads and the same key the tests assert.
