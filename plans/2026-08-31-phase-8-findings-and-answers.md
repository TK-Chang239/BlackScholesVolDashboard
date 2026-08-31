# Phase 8 — Findings and the Ten Answers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer all ten learning-log questions with evidence this project generated, and write the README's Findings section for the five public ones — closing the loop the repo opened before its first line of code.

**Architecture:** Three of the ten questions have no measured answer yet. Each gets a small committed probe script under `scripts/`, following the pattern the `verify_*.py` probes already established: offline, read-only against the stored archive, printing numbers a human reads and records. No new panels, no new analytics modules, no changes to the render or pipeline. The answers themselves are prose in `LEARNING_LOG.md` and `README.md`.

**Tech Stack:** Existing only — NumPy, pandas, this repo's own `src/models/black_scholes.py` and `src/analytics/`. No new dependencies.

**Spec:** `SPEC.md` — §5 row 8 (the phase and its exit criterion), §7 (no pricing libraries), and the ten questions as listed in `LEARNING_LOG.md:7-28`.

## Global Constraints

- **No pricing libraries (SPEC:25, §7).** Every number comes from this repo's own code. A library is permitted *only* to cross-check an answer, never to produce one — and this phase does not need one.
- **Offline and deterministic.** Every probe reads `data/chains/` and `data/underlying.parquet` and touches no network. `filterwarnings = ["error"]` in `pyproject.toml` makes any pandas/NumPy warning a hard failure.
- **Do not run `python -m src.run_daily` during this phase.** It is a pre-market environment; the overwrite-shrink guard correctly refuses, and nothing here needs a fresh page. The scheduled job renders the page.
- **No number without its sample.** This project's standing discipline: an answer states what it measured, on how much data, and what it cannot support. Several of these questions have small samples; say so in the answer rather than in a footnote.
- **Answers cite evidence.** Each of the ten answers names the panel, log entry, test, or probe output it rests on. An answer with no citation is not finished.
- **Test command:** `.venv/bin/python -m pytest -q`. Green at **432 passing**.
- **Commit style:** conventional commits, imperative mood, no trailing period.

---

## Context an implementer needs

**The ten questions** are at `LEARNING_LOG.md:7-28`. Questions 1–5 are the public guiding questions, also printed in `README.md`; 6–10 are the private list.

**What is already answered, and where** — verified by reading the log, so do not redo this work:

| Q | State | Where the evidence is |
|---|---|---|
| 1 | Answered | Panel P1 (input sensitivity); the Phase 4 "Sensitivity" entry |
| 2 | Answered | Panels P2, P3, P6, P9; the "Model vs market" and skew entries |
| 3 | Answered, and the sample just grew — see below | Panel P5; the Phase 4 entry at `LEARNING_LOG.md:269-276` |
| 4 | Answered | Panel P8; the Phase 6 entry at `LEARNING_LOG.md:391-434` |
| 5 | Answered | Panel P7 (put-call parity); the parity and early-exercise entries |
| 6 | **Partly** — the φ(d₁) geometry is explained in the "Greeks" entry, but its consequence for inversion is only asserted | Needs measurement — Task 2 |
| 7 | **Fully answered** | The "Inverting the formula" entry: the real failure at a 1.4e-60 price, the `_MIN_VOL` collapse, and the one-line reorder that fixed it |
| 8 | **Essentially unanswered** — one passing mention in the whole log | Needs measurement — Task 1 |
| 9 | **Partly** — the early-exercise bound and the parity residue exist, but "how large" and "where it concentrates" are not quantified | Needs measurement — Task 3 |
| 10 | **Fully answered** | The Phase 6 entry: the gamma-P&L derivation, the exact P&L identity, and the discrete-hedging residual |

**Q3's sample changed on 2026-08-31 and the answer must use the new numbers.** `data/daily_metrics.parquet` now holds 64 sessions; `iv_rv_summary` reports **37 evaluable days** at a **81%** share, against the "100% of 1" the Phase 4 entry had. Re-derive it yourself rather than copying — see `src/analytics/iv_rv.py`.

**The archive.** 64 chain files: an isolated 2024-09-03 probe plus a continuous 2026-06-01 → 2026-08-28 run. Each carries `bid`, `ask`, `mid`, `close`, `volume`, `open_interest` per contract (`src/data/base.py:CHAIN_COLUMNS`). Most are `massive-backfill` (close-based, and their `bid`/`ask` may be absent or unreliable); the 2026-08-28 file is `yfinance` with real bid/ask on all 1,452 rows. **That single session is the only place a bid-vs-ask question can honestly be asked**, and Task 1 must check this rather than assume it.

**Existing probe pattern.** `scripts/verify_eodhd.py`, `verify_massive.py`, `verify_alphavantage.py` are committed, offline-ish, single-purpose scripts that print findings for a human to read and record in the log. Match their shape: a module docstring saying what question it answers, a `main()`, no side effects on `data/` or `docs/`.

---

## File structure

| File | Responsibility | Task |
|---|---|---|
| `scripts/probe_quote_iv_spread.py` (create) | Q8: implied vol solved from bid vs mid vs ask, across the chain | 1 |
| `scripts/probe_vega_inversion.py` (create) | Q6: vega's geometry and what it does to inversion reliability | 2 |
| `scripts/probe_american_error.py` (create) | Q9: size and location of the European-model approximation | 3 |
| `LEARNING_LOG.md` (modify) | the ten answers | 4 |
| `README.md` (modify) | Findings for the five public questions; phase tick | 5 |

---

### Task 1: Q8 — what a bid-ask spread is worth in vol points

**Files:**
- Create: `scripts/probe_quote_iv_spread.py`

**The question** (`LEARNING_LOG.md:22-23`): *"Does implied vol differ when solved from bid vs mid vs ask, and by how much across the chain?"* SPEC §2.2 stores all three columns specifically so this could be asked.

**Why it matters, and what makes it a real finding rather than an arithmetic exercise:** every implied vol on the dashboard is solved from one price. If the bid-to-ask width in *vol* terms is comparable to the effects the panels measure — the 25-delta skew is about 4 vol points, the IV-minus-RV spread about 1.4 — then some of what those panels show is quote noise rather than signal, and the honest reading of them changes. Vega is the conversion factor: `Δσ ≈ Δprice / vega`, so the spread should blow up exactly where vega is small, which ties this question to Q6.

- [ ] **Step 1: Establish which sessions can answer the question at all**

Before measuring anything, check the archive for usable bid/ask. Read every stored chain and report, per session: row count, source, how many rows have a positive bid *and* a non-null ask, and the median relative spread `(ask − bid) / mid`. Print the table.

Expect that only the `yfinance` session carries real two-sided quotes and the `massive-backfill` sessions do not. **If that is what you find, say so and scope the rest of the task to the sessions that qualify** — a single-session answer, clearly labelled, is the honest result here, and pretending otherwise would be the defect. If more sessions qualify than expected, use them all.

- [ ] **Step 2: Solve the three implied vols**

For every qualifying contract, solve implied vol three times with `src.models.black_scholes.implied_vol` — from `bid`, from `mid`, from `ask` — at that session's `r` and `q` (take them from `docs/status.json`). Keep the contract's `kind`, `strike`, `dte`, `mid`, `moneyness = strike / spot`, the three vols, and `vega` at the mid vol from `greeks`.

Handle non-convergence honestly: a bid below intrinsic has no implied vol, and that is a finding about the quote, not an error to suppress. Count and report how many contracts fail on each of the three prices.

- [ ] **Step 3: Report the distribution, sliced the ways that matter**

Print:
- the overall median and 90th-percentile `iv_ask − iv_bid` in **vol points**, over contracts where all three solved;
- the same, bucketed by moneyness (at least: deep OTM, near the money within ±2%, deep ITM) and by DTE;
- the same, expressed as a fraction of that contract's own mid IV, since a 2-point spread on a 12% vol is a very different claim from 2 points on a 60% vol;
- the relationship to vega: report the median vol spread for the lowest and highest vega deciles, which is the direct test of the `Δσ ≈ Δprice / vega` prediction;
- an explicit comparison against the two effects the dashboard reports — the ~4-point 25-delta skew and the ~1.4-point IV-minus-RV spread — so the answer can state whether those survive the quote noise.

- [ ] **Step 4: Run it and read the output**

Run: `.venv/bin/python scripts/probe_quote_iv_spread.py`

Record the numbers. They go into the Q8 answer in Task 4. If any result surprises you, say so in your report — a surprise here is the most valuable thing this task can produce.

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_quote_iv_spread.py
git commit -m "feat: probe implied vol solved from bid, mid and ask"
```

---

### Task 2: Q6 — vega's geometry, and what it does to inversion

**Files:**
- Create: `scripts/probe_vega_inversion.py`

**The question** (`LEARNING_LOG.md:18-20`): *"Why is vega largest at-the-money and near zero deep ITM/OTM — and what does that geometry do to anything that tries to invert price into vol?"*

The first half is already explained in the "The Greeks" log entry: vega and gamma share the φ(d₁) kernel, so both peak at the money and decay outward. The second half is asserted there and never measured. This task measures it, and the result connects directly to Q7's real Newton failure and to Q8's spread blow-up.

- [ ] **Step 1: Measure vega's shape on real quotes**

For the ~30-DTE expiry of the most recent stored chain, print vega (per vol point, i.e. raw vega / 100) against strike for calls and puts, and report where it peaks relative to spot and how far it has decayed at the moneyness limits the chain filter allows (`config.yaml`'s `moneyness_min`/`moneyness_max`).

- [ ] **Step 2: Measure what that does to inversion**

Two measurements, both on the real chain:

1. **Convergence against vega.** Bucket every contract in the session by vega decile and report the IV solver's convergence rate per bucket. If the failures concentrate in the low-vega buckets, that is the geometry's consequence, measured.
2. **Conditioning.** For each contract, compute how many vol points a *one-cent* price error moves the implied vol — `0.01 / vega_per_vol_point`. Report the median by moneyness bucket. This is the number that says why a deep-OTM quote cannot carry a trustworthy vol, in units anyone can check.

- [ ] **Step 3: Run it, record the numbers**

Run: `.venv/bin/python scripts/probe_vega_inversion.py`

- [ ] **Step 4: Commit**

```bash
git add scripts/probe_vega_inversion.py
git commit -m "feat: probe vega geometry against IV inversion reliability"
```

---

### Task 3: Q9 — how wrong is the European model on American options

**Files:**
- Create: `scripts/probe_american_error.py`

**The question** (`LEARNING_LOG.md:24-26`): *"How large is the error from pricing American-style SPY options with a European model (and continuous-yield q instead of discrete dividends), and where in the chain does it concentrate?"*

**Scope this honestly.** A rigorous answer prices the same contracts under an American model and differences them, which needs a binomial tree this project has not built. Writing one is permitted by SPEC §7 (it would be our own code, not a library) but it is a substantial build and is **out of scope for this phase**. What is in scope is to *bound and locate* the error using evidence the project already produces, and to state plainly what a bound is not.

- [ ] **Step 1: Locate the error using the parity residue**

`src/analytics/parity.py` already computes, per strike and expiry: the forward-calibrated parity deviation, the early-exercise bound `K(1 − e^{−rT}) − S(1 − e^{−qT})`, and which violations that bound explains. Run it over the most recent chain and report the deviation's distribution by moneyness and by DTE — in dollars *and*, dividing by vega, in vol points.

European put-call parity is model-free: it holds for European options by arbitrage alone. So a systematic deviation that grows exactly where early exercise is most valuable is the American premium showing itself, and its size is a *lower bound* on the pricing error there. Say both halves of that: what it demonstrates, and that a lower bound is not the error.

- [ ] **Step 2: Bound the premium from the theory side**

For every contract, compute the rigorous ceiling on an American put's early-exercise premium, `K(1 − e^{−rT})`, and the tighter bound the project actually uses. Report both by moneyness and DTE, in dollars and vol points. This brackets the answer from above.

- [ ] **Step 3: Isolate the continuous-q approximation**

The question has a second half. SPY pays discrete quarterly dividends; the model uses a continuous yield `q`. Using `data/underlying.parquet` and the dividend history the pipeline already fetches (see `src/data/eodhd_provider.py`), report how much of the ~30-DTE window typically contains an ex-dividend date, and price one representative contract both ways — continuous `q` versus a discrete dividend subtracted at its actual date — reporting the difference in dollars and vol points. If the dividend data needed is not available offline, say so and report what you could measure instead.

- [ ] **Step 4: Run it, record the numbers**

Run: `.venv/bin/python scripts/probe_american_error.py`

- [ ] **Step 5: Commit**

```bash
git add scripts/probe_american_error.py
git commit -m "feat: probe and bound the European-model approximation error"
```

---

### Task 4: The ten answers

**Files:**
- Modify: `LEARNING_LOG.md`

Add a new top-level section, `## The ten answers`, placed immediately after `## The ten questions` so a reader meets the answers where the questions were posed. One subsection per question, in order.

**Each answer must:**
- state the answer in its first sentence, before any qualification;
- cite its evidence by name — the panel, the log entry, the test, or the probe script and its output;
- give the number, with the sample it came from;
- say what it does not establish.

**Length is not the goal; sufficiency is.** Q7 and Q10 are already answered at length in their own entries — their subsections here should be short and point there rather than restating them. Q6, Q8 and Q9 carry the new measurements and will be longer.

**Specific care:**
- **Q3** must use the current numbers (37 evaluable days, 81%), not the Phase 4 entry's "100% of 1" — and should note explicitly that the earlier entry's caution about a one-point sample was right and has now been superseded by data, since that is a better lesson than either number alone.
- **Q4** must not claim the scatter shows a relationship. Two settled trades, and the page itself declines to claim the fit.
- **Q8** should state whether the measured spread is large or small *relative to the effects the dashboard reports*, because that is what determines whether the panels' findings survive it.
- **Q9** must distinguish a bound from a measurement.

- [ ] **Step 1: Write the section**

- [ ] **Step 2: Check every citation resolves**

For each answer, confirm the panel exists, the log line says what you claim, the test name is real, and any number matches its probe's output. A dead citation in the section whose whole purpose is evidence is the defect this task most needs to avoid.

- [ ] **Step 3: Full suite, then commit**

Run: `.venv/bin/python -m pytest -q` — Expected: PASS (432)

```bash
git add LEARNING_LOG.md
git commit -m "docs: answer the ten learning-log questions with the project's own evidence"
```

---

### Task 5: README Findings, and closing the phase

**Files:**
- Modify: `README.md`

`README.md:123-126` currently holds a placeholder: *"Filled in as evidence accumulates — one subsection per guiding question, each answered with this dashboard's own charts."* Replace it with the real thing.

- [ ] **Step 1: Write Findings for the five public questions**

Five short subsections, one per guiding question, written for **a reader who has not read the learning log and will not** — a recruiter, a classmate, someone who found the repo. Each: the answer in plain language, the number that supports it, and a pointer to the panel on the live dashboard that shows it. Two to four sentences each. This is the most-read prose in the repository; the learning log is where the depth lives, and this section should link to it rather than compress it.

Keep the README's existing voice. Do not restructure sections other than Findings and the Status list.

- [ ] **Step 2: Check the claims against the site**

Every number in Findings must match what the live dashboard actually shows, or say which session it is from. Note that the published page may lag the repository — `docs/status.json` in the repo is the authority for what was last rendered.

- [ ] **Step 3: Tick Phase 8 in the Status list**

Match how Phases 2–7 are written. Note that Phase 7's line was corrected earlier to say its exit criterion is still accruing — do not undo that.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: write the Findings section for the five guiding questions"
```

---

## Self-review

**Spec coverage:**

| Spec clause | Task |
|---|---|
| §5 row 8: answer all ten learning-log questions | 4, resting on 1–3 for the three that lacked evidence |
| §5 row 8: with links to own panels | 4 (citation requirement), 5 (dashboard pointers) |
| §5 row 8: README "Findings" section for the five public ones | 5 |
| §5 row 8 exit criterion: every question has an evidence-backed written answer | 4, Step 2's citation check |
| §7: no pricing libraries | Global constraint; Task 3 explicitly declines to build or import an American pricer and says what that costs |

**Deliberately not done:** a binomial American pricer, which would convert Q9's bound into a measurement. It is permitted by SPEC §7 and it is a real build; this phase bounds the answer instead and says so. Recorded here so the choice is visible rather than silent.

**Placeholder scan:** Tasks 1–3 specify what to measure and how to slice it, but not the numbers — those are outputs, and pre-writing them would be fabrication. Tasks 4–5 specify the structure and the constraints on each answer rather than the prose, which is the correct level for a writing task.

**Type consistency:** the three probes are standalone scripts sharing no interface; they consume `src.models.black_scholes.implied_vol`/`greeks`, `src.analytics.chain_iv.compute_chain_iv`, `src.analytics.parity.compute_parity`, and the stored parquet files, all of which exist unchanged.
