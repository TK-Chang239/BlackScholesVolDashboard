# SPEC — vol-lens: A Daily Black-Scholes Reality-Check Dashboard

> Working name `vol-lens` — rename freely. This spec is the build contract:
> what gets built, how, in what order, and what "done" means for each piece.
> The README carries the public story; this file carries the engineering.

---

## 1. Purpose

Build an automated dashboard that (a) prices options with a from-scratch
Black-Scholes implementation, (b) compares those model outputs against live
US options market data every trading day, and (c) uses the daily divergence
between model and market as the raw material for answering five guiding
questions about how useful Black-Scholes actually is in the real world
(see README).

Dual goal, explicitly ordered (revised at spec review, 2026-08-27):

1. **Resume**: a permanently-live, daily-updating public artifact with a
   commit history proving it runs, suitable for interviews for
   markets/S&T and finance roles.
2. **Learning**: the author learns Black-Scholes *from the finished
   artifact*. The code is built by an AI pair (Claude Code); the
   "hand-built" rule still binds the codebase (no pricing libraries —
   every formula implemented from scratch in this repo), and
   `LEARNING_LOG.md` is written as a **teaching document**: every formula
   explained, every numerical failure documented, so the author can study
   it to meet the §9 interview bar.

### Non-goals (v1)

- No intraday/streaming data — end-of-day snapshots only.
- No exotic option pricing (American early exercise treated as a documented
  approximation issue, not solved).
- No stochastic-vol models (Heston/SABR). We diagnose *why* they exist;
  we don't build them.
- No trading, no broker connectivity, no investment advice. Simulated P&L only.
- One underlying done deeply (SPY). TXO (Taiwan index options) is a
  stretch-goal comparison panel, not a dependency.

---

## 2. Architecture

```
┌─────────────────────────── GitHub Actions (cron, daily) ───────────────────────────┐
│                                                                                    │
│  fetch (EODHD API) ──► store (data/ in repo) ──► compute (src/) ──► render (docs/) │
│                                                                                    │
└──────────────────────────────────────┬─────────────────────────────────────────────┘
                                       │ commit + push
                                       ▼
                          GitHub Pages serves docs/index.html
```

Four stages, strictly separated so each is testable alone:

| Stage   | Module            | Responsibility                                                    |
|---------|-------------------|-------------------------------------------------------------------|
| Fetch   | `src/data/`       | Pull option chain + underlying EOD data from EODHD; normalize     |
| Store   | `data/`           | Append today's filtered snapshot to on-repo history (Parquet/CSV) |
| Compute | `src/models/`, `src/analytics/` | BS pricer, Greeks, IV solver, realized vol, panel metrics |
| Render  | `src/render/`     | Build one self-contained static `docs/index.html` with Plotly     |

### 2.1 Automation & hosting

- **Scheduler**: GitHub Actions workflow, cron `30 1 * * 2-6` UTC
  (≈ 21:30 ET, after US close and EOD data settlement; Tue–Sat UTC covers
  Mon–Fri US sessions). Also `workflow_dispatch` for manual runs.
- **Publishing**: the workflow commits updated `data/` and `docs/` back to
  `main`; GitHub Pages serves the `docs/` folder. No servers, no cost.
- **Secrets**: `EODHD_API_TOKEN` stored as a GitHub Actions secret. Never
  in code, never in committed data.
- **Failure policy**: if the fetch fails (API down, holiday, rate limit),
  the job exits gracefully *without* committing, leaving yesterday's
  dashboard live. A `status.json` (last successful run, row counts) is
  rendered into the page footer so staleness is visible, not silent.

### 2.2 Data layer (`src/data/`)

**Primary source: EODHD.** Endpoints used:

- US options EOD: contracts + end-of-day quotes for SPY
  (EODHD US Options API — verify plan/marketplace access in Phase 0;
  confirm available fields: bid, ask, last, volume, open interest, and
  whether vendor-computed IV/Greeks are present — if present, they are
  stored for cross-checking but **never** used in our computations).
- Underlying EOD history: SPY adjusted close (for spot + realized vol).
- Risk-free rate: 3-month US T-bill (EODHD bill-rate endpoint or a static
  monthly-updated value in `config.yaml` as fallback — precision here
  barely matters, and *demonstrating that* is part of the sensitivity panel).
- Dividend yield `q`: trailing 12-month sum of SPY distributions ÷ spot,
  computed daily from EODHD's dividends endpoint, with a static
  `config.yaml` fallback (mirroring the risk-free-rate pattern). For SPY,
  q ≈ 1.2–1.4% and is material to parity and deep-ITM pricing. The
  discrete-dividend-reality vs continuous-yield-model gap is itself a
  model-vs-reality point worth a caption.

**Abstraction**: a single `DataProvider` interface
(`get_option_chain(symbol, date)`, `get_underlying_history(symbol)`,
`get_risk_free_rate()`), with `EODHDProvider` as the implementation and
room for a `YFinanceProvider` fallback if plan limits bite. Nothing outside
`src/data/` may know which vendor is in use.

**Chain filtering** (keeps the repo small and the analysis honest):

- Expiries: nearest ~6 monthly expiries within 7–365 DTE.
- Strikes: 0.70–1.30 moneyness (K/S).
- Liquidity floor: drop quotes with bid = 0 or missing ask.
- Store bid, ask, and mid — the IV solver runs on all three at least once
  (guiding question Q-private-8), mid by default.

**Storage format**: one Parquet file per day in `data/chains/YYYY-MM-DD.parquet`
(~tens of KB filtered), plus rolling `data/underlying.parquet` and derived
`data/daily_metrics.parquet` (one row per day: ATM IV by expiry, skew metric,
trailing realized vol, forward realized vol back-filled as days mature —
see §3 P5). Time-series panels read `daily_metrics`; the hedge sim (§3 P8)
instead replays raw chain files, so chains it needs are the one exception
to the "old chains are prunable" rule.

**Cold-start / backfill**: time-series panels (IV vs RV, skew history,
hedge P&L) need history. Phase 2 includes a one-time backfill script pulling
as much historical options EOD as the EODHD plan allows (target ≥ 6 months;
if unavailable, panels state "accumulating since <date>" and grow live —
which is itself honest and fine).

### 2.3 Model layer (`src/models/`) — hand-built, the heart of the project

`black_scholes.py` — no pricing libraries. NumPy + `scipy.stats.norm.cdf` only.

**API shape**: plain pure functions (no classes), every argument accepting
NumPy arrays with broadcasting — one call prices the entire chain.

- `bs_price(S, K, T, r, sigma, q, kind)` — European call/put with continuous
  dividend yield q:
  - `d1 = (ln(S/K) + (r − q + σ²/2)T) / (σ√T)`, `d2 = d1 − σ√T`
  - call `= S·e^{−qT}·N(d1) − K·e^{−rT}·N(d2)`; put via the mirror formula.
  - **Edge contract**: as `T → 0` or `σ → 0`, price converges to discounted
    intrinsic on the *forward*, `max(S·e^{−qT} − K·e^{−rT}, 0)` for calls
    (mirror for puts); `σ√T` guarded against underflow. Invalid inputs
    (negative T or σ, nonpositive prices) return NaN — never raise
    mid-pipeline; NaN counts surface in `status.json`.
- `greeks(...)` — closed-form delta, gamma, vega, theta, rho. Vectorized
  over strikes (NumPy broadcasting — no Python loops over the chain).
  **Convention**: the model layer returns raw calculus derivatives (theta
  per year, vega per unit vol, rho per unit rate) so unit tests match
  textbook values exactly; the render layer owns display scaling
  (theta/day ÷365, vega per vol point ÷100, rho per 1% ÷100).
- `implied_vol(price, S, K, T, r, q, kind)` — algorithm, in order:
  1. No-arbitrage pre-check: price must lie in
     `[discounted forward intrinsic, S·e^{−qT}]` for calls (mirror for
     puts); violation → NaN immediately, no iteration.
  2. Initial guess: Brenner–Subrahmanyam `σ₀ ≈ √(2π/T)·(price/S)`,
     clamped into [1e-4, 5.0].
  3. Newton–Raphson using our own vega, tolerance 1e-8 in price,
     max ~20 iterations, run vectorized across the whole chain at once.
  4. Fallback to `scipy.optimize.brentq` on [1e-4, 5.0] (scalar loop over
     the stragglers) when Newton fails to converge, steps out of bounds,
     or vega < ~1e-10 — deep OTM options with tiny vega WILL break naive
     Newton; that failure and its fix go in the learning log.
  Returns NaN, never garbage, when price violates no-arbitrage bounds.
- SPY note: options are American-style; we treat them as European and
  *document* the approximation (small for SPY calls given dividend
  handling; larger for deep ITM puts) — this is a feature of the "model
  vs reality" story, not a bug to hide.

`realized_vol.py` — close-to-close log-return realized vol on **adjusted**
close (unadjusted closes would book dividends as fake volatility),
annualized (√252), windows 20d (primary) and 60d. Fancier estimators
(Parkinson, Yang-Zhang) are out of scope per §1.

### 2.4 Analytics layer (`src/analytics/`)

Panel-specific computations, each a pure function
`(chain, history, config) -> tidy DataFrame`, individually unit-testable.
Detailed per-panel specs in §3.

### 2.5 Render layer (`src/render/`)

- Plotly figures serialized into **one self-contained `docs/index.html`**
  (decide bundling in Phase 3; keep page < 5 MB. Leading candidate:
  Plotly's cartesian-only *partial bundle* (~1 MB) inlined — truly
  self-contained, covers every chart type the nine panels need. Full
  plotly.js is ~3.5 MB and a CDN page would not be self-contained.).
- Layout: header (title, underlying, spot, last-updated) → the five guiding
  questions as section headings → panels grouped under the question they
  evidence → footer (status.json, disclaimer, GitHub link).
- Every panel carries a 2–3 sentence caption: what it shows and which
  guiding question it speaks to. The captions ARE the interview prep.
- Must be readable on mobile (interviewers open links on phones).

---

## 3. Panels

Nine panels (the four "core" + four "differentiator" panels chosen at
scoping, plus the P1 sensitivity panel added to evidence Q1). Each spec:
computation → output → acceptance criteria.
"Q1–Q5" reference the README's guiding questions.

### P1. Input sensitivity — *evidences Q1*

- **Compute**: for the ~30-DTE ATM call, price under ±20% perturbation of
  each input (σ, S, r, q, T) holding others fixed; express as % price change.
- **Output**: tornado chart, one bar pair per input.
- **Accept**: σ bar visibly dominates; r bar nearly invisible. Caption states
  the point: vol is the only input worth arguing about.

### P2. IV smile / skew by expiry — *evidences Q2* (centerpiece)

- **Compute**: our IV solver on mid prices for 3 selected expiries
  (~30, ~90, ~180 DTE); x-axis moneyness or delta.
- **Output**: one line per expiry; ATM marker; optional bid/ask IV band toggle.
- **Accept**: solver converges for ≥ 95% of liquid quotes; puts show the
  classic downward skew; curves smooth (no zigzag artifacts from bad quotes).

### P3. ATM IV term structure — *evidences Q2*

- **Compute**: interpolated ATM IV per stored expiry (linear in strike
  around spot), plotted vs DTE; overlay yesterday's curve in grey.
- **Output**: line chart with day-over-day shift visible.
- **Accept**: kinks around known event dates (FOMC etc.) are visible when
  present; caption teaches the reader to look for them.

### P4. Greeks panel — *evidences Q1/Q4 mechanics*

- **Compute**: delta/gamma/vega/theta for the ~30-DTE ATM call & put, with
  1-day change; plus small delta-vs-strike and gamma-vs-strike curves.
- **Output**: stat tiles + two mini-charts.
- **Accept**: values match closed-form unit tests; call delta ≈ 0.5 ATM;
  gamma peaks ATM.

### P5. Implied vs realized volatility — *evidences Q3* (intellectual centerpiece)

- **Compute**: daily time series of ~30-DTE ATM IV vs 20-day *trailing*
  realized vol; spread (IV − RV) shaded; running mean of spread. **Plus a
  forward-RV series**: for each day t at least ~30 calendar days old, the
  vol actually realized over [t, t+30d] (one extra column in
  `daily_metrics`, back-filled as days mature). Trailing RV is the live
  visual; forward RV is the honest forecast evaluation — after a vol
  spike, trailing RV makes IV look "too low" when it wasn't a bad
  forecast.
- **Output**: chart with IV, trailing RV, and the lagged forward-RV series
  + shaded spread; summary stat computed against **forward** RV: "IV has
  exceeded subsequently-realized vol on X% of evaluable days."
- **Accept**: with ≥ 3 months of history, the positive-spread regime is
  visible in the forward comparison; caption names it: the variance risk
  premium.

### P6. Skew metric time series — *evidences Q2/Q5-discussion*

- **Compute**: daily 25-delta put IV − 25-delta call IV (interpolate in
  delta space) at ~30 DTE. One number per day, appended to `daily_metrics`.
- **Output**: time series with annotations on spike days (manual annotation
  file `data/annotations.yaml` — adding a news note is a 1-line commit).
- **Accept**: metric is stable day-to-day in calm markets (< a few vol
  points of noise); spikes align with real risk-off days.

### P7. Put-call parity checker — *evidences Q5*

- **Compute**: for each strike/expiry pair: `C − P` vs `S·e^{−qT} − K·e^{−rT}`;
  violation = deviation beyond the combined bid-ask spread (i.e. only
  *tradeable* violations count). Also: back out the implied carry
  (r − q) that makes parity hold exactly ATM — reverse-engineering the
  market's funding assumption (stretch, Phase 5).
- **Output**: heatmap of deviations (strike × expiry) with the
  within-spread region visually muted; count of tradeable violations
  (expected: ~zero).
- **Accept**: near-zero tradeable violations on liquid strikes; caption
  makes the model-free vs model-dependent distinction. Caption must also
  attribute deep-ITM put "violations" to American early-exercise premium
  (European parity is only approximate for American options) rather than
  calling them arbitrage.

### P8. Delta-hedged P&L simulation — *evidences Q4* (empirical climax)

- **Setup**: on the first trading day each month, "sell" one ~30-DTE ATM
  straddle at mid; record entry IV.
- **Daily**: re-hedge to delta-neutral using **our own** BS deltas at
  current market IV; accrue hedge position P&L from underlying moves;
  mark the short straddle to market (mid). Cash earns r. No transaction
  costs in v1 (v2 toggle: X bps per hedge trade).
- **At expiry**: settle, record round-trip P&L and the trade's entry IV vs
  realized vol over its life.
- **Output**: (a) cumulative P&L across all monthly trades; (b) scatter of
  per-trade P&L vs (entry IV − lifetime RV) — the money chart: the
  regression line through it is the whole point of the panel;
  (c) daily P&L distribution histogram.
- **Accept**: scatter shows clear positive relationship between P&L and
  (IV − RV); caption states the gamma-P&L result — daily hedged P&L
  ≈ ½ Γ S² (σ²_realized − σ²_implied) dt, with vol-mark-to-market noise
  around it — and flags residual noise as discrete-hedging error.
- **Design**: the simulation is **stateless** — recomputed in full from
  stored chain history (`data/chains/`) on every run, never accruing
  marks day by day. The day Phase 6 lands, it replays every monthly trade
  the stored/backfilled history supports, so the scatter starts populated;
  any later bug fix retroactively heals the whole P&L history, and there
  is no sim state to corrupt. This removes the need to start Phase 6
  early (see §8).
- **Note**: this panel is a *simulation for learning*, labeled as such
  on the page. It is not tracking real positions or advice.

### P9. Model-vs-market price heatmap — *evidences Q2*

- **Compute**: BS price for every stored contract using a single flat vol
  (today's ~30-DTE ATM IV) vs market mid; deviation in % of mid (or in vol
  points, toggle), on a strike × expiry grid, calls and puts separately.
- **Output**: heatmap; within-bid-ask deviations visually muted (consistent
  with P7's convention).
- **Accept**: the mispricing pattern is visibly *structured* — wings
  systematically rich vs the flat-vol model, stable across days — not
  random noise. Caption: structured error means a missing feature of the
  world (fat tails, stochastic vol), which is why models like Heston/SABR
  exist; this panel is the smile of P2 re-expressed in price space.

### P10 (stretch). TXO comparison

Static or weekly-refreshed TXO smile beside SPY's, if a workable data
path exists (EODHD TW coverage or manual TAIFEX download). Talking-point
hook for Taiwan-desk conversations. Explicitly allowed to be cut.

---

## 4. Testing

- **Unit tests (pytest)**, written in Phase 1 *with* the pricer, not after:
  - Prices vs published textbook values (e.g. Hull examples) to 4 decimals.
  - Put-call parity as an internal identity test of the pricer (machine ε).
  - Greeks vs finite-difference bumps of our own price function.
  - IV round-trip: `implied_vol(bs_price(σ)) == σ` to 1e-6 across the
    normal strike/tenor grid. Near-degenerate corners (T→0, deep OTM,
    vega underflow) assert the *contract* instead: recover σ to 1e-6
    **or** return NaN — never silently return a wrong number (1e-6
    recovery is mathematically unachievable where the price carries no
    vol information).
  - Boundary behavior: σ→0 gives intrinsic (discounted-forward) value;
    price within no-arbitrage bounds everywhere.
- **Pipeline test**: full run against a frozen fixture chain committed in
  `tests/fixtures/`, so CI can run without network or API key.
- **CI**: tests run on every push (separate lightweight workflow from the
  daily cron); the daily job also runs tests before rendering — a broken
  model never publishes.

---

## 5. Build phases

Each phase is shippable and its exit criterion is checkable. **Learning
rule** (revised at spec review): `LEARNING_LOG.md` is written *by the AI
pair as a teaching document*, in the same phase as the code it explains —
every Phase 1/3 formula derived and explained in it, every numerical
failure encountered documented. The author studies it to meet the §9 bar;
no code is gated on prior hand-derivation.

| Phase | Deliverable | Exit criterion |
|-------|-------------|----------------|
| 0 | Repo scaffold: layout below, `config.yaml`, README, this spec, learning log with the 10 questions; **verify EODHD options-data access for SPY** (fields, history depth, rate limits, EOD-data settlement time vs the cron hour, and that a plain REST API token exists for the Actions runner) | One manual script pulls and prints today's SPY chain |
| 1 | `black_scholes.py` + Greeks + full unit-test suite | All §4 model tests green |
| 2 | Data layer + storage + backfill script | ≥ 1 real daily snapshot stored; backfill run attempted and depth documented |
| 3 | IV solver → P2 smile + P3 term structure rendered to HTML | Convergence ≥ 95% on liquid quotes; first `index.html` published to Pages |
| 4 | Realized vol → P5 IV-vs-RV; P1 sensitivity; P4 Greeks panel | Panels render from real stored history |
| 5 | P6 skew series, P7 parity checker (+ implied-carry stretch), P9 heatmap | Daily metrics appending correctly; tradeable-violation count ≈ 0 |
| 6 | P8 hedge simulation | First simulated trade running mark-to-market daily; scatter renders (sparse is fine) |
| 7 | GitHub Actions automation + Pages polish + captions + mobile pass | Five consecutive unattended green daily runs |
| 8 | Reflection: answer all 10 learning-log questions with links to own panels; README "Findings" section for the 5 public ones | Every question has an evidence-backed written answer |

Sequencing rationale: the model (Phase 1) precedes any market data use, so
learning happens against clean known-answer tests before real-world mess
arrives. Automation (Phase 7) is late deliberately — run the pipeline
manually for a while first; babysitting it is how you find the failure modes
the Action must handle.

## 6. Repository layout

```
vol-lens/
├── README.md               # public story + 5 guiding questions + findings
├── SPEC.md                 # this file
├── LEARNING_LOG.md         # all 10 questions; dated entries; answered in Phase 8
├── config.yaml             # symbol, filters, expiries, windows, sim params
├── data/                   # committed daily snapshots + daily_metrics + annotations.yaml
├── docs/                   # GitHub Pages root (index.html, status.json)
├── src/
│   ├── data/               # DataProvider interface, EODHDProvider, backfill
│   ├── models/             # black_scholes.py, realized_vol.py  ← hand-built core
│   ├── analytics/          # one module per panel (P1–P8)
│   ├── render/             # figure builders, page assembly
│   └── run_daily.py        # orchestrator: fetch → store → compute → render
├── tests/                  # unit + fixture pipeline tests
└── .github/workflows/      # daily.yml (cron), ci.yml (tests on push)
```

## 7. Tech stack

Python 3.12 · NumPy · SciPy (`stats.norm`, `optimize.brentq`) · pandas ·
Plotly · pyarrow (Parquet) · requests · pytest · GitHub Actions · GitHub
Pages. **Deliberately excluded**: any options-pricing library (QuantLib,
py_vollib) — allowed later *only* to cross-check answers, never to produce
them; heavy frameworks (no Streamlit/Dash — static HTML is the point).

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| EODHD plan lacks options data / thin history | Phase 0 verifies before anything is built on it; `DataProvider` abstraction keeps a yfinance fallback one class away |
| Time-series panels look empty for weeks | Backfill (Phase 2); "accumulating since" labels; hedge sim is a stateless replay over stored chain history (§3 P8), so its scatter starts as populated as the chain archive allows — no early start needed |
| IV solver instability on junk quotes | Liquidity filters + no-arb bound checks + Brent safeguard; failures logged, counted in status.json, and written up |
| Silent staleness (Action dies quietly) | Visible last-updated stamp + status.json; failed runs don't commit |
| Repo bloat from daily data | Filtered chains, Parquet, prunable design (§2.2) |
| Scope creep (the real one) | Non-goals list §1; P10 explicitly cuttable; v2 ideas go to a `FUTURE.md`, not the codebase |

## 9. Definition of done (v1)

- Dashboard publicly live on GitHub Pages, all 9 panels rendering, updated
  automatically every US trading day for ≥ 2 unattended weeks.
- Test suite green; model layer 100% hand-built and test-covered.
- All 10 learning-log questions answered in writing with evidence from the
  dashboard's own panels; README Findings section complete for the 5
  public questions.
- The author can explain every formula, every panel, and every failure
  encountered — from memory, in an interview.
