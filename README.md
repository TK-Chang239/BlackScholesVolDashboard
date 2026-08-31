# vol-lens

**A daily reality-check of the Black-Scholes model against the live US options market.**

Every trading day, an automated pipeline pulls the end-of-day SPY option
chain, prices every contract with a Black-Scholes implementation written
from scratch, backs out implied volatilities, and publishes a static
dashboard comparing what the model says to what the market does. The gap
between the two — measured daily, in nine panels — is the subject of this
project.

**Live dashboard:** https://tk-chang239.github.io/BlackScholesVolDashboard/ ·
Updated automatically after each US close via GitHub Actions.

---

## Why this project exists

Black-Scholes is the most famous formula in finance, and everyone has it.
That raises an obvious puzzle: if the formula is public, why does anyone
disagree about option prices? And if its assumptions are known to be false,
why does every trading desk still use it every day?

I built this project to learn Black-Scholes properly — by implementing it,
not reading about it — and then to hold it up against real market data long
enough to answer that puzzle with evidence I generated myself. The model
layer uses no pricing libraries: every price, Greek, and implied vol on the
dashboard comes from code in [`src/models/`](src/models/), unit-tested
against textbook values and no-arbitrage identities.

## Five guiding questions

These were written down **before** the first line of code. Each panel on
the dashboard exists to put evidence behind one of them; the Findings
section below is filled in as the data accumulates.

**1. Everyone has the same formula. Why does anyone disagree about prices?**
Because the formula needs five inputs, and one of them — future volatility
— is not a fact anyone can look up. Everyone plugs in their own guess, so
everyone gets a different price. The disagreement is never about the math;
it is about the volatility number you feed it. *Shown by: the
input-sensitivity panel — volatility moves the price far more than any
other input.*

**2. Where does the market ignore the model, and why?**
Black-Scholes says one volatility number should price every option on the
same stock. The market instead uses a different volatility for every strike
and every expiry date. This project measures that pattern daily and asks
what the market knows that the model doesn't. *Shown by: the smile/skew
panel, the term-structure panel, and the model-vs-market price heatmap.*

**3. Is implied volatility a prediction, or a price?**
If implied volatility were just the market's prediction of future
volatility, it would be right on average. Comparing it every day against
the volatility that actually happens shows it usually runs too high — and
the interesting question is why sellers get paid that extra margin. *Shown
by: the implied-vs-realized panel and the hedge simulation's per-trade
P&L.*

**4. The model claims you can hedge an option perfectly. Can you?**
Only if you trade continuously, which nobody can. This project simulates
the real version: sell an option, re-hedge once a day using this project's
own deltas, and record the profit or loss. How far that P&L is from "riskless"
— and what drives it — is the most honest test of the model there is.
*Shown by: the delta-hedged P&L simulation panel.*

**5. Given all these flaws, why does every trading desk still use Black-Scholes?**
Two reasons this project checks directly. First, some of its relationships
(like put-call parity) don't depend on the model's assumptions at all, and
they hold almost perfectly in real data. Second, even where the model is
wrong, it works as a shared language: traders quote volatility to each
other instead of dollar prices, using the formula to translate. *Shown by:
the put-call parity checker.*

A longer private list of ten questions lives in
[`LEARNING_LOG.md`](LEARNING_LOG.md), answered as an end-of-project
reflection.

## What the dashboard shows

| Panel | What it is |
|-------|------------|
| Input sensitivity | Which Black-Scholes input actually moves the price (tornado chart) |
| IV smile / skew | Implied vol across strikes for three expiries, solved daily from market mids |
| ATM term structure | ATM implied vol vs time-to-expiry, with day-over-day shift |
| Greeks | Delta, gamma, vega, theta for the ~30-day ATM contracts, with daily changes |
| Implied vs realized vol | ~30-day ATM IV vs 20-day realized vol — the variance risk premium, live |
| Skew time series | Daily 25-delta put-call IV spread — the market's price of crash protection |
| Model-vs-market heatmap | Where a flat-vol Black-Scholes price diverges from market prices, across the whole chain |
| Put-call parity checker | Model-free arbitrage relationships, tested against actual bid-ask spreads |
| Delta-hedged P&L sim | A monthly simulated short straddle, re-hedged daily with this project's own deltas |

## How it works

```
GitHub Actions (daily cron, after US close)
  → fetch SPY option chain + underlying EOD data (EODHD API)
  → append filtered snapshot to on-repo history (Parquet)
  → compute: hand-built BS pricer / Greeks / IV solver / realized vol / panel metrics
  → render one static, self-contained HTML page
  → commit + publish via GitHub Pages
```

No servers, no database, no frameworks: Python (NumPy, SciPy, pandas),
Plotly for charts, GitHub Actions for scheduling, GitHub Pages for hosting.
Failed runs never publish — the model test suite gates every deploy, and
the page footer always shows when data was last refreshed. Full
architecture, per-panel acceptance criteria, and the phased build plan are
in [`SPEC.md`](SPEC.md).

## Status

- [x] Phase 0 — repo scaffold, data-access verification
- [x] Phase 1 — Black-Scholes pricer + Greeks, fully unit-tested
- [x] Phase 2 — data pipeline + history backfill
- [x] Phase 3 — IV solver; smile + term-structure panels live on Pages
- [x] Phase 4 — implied-vs-realized, sensitivity, Greeks panels
- [x] Phase 5 — skew series + parity checker
- [x] Phase 6 — delta-hedged P&L simulation
- [ ] Phase 7 — full daily automation, five unattended green runs
- [ ] Phase 8 — findings written up; all ten learning-log questions answered

## Findings

*Filled in as evidence accumulates — one subsection per guiding question,
each answered with this dashboard's own charts.*

## Disclaimers

Educational project. The hedging "P&L" is a simulation against end-of-day
mid prices — no real positions, no transaction costs in v1, and nothing
here is investment advice. SPY options are American-style and are priced
here with a European model; the size and location of that approximation
error is discussed on the dashboard rather than hidden.

## Author

TK — Finance & Informatics, University of Washington '27. Built to learn
Black-Scholes from first principles and to find out where it stops working.
