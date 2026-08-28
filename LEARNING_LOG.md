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

### 2026-08-28 — Verification run (`scripts/verify_eodhd.py`)

**Account/token.** `GET /api/user` → HTTP 200. The token is valid and the
account is on a paid monthly subscription (`subscriptionType: "monthly"`,
`subscriptionMode: "paid"`). Rate-limit fields present: `dailyRateLimit:
100000`, `extraLimit: 500`, `apiRequests: 252` used so far that day. No
per-minute limit is exposed in this payload, and the response contains no
field indicating which EODHD add-ons/marketplace products the account is
entitled to — that had to be discovered empirically via the options probes
below.

**Underlying EOD (`eod/SPY.US`).** HTTP 200, 22 rows for a 30-day window.
Fields: `date, open, high, low, close, adjusted_close, volume` — everything
needed for spot and realized vol.

**Dividends (`div/SPY.US`).** HTTP 200, 4 rows over the trailing ~13 months.
Fields: `date, declarationDate, recordDate, paymentDate, period, value,
unadjustedValue, currency`. Quarterly SPY distributions in the $1.80–$2.00
range, consistent with the SPEC §2.2 estimate of q ≈ 1.2–1.4% once summed
trailing-12m and divided by spot.

**Options data — both probes failed; no options endpoint returns 200 on
this account.**

- `mp/unicornbay/options/eod` (filter `underlying_symbol=SPY`) → **HTTP
  403 Forbidden**. Body is an HTML error page (title "Forbidden"), not
  JSON — this is a hard access-denial, not a malformed-request error.
  Consistent with the account not being subscribed to the UnicornBay
  options marketplace add-on.
- `options/SPY.US` (legacy endpoint) → **HTTP 404**, body `"Ticker Not
  Found."` — distinct failure mode from the 403 above; either the legacy
  options endpoint has been fully retired for new/this-tier accounts, or
  it never recognized the `SPY.US` ticker format. Either way it is not a
  usable path.
- History-depth probe (`mp/unicornbay/options/eod`, `date_from`/`date_to`
  ~180 days back) → **HTTP 403 Forbidden**, same error page as the current
  probe. This confirms the failure is account/entitlement-based, not
  date-range-based — going further back does not unlock the endpoint.

Because no options endpoint ever returned data, the remaining items the
script's closing message asks for are **not observable from this
account/plan**: no field inventory (bid/ask/last/volume/open_interest/
expiry/strike/type), no vendor IV/Greeks presence, no options history
depth, and no read on when today's options rows would settle relative to
the `30 1 * * 2-6` UTC cron — there is no options "today" to check against.

**Risk-free rate.** `eod/US3M.INDX` → HTTP 200 on the first candidate
(`US3M.GBOND` was not needed), 11 rows over 14 days, `close` around 3.8%.
This is a viable risk-free-rate source independent of the options-access
question.

**Bottom line / open decision for the controller (SPEC §8).** EOD
underlying, dividends, and a risk-free-rate proxy are all confirmed
working on this plan. SPY options data is not reachable through either
endpoint this script tried, under the current subscription. Per SPEC §8
this is exactly the scenario the `DataProvider` abstraction was built to
absorb (`YFinanceProvider` fallback "one class away"), but before Phase 2
picks a path someone needs to decide between: (a) upgrading the EODHD
subscription to include the UnicornBay options marketplace add-on and
re-running this script to confirm 200s and inspect real fields/history
depth, or (b) building `EODHDProvider` for underlying/dividends/rate only
and pairing it with a `YFinanceProvider` (or similar) for the options leg.
This script and this record are reusable either way — rerun it after any
plan change to fill in the field inventory and history-depth sections
above.

### 2026-08-28 — Addendum: Alpha Vantage probe (`scripts/verify_alphavantage.py`)

Alpha Vantage was tried next on the belief that `HISTORICAL_OPTIONS` was
free-tier. It is not anymore: all four probes (latest session, ~6 months
back, ~5 years back, holiday) returned HTTP 200 with **no data** and an
`Information` body — two explicitly saying "This is a premium endpoint,"
the other two showing the free-key throttle message (25 requests/day,
1/second). Note the vendor pattern: Alpha Vantage signals both errors and
paywalls *inside* a 200 response, so any provider built on it must parse
the body, never trust the status code.

### 2026-08-28 — Addendum: Massive (ex-Polygon) probe (`scripts/verify_massive.py`)

The old `api.polygon.io` domain still answers post-rebrand. Free-tier
findings: reference contracts ✅ (strikes/expiries/tickers, paginated);
**chain snapshot ❌ 403 `NOT_AUTHORIZED`** ("You are not entitled to this
data. Please upgrade your plan at https://massive.com/pricing");
per-contract previous-day aggregates ✅ (`o,h,l,c,v,vw,n` — OHLC only,
**no bid/ask**); historical aggregate ranges authorized. Consequences:
the free tier's only daily-fetch path is one call per contract (~400–700
contracts ≈ 80–140 min/day at 5 calls/min) and yields no bid/ask —
which structurally breaks P7's beyond-the-spread violation test and
question 8. Verdict: free tier insufficient for this spec; the Options
Starter tier plausibly unlocks the snapshot (with bid/ask, IV, OI) and
~2 years of history — must be re-verified with this same script after
any upgrade before Phase 2 builds on it.

**Post-upgrade re-verification (same day, Options Starter):** chain
snapshot flipped 403→200 (250 rows/page, `day` OHLC + `open_interest`;
vendor `greeks` and `implied_volatility` populated on liquid ATM strikes —
stored for cross-checks only, per §2.2). **`last_quote` (bid/ask) is
still absent even on a 17,974-OI ATM contract — quotes are a
higher-tier entitlement, not part of Starter.** Backfill path verified
end-to-end: expired-contract reference (`expired=true`) works and
historical daily aggregates on a September-2025 contract returned real
rows a year back. Decision: **hybrid** — yfinance serves the daily live
chain (real bid/ask → P7's beyond-the-spread test and Q8 keep full
fidelity), Massive Starter serves the one-time backfill (close-based
IVs, labeled as such) and acts as fallback if yfinance breaks; EODHD
keeps underlying/dividends/rates. Three probes, three vendors, one
lesson: entitlements are discovered by calling the API, never by
reading the pricing page.

### 2026-08-28 — Addendum: yfinance spike (throwaway)

A five-line yfinance spike confirmed the free path works: 29 SPY
expirations listed, per-expiry chains with `strike, bid, ask, lastPrice,
volume, openInterest, impliedVolatility` (~80 call rows on a mid-dated
expiry), live spot via `fast_info`. ATM quotes looked sane (penny-wide
markets, positive OI/volume). Known limitations for Phase 2 design:
current chain only — **no historical backfill exists on this path** — and
it is an unofficial API scraping Yahoo, so breakage is a when, not an if
(the fetch stage must fail loudly and skip the day, per SPEC §2.1's
failure policy). Data-source decision for Phase 2 pending.

## Entries

*(Dated entries appended as each piece of the model layer is built.)*

### 2026-08-27 — The pricing formula

The Black-Scholes formula prices a European option as the discounted expected payoff under the risk-neutral measure, where both the spot and strike are discounted at their appropriate rates. The key terms are d1 and d2: d2 represents the risk-neutral probability that the option finishes in-the-money under the stock's forward distribution, while d1 is d2 shifted by σ√T, reflecting the difference in measuring probabilities under two numeraires (the bond vs the stock). Concretely, a call price is `S e^{-qT} N(d1) - K e^{-rT} N(d2)`, where the spot discount `e^{-qT}` appears naturally from the continuous dividend yield q; the test `test_dividend_yield_is_spot_shift` validates this by proving that BSM(S, q) is exactly equivalent to pricing with an effective spot `S e^{-qT}` and zero dividend, showing that dividend yield acts purely as a forward adjustment.

Put-call parity, `C - P = S e^{-qT} - K e^{-rT}`, holds to machine epsilon here by algebraic construction: since both calls and puts are built from the same d1 and d2 values and discount factors, their difference cancels all the Normal-CDF terms and reduces to the forward-strike identity. In real markets, parity typically only holds to the bid-ask spread because of transaction costs, market fragmentation, and discrete settlement; our implementation achieves machine-precision parity (validated in `TestIdentities`) because there are no such frictions. This distinction clarifies that parity is not a robust market law but an artifact of frictionless pricing.

As volatility approaches zero, the option price converges to the discounted intrinsic value of the forward—not the spot—which emerges naturally from the limiting behavior of d1 and d2. When σ → 0, both d1 and d2 approach negative infinity for out-of-the-money options (shrinking the CDF to zero) or positive infinity for in-the-money options (shrinking the CDF to one), causing the price to flatten to `max(S e^{-qT} - K e^{-rT}, 0)` for calls. This forward-intrinsic limit, validated in `TestBoundaries`, shows why options have real value even at zero vol when the forward is above strike; the discount factors and dividend yield are essential to the boundary behavior, not incidental.

The implementation safeguards underflow in the denominator (σ√T) with a small epsilon guard and uses NumPy broadcasting to handle scalar and array inputs uniformly. Invalid numeric inputs (negative spot, strike, or time; negative or infinite volatility) yield NaN elementwise rather than raising, allowing the downstream pipeline to track data quality through status counters. This "NaN-never-raise" contract preserves vectorization and handles bad market data gracefully without breaking batch processing.

### 2026-08-27 — The Greeks

The five Greeks capture how option prices change with moves in the underlying market variables. **Delta** (∂P/∂S) measures the option's sensitivity to the stock price—a call's delta equals `e^{-qT} N(d1)` and a put's equals `e^{-qT} (N(d1) - 1)`, both of which lie between 0 and 1 for calls and between -1 and 0 for puts. **Gamma** (∂²P/∂S²) is the rate of change of delta with respect to spot; it is always positive for both calls and puts (reflected in the formula as `e^{-qT} φ(d1) / (S σ √T)`) and represents the curvature of the price surface. **Vega** (∂P/∂σ) measures sensitivity to volatility and equals `S e^{-qT} φ(d1) √T`; like gamma, it is always positive. **Theta** (∂P/∂t) is the time decay per **year**; its sign depends on moneyness and whether the position is a call or put (for long options it is typically negative, meaning time decay erodes value). **Rho** (∂P/∂r) captures interest-rate sensitivity and differs between calls and puts: a call's rho is positive (`K T e^{-rT} N(d2)`), while a put's is negative.

A classic interview trap asks why an at-the-money call delta sits slightly above 0.5 rather than exactly at 0.5. The answer lies in the drift term in d1: `d1 = (ln(S/K) + (r - q + σ²/2) T) / (σ √T)`. When S = K at-the-money, d1 = `(r - q + σ²/2) √T / σ` is positive (assuming r > q and positive vol), which means N(d1) > 0.5. This drift—the excess of the risk-free rate over the dividend yield, plus half the variance—pushes the forward above the strike, raising the probability of finishing in-the-money slightly. This same r−q+σ²/2 drift is one of the trickier concepts in interview questions (questions 1 and 6 in the guide) because it is not the stock's actual drift (which is r under the risk-neutral measure) but the forward's drift relative to the discounted spot.

Both gamma and vega share the probability-density-function kernel φ(d1), which is why both peak at-the-money and decay toward zero deep in-the-money or out-of-the-money. This shared geometry is more than notational accident: gamma measures the sensitivity of the hedge ratio (delta) to moves in the stock, while vega measures the sensitivity of price to moves in volatility expectations. Both are largest when the option is at-the-money because that is where small moves in the underlying have the largest fractional impact on the probability of finishing in-the-money (the maximum slope of the CDF N(d1)). At-the-money options are the most "gamma-rich" and "vega-rich," which is why short-selling at-the-money options to collect time decay (selling theta) is most dangerous there—the buyer of that protection is buying the most convex exposure. This insight seeds question 6, but not as a theory of the smile itself: because vega decays toward zero in the wings, any procedure that inverts price into vol—Newton, Brent, a broker's own solver—becomes numerically fragile exactly where the smile is most interesting to read. The geometry explains why wing IVs are noisy to recover, not why they're higher in the first place.

The theta convention chosen here is ∂P/∂t per **year**, not per day. This is mathematically clean (the formulas differentiate T directly) but requires the render layer to scale theta for human display—dividing by 365 to show per-day decay. The brief explicitly marks this as a raw derivative; downstream pricing feeds and dashboards must handle the scaling. For a long call, theta is negative (time decay erodes the option's value toward expiry), while for a short call theta is positive profit. The formula accounts for both the decay of the Gaussian term and the changing discount factors as T shrinks.

Finally, differentiating put-call parity with respect to spot yields the delta-parity relation: `delta_call - delta_put = e^{-qT}`. This holds exactly in our implementation (validated in `TestShapes`) because delta for a call is `e^{-qT} N(d1)` and for a put is `e^{-qT} (N(d1) - 1)`, so their difference is exactly `e^{-qT}`. This relation is important for market-making: if you are short one call and long one put at the same strike and expiry, your net delta exposure is `-e^{-qT}` (a short position in the forward discounted at the dividend yield). This follows algebraically from parity but often surprises practitioners who confuse delta with the more intuitive notion of "how many shares behave like this option."

### 2026-08-27 — Inverting the formula: implied vol

Implied vol only makes sense as a concept because `bs_price` is strictly increasing in σ — `TestBoundaries::test_monotone_increasing_in_vol` from Task 3 is the proof, and it's the whole reason inversion is well-posed rather than a guessing game. A strictly monotone continuous function from `(0, ∞)` onto `(intrinsic, upper_bound)` has a unique inverse on that range, so "the σ that produces this price" is a well-defined single number, not a family of candidates — provided the price actually falls in the reachable range. The Newton step needs a starting point, and the Brenner–Subrahmanyam guess `√(2π/T) · price/S` comes from linearizing the ATM price formula: near the money, `C ≈ 0.4 · S · σ√T` (the constant is `1/√(2π) ≈ 0.3989`), so solving for σ gives `σ ≈ √(2π/T) · C/S`. It's derived for ATM options specifically, which is why the docstring calls it "near-exact ATM, decent start elsewhere" — everywhere else it's just a reasonable place to start Newton, not a real estimate.

Question 7 asks what actually breaks Newton deep OTM, and I hit the real failure rather than a hypothetical one. The naive first draft checked convergence (`|diff| < tol`) *before* checking whether vega was too small to trust. For `test_deep_otm_tiny_vega_does_not_return_garbage` (S=100, K=300, T=0.05, true σ=0.30), the true price is ~1.4e-60 — so small it's indistinguishable from 0 in float64. The Brenner–Subrahmanyam guess for that price collapses to the `_MIN_VOL` floor, and at `_MIN_VOL` the computed price is *also* 0.0 in float64, so `diff = -1.4e-60`, which is comfortably under `tol=1e-10`. The loop declared victory on iteration one and returned σ = 1e-4 — a plausible-looking number that was completely wrong, exactly the "silently wrong" outcome the corner contract forbids. The fix was a one-line reorder: check `vega < floor` and release to Brent *before* evaluating convergence, so a near-zero-vega point can never be accepted just because its price happens to round to the target.

That fix didn't fully solve it, though, which is the more interesting failure. Because `tol` lives in price space, the resulting σ error is only ever ~`tol/vega` — reasonable off-ATM, but the docstring's own claim implies a lower bound on how small vega can be before the guaranteed accuracy evaporates. I'd set the "vega is too small to trust" cutoff to a bare `1e-10`, matching the underflow guard already used elsewhere in the file — but a point with vega = 8.8e-10 sailed past that gate, "converged" on a diff of 5.4e-11, and returned σ = 3.15 for a true σ of 3.00 (a K=300, T=1-day put): an error of 0.15, three orders of magnitude past the 1e-6 promise. `tol/vega ≈ 1e-10/8.8e-10 ≈ 0.11` — the docstring's own formula predicted this exact failure once I did the arithmetic; the cutoff just wasn't derived from the accuracy target it was supposed to protect. The real gate has to be `vega ≥ tol / target_accuracy = 1e-10 / 1e-6 = 1e-4`, not an arbitrary underflow-sized number — I checked this against the normal grid's minimum vega (~0.0068) to confirm the tighter floor never falsely disqualifies a healthy point.

Brent can't fail to find a root that mathematically exists inside a bracket with a genuine sign change — that's the whole content of the intermediate value theorem, and it's why it's the designated fallback instead of a second Newton variant. But "genuine sign change" is doing real work in that sentence: for the deepest, shortest-dated corners (K=300 vs S=100, T=1 day), `bs_price(σ)` is *bit-identical* to the target price across almost the entire domain from `_MIN_VOL` up to σ≈3, and only diverges past that. Brent, searching for where the float64-evaluated sign flips, correctly converges to the edge of that plateau — not to whatever σ the test happened to generate the price from, because within the plateau every σ is an equally valid "root" of the floating-point function. I added a post-check mirroring the Newton fix: compute vega at whatever Brent returns, and discard the root (leave NaN) if it's below the same `vega_floor` derived above, rather than trust a technically-valid root that carries no real information. This is question 6's answer applied to the solver rather than the pricer: vega's peak-at-the-money, decays-to-zero-in-the-wings geometry (Task 4) isn't a quirk of this codebase, it's why *any* numerical inversion — ours, a broker's, a Bloomberg terminal's — gets noisy or undefined in the deep wings. The flatness is in the option's economics, not in our loop, which is also the argument for why no-arbitrage violations return NaN immediately rather than a clamped σ: a price above the upper bound or below intrinsic isn't "a vol we solved slightly wrong," it's a price that no σ could have produced, so *any* number returned there is fabricated. Clamping it to `_MAX_VOL` or `_MIN_VOL` would look like a real answer downstream — the smile panel would happily plot a fictitious point next to real ones, and nothing would flag it as garbage. NaN is the only value that's honest about "no σ answers this," the same discipline as the `_invalid_mask` in Tasks 3–4: bad inputs don't get a best-effort number, they get a value the caller is forced to check for.

### 2026-08-28 — Phase 2: first real data

**Daily snapshot (`python -m src.run_daily`, 2026-08-28).** yfinance served the live chain — no fallback needed. Status: `source: yfinance`, spot 771.10, risk_free_rate 3.78% (EODHD `US3M.INDX`), dividend_yield 0.976% (EODHD trailing-12m). Stored chain: 40 rows, 6 expiries, 1 source (sanity check from the brief passed exactly). That row count is the first real surprise: the brief expected "hundreds," but the raw yfinance chain for SPY that day was 2,277 rows across those 6 expiries, and the `chain_filter` moneyness gate alone kept 1,492 of them (K/S in [0.70, 1.30] is wide). What actually gutted the count was the *liquidity* gate for live sources — `bid > 0 & ask.notna()` — which only 167 of the 2,277 raw rows passed. Combined with moneyness, 40 survived. **Important caveat this session got wrong on first pass: this run happened at `last_success_utc` 06:58 UTC = 02:58 ET — the middle of the night, hours after the US market closed and hours before it reopens.** SPY market-makers legitimately pull two-sided quotes overnight, so a near-total absence of live `bid > 0` markets is expected at that hour and is **not yet validated against a live session**. The kill-rate breakdown above (2,277 raw → 1,492 post-moneyness → 167 post-liquidity → 40 final) is real and worth keeping, but its attribution needs correcting: it may be showing "the book is closed right now" more than "the filter is stricter than expected in general." The production cron is scheduled for 21:30 ET, shortly after the 16:00 ET close, when the book still carries end-of-day two-sided quotes — that run is expected to see a meaningfully higher post-liquidity count than this overnight one did. Re-validate the kill-rate numbers against a 21:30 ET (or intraday) run before drawing conclusions about Phase 3+ smile-panel density.

**Backfill (`python scripts/backfill.py`) — small window and depth probe.** Ran `--start 2026-08-14 --end 2026-08-27` (10 trading days) and, separately, a single-day depth probe at `--start 2024-09-03 --end 2024-09-03` (a Tuesday ~2 years back). The results split sharply by *how recent* the target date is, which was not the axis I expected trouble on.

- **Depth probe: success.** 2024-09-03 completed cleanly in **5m05s** for one day and stored **977 rows** across the same 6 monthly expiries (2024-09-20 through 2025-03-21), source `massive-backfill`, `open_interest` all NaN as documented (Starter tier has no historical OI — volume-only liquidity gate, `PRE_FILTER_COLUMNS`/`filter_chain`'s `liquid_close` path). Kind split was close to even (493 puts / 484 calls). This confirms Massive's historical reference+aggregates path genuinely reaches **at least 2 years back** with usable per-contract daily bars — the backfill's core premise holds for older data.
- **Small window: failed, reproducibly, 5/5 attempts — but only on recent dates.** `get_historical_chain` for one day issues ~3s of contract-reference calls (8,596 total contracts near spot ≈776, 1,876 surviving the monthly-expiry/DTE filter) followed by **one sequential HTTP GET per surviving contract** to `/v2/aggs/ticker/{ticker}/range/1/day/{date}/{date}` — no batching, no delay, no retry. A direct timing probe (5-contract sample) measured ~0.21–0.30s/call, projecting **~7.2 min for 1,876 calls** — which matches the successful 2024-09-03 run's 5m05s. But every attempt at a *recent* date in the small window failed before finishing: 2026-08-14 timed out or reset the connection on attempts 1–4 (`ReadTimeoutError`, read timeout=30s, three times; one `RemoteDisconnected('Remote end closed connection without response')`), taking anywhere from 49s to 8m28s to fail. 2026-08-27 (the window's last day) failed the same way on its first attempt (3m06s, `ReadTimeoutError`). Zero chain files were written for the 14-day window; `data/underlying.parquet` did get upserted each time (that write happens before the per-day loop), so no corruption, just no chain data. Because a mid-day HTTP failure aborts the whole day (no per-contract resumability, only per-*day* resumability via `chain_exists`), every retry re-fetched all ~1,876 contracts from scratch — a real cost of the "resumability = skip existing files" design when a day never finishes once.
- **The surprise is the direction of the split.** Naively I'd have guessed older dates are the ones with sketchy retention/latency and recent dates would be fast (closer to a live cache). Observed exactly the opposite: the settled, 2-year-old date sailed through in one shot; five separate attempts (across two different dates within days of "today") in the recent window all choked on the exact same endpoint. Plausible explanation: Massive/Polygon likely serves long-settled historical aggregates from a static, pre-computed store, while near-live dates may still be getting rolled up from tick data on demand — making them slower and more failure-prone on this tier, the opposite of the "recent data is cheap, historical is expensive" assumption baked into the brief's "expect minutes per day" framing.

**Wide-backfill decision (deliberately not made here).** Per-day cost when a day *succeeds* is consistent with the ~7 min projection (5m05s observed). At that rate a full 730-day run is **~85 hours of successful-day time alone (not the "hours" the brief guessed — more like 3.5+ days of continuous runtime)**, and that's the optimistic case: the observed 0/5 success rate on recent dates in this session means a naive unattended overnight run risks burning hours retrying the same few near-term days without ever finishing them, since each failed attempt restarts that day's ~1,876 requests from zero. The implementation as shipped (verbatim from the brief — no retry/backoff/rate-limiting added) is not reliable enough for an unattended multi-day run against recent dates without added resilience (e.g., per-contract retry, request pacing, or a bulk-aggregates endpoint if Massive offers one at a higher tier). Recommendation for whoever schedules the real 730-day run: expect it to take days of wall-clock time even under ideal conditions, budget for manual restarts around recent-date failures, and consider prioritizing the older 2/3 of the window first (where this session saw 1/1 success) before fighting the flakier recent tail.

**Retry-fix validation (post-review addendum).** Code review ruled that `MassiveProvider._get_json` needed transport-level retry (`requests.exceptions.RequestException` only — never the `RuntimeError` used for a bad body status like `NOT_AUTHORIZED`, which must still fail immediately) — 3 attempts total, `time.sleep(2)` then `time.sleep(8)` between attempts, re-raising the last exception if all three fail. Added with 3 new fixture tests (`TestRetry` in `tests/test_massive_provider.py`) covering: two transient failures then success (3 calls, sleeps `[2, 8]`), three transient failures exhausting retries (propagates, 3 calls), and a `NOT_AUTHORIZED` body making exactly 1 call with no sleep. With the fix in place, re-ran the same small window that had failed 5/5 times before (`--start 2026-08-14 --end 2026-08-27`): **`data/chains/2026-08-14.parquet` completed successfully on the first attempt** — this is the exact date that timed out or reset the connection on all 4 of its prior (pre-fix) attempts, so it never got past this day before. With day 1 no longer blocking, the run continued straight into day 2 and **`data/chains/2026-08-17.parquet` also completed successfully** (a date never previously reached at all, since no pre-fix run got past 2026-08-14). Two-for-two on the first two days is direct before/after evidence that the retry/backoff fix resolves the recent-date transport failures this session hit. The remaining days in the window were left running/resumable rather than waited on to completion — per the backfill's design, `chain_exists` skip-on-resume means whoever runs it next (or lets it keep running) picks up exactly where it left off with no repeated work on the days already written.

Data committed from this session: `data/chains/2026-08-28.parquet` (40 rows, yfinance), `data/chains/2024-09-03.parquet` (977 rows, massive-backfill), `data/underlying.parquet` (8,452 rows through 2026-08-27), `docs/status.json`.

**Final-review correction (relabel).** The file logged above as `data/chains/2026-08-28.parquet` was itself an instance of the off-hours caveat two paragraphs up, taken to its logical conclusion: captured 2026-08-28 02:58 ET using 2026-08-27's close as spot, it never described the 2026-08-28 session at all — it described 2026-08-27's, labeled with the wall-clock date the script happened to run on instead of the market date the data was actually *of*. Final review caught this as a general bug (`src/run_daily.py`'s `run()` derived `snapshot_date` from `dt.date.today()` rather than from the underlying it had just fetched) and fixed it at the source: `snapshot_date` is now the last date in the freshly-upserted underlying history, with a >5-day staleness guard against wall clock. The one committed file this bug had already mislabeled was renamed to its true session, `data/chains/2026-08-27.parquet` (`dte` recomputed against the corrected `snapshot_date`), and `docs/status.json` updated to match.

**Backfill, continued.** `data/chains/2026-08-18.parquet` and `data/chains/2026-08-19.parquet` are two more post-retry-fix successes in the same run documented above (`2026-08-14` and `2026-08-17` were the first two) — the transport-retry fix continues to hold on the recent-date tail that failed 5/5 times before it.

**Parked gap: holiday-shifted monthlies.** `is_monthly_expiry` treats "third Friday of the month" as the entire definition of a standard monthly. When an exchange holiday pushes that expiry to the preceding Thursday (observed in the backfill window), this rule misses it — a real gap, worth roughly one month a year across the backfill. Deliberately left unfixed for now: a naive "third Friday-or-Thursday" rule would also match SPY's daily expiries, which trade on every weekday including plenty of Thursdays, so loosening the day-of-week check without also disambiguating "the standard monthly contract" from "a Thursday that happens to carry a daily expiry" would misclassify far more than it fixes.

### 2026-08-28 — Phase 3: the first page

**The real run (`python -m src.run_daily`, `last_success_utc` 2026-08-28T14:38:34Z = 10:38 ET — an hour after the open, live market hours).** `source: yfinance`, no fallback needed. Session labeled `2026-08-27` again (same spot, 771.10, as the prior overnight entry — the underlying's latest settled close hadn't rolled to today yet), so this run atomically overwrote `data/chains/2026-08-27.parquet` in place, per the brief's expectation. The number that matters: **1,459 rows stored, vs. the prior overnight run's 40** — a 36x jump from the exact same source (`yfinance`) and the exact same filter (`bid > 0 & ask.notna()`), with the only variable being whether the book was open. That single before/after pair is the cleanest evidence yet for the Phase 2 entry's hedge: the overnight 40-row result really was "the book is closed right now," not "the liquidity filter is stricter than expected in general." A live SPY book is stunningly deep by comparison.

**Convergence: 99.31% (1,449/1,459) vs. the ≥95% exit criterion — passed with room to spare, and for the first time on real (not synthetic) data.** The self-review notes flagged this exact gap: P2's skew-visible criterion had only been tested synthetically before this task. The 10 non-converging rows are not solver noise — I pulled them out and checked by hand. All 10 are **deep in-the-money calls** (e.g. K=641 vs spot=771.10, 22 DTE, quoted mid 124.695) whose quoted mid sits *below* the discounted-intrinsic no-arbitrage floor (intrinsic ≈ S−K ≈ 130.1 here). This is exactly the "no-arbitrage violations return NaN immediately" contract from the 2026-08-27 implied-vol entry, now firing on genuine market data instead of a synthetic corner case: the solver correctly refuses to fabricate a vol for a quote that no σ could have produced, rather than clamping to something plausible-looking. Real deep-ITM quotes apparently do sit on the wrong side of that bound often enough (10 times in one chain) to be worth knowing the pipeline handles it the way the model layer promised it would.

**The smile (P2, front expiry 2026-09-18, 22 DTE, 254 points, strikes $540–$905 in $1 increments near the money widening to $5 further out).** The put skew is unmistakable and steep: near-ATM IV sits at **~11.3–11.4%** (strikes 769–774, moneyness ≈1.0), a deep OTM put at K=540 (moneyness 0.70) prices at **~50.5% IV**, while the symmetric deep OTM call at K=905 (moneyness 1.17) is only **~21%** — the downside wing rises roughly 2.4x steeper in IV-per-unit-moneyness than the upside wing. This is the textbook equity/index skew (crash-insurance premium on downside puts), and it's the first time this codebase has shown it on data it didn't synthesize itself.

**The term structure (P3, 6 expiries, 22–204 DTE).** Clean contango, no kinks or inversions: ATM IV rises monotonically from 11.4% (22 DTE) → 12.7% (50 DTE) → 14.1% (85 DTE) → 14.8% (113 DTE) → 15.2% (141 DTE) → 16.4% (204 DTE). A calm-market curve — nothing here to stress-test the overlay logic beyond what P3's own synthetic tests already covered, since there was no prior-day chain new enough to overlay against a materially different vol regime.

**Page: 1.40 MB, self-contained.** One nuance in verifying that: the brief's literal Step 2 snippet (`assert "https://cdn.plot.ly" not in html`) fails against the real rendered page — but the failure is a known false positive already handled by the codebase's own test (`tests/test_render.py::test_self_contained_no_external_refs`, added in commit `12efb25`). The string exists exactly once, and it's the vendored `plotly-cartesian-2.35.2.min.js` bundle's own inert default value for `topojsonURL` (a geo/choropleth config this dashboard never invokes) — not a live `<script src>` or fetch. Scoping the check the same way the test does (`html.replace(_BUNDLE.read_text(), "")` before searching) confirms zero external references and zero external script tags in the page's own markup. Worth flagging so a future naive copy-paste of the brief's snippet doesn't get treated as a real regression.

**Honest ugliness.** Two things worth a future look, neither blocking: (1) the front-month smile is put-heavy — 184 put points vs. 70 call points out of 254 — so the rendered curve will visually skew long-and-dense on the left with a shorter, sparser right wing; real, not a rendering bug, but asymmetric enough that a first-time viewer might wonder if the call side is under-sampled rather than genuinely thinner in the listed strike ladder. (2) The 10 NaN'd deep-ITM points are silently dropped from the smile rather than shown-and-flagged — correct per the no-fabrication contract, but it means a viewer staring at the P2 panel has no visual cue that a handful of real quotes existed at those strikes and were deliberately excluded; that's a reasonable "optional" follow-up (visible gap markers) rather than a Phase 3 blocker, in the same spirit as the deliberately-deferred bid/ask IV band toggle.

Data committed from this run: `data/chains/2026-08-27.parquet` (1,459 rows, yfinance, overwritten), `data/chains/2026-08-24.parquet`, `data/chains/2026-08-25.parquet`, `data/chains/2026-08-26.parquet` (background-backfill additions picked up at commit time), `docs/index.html` (1.40 MB), `docs/status.json`.


