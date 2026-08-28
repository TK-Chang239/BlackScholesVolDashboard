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

### 2026-08-27 — The pricing formula

The Black-Scholes formula prices a European option as the discounted expected payoff under the risk-neutral measure, where both the spot and strike are discounted at their appropriate rates. The key terms are d1 and d2: d2 represents the risk-neutral probability that the option finishes in-the-money under the stock's forward distribution, while d1 is d2 shifted by σ√T, reflecting the difference in measuring probabilities under two numeraires (the bond vs the stock). Concretely, a call price is `S e^{-qT} N(d1) - K e^{-rT} N(d2)`, where the spot discount `e^{-qT}` appears naturally from the continuous dividend yield q; the test `test_dividend_yield_is_spot_shift` validates this by proving that BSM(S, q) is exactly equivalent to pricing with an effective spot `S e^{-qT}` and zero dividend, showing that dividend yield acts purely as a forward adjustment.

Put-call parity, `C - P = S e^{-qT} - K e^{-rT}`, holds to machine epsilon here by algebraic construction: since both calls and puts are built from the same d1 and d2 values and discount factors, their difference cancels all the Normal-CDF terms and reduces to the forward-strike identity. In real markets, parity typically only holds to the bid-ask spread because of transaction costs, market fragmentation, and discrete settlement; our implementation achieves machine-precision parity (validated in `TestIdentities`) because there are no such frictions. This distinction clarifies that parity is not a robust market law but an artifact of frictionless pricing.

As volatility approaches zero, the option price converges to the discounted intrinsic value of the forward—not the spot—which emerges naturally from the limiting behavior of d1 and d2. When σ → 0, both d1 and d2 approach infinity for out-of-the-money options (shrinking the CDF to zero) or negative infinity for in-the-money options (shrinking the CDF to one), causing the price to flatten to `max(S e^{-qT} - K e^{-rT}, 0)` for calls. This forward-intrinsic limit, validated in `TestBoundaries`, shows why options have real value even at zero vol when the forward is above strike; the discount factors and dividend yield are essential to the boundary behavior, not incidental.

The implementation safeguards underflow in the denominator (σ√T) with a small epsilon guard and uses NumPy broadcasting to handle scalar and array inputs uniformly. Invalid numeric inputs (negative spot, strike, or time; negative or infinite volatility) yield NaN elementwise rather than raising, allowing the downstream pipeline to track data quality through status counters. This "NaN-never-raise" contract preserves vectorization and handles bad market data gracefully without breaking batch processing.


