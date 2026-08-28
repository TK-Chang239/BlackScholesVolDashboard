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
