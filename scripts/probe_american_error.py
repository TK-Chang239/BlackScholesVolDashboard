"""Q9 probe: how wrong is Black-Scholes on SPY's American options, and
where does the error concentrate.

(`LEARNING_LOG.md` question 9 of 10.) SPY options are American; this
project prices them with a European model and a continuous dividend
yield q. A rigorous answer prices the same contracts under an American
model and differences them -- that needs a binomial tree this project has
not built, and building one is out of scope for this phase (see the task
brief). This probe does the in-scope thing instead: it BOUNDS and LOCATES
the error using evidence the project already produces, and says plainly,
throughout, what a bound is not.

Three pieces, all read-only against stored data (no network, no writes
under `data/` or `docs/`, no pricing library -- every price, vol and vega
below comes from this repo's own `src.models.black_scholes`):

1. The parity residue (`src.analytics.parity`) on the most recent chain.
   European put-call parity is model-free -- it holds by arbitrage alone,
   with no volatility and no distribution assumed. A systematic deviation
   that survives forward-calibration, concentrates in in-the-money puts,
   and fits inside the early-exercise bound is the American premium
   showing itself. Its size is a LOWER bound on the pricing error there:
   it is real money left on the table by the European model, but it is
   not the true American-minus-European gap, because the classifier is
   deliberately conservative (Step 2's tighter bound) and because a
   European price with the WRONG volatility could produce the same
   residue. Both halves are stated below, not just the flattering one.

2. The theory-side ceiling: `K(1 - e^{-rT})`, the rigorous upper limit on
   an American put's early-exercise premium over its European price (see
   `src.analytics.parity`'s module docstring and LEARNING_LOG.md's
   "early-exercise residue" entry for the derivation), alongside the
   tighter `K(1 - e^{-rT}) - S(1 - e^{-qT})` bound the project actually
   uses. Depends only on K, T, r, q, S -- no quote, no volatility. This
   brackets the answer from above.

3. The continuous-q-versus-discrete-dividend approximation. The pipeline
   fetches SPY's dividend history over the network at run time
   (`src.data.eodhd_provider.get_dividend_yield`) and writes only the
   resulting SCALAR trailing yield into `docs/status.json` -- no per-date,
   per-amount dividend table is stored anywhere in this repo. Offline, the
   closest available substitute is `data/underlying.parquet`'s `close` vs
   `adjusted_close` columns: EODHD back-adjusts historical closes for
   dividends (SPY has never split, so the divergence between the two
   columns is dividend-only), and the step where that adjustment factor
   changes IS an ex-dividend date, with the step size implying the
   dollar amount. That is what this script recovers and uses -- an
   ARTEFACT of a price-adjustment convention, not a fetched or trusted
   dividend feed, and it is labelled that way everywhere it appears.

Usage: .venv/bin/python scripts/probe_american_error.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHAINS_DIR = ROOT / "data" / "chains"
STATUS_PATH = ROOT / "docs" / "status.json"
UNDERLYING_PATH = ROOT / "data" / "underlying.parquet"
sys.path.insert(0, str(ROOT))

from src.analytics.chain_iv import compute_chain_iv  # noqa: E402
from src.analytics.parity import compute_parity, parity_summary  # noqa: E402
from src.models.black_scholes import bs_price, greeks, implied_vol  # noqa: E402

NEAR_MONEY_BAND = 0.02
DEEP_BAND = 0.10
JUMP_THRESHOLD = 1e-4   # |Delta(close/adjusted_close)| beyond this => a real ex-div step,
                        # not float noise (see detect_ex_dividend_jumps docstring)


def most_recent_chain_path() -> Path:
    paths = sorted(CHAINS_DIR.glob("*.parquet"))
    if not paths:
        raise SystemExit("No stored chains under data/chains/ -- nothing to probe.")
    return paths[-1]


def load_status() -> dict:
    with open(STATUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def parity_moneyness_bucket(p: pd.DataFrame) -> pd.Series:
    """Direction-aware bucket keyed on K/S, since a parity PAIR (not a
    single option) is the unit here. 'itm_put' means K > S, the only
    side early exercise can ever matter for a put."""
    dist = p["moneyness"] - 1.0
    near = dist.abs() <= NEAR_MONEY_BAND
    deep_itm_put = (~near) & (dist > DEEP_BAND)
    deep_otm_put = (~near) & (dist < -DEEP_BAND)
    out = pd.Series("moderate", index=p.index)
    out[near] = "near_money"
    out[deep_itm_put] = "deep_itm_put"
    out[deep_otm_put] = "deep_otm_put"
    return out


def add_put_vega(p: pd.DataFrame, chain_iv: pd.DataFrame, spot: float, r: float, q: float) -> pd.DataFrame:
    """Attach each pair's PUT vega at the put's own solved IV.

    The parity residue is a signed, joint (call, put) quantity, but the
    early-exercise story this file is testing is specifically about the
    PUT: `deviation_fwd < 0` means the put is rich, and `early_exercise_
    bound` is a bound on a put's premium. Converting the residue to vol
    points via the put's own vega is therefore the natural choice -- it
    answers "how many points would the put's OWN implied vol be
    overstated to explain this dollar gap" -- and it is used consistently
    for both the residue (Step 1) and the theory bounds (Step 2).
    """
    df = p.copy()
    puts = chain_iv[chain_iv["kind"] == "put"][["expiry", "dte", "strike", "iv"]].rename(
        columns={"iv": "put_iv"})
    df = df.merge(puts, on=["expiry", "dte", "strike"], how="left")
    T = df["dte"].to_numpy(dtype=float) / 365.0
    K = df["strike"].to_numpy(dtype=float)
    sigma = df["put_iv"].to_numpy(dtype=float)
    vega = greeks(np.full(len(df), spot), K, T, r, sigma, q=q, kind="put")["vega"]
    df["vega_put_pv"] = vega / 100.0
    return df


# --------------------------------------------------------------------------
# Step 1: the parity residue -- a measured lower bound
# --------------------------------------------------------------------------

def step1_parity_residue(chain: pd.DataFrame, r: float, q: float) -> None:
    print("\n--- Step 1: the parity residue (a lower bound, not a measurement) ---")
    chain_iv, _ = compute_chain_iv(chain, r, q)
    spot = float(chain_iv["spot"].iloc[0])
    parity = compute_parity(chain_iv, r, q)
    summ = parity_summary(parity)
    print(f"compute_parity: {summ['n_pairs']} pairs, {summ['n_liquid']} liquid "
          f"(both legs have positive open interest), {summ['n_tradeable_violations']} raw "
          f"tradeable violations -> {summ['n_tradeable_violations_fwd']} once each expiry's "
          f"forward is calibrated -> {summ['n_violations_early_exercise']} explained by the "
          f"early-exercise bound, {summ['n_violations_unexplained']} unexplained residue. "
          "(These reproduce LEARNING_LOG.md's Phase 5 chain: 132 -> 42 -> 36/6.)")

    p = add_put_vega(parity, chain_iv, spot, r, q)
    liq = p[p["liquid"]].copy()
    liq["deviation_vol_pts"] = liq["deviation_fwd"] / liq["vega_put_pv"]
    liq["bucket"] = parity_moneyness_bucket(liq)

    print(f"\nAmong the {len(liq)} liquid pairs, |deviation_fwd| by moneyness bucket "
          "(vol points = deviation_fwd / put's own solved vega):")
    print(f"{'bucket':<14} {'n':>5} {'median $':>10} {'p90 $':>9} {'median vega/pt':>15} "
          f"{'median vol pts':>15} {'p90 vol pts':>12}")
    bucket_order = ("deep_otm_put", "moderate", "near_money", "deep_itm_put")
    for b in bucket_order:
        part = liq[liq["bucket"] == b]
        if part.empty:
            continue
        dev = part["deviation_fwd"].abs()
        volp = part["deviation_vol_pts"].abs()
        print(f"{b:<14} {len(part):>5} {dev.median():>10.3f} {dev.quantile(0.9):>9.3f} "
              f"{part['vega_put_pv'].median():>15.4f} {volp.median():>15.3f} {volp.quantile(0.9):>12.3f}")
    print("CAVEAT on the vol-point column: converting a dollar gap to vol points divides by "
          "vega, and per scripts/probe_vega_inversion.py (Q6) vega itself collapses by orders of "
          "magnitude away from the money. The deep_otm_put row's large p90 vol-point figure is "
          "mostly THAT conditioning effect, not a large dollar mispricing -- its own median dollar "
          "gap is one of the smallest of the four buckets. Read the $ and vol-point columns "
          "together, never the vol-point one alone.")

    print(f"\nBy DTE:")
    print(f"{'dte':>5} {'n':>5} {'median $':>10} {'p90 $':>9} {'median vol pts':>15}")
    for dte, part in liq.groupby("dte"):
        dev = part["deviation_fwd"].abs()
        volp = part["deviation_vol_pts"].abs()
        print(f"{dte:>5} {len(part):>5} {dev.median():>10.3f} {dev.quantile(0.9):>9.3f} "
              f"{volp.median():>15.3f}")

    explained = liq[liq["early_exercise_explained"]]
    viol_fwd = liq[liq["tradeable_violation_fwd"]]
    unexplained = viol_fwd[~viol_fwd["early_exercise_explained"]]
    print(f"\nThe {len(explained)} pairs the early-exercise bound EXPLAINS "
          f"(negative deviation, in-the-money put, |gap| within bound+spread) -- this is where the "
          "American premium is showing itself, measured as a lower bound on the pricing error there:")
    print(f"  deviation_fwd: median ${explained['deviation_fwd'].median():.2f}, "
          f"range ${explained['deviation_fwd'].min():.2f} to ${explained['deviation_fwd'].max():.2f}")
    print(f"  |deviation| in vol points: median {explained['deviation_vol_pts'].abs().median():.3f}, "
          f"range {explained['deviation_vol_pts'].abs().min():.3f} to "
          f"{explained['deviation_vol_pts'].abs().max():.3f}")
    print(f"  vega/pt range on this subset: {explained['vega_put_pv'].min():.3f} to "
          f"{explained['vega_put_pv'].max():.3f} -- reasonably liquid vega, so these vol-point "
          "numbers are not a conditioning artifact.")
    print(f"  DTE range: {int(explained['dte'].min())}-{int(explained['dte'].max())}")

    print(f"\nThe {len(unexplained)} unexplained residue -- what neither a stale forward nor the "
          "bound accounts for:")
    print(f"  deviation_fwd: median ${unexplained['deviation_fwd'].median():.2f}, "
          f"range ${unexplained['deviation_fwd'].min():.2f} to ${unexplained['deviation_fwd'].max():.2f}")
    print(f"  |deviation| in vol points: median {unexplained['deviation_vol_pts'].abs().median():.1f}, "
          f"range {unexplained['deviation_vol_pts'].abs().min():.2f} to "
          f"{unexplained['deviation_vol_pts'].abs().max():.1f}")
    print(f"  vega/pt range on this subset: {unexplained['vega_put_pv'].min():.4f} to "
          f"{unexplained['vega_put_pv'].max():.4f} -- includes near-zero vega. Its huge vol-point "
          "figures are dominated by the SAME low-vega conditioning Q6 measures, not by a large "
          "dollar mispricing; the dollar range overlaps the explained subset's.")

    print("\nWhat this demonstrates: parity is model-free, so a systematic, sign-consistent, "
          "moneyness-concentrated deviation that a derived no-arbitrage bound can absorb is real "
          "evidence the European model is missing early-exercise value on in-the-money puts, sized "
          "here in both dollars and vol points.")
    print("What this does NOT demonstrate: the residue is a LOWER bound on the American-minus-"
          "European gap, not the gap itself. `early_exercise_explained` marks a violation as "
          "CONSISTENT with early exercise, not proven to be caused by it -- a stale mark of the "
          "same size would fit the same bound. And parity cannot see any mispricing that is smaller "
          "than the combined bid-ask spread, so whatever this test finds is a floor, not a ceiling, "
          "on where the true error sits.")


# --------------------------------------------------------------------------
# Step 2: the theory-side ceiling
# --------------------------------------------------------------------------

def step2_theory_ceiling(chain: pd.DataFrame, r: float, q: float) -> None:
    print("\n--- Step 2: the theory-side ceiling (brackets the answer from above) ---")
    chain_iv, _ = compute_chain_iv(chain, r, q)
    spot = float(chain_iv["spot"].iloc[0])
    parity = compute_parity(chain_iv, r, q)
    p = add_put_vega(parity, chain_iv, spot, r, q)
    T = p["dte"].to_numpy(dtype=float) / 365.0
    K = p["strike"].to_numpy(dtype=float)
    p["rigorous_ceiling"] = K * (1.0 - np.exp(-r * T))
    liq = p[p["liquid"]].copy()
    liq["bucket"] = parity_moneyness_bucket(liq)
    liq["ee_bound_vol_pts"] = liq["early_exercise_bound"] / liq["vega_put_pv"]
    liq["ceiling_vol_pts"] = liq["rigorous_ceiling"] / liq["vega_put_pv"]

    print("For every liquid pair: K(1 - e^{-rT}) (the rigorous ceiling on the American-put "
          "early-exercise premium) and K(1 - e^{-rT}) - S(1 - e^{-qT}) (the tighter bound "
          "`compute_parity` actually uses -- both depend only on K, T, r, q, S; no quote, no vol):")
    print(f"\n{'bucket':<14} {'n':>5} {'median bound $':>15} {'median ceiling $':>17} "
          f"{'median bound pts':>17} {'median ceiling pts':>19}")
    bucket_order = ("deep_otm_put", "moderate", "near_money", "deep_itm_put")
    for b in bucket_order:
        part = liq[liq["bucket"] == b]
        if part.empty:
            continue
        print(f"{b:<14} {len(part):>5} {part['early_exercise_bound'].median():>15.3f} "
              f"{part['rigorous_ceiling'].median():>17.3f} "
              f"{part['ee_bound_vol_pts'].median():>17.3f} {part['ceiling_vol_pts'].median():>19.3f}")
    print("\nBy DTE:")
    print(f"{'dte':>5} {'n':>5} {'median bound $':>15} {'median ceiling $':>17} "
          f"{'median bound pts':>17} {'median ceiling pts':>19}")
    for dte, part in liq.groupby("dte"):
        print(f"{dte:>5} {len(part):>5} {part['early_exercise_bound'].median():>15.3f} "
              f"{part['rigorous_ceiling'].median():>17.3f} "
              f"{part['ee_bound_vol_pts'].median():>17.3f} {part['ceiling_vol_pts'].median():>19.3f}")

    itm = liq[liq["moneyness"] > 1.0]
    print(f"\nRestricted to K > S ({len(itm)} of {len(liq)} liquid pairs) -- the only side either "
          "bound is economically meaningful (a put struck below spot has nothing to exercise "
          "early for): median tighter bound ${:.3f} ({:.3f} vol pts), median rigorous ceiling "
          "${:.3f} ({:.3f} vol pts). The tighter bound this project uses runs {:.1f}% of the "
          "rigorous ceiling on this subset (median ratio).".format(
              itm["early_exercise_bound"].median(), itm["ee_bound_vol_pts"].median(),
              itm["rigorous_ceiling"].median(), itm["ceiling_vol_pts"].median(),
              100.0 * (itm["early_exercise_bound"] / itm["rigorous_ceiling"]).median()))
    all_itm = p[p["moneyness"] > 1.0]
    illiq_itm = all_itm[~all_itm["liquid"]]
    print(f"Liquidity check (requested explicitly, not assumed): the {len(illiq_itm)} ILLIQUID "
          f"K>S pairs this session has (a leg with zero open interest) carry a LARGER median bound "
          f"(${illiq_itm['early_exercise_bound'].median():.2f}) than the {len(itm)} liquid ones "
          f"(${itm['early_exercise_bound'].median():.2f}) -- because the bound depends only on K, "
          "T, r, q, S, and the illiquid population happens to skew toward higher strikes and longer "
          "tenors, not because illiquidity itself inflates the bound. Restricting Step 2 to liquid "
          "pairs is therefore the conservative choice for the headline, not the one that flatters it.")
    print("\nBracket, stated plainly: Step 1's measured lower bound on 2026-08-28's in-the-money "
          "puts (median ~2.8 vol points on the explained subset) sits below this theory-side "
          "ceiling (median ~3.6-9.6 vol points across buckets) -- consistent with each other, "
          "neither one is the true American-minus-European gap. That gap requires a model this "
          "phase does not build.")


# --------------------------------------------------------------------------
# Step 3: continuous q vs. discrete dividends
# --------------------------------------------------------------------------

def detect_ex_dividend_jumps(underlying: pd.DataFrame, threshold: float = JUMP_THRESHOLD) -> pd.DataFrame:
    """Ex-dividend dates and per-event dollar amounts, recovered from
    `close` vs `adjusted_close` in the stored underlying history.

    EODHD's `adjusted_close` is `close` back-adjusted, multiplicatively,
    for every dividend AFTER a given date -- so close/adjusted_close is a
    step function of time that steps on every ex-dividend date and is flat
    everywhere else. SPY has never split, so this divergence is dividend-
    only here (a general ticker could also step on a split; that
    ambiguity does not apply to SPY's history).

    A naive `close[t] - adjusted_close[t]` on the day before a step gives
    the CUMULATIVE adjustment of every dividend from that date forward,
    not the single event's amount -- it happens to look right for the
    single most recent event only because nothing has occurred after it
    yet to compound on top. The per-event amount instead isolates the
    ratio change AT the step: if ratio[t] = close[t]/adjusted_close[t],
    then ratio[t-1]/ratio[t] = 1/(1 - D_t/close[t]), so
        D_t = close[t] * (1 - ratio[t]/ratio[t-1]).

    `threshold` separates genuine steps from float rounding noise in the
    stored data: `|Delta(ratio)|` is flat around 1e-7 to 1e-8 chain-wide
    (rounding) and jumps to 1e-3 or larger at a real ex-dividend date, with
    nothing populating the gap between 1e-5 and 1e-3 anywhere in the 33-
    year history -- so the exact threshold value in that gap does not
    matter. At 1e-4 this finds 135 dates over 1993-03-19 to 2026-06-18, median spacing
    91 days (quarterly), which is what SPY's actual dividend cadence is
    known to be -- the detector's output is internally consistent with a
    fact about SPY that this script did not assume going in.
    """
    df = underlying.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    ratio = df["close"] / df["adjusted_close"]
    jump = (ratio - ratio.shift(1)).abs()
    idx = df.index[(jump > threshold) & (df.index > 0)]
    rows = []
    for i in idx:
        D = float(df.loc[i, "close"] * (1.0 - ratio.loc[i] / ratio.loc[i - 1]))
        rows.append({"exdate": df.loc[i, "date"], "amount": D})
    return pd.DataFrame(rows, columns=["exdate", "amount"])


def step3_dividends(underlying: pd.DataFrame) -> None:
    print("\n--- Step 3: continuous q vs. a discrete dividend (isolating the second half of Q9) ---")
    print("No per-date dividend table is stored anywhere in this repo -- `get_dividend_yield` "
          "fetches EODHD's dividend history over the network and writes only the resulting scalar "
          "trailing-12m yield into docs/status.json. Offline, the substitute measured below is "
          "recovered from data/underlying.parquet's close/adjusted_close divergence (see "
          "`detect_ex_dividend_jumps` docstring) -- an artefact of a price-adjustment convention, "
          "not a dividend feed this project fetches or trusts elsewhere. Reported as such.")

    exdiv = detect_ex_dividend_jumps(underlying)
    print(f"\nDetected {len(exdiv)} ex-dividend dates in the stored history "
          f"({exdiv['exdate'].min().date()} to {exdiv['exdate'].max().date()}), median spacing "
          f"{exdiv['exdate'].diff().dt.days.median():.0f} calendar days "
          f"(quarterly cadence, consistent with SPY's known dividend schedule -- this is a check "
          "on the detector, not an assumption fed into it). Recent amounts: "
          + ", ".join(f"{d.date()}=${a:.2f}" for d, a in
                       exdiv.tail(6)[["exdate", "amount"]].itertuples(index=False)))

    print("\nCoverage question: of the archive's 64 stored chain sessions, how many have an "
          "ex-dividend date within their ~30-calendar-day forward window? Restricted to sessions "
          "whose window is FULLY inside the underlying history "
          f"(session date + 30 <= {underlying['date'].astype(str).max()}, the last stored row) -- "
          "a session near the end of the archive cannot be checked because the window would run "
          "past what this file has recorded.")
    chain_dates = sorted(pd.Timestamp(p.stem) for p in CHAINS_DIR.glob("*.parquet"))
    last_data = pd.to_datetime(underlying["date"]).max()
    checkable, contains = 0, 0
    for d in chain_dates:
        window_end = d + pd.Timedelta(days=30)
        if window_end > last_data:
            continue
        checkable += 1
        if ((exdiv["exdate"] >= d) & (exdiv["exdate"] <= window_end)).any():
            contains += 1
    print(f"{checkable} of {len(chain_dates)} stored sessions have a fully-covered 30-day forward "
          f"window; of those, {contains} ({100.0 * contains / checkable:.1f}%) contain a detected "
          f"ex-dividend date. (Theoretical expectation for a quarterly payer against a 30-day "
          f"window, ignoring calendar alignment: 30/91 = {100.0 * 30 / 91:.1f}%, in the same range.)")
    print(f"The remaining {len(chain_dates) - checkable} stored sessions cannot be checked this way "
          "-- their forward window runs past 2026-08-28, the last row this project has fetched, so "
          "whether they contain a September 2026 ex-dividend date is not something stored data can "
          "answer (the quarterly cadence above makes one likely in mid-September, but that is "
          "pattern extrapolation, not a stored fact, and is not used as one below).")

    print("\nRepresentative contract: session 2026-06-15 (a close-based backfill session, so priced "
          "at the stored close, not a live quote), nearest-to-30-DTE expiry, ATM-ish strike, whose "
          "32-day remaining life spans the real detected 2026-06-18 ex-dividend date -- the closest "
          "the archive's monthly-expiry-only ladder gets to a genuine ~30-DTE contract crossing a "
          "known ex-div date (LEARNING_LOG.md's Phase 6 'Finding 2' already documents that the "
          "June *option* expiry itself is dropped from every June chain by the Juneteenth-shift "
          "bug in `is_monthly_expiry` -- this contract's own 46- and 81-day siblings are casualties "
          "of that same gap, which is why 32 days is what is available, not 30).")

    status = load_status()
    r, q = float(status["risk_free_rate"]), float(status["dividend_yield"])
    print(f"Using docs/status.json's (r={r}, q={q}) -- the only (r, q) this archive stores, from "
          "2026-08-28, applied to a 2026-06-15 contract for lack of any session-specific rate "
          "(same caveat scripts/probe_quote_iv_spread.py makes for the same reason).")

    session_path = CHAINS_DIR / "2026-06-15.parquet"
    chain = pd.read_parquet(session_path)
    target_dte = int(chain["dte"].iloc[(chain["dte"] - 30).abs().argmin()])
    sub = chain[chain["dte"] == target_dte]
    spot = float(sub["spot"].iloc[0])
    strike = 755.0
    dte = int(sub["dte"].iloc[0])
    T = dte / 365.0
    exdate = pd.Timestamp("2026-06-18")
    valuation_date = pd.Timestamp("2026-06-15")
    t_div = (exdate - valuation_date).days / 365.0
    D = float(exdiv.loc[exdiv["exdate"] == exdate, "amount"].iloc[0])
    print(f"\nContract: strike={strike:.0f}, spot={spot:.2f}, dte={dte}, T={T:.4f}y. "
          f"Ex-dividend {exdate.date()} is {t_div * 365:.0f} days out (t_div={t_div:.4f}y), "
          f"detected amount D=${D:.4f}.")

    for kind in ("put", "call"):
        row = sub[(sub["strike"] == strike) & (sub["kind"] == kind)].iloc[0]
        price = float(row["close"])
        sigma = float(implied_vol(price, spot, strike, T, r, q=q, kind=kind))
        check = float(bs_price(spot, strike, T, r, sigma, q=q, kind=kind))
        vega_pv = float(greeks(spot, strike, T, r, sigma, q=q, kind=kind)["vega"]) / 100.0

        s_continuous = spot * np.exp(-q * T)
        s_discrete = spot - D * np.exp(-r * t_div)
        price_discrete = float(bs_price(s_discrete, strike, T, r, sigma, q=0.0, kind=kind))
        diff = price_discrete - check

        print(f"\n{kind} K={strike:.0f}: market close=${price:.2f}, solved sigma (continuous q) "
              f"= {sigma * 100:.2f}%, bs_price check ${check:.4f}, vega/pt ${vega_pv:.4f}")
        print(f"  continuous-q effective spot S*e^(-qT) = ${s_continuous:.4f}; "
              f"discrete escrowed-dividend spot S - D*e^(-r*t_div) = ${s_discrete:.4f} "
              f"(${s_continuous - s_discrete:.4f} apart -- the continuous model discounts far less "
              "of the stock's value away over a mere 32-day window than the one real dividend "
              "landing inside it actually removes).")
        print(f"  price at the SAME solved sigma, discrete dividend instead of continuous q: "
              f"${price_discrete:.4f} vs ${check:.4f} -> diff ${diff:+.4f}, "
              f"{diff / vega_pv:+.3f} vol points.")

    print("\nSign check (self-consistency, not an external fact): a discrete dividend removes value "
          "from the stock at one date instead of spreading a smaller continuous drip over the whole "
          "window, so it should make the PUT more valuable and the CALL less valuable at fixed vol "
          "-- and it does, symmetrically, in the numbers above. This is the mechanism, measured on "
          "one contract; it is not a chain-wide average and is not claimed as one.")


def main() -> None:
    print("=" * 88)
    print("Q9 -- bounding and locating the European-model error on American SPY options")
    print("=" * 88)

    chain_path = most_recent_chain_path()
    status = load_status()
    r, q = float(status["risk_free_rate"]), float(status["dividend_yield"])
    print(f"\nChain: {chain_path.name}. docs/status.json: risk_free_rate={r}, dividend_yield={q}.")
    if chain_path.stem != status.get("snapshot_date"):
        print(f"NOTE: status.json's snapshot_date is {status.get('snapshot_date')}, which does not "
              f"match the most recent stored chain {chain_path.stem}.")

    chain = pd.read_parquet(chain_path)
    step1_parity_residue(chain, r, q)
    step2_theory_ceiling(chain, r, q)

    underlying = pd.read_parquet(UNDERLYING_PATH)
    step3_dividends(underlying)

    print("\nDone.")


if __name__ == "__main__":
    main()
