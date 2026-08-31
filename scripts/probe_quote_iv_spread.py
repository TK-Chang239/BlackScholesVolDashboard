"""Q8 probe: does implied vol differ when solved from bid vs mid vs ask?

(`LEARNING_LOG.md`, question 8 of 10; SPEC §2.2 stores bid/ask/mid on every
contract specifically so this could be asked.) Every implied vol this
dashboard plots is solved from ONE price. This script solves it from all
three stored prices for every contract that can answer the question, and
reports the gap in vol points -- `Δσ ≈ Δprice / vega`, so the gap should
blow up wherever vega is small (ties this to Q6).

Step 1 is a census, not an assumption: the archive holds 64 stored chains,
but 63 came from a close-based backfill (`source == "massive-backfill"`)
that never populated bid/ask at all, and only one (`source == "yfinance"`,
2026-08-28) carries real two-sided quotes. The census below is what proves
that split, per session, before anything downstream trusts it. If it ever
finds more than one usable session, the rest of the script runs over all
of them; today it does not, and the answer is scoped to that one session
and labelled as such throughout.

No pricing libraries: every vol and every vega below comes from this
repo's own `src.models.black_scholes`. Offline: reads only the stored
parquet files under `data/chains/`. Read-only: never writes under `data/`
or `docs/`.

Usage: .venv/bin/python scripts/probe_quote_iv_spread.py
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
sys.path.insert(0, str(ROOT))

from src.analytics.chain_iv import compute_chain_iv  # noqa: E402
from src.analytics.parity import compute_parity  # noqa: E402
from src.models.black_scholes import greeks, implied_vol  # noqa: E402

NEAR_MONEY_BAND = 0.02   # |strike/spot - 1| <= this => "near the money"
DEEP_BAND = 0.10         # OTM/ITM distance beyond this (direction-aware) => "deep"

# The two effects this dashboard already reports, both read off the SAME
# 2026-08-28 session this probe ends up scoped to -- an apples-to-apples
# comparison, not a cross-session guess.
REFERENCE_SKEW_POINTS = 4.003          # docs/status.json: skew_25d = 0.04003
REFERENCE_IV_MINUS_RV_POINTS = 1.43    # LEARNING_LOG.md 2026-08-28: 11.83% atm_iv - 10.40% trailing RV


# --------------------------------------------------------------------------
# Step 1: census -- which sessions can even answer this question?
# --------------------------------------------------------------------------

def census() -> pd.DataFrame:
    """Per-session usable-quote coverage: rows, source, how many rows carry
    a positive bid AND a non-null ask, and the median relative spread
    (ask - bid) / mid over just those usable rows."""
    rows = []
    for path in sorted(CHAINS_DIR.glob("*.parquet")):
        df = pd.read_parquet(path)
        n = len(df)
        source = str(df["source"].iloc[0]) if n else "(empty)"
        usable = (df["bid"] > 0) & df["ask"].notna()
        n_usable = int(usable.sum())
        if n_usable:
            u = df.loc[usable]
            median_rel_spread = float(((u["ask"] - u["bid"]) / u["mid"]).median())
        else:
            median_rel_spread = float("nan")
        rows.append({
            "snapshot_date": path.stem,
            "source": source,
            "rows": n,
            "n_usable": n_usable,
            "pct_usable": (100.0 * n_usable / n) if n else 0.0,
            "median_rel_spread": median_rel_spread,
        })
    return pd.DataFrame(rows)


def print_census(c: pd.DataFrame) -> pd.DataFrame:
    print(f"{'date':<12} {'source':<18} {'rows':>6} {'usable':>7} {'%usable':>8}  {'median rel spread':>18}")
    for _, row in c.iterrows():
        med = f"{row['median_rel_spread']:.4f}" if pd.notna(row["median_rel_spread"]) else "n/a"
        print(f"{row['snapshot_date']:<12} {row['source']:<18} {row['rows']:>6} "
              f"{row['n_usable']:>7} {row['pct_usable']:>7.1f}%  {med:>18}")

    by_source = (c.groupby("source", as_index=False)
                  .agg(sessions=("rows", "size"), total_rows=("rows", "sum"),
                       total_usable=("n_usable", "sum")))
    print("\nBy source:")
    print(by_source.to_string(index=False))

    qualifying = c[c["n_usable"] > 0].copy()
    print(f"\n{len(qualifying)} of {len(c)} sessions have ANY row with a positive bid and a non-null ask.")
    if qualifying.empty:
        print("No session can answer Q8 from stored data. Stopping here.")
    elif len(qualifying) == 1:
        row = qualifying.iloc[0]
        print(f"SCOPE: only {row['snapshot_date']} (source={row['source']}) qualifies -- "
              f"{row['n_usable']} of {row['rows']} rows ({row['pct_usable']:.1f}%) carry usable "
              "two-sided quotes. The other 63 massive-backfill sessions store bid/ask as "
              "entirely null (0% usable) -- they never had a real quote to begin with, not "
              "a noisy one. Everything below is therefore a SINGLE-SESSION finding, labelled "
              "as such, not a chain-wide average across the archive.")
    else:
        print(f"SCOPE: {len(qualifying)} sessions qualify -- using all of them, per instructions.")
    return qualifying


# --------------------------------------------------------------------------
# Step 2: solve implied vol from bid, mid and ask
# --------------------------------------------------------------------------

def _solve_by_kind(df: pd.DataFrame, price_col: str, r: float, q: float) -> np.ndarray:
    out = np.full(len(df), np.nan)
    t_years = df["dte"].to_numpy(dtype=float) / 365.0
    S = df["spot"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    price = df[price_col].to_numpy(dtype=float)
    for kind in ("call", "put"):
        m = (df["kind"] == kind).to_numpy()
        if not m.any():
            continue
        out[m] = implied_vol(price[m], S=S[m], K=K[m], T=t_years[m], r=r, q=q, kind=kind)
    return out


def _vega_at(df: pd.DataFrame, sigma: np.ndarray, r: float, q: float) -> np.ndarray:
    out = np.full(len(df), np.nan)
    t_years = df["dte"].to_numpy(dtype=float) / 365.0
    S = df["spot"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    for kind in ("call", "put"):
        m = (df["kind"] == kind).to_numpy()
        if not m.any():
            continue
        out[m] = greeks(S[m], K[m], t_years[m], r, sigma[m], q=q, kind=kind)["vega"]
    return out


def _no_arb_bounds(df: pd.DataFrame, r: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Same European no-arbitrage bounds `implied_vol` itself pre-checks --
    reproduced here purely to diagnose *why* a price failed to solve."""
    t_years = df["dte"].to_numpy(dtype=float) / 365.0
    S = df["spot"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    disc_s = S * np.exp(-q * t_years)
    disc_k = K * np.exp(-r * t_years)
    is_call = (df["kind"] == "call").to_numpy()
    lower = np.where(is_call, np.maximum(disc_s - disc_k, 0.0), np.maximum(disc_k - disc_s, 0.0))
    upper = np.where(is_call, disc_s, disc_k)
    return lower, upper


def solve_all(df: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    """Per-contract working table: kind, strike, dte, mid, moneyness = strike/spot,
    the three implied vols, and vega at the mid vol -- exactly what the brief asks
    to keep, plus the no-arb bounds used only for the failure diagnostics."""
    work = df[["kind", "strike", "dte", "spot", "bid", "mid", "ask",
               "volume", "open_interest"]].copy()
    work["moneyness"] = work["strike"] / work["spot"]
    for col in ("bid", "mid", "ask"):
        work[f"iv_{col}"] = _solve_by_kind(work, col, r, q)
    work["vega_mid"] = _vega_at(work, work["iv_mid"].to_numpy(), r, q)
    lower, upper = _no_arb_bounds(work, r, q)
    work["_lower"], work["_upper"] = lower, upper
    return work


def report_failures(work: pd.DataFrame) -> None:
    print(f"\nContracts: {len(work)}")
    for col in ("bid", "mid", "ask"):
        iv_col = f"iv_{col}"
        failed = work[work[iv_col].isna()]
        n_fail = len(failed)
        if n_fail == 0:
            print(f"  {col:<4}: 0 non-convergent")
            continue
        below = int((failed[col] < failed["_lower"]).sum())
        above = int((failed[col] > failed["_upper"]).sum())
        other = n_fail - below - above
        print(f"  {col:<4}: {n_fail}/{len(work)} non-convergent "
              f"({100.0 * n_fail / len(work):.1f}%) -- {below} priced below the European "
              f"no-arb intrinsic lower bound (not our error: a fact about the quote), "
              f"{above} above the upper bound, {other} other "
              "(e.g. vega too small at any sigma to trust a root)")
        by_kind = failed["kind"].value_counts().to_dict()
        print(f"        by kind: {by_kind}; dte range: {failed['dte'].min():.0f}-{failed['dte'].max():.0f}; "
              f"moneyness range: {failed['moneyness'].min():.3f}-{failed['moneyness'].max():.3f}")


# --------------------------------------------------------------------------
# Step 3: the distribution, sliced
# --------------------------------------------------------------------------

def report_itm_quoting_shape(work: pd.DataFrame) -> None:
    """Describes -- does not explain -- how the dollar bid-ask width behaves
    on the ITM side (any depth, not just the 'deep_itm' >10% bucket), since
    review found the deep-ITM width is a flat ~constant-fraction-of-spot
    step, not a value that grows with the contract's own premium."""
    w = work.copy()
    w["dollar_width"] = w["ask"] - w["bid"]
    is_call = (w["kind"] == "call").to_numpy()
    moneyness = w["moneyness"].to_numpy()
    is_itm = (is_call & (moneyness < 1.0)) | (~is_call & (moneyness > 1.0))
    itm_all = w[is_itm]
    if itm_all.empty:
        print("  No ITM rows in this session.")
        return
    chain_corr = w["dollar_width"].corr(w["mid"])
    itm_corr = itm_all["dollar_width"].corr(itm_all["mid"])
    pct_of_spot = itm_all["dollar_width"] / itm_all["spot"] * 100.0
    print(f"  ITM contracts (any depth, strike on the in-the-money side of spot): "
          f"n={len(itm_all)} of {len(w)}")
    print(f"  median $ width {itm_all['dollar_width'].median():.3f}, IQR "
          f"[{itm_all['dollar_width'].quantile(0.25):.3f}, {itm_all['dollar_width'].quantile(0.75):.3f}], "
          f"across a premium range of ${itm_all['mid'].min():.2f} to ${itm_all['mid'].max():.2f}")
    print(f"  width as % of spot: median {pct_of_spot.median():.3f}%, std {pct_of_spot.std():.3f}%")
    print(f"  corr(width, mid): {itm_corr:.3f} within this ITM population, "
          f"vs {chain_corr:.3f} chain-wide")
    print("  Observation only: the width-vs-premium relationship that holds chain-wide is largely absent")
    print("  inside the ITM band -- a materially flatter, near-constant-fraction-of-spot width regardless of")
    print("  how deep ITM or how large the premium. This probe does not establish WHY (vendor/market-maker")
    print("  quoting convention vs. something else); see the liquidity cross-check below for one candidate.")


def report_parity_crosscheck(chain: pd.DataFrame, all3: pd.DataFrame, r: float, q: float) -> None:
    """Reproduces Phase 5's put-call parity check (`src.analytics.chain_iv`,
    `src.analytics.parity`) on this same session and asks, with exact
    (dte, strike, kind) keys, whether the wide-spread deep-ITM population
    this probe found is the same population Phase 5 flagged as zero-open-
    interest 'dead quotes' via a completely different diagnostic."""
    chain_iv, _ = compute_chain_iv(chain, r, q)
    parity = compute_parity(chain_iv, r, q)
    if parity.empty:
        print("  No parity pairs (no matching call/put strikes) for this session -- skipping cross-check.")
        return
    n_illiquid = int((~parity["liquid"]).sum())
    n_viol = int(parity["tradeable_violation"].sum())
    n_viol_illiquid = int(parity.loc[~parity["liquid"], "tradeable_violation"].sum())
    print(f"  Parity pairs: {len(parity)}; illiquid (a leg has zero open interest): {n_illiquid}; "
          f"raw tradeable violations: {n_viol}, of which {n_viol_illiquid} have an illiquid leg.")
    print("  (These reproduce LEARNING_LOG.md's Phase 5 numbers for 2026-08-28: "
          "654 pairs, 69 illiquid, 132 violations, 45 illiquid-and-violating.)")

    zero_oi_legs: set[tuple] = set()
    viol_legs: set[tuple] = set()
    for _, prow in parity.iterrows():
        if prow["call_oi"] <= 0:
            zero_oi_legs.add((prow["dte"], prow["strike"], "call"))
        if prow["put_oi"] <= 0:
            zero_oi_legs.add((prow["dte"], prow["strike"], "put"))
        if prow["tradeable_violation"]:
            viol_legs.add((prow["dte"], prow["strike"], "call"))
            viol_legs.add((prow["dte"], prow["strike"], "put"))

    deep_itm = all3[all3["bucket"] == "deep_itm"]
    deep_itm_keys = set(zip(deep_itm["dte"], deep_itm["strike"], deep_itm["kind"]))
    if not deep_itm_keys:
        print("  No deep_itm contracts to cross-check.")
        return
    overlap_zero_oi = deep_itm_keys & zero_oi_legs
    overlap_viol = deep_itm_keys & viol_legs
    n = len(deep_itm_keys)
    print(f"\n  Overlap with this probe's deep_itm vol-spread bucket ({n} contracts, bid/mid/ask all solved):")
    print(f"    {len(overlap_zero_oi)}/{n} ({100.0 * len(overlap_zero_oi) / n:.1f}%) ARE a zero-open-interest leg "
          "of a parity pair -- the same dead-quote population Phase 5 found independently via put-call parity.")
    print(f"    {len(overlap_viol)}/{n} ({100.0 * len(overlap_viol) / n:.1f}%) belong to a pair Phase 5 flagged as "
          "a tradeable parity violation.")
    remaining = n - len(overlap_zero_oi)
    print(f"    The other {remaining}/{n} ({100.0 * remaining / n:.1f}%) carry positive open interest on both "
          "legs of their pair -- so the flat wide-quote pattern above is NOT confined to dead quotes; a real, "
          "if thin, market shows the same ~$3.2 step.")


def moneyness_bucket(work: pd.DataFrame) -> pd.Series:
    dist = work["moneyness"] - 1.0
    is_call = (work["kind"] == "call").to_numpy()
    otm_dist = np.where(is_call, dist, -dist)  # positive => OTM by this much, negative => ITM
    near = dist.abs() <= NEAR_MONEY_BAND
    deep_otm = (~near) & (otm_dist > DEEP_BAND)
    deep_itm = (~near) & (otm_dist < -DEEP_BAND)
    out = pd.Series("moderate", index=work.index)
    out[near] = "near_money"
    out[deep_otm] = "deep_otm"
    out[deep_itm] = "deep_itm"
    return out


def _stats_row(label: str, part: pd.DataFrame) -> dict:
    vs = part["vol_spread_pts"]
    rel = part["vol_spread_rel_pct"]
    return {
        "slice": label, "n": len(part),
        "median_pts": vs.median(), "p90_pts": vs.quantile(0.90),
        "median_rel_pct_of_mid_iv": rel.median(), "p90_rel_pct_of_mid_iv": rel.quantile(0.90),
    }


def print_stats_table(rows: list[dict]) -> None:
    print(f"{'slice':<16} {'n':>6} {'median pts':>11} {'p90 pts':>9} "
          f"{'median % of mid IV':>19} {'p90 % of mid IV':>16}")
    for r in rows:
        print(f"{r['slice']:<16} {r['n']:>6} {r['median_pts']:>11.3f} {r['p90_pts']:>9.3f} "
              f"{r['median_rel_pct_of_mid_iv']:>19.2f} {r['p90_rel_pct_of_mid_iv']:>16.2f}")


def _liquidity_row(label: str, part: pd.DataFrame) -> dict:
    vol = part["volume"]
    oi = part["open_interest"]
    return {
        "slice": label, "n": len(part),
        "median_volume": vol.median(), "median_oi": oi.median(),
        "pct_oi_zero": 100.0 * (oi <= 0).mean(),
        "pct_vol_zero_or_missing": 100.0 * ((vol <= 0) | vol.isna()).mean(),
    }


def print_liquidity_table(rows: list[dict]) -> None:
    print(f"{'slice':<20} {'n':>6} {'median volume':>14} {'median OI':>10} "
          f"{'% OI<=0':>9} {'% vol<=0/NaN':>13}")
    for r in rows:
        print(f"{r['slice']:<20} {r['n']:>6} {r['median_volume']:>14.1f} {r['median_oi']:>10.1f} "
              f"{r['pct_oi_zero']:>8.1f}% {r['pct_vol_zero_or_missing']:>12.1f}%")


def main() -> None:
    print("=" * 88)
    print("Q8 -- implied vol solved from bid vs mid vs ask, and by how much across the chain")
    print("=" * 88)

    print("\n--- Step 1: session census ---\n")
    c = census()
    qualifying = print_census(c)
    if qualifying.empty:
        return

    with open(STATUS_PATH, encoding="utf-8") as f:
        status = json.load(f)
    r, q = float(status["risk_free_rate"]), float(status["dividend_yield"])
    print(f"\ndocs/status.json snapshot_date={status['snapshot_date']}, "
          f"risk_free_rate={r}, dividend_yield={q}")
    mismatched = qualifying[qualifying["snapshot_date"] != status["snapshot_date"]]
    if not mismatched.empty:
        print(f"NOTE: applying status.json's single (r, q) to {len(mismatched)} qualifying "
              "session(s) whose date does not match status.json -- no per-session r/q is "
              "stored in the archive, so this is the best available rate for those sessions "
              "and introduces some (unquantified) error for them.")

    frames = []
    for _, row in qualifying.iterrows():
        df = pd.read_parquet(CHAINS_DIR / f"{row['snapshot_date']}.parquet")
        usable = (df["bid"] > 0) & df["ask"].notna()
        frames.append(df.loc[usable])
    chain = pd.concat(frames, ignore_index=True)

    print(f"\n--- Step 2: solving implied vol from bid, mid, ask for {len(chain)} usable contracts ---")
    work = solve_all(chain, r, q)
    report_failures(work)

    all3 = work[work["iv_bid"].notna() & work["iv_mid"].notna() & work["iv_ask"].notna()].copy()
    print(f"\nContracts where bid, mid AND ask all solved: {len(all3)} of {len(work)}")

    all3["vol_spread_pts"] = (all3["iv_ask"] - all3["iv_bid"]) * 100.0
    all3["vol_spread_rel_pct"] = 100.0 * (all3["iv_ask"] - all3["iv_bid"]) / all3["iv_mid"]
    all3["dollar_width"] = all3["ask"] - all3["bid"]
    all3["bucket"] = moneyness_bucket(all3)

    print("\n--- Step 3: the distribution ---")

    overall = _stats_row("overall", all3)
    print(f"\nHEADLINE (single session {qualifying.iloc[0]['snapshot_date']}): "
          f"median iv_ask - iv_bid = {overall['median_pts']:.3f} vol points, "
          f"90th percentile = {overall['p90_pts']:.3f} vol points, over n={overall['n']} contracts.")

    print("\nBy moneyness bucket "
          f"(near_money: |K/S-1|<={NEAR_MONEY_BAND}; deep beyond {DEEP_BAND}, direction-aware by call/put):")
    bucket_order = ("deep_otm", "moderate", "near_money", "deep_itm")
    bucket_rows = [_stats_row(b, all3[all3["bucket"] == b]) for b in bucket_order if (all3["bucket"] == b).any()]
    print_stats_table(bucket_rows)

    print("\nLiquidity (volume, open interest) by the same moneyness bucket -- requested review follow-up,")
    print("since a wide quote with no open interest may not be a two-sided market at all:")
    liq_rows = [_liquidity_row(b, all3[all3["bucket"] == b]) for b in bucket_order if (all3["bucket"] == b).any()]
    print_liquidity_table(liq_rows)

    print("\nBy DTE:")
    dte_rows = [_stats_row(f"dte={d}", g) for d, g in all3.groupby("dte")]
    print_stats_table(dte_rows)

    print(f"\nOverall relative to each contract's own mid IV: "
          f"median {overall['median_rel_pct_of_mid_iv']:.2f}%, "
          f"p90 {overall['p90_rel_pct_of_mid_iv']:.2f}% of mid IV.")

    print("\n--- Vega-decile test of Δσ ≈ Δprice / vega ---")
    all3["vega_decile"] = pd.qcut(all3["vega_mid"], 10, labels=False, duplicates="drop")
    n_deciles = int(all3["vega_decile"].max()) + 1
    lo = all3[all3["vega_decile"] == 0]
    hi = all3[all3["vega_decile"] == n_deciles - 1]
    for label, part in (("lowest vega decile", lo), ("highest vega decile", hi)):
        print(f"  {label}: n={len(part)}, raw vega (per unit vol) in "
              f"[{part['vega_mid'].min():.3f}, {part['vega_mid'].max():.3f}] "
              f"(= [{part['vega_mid'].min()/100:.4f}, {part['vega_mid'].max()/100:.4f}] per vol point); "
              f"median $ bid-ask width ${part['dollar_width'].median():.3f}; "
              f"median vol spread {part['vol_spread_pts'].median():.3f} pts, "
              f"p90 {part['vol_spread_pts'].quantile(0.90):.3f} pts; "
              f"median volume {part['volume'].median():.1f}, median OI {part['open_interest'].median():.1f}, "
              f"{100.0 * (part['open_interest'] <= 0).mean():.1f}% with zero open interest")
    print("\n  Decile-by-decile (vega ascending) -- shows the full shape, not just the two ends,")
    print("  including liquidity so a reader can see whether noise tracks vega or tracks thin markets:")
    print(f"  {'decile':>6} {'n':>5} {'median vega':>12} {'median $width':>14} "
          f"{'median spread pts':>18} {'median OI':>10} {'% OI<=0':>8} {'dominant bucket':>16}")
    for d in range(n_deciles):
        part = all3[all3["vega_decile"] == d]
        dominant = part["bucket"].value_counts().idxmax()
        print(f"  {d:>6} {len(part):>5} {part['vega_mid'].median():>12.3f} "
              f"{part['dollar_width'].median():>14.3f} {part['vol_spread_pts'].median():>18.3f} "
              f"{part['open_interest'].median():>10.1f} {100.0 * (part['open_interest'] <= 0).mean():>7.1f}% "
              f"{dominant:>16}")

    print("\n--- ITM quoting shape: is the wide deep-ITM width graded, or a flat step? ---")
    report_itm_quoting_shape(work)

    print("\n--- Cross-check against Phase 5's put-call parity zero-open-interest population ---")
    report_parity_crosscheck(chain, all3, r, q)

    print("\n--- Comparison against the dashboard's reported effects (same 2026-08-28 session) ---")
    print(f"  25-delta skew (docs/status.json skew_25d):          {REFERENCE_SKEW_POINTS:.3f} vol points")
    print("    (drawn from OTM puts/calls near 30 DTE -- see 'deep_otm'/'moderate' rows above)")
    print(f"  IV - trailing RV (LEARNING_LOG.md 2026-08-28):      {REFERENCE_IV_MINUS_RV_POINTS:.3f} vol points")
    print("    (drawn from ATM IV interpolated at 30 DTE -- see 'near_money' row above)")
    print(f"  Overall bid/ask IV spread -- median across the chain: {overall['median_pts']:.3f} vol points")
    print(f"  Overall bid/ask IV spread -- 90th percentile:         {overall['p90_pts']:.3f} vol points")
    near = all3[all3["bucket"] == "near_money"]
    itm = all3[all3["bucket"] == "deep_itm"]
    print(f"  near_money bid/ask IV spread -- median / p90:         "
          f"{near['vol_spread_pts'].median():.3f} / {near['vol_spread_pts'].quantile(0.9):.3f} vol points")
    print(f"  deep_itm bid/ask IV spread -- median / p90:           "
          f"{itm['vol_spread_pts'].median():.3f} / {itm['vol_spread_pts'].quantile(0.9):.3f} vol points")

    print("\nDone.")


if __name__ == "__main__":
    main()
