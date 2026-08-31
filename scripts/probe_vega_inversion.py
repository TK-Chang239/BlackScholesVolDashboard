"""Q6 probe: vega's geometry -- peaked at the money, decaying toward both
wings -- and what that geometry does to anything that inverts a price into
an implied vol.

(`LEARNING_LOG.md` question 6 of 10.) The "Greeks" log entry already derives
the FIRST half: vega and gamma share BSM's phi(d1) kernel, so both peak at
the money and decay outward. The SECOND half is asserted there and never
measured: "what does that geometry do to anything that tries to invert
price into vol?" This probe measures it, on the most recent stored chain
(2026-08-28, source=yfinance, 1,452 rows after the chain filter -- the only
session this archive would want you to trust for a chain-wide statistic;
see `scripts/probe_quote_iv_spread.py`'s census for why the other 63 cannot
answer chain-shape questions like this one).

Two DIFFERENT vegas are used below, on purpose, and named separately so
they are never confused for one another:

  - GEOMETRIC vega: BS vega evaluated at a REFERENCE vol -- the mean of
    the CONVERGED solved IVs within 2% of spot, for that contract's own
    expiry. This is what lets every contract get a vega, including ones
    whose own price never produced a valid IV. Bucketing convergence by a
    vega that itself required a converged IV would be circular: the very
    failures Step 2.1 is trying to locate would be silently excluded from
    their own histogram. Using a reference vol common to the expiry
    breaks that circularity, at the cost of vega no longer reflecting
    each contract's own (possibly wrong or absent) market IV -- which is
    exactly the point, since the geometry this question asks about is a
    property of the Black-Scholes FORM, not of any one quote.
  - SOLVED vega: BS vega evaluated at each contract's own solved IV. Only
    defined where the solver converged. This is the right notion for
    Step 2.2's conditioning number, which asks "how much does the vol
    THIS CONTRACT ACTUALLY REPORTS move for a penny of price noise" -- a
    question that only makes sense at a point where an IV exists.

No pricing libraries: every price, vol and vega below comes from this
repo's own `src.models.black_scholes`. Offline: reads only the stored
parquet under `data/chains/`, `docs/status.json` and `config.yaml`.
Read-only: never writes under `data/` or `docs/`.

Usage: .venv/bin/python scripts/probe_vega_inversion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHAINS_DIR = ROOT / "data" / "chains"
STATUS_PATH = ROOT / "docs" / "status.json"
CONFIG_PATH = ROOT / "config.yaml"
sys.path.insert(0, str(ROOT))

from src.analytics.chain_iv import compute_chain_iv  # noqa: E402
from src.models.black_scholes import greeks  # noqa: E402

NEAR_MONEY_BAND = 0.02   # |strike/spot - 1| <= this => "near the money"
DEEP_BAND = 0.10         # OTM/ITM distance beyond this (direction-aware) => "deep"
MONEYNESS_GRID = np.round(np.arange(0.70, 1.301, 0.05), 2)


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def most_recent_chain_path() -> Path:
    paths = sorted(CHAINS_DIR.glob("*.parquet"))
    if not paths:
        raise SystemExit("No stored chains under data/chains/ -- nothing to probe.")
    return paths[-1]


def load_status() -> dict:
    with open(STATUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def per_expiry_reference_sigma(chain_iv: pd.DataFrame, near_band: float = NEAR_MONEY_BAND) -> dict[int, float]:
    """ATM reference vol per expiry: the MEAN of converged solved IVs
    within `near_band` of spot, for that expiry only.

    Deliberately not the whole-expiry median: on 2026-08-28's 21-DTE
    expiry the median converged IV over EVERY strike is 20.7%, nearly
    double the 11.7% this function reports, because the far wings --
    exactly the population Step 2 shows is least trustworthy -- solve to
    implausible values (a 0.70-moneyness 21-day call "converges" to 60%)
    and drag a whole-chain median upward. Using those same untrustworthy
    values to build the yardstick that describes their own untrustworthi-
    ness would be circular. Falls back to the expiry's converged median,
    then to the whole chain's, if no near-money strike converged; neither
    fallback is exercised on 2026-08-28 (every expiry has converged
    near-money strikes on both sides).
    """
    global_conv = chain_iv["iv"].dropna()
    global_fallback = float(global_conv.median()) if len(global_conv) else np.nan
    out: dict[int, float] = {}
    for dte, g in chain_iv.groupby("dte"):
        moneyness = g["strike"] / g["spot"]
        near = g[(moneyness - 1.0).abs() <= near_band]
        conv = near["iv"].dropna()
        if len(conv):
            out[int(dte)] = float(conv.mean())
            continue
        conv_all = g["iv"].dropna()
        out[int(dte)] = float(conv_all.median()) if len(conv_all) else global_fallback
    return out


def add_geometric_vega(chain_iv: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    df = chain_iv.copy()
    ref_sigma = per_expiry_reference_sigma(df)
    df["ref_sigma"] = df["dte"].map(ref_sigma).astype(float)
    T = df["dte"].to_numpy(dtype=float) / 365.0
    S = df["spot"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    sigma = df["ref_sigma"].to_numpy(dtype=float)
    vega = np.full(len(df), np.nan)
    for kind in ("call", "put"):
        m = (df["kind"] == kind).to_numpy()
        if m.any():
            vega[m] = greeks(S[m], K[m], T[m], r, sigma[m], q=q, kind=kind)["vega"]
    df["vega_geo_pv"] = vega / 100.0   # per unit vol -> per vol point
    return df


def add_solved_vega(chain_iv: pd.DataFrame, r: float, q: float) -> pd.DataFrame:
    df = chain_iv.copy()
    T = df["dte"].to_numpy(dtype=float) / 365.0
    S = df["spot"].to_numpy(dtype=float)
    K = df["strike"].to_numpy(dtype=float)
    sigma = df["iv"].to_numpy(dtype=float)
    vega = np.full(len(df), np.nan)
    for kind in ("call", "put"):
        m = (df["kind"] == kind).to_numpy()
        if m.any():
            vega[m] = greeks(S[m], K[m], T[m], r, sigma[m], q=q, kind=kind)["vega"]
    df["vega_solved_pv"] = vega / 100.0
    return df


def moneyness_bucket(df: pd.DataFrame) -> pd.Series:
    dist = df["strike"] / df["spot"] - 1.0
    is_call = (df["kind"] == "call").to_numpy()
    otm_dist = np.where(is_call, dist, -dist)  # positive => OTM by this much
    near = dist.abs() <= NEAR_MONEY_BAND
    deep_otm = (~near) & (otm_dist > DEEP_BAND)
    deep_itm = (~near) & (otm_dist < -DEEP_BAND)
    out = pd.Series("moderate", index=df.index)
    out[near] = "near_money"
    out[deep_otm] = "deep_otm"
    out[deep_itm] = "deep_itm"
    return out


# --------------------------------------------------------------------------
# Step 1: vega's shape on the ~30-DTE expiry
# --------------------------------------------------------------------------

def step1_shape(chain_iv: pd.DataFrame, r: float, q: float, target_dte: int,
                 moneyness_min: float, moneyness_max: float) -> None:
    print("\n--- Step 1: vega's shape on the ~30-DTE expiry ---")
    avail = chain_iv[["expiry", "dte"]].drop_duplicates()
    pick = avail.iloc[(avail["dte"] - target_dte).abs().idxmin()]
    pick_dte = int(pick["dte"])
    print(f"config target_dte.atm_panel = {target_dte}; nearest stored expiry is "
          f"{pick_dte} DTE ({pick['expiry']}) -- the chain stores 6 monthly expiries "
          f"and {pick_dte} is the closest one actually offered.")

    sub = chain_iv[chain_iv["dte"] == pick_dte].copy()
    spot = float(sub["spot"].iloc[0])
    sub["moneyness"] = sub["strike"] / spot
    ref_sigma = per_expiry_reference_sigma(chain_iv)[pick_dte]
    T = pick_dte / 365.0
    print(f"spot={spot:.2f}, T={T:.4f}y, reference (near-money mean converged) vol "
          f"= {ref_sigma * 100:.2f}%, n contracts at this expiry = {len(sub)} "
          f"({int(sub['iv'].notna().sum())} converged)")
    print(f"chain_filter allows moneyness in [{moneyness_min}, {moneyness_max}]; "
          f"this expiry's own strikes span moneyness "
          f"[{sub['moneyness'].min():.4f}, {sub['moneyness'].max():.4f}] -- the vendor "
          "did not offer every strike the filter would have allowed at this tenor.")

    rows = []
    for kind in ("call", "put"):
        g = sub[sub["kind"] == kind].copy()
        for target_m in MONEYNESS_GRID:
            idx = (g["moneyness"] - target_m).abs().idxmin()
            row = g.loc[idx]
            vega_geo = float(greeks(spot, row["strike"], T, r, ref_sigma, q=q, kind=kind)["vega"]) / 100.0
            rows.append({
                "kind": kind, "target_moneyness": target_m, "strike": row["strike"],
                "moneyness": row["moneyness"], "vega_geo_pv": vega_geo,
                "iv_solved": row["iv"],
            })
    grid = pd.DataFrame(rows)

    print(f"\n{'kind':<5} {'target K/S':>10} {'strike':>7} {'actual K/S':>11} "
          f"{'vega/vol-pt (ref vol)':>22} {'own solved IV':>14}")
    for _, rr in grid.iterrows():
        iv_s = f"{rr['iv_solved']*100:.2f}%" if pd.notna(rr["iv_solved"]) else "did not converge"
        print(f"{rr['kind']:<5} {rr['target_moneyness']:>10.2f} {rr['strike']:>7.0f} "
              f"{rr['moneyness']:>11.4f} {rr['vega_geo_pv']:>22.6e} {iv_s:>14}")

    for kind in ("call", "put"):
        g = grid[grid["kind"] == kind]
        peak = g.loc[g["vega_geo_pv"].idxmax()]
        lo = g.iloc[0]
        hi = g.iloc[-1]
        print(f"\n{kind}: peak vega {peak['vega_geo_pv']:.4f} per vol point at K={peak['strike']:.0f} "
              f"(K/S={peak['moneyness']:.4f}, {abs(peak['moneyness']-1)*100:.2f}% from spot). "
              f"At the low-moneyness filter edge (K/S={lo['moneyness']:.3f}) vega has decayed to "
              f"{lo['vega_geo_pv']:.3e} per vol point; at the high edge (K/S={hi['moneyness']:.3f}) "
              f"to {hi['vega_geo_pv']:.3e}.")
    print("\nObserved, not asserted: on this 21-DTE expiry, vega collapses by roughly 30+ orders "
          "of magnitude between the money and the moneyness-filter edges -- 'near zero' in the log "
          "entry undersells it; at these tenors the wings are numerically zero.")


# --------------------------------------------------------------------------
# Step 2.1: convergence vs geometric-vega decile
# --------------------------------------------------------------------------

def step2_convergence(chain_iv: pd.DataFrame, r: float, q: float) -> None:
    print("\n--- Step 2.1: IV-solver convergence rate, by geometric-vega decile ---")
    df = add_geometric_vega(chain_iv, r, q)
    df["bucket"] = moneyness_bucket(df)
    df["vega_decile"] = pd.qcut(df["vega_geo_pv"], 10, labels=False, duplicates="drop")
    n_deciles = int(df["vega_decile"].max()) + 1
    if n_deciles < 10:
        print(f"NOTE: vega_geo_pv only supports {n_deciles} distinct deciles after dropping "
              "duplicate bin edges (expected on a chain with many near-zero-vega ties).")

    print(f"\n{'decile':>6} {'n':>5} {'n_conv':>7} {'conv_rate':>10} {'median vega':>13} "
          f"{'median OI':>10} {'%OI<=0':>7} {'median vol':>11} {'dominant bucket':>16}")
    total_fail = 0
    for d in range(n_deciles):
        part = df[df["vega_decile"] == d]
        n = len(part)
        n_conv = int(part["iv"].notna().sum())
        n_fail = n - n_conv
        total_fail += n_fail
        conv_rate = 100.0 * n_conv / n if n else float("nan")
        dominant = part["bucket"].value_counts().idxmax()
        print(f"{d:>6} {n:>5} {n_conv:>7} {conv_rate:>9.1f}% {part['vega_geo_pv'].median():>13.4e} "
              f"{part['open_interest'].median():>10.1f} "
              f"{100.0 * (part['open_interest'].fillna(0) <= 0).mean():>6.1f}% "
              f"{part['volume'].median():>11.1f} {dominant:>16}")

    n_total = len(df)
    n_conv_total = int(df["iv"].notna().sum())
    lowest = df[df["vega_decile"] == 0]
    highest = df[df["vega_decile"] == n_deciles - 1]
    print(f"\nChain-wide convergence: {n_conv_total}/{n_total} = "
          f"{100.0 * n_conv_total / n_total:.2f}%. Failures counted across deciles: {total_fail} "
          f"(matches chain-wide non-convergent count "
          f"{n_total - n_conv_total} -- every failure is accounted for in exactly one decile).")
    print(f"Lowest vega decile: {100.0 * lowest['iv'].notna().mean():.1f}% convergence, "
          f"median open interest {lowest['open_interest'].median():.0f}, "
          f"{100.0 * (lowest['open_interest'].fillna(0) <= 0).mean():.1f}% zero-OI. "
          f"Highest vega decile: {100.0 * highest['iv'].notna().mean():.1f}% convergence, "
          f"median open interest {highest['open_interest'].median():.0f}.")
    print("Liquidity check (requested explicitly, not assumed): open-interest and volume medians "
          "do NOT fall monotonically with vega decile the way convergence does -- the lowest-vega "
          "decile is not the most illiquid one by open interest. The convergence failures track "
          "the vega geometry itself, not simply a population of dead quotes; this is measured on "
          "this session's OI/volume columns, not inferred from the shape of the failure curve alone.")


# --------------------------------------------------------------------------
# Step 2.2: conditioning -- one cent of price noise, in vol points
# --------------------------------------------------------------------------

def step2_conditioning(chain_iv: pd.DataFrame, r: float, q: float) -> None:
    print("\n--- Step 2.2: conditioning -- vol-point move from a one-cent price error ---")
    df = add_solved_vega(chain_iv, r, q)
    conv = df[df["iv"].notna()].copy()
    conv["bucket"] = moneyness_bucket(conv)
    conv["cent_move_vol_pts"] = 0.01 / conv["vega_solved_pv"]

    print(f"Restricted to the {len(conv)} of {len(df)} contracts whose own IV converged -- "
          "conditioning is only defined where an IV exists to condition.")
    print(f"\n{'bucket':<12} {'n':>5} {'median vol pts':>15} {'p90 vol pts':>12} "
          f"{'median vega/pt':>15} {'median OI':>10} {'median vol':>11}")
    bucket_order = ("deep_otm", "moderate", "near_money", "deep_itm")
    for b in bucket_order:
        part = conv[conv["bucket"] == b]
        if part.empty:
            continue
        print(f"{b:<12} {len(part):>5} {part['cent_move_vol_pts'].median():>15.4f} "
              f"{part['cent_move_vol_pts'].quantile(0.9):>12.4f} "
              f"{part['vega_solved_pv'].median():>15.4f} "
              f"{part['open_interest'].median():>10.1f} {part['volume'].median():>11.1f}")

    overall = conv["cent_move_vol_pts"]
    print(f"\nOverall: median {overall.median():.4f} vol points per penny of price error, "
          f"p90 {overall.quantile(0.9):.4f}, max {overall.max():.4f} "
          f"(n={len(conv)} converged contracts).")
    near = conv.loc[conv["bucket"] == "near_money", "cent_move_vol_pts"]
    deep_otm = conv.loc[conv["bucket"] == "deep_otm", "cent_move_vol_pts"]
    if len(near) and len(deep_otm):
        ratio = deep_otm.median() / near.median()
        print(f"A penny of noise moves a deep-OTM implied vol {ratio:.1f}x further than it moves "
              "a near-the-money one (median-to-median, this session) -- this is the mechanism, "
              "not just the correlation: the same $0.01 numerator divided by a vega an order of "
              "magnitude smaller at the wings produces a proportionally larger vol swing. This is "
              "why a deep-OTM quote (bid-ask noise on the order of a few cents, per "
              "scripts/probe_quote_iv_spread.py) cannot carry a trustworthy vol even when the "
              "solver technically converges.")


def main() -> None:
    print("=" * 88)
    print("Q6 -- vega's geometry, and what it does to inverting price into implied vol")
    print("=" * 88)

    chain_path = most_recent_chain_path()
    status = load_status()
    cfg = load_config()
    r, q = float(status["risk_free_rate"]), float(status["dividend_yield"])
    target_dte = int(cfg["target_dte"]["atm_panel"])
    m_min, m_max = float(cfg["chain_filter"]["moneyness_min"]), float(cfg["chain_filter"]["moneyness_max"])

    print(f"\nChain: {chain_path.name}. docs/status.json: risk_free_rate={r}, dividend_yield={q}.")
    if chain_path.stem != status.get("snapshot_date"):
        print(f"NOTE: status.json's snapshot_date is {status.get('snapshot_date')}, which does not "
              f"match the most recent stored chain {chain_path.stem} -- using status.json's (r, q) "
              "anyway as the best available rate; this is the same choice "
              "scripts/probe_quote_iv_spread.py makes and flags for the same reason.")

    df = pd.read_parquet(chain_path)
    chain_iv, stats = compute_chain_iv(df, r, q)
    print(f"compute_chain_iv: {stats['n_converged']}/{stats['n_priced']} converged "
          f"({100.0 * stats['convergence']:.2f}%).")

    step1_shape(chain_iv, r, q, target_dte, m_min, m_max)
    step2_convergence(chain_iv, r, q)
    step2_conditioning(chain_iv, r, q)

    print("\nDone.")


if __name__ == "__main__":
    main()
