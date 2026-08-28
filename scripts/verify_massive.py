"""Phase 0 addendum: verify Massive (ex-Polygon) options access for vol-lens.

Answers, empirically, whether the FREE options tier covers what the
pipeline needs (SPEC §2.2): a filtered end-of-day SPY chain with usable
prices — and whether bid/ask quotes are included or paywalled. Makes 5
requests, spaced 13s apart to respect the free tier's 5 calls/minute.

Usage: MASSIVE_API_KEY=... python scripts/verify_massive.py
"""
import datetime as dt
import json
import os
import sys
import time

import requests

BASES = ("https://api.polygon.io", "https://api.massive.com")
TIMEOUT = 30
PAUSE = 13  # free tier: 5 calls/min


def key() -> str:
    k = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not k:
        sys.exit("MASSIVE_API_KEY is not set. Export it and re-run.")
    return k


def get(base: str, path: str, **params) -> tuple[int, object]:
    params.setdefault("apiKey", key())
    r = requests.get(f"{base}{path}", params=params, timeout=TIMEOUT)
    try:
        body = r.json()
    except ValueError:
        body = r.text[:500]
    return r.status_code, body


def show(title: str, status: int, body: object, n_rows: int = 2) -> None:
    print(f"\n=== {title} — HTTP {status} ===")
    if isinstance(body, dict):
        for k in ("status", "error", "message"):
            if k in body and not isinstance(body[k], (dict, list)):
                print(f"{k}: {body[k]}")
        results = body.get("results")
        if isinstance(results, list):
            print(f"rows: {len(results)}", "| next_url present" if body.get("next_url") else "| no next_url")
            for row in results[:n_rows]:
                print(json.dumps(row, indent=2)[:1500])
            if results:
                print("field inventory:", sorted(results[0].keys()))
        elif isinstance(results, dict):
            print(json.dumps(results, indent=2)[:1500])
        elif results is None and "results" not in body:
            print(json.dumps(body, indent=2)[:800])
    else:
        print(str(body)[:500])


def main() -> None:
    # Resolve which base answers (rebrand kept the old domain alive; verify).
    base = None
    for b in BASES:
        try:
            status, body = get(b, "/v3/reference/options/contracts",
                               underlying_ticker="SPY", limit=1)
            base = b
            print(f"Using base: {b} (probe HTTP {status})")
            break
        except requests.RequestException as e:
            print(f"{b}: unreachable ({type(e).__name__})")
    if base is None:
        sys.exit("No API base reachable.")

    today = dt.date.today()

    # 1. Reference contracts — strikes/expiries inventory (counts as the probe above)
    show("Reference contracts (limit=1 probe)", status, body)
    time.sleep(PAUSE)

    # 2. THE make-or-break call: full chain snapshot, one page.
    #    If this works on the free key, the daily fetch is ~4-8 calls total.
    status, body = get(base, "/v3/snapshot/options/SPY", limit=250)
    show("Option chain snapshot (/v3/snapshot/options/SPY, limit=250)", status, body, n_rows=1)
    snap_results = body.get("results") if isinstance(body, dict) else None
    contract = None
    if isinstance(snap_results, list) and snap_results:
        contract = snap_results[0].get("details", {}).get("ticker")
        sample = snap_results[0]
        print("snapshot sub-objects present:",
              {k: (k in sample) for k in ("day", "last_quote", "last_trade", "greeks", "implied_volatility", "open_interest")})
    time.sleep(PAUSE)

    # 3. Fallback path pricing: previous-day aggregate for one contract.
    if contract is None:
        # Construct nothing — ask reference for one liquid contract instead.
        status, body = get(base, "/v3/reference/options/contracts",
                           underlying_ticker="SPY", limit=1,
                           **{"expiration_date.gte": str(today + dt.timedelta(days=20))})
        rows = body.get("results") if isinstance(body, dict) else None
        contract = rows[0]["ticker"] if rows else None
        time.sleep(PAUSE)
    if contract:
        status, body = get(base, f"/v2/aggs/ticker/{contract}/prev")
        show(f"Prev-day aggregate for {contract}", status, body, n_rows=1)
        time.sleep(PAUSE)

        # 4. History depth: daily aggregates ~2 years back for the underlying
        #    contract family can't be probed directly (contracts expire), so
        #    probe an old date range on this contract's aggregates instead —
        #    expect empty-but-authorized (status OK) vs 403 (paywalled).
        d0 = today - dt.timedelta(days=700)
        status, body = get(base, f"/v2/aggs/ticker/{contract}/range/1/day/{d0}/{d0 + dt.timedelta(days=10)}")
        show(f"Aggregates range probe ({d0}, expect empty-but-authorized)", status, body, n_rows=1)

    print(
        "\nDone. Record in LEARNING_LOG.md: does the free key get the chain"
        "\nsnapshot (and does it include last_quote bid/ask, greeks, IV, OI)?"
        "\nOr only reference + per-contract aggregates (close, no bid/ask)?"
        "\nAny 403/NOT_AUTHORIZED bodies verbatim; rate-limit behavior; and"
        "\nthe implied daily call budget for a filtered SPY chain each path."
    )


if __name__ == "__main__":
    main()
