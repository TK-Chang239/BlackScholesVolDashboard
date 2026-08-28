"""Phase 0 addendum: verify Alpha Vantage options access for vol-lens.

The EODHD account has no options entitlement (see LEARNING_LOG Phase 0
findings), so option chains move to Alpha Vantage HISTORICAL_OPTIONS
(SPEC §2.2, revised 2026-08-28). This probe confirms, before Phase 2
builds on it: endpoint access, field inventory, history depth, and the
free-tier rate-limit behavior. Read-only; makes 4 requests.

Usage: ALPHAVANTAGE_API_KEY=... python scripts/verify_alphavantage.py
"""
import datetime as dt
import json
import os
import sys

import requests

BASE = "https://www.alphavantage.co/query"
TIMEOUT = 30


def key() -> str:
    k = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    if not k:
        sys.exit("ALPHAVANTAGE_API_KEY is not set. Export it and re-run.")
    return k


def get(**params) -> tuple[int, object]:
    params.setdefault("apikey", key())
    r = requests.get(BASE, params=params, timeout=TIMEOUT)
    try:
        body = r.json()
    except ValueError:
        body = r.text[:500]
    return r.status_code, body


def show(title: str, status: int, body: object, n_rows: int = 2) -> None:
    print(f"\n=== {title} — HTTP {status} ===")
    if isinstance(body, dict):
        # AV signals errors/limits inside a 200 body; surface those keys.
        for k in ("Error Message", "Information", "Note"):
            if k in body:
                print(f"{k}: {body[k]}")
        data = body.get("data")
        if isinstance(data, list):
            print(f"rows: {len(data)}")
            for row in data[:n_rows]:
                print(json.dumps(row, indent=2)[:1200])
            if data:
                print("field inventory:", sorted(data[0].keys()))
        elif "data" not in body and not any(
            k in body for k in ("Error Message", "Information", "Note")
        ):
            print(json.dumps(body, indent=2)[:1500])
    else:
        print(str(body)[:500])


def main() -> None:
    today = dt.date.today()

    # 1. Latest available chain (no date param -> previous session)
    status, body = get(function="HISTORICAL_OPTIONS", symbol="SPY")
    show("SPY chain, latest session (no date param)", status, body)

    # 2. ~6 months back — the SPEC §2.2 backfill target depth
    d6 = today - dt.timedelta(days=182)
    while d6.weekday() > 4:  # walk to a weekday
        d6 -= dt.timedelta(days=1)
    status, body = get(function="HISTORICAL_OPTIONS", symbol="SPY", date=str(d6))
    show(f"SPY chain, ~6 months back ({d6})", status, body, n_rows=1)

    # 3. ~5 years back — probes the claimed deep history
    d5y = today - dt.timedelta(days=5 * 365)
    while d5y.weekday() > 4:
        d5y -= dt.timedelta(days=1)
    status, body = get(function="HISTORICAL_OPTIONS", symbol="SPY", date=str(d5y))
    show(f"SPY chain, ~5 years back ({d5y})", status, body, n_rows=1)

    # 4. A known market holiday — how does "no data" present vs an error?
    status, body = get(
        function="HISTORICAL_OPTIONS", symbol="SPY", date=f"{today.year}-01-01"
    )
    show(f"SPY chain on a holiday ({today.year}-01-01)", status, body, n_rows=1)

    print(
        "\nDone. Record in LEARNING_LOG.md: whether HISTORICAL_OPTIONS answered"
        "\nwith data on the free key, the field inventory (bid/ask/mark/volume/"
        "\nopen_interest/expiration/strike/type, vendor IV+greeks), observed"
        "\nhistory depth (6mo and 5y probes), how holidays/no-data present,"
        "\nand any rate-limit 'Information'/'Note' messages encountered."
    )


if __name__ == "__main__":
    main()
