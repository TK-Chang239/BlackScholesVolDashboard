"""Phase 0 gate: verify EODHD data access for vol-lens (SPEC §5 Phase 0).

Probes every data source the pipeline needs and prints a report to paste
into LEARNING_LOG.md. Read-only; makes a handful of requests.

Usage: EODHD_API_TOKEN=... python scripts/verify_eodhd.py
"""
import datetime as dt
import json
import os
import sys

import requests

BASE = "https://eodhd.com/api"
TIMEOUT = 30


def token() -> str:
    tok = os.environ.get("EODHD_API_TOKEN", "").strip()
    if not tok:
        sys.exit("EODHD_API_TOKEN is not set. Export it and re-run.")
    return tok


def get(path: str, **params) -> tuple[int, object]:
    params.setdefault("api_token", token())
    params.setdefault("fmt", "json")
    r = requests.get(f"{BASE}/{path}", params=params, timeout=TIMEOUT)
    try:
        body = r.json()
    except ValueError:
        body = r.text[:500]
    return r.status_code, body


def show(title: str, status: int, body: object, n: int = 1) -> None:
    print(f"\n=== {title} — HTTP {status} ===")
    if isinstance(body, list):
        print(f"rows: {len(body)}")
        for row in body[:n]:
            print(json.dumps(row, indent=2)[:1500])
    elif isinstance(body, dict):
        print(json.dumps(body, indent=2)[:2000])
    else:
        print(str(body)[:500])


def main() -> None:
    today = dt.date.today()

    # 1. Account/plan details (also confirms the token is valid)
    status, body = get("user")
    show("Account details (plan, rate limits, credits)", status, body)

    # 2. Underlying EOD history — needed for spot + realized vol
    status, body = get("eod/SPY.US", **{"from": str(today - dt.timedelta(days=30))})
    show("SPY underlying EOD (last 30d)", status, body, n=2)

    # 3. Dividends — needed for trailing-12m dividend yield q (SPEC §2.2)
    status, body = get("div/SPY.US", **{"from": str(today - dt.timedelta(days=400))})
    show("SPY dividends (trailing ~13 months)", status, body, n=3)

    # 4. Options chain — try the current marketplace endpoint first,
    #    then the legacy endpoint. Whichever answers 200 wins.
    status, body = get(
        "mp/unicornbay/options/eod",
        **{"filter[underlying_symbol]": "SPY", "page[limit]": 3},
    )
    show("Options EOD — marketplace (mp/unicornbay/options/eod)", status, body)
    if status != 200:
        status, body = get("options/SPY.US")
        show("Options EOD — legacy (options/SPY.US)", status, body)

    # 5. Options history depth — same endpoint, ~6 months back (SPEC §2.2 backfill)
    past = today - dt.timedelta(days=180)
    status, body = get(
        "mp/unicornbay/options/eod",
        **{
            "filter[underlying_symbol]": "SPY",
            "filter[date_from]": str(past),
            "filter[date_to]": str(past + dt.timedelta(days=4)),
            "page[limit]": 3,
        },
    )
    show(f"Options history probe ({past})", status, body)

    # 6. Risk-free rate candidates (3m T-bill; fallback lives in config.yaml)
    for sym in ("US3M.INDX", "US3M.GBOND"):
        status, body = get(f"eod/{sym}", **{"from": str(today - dt.timedelta(days=14))})
        show(f"Risk-free candidate {sym}", status, body, n=1)
        if status == 200:
            break

    print(
        "\nDone. Record in LEARNING_LOG.md: which options endpoint answered 200,"
        "\nthe field names present (bid/ask/last/volume/open_interest/expiry/"
        "\nstrike/type, any vendor IV/greeks), history depth, rate limits from"
        "\nthe account payload, and roughly when today's options rows appeared"
        "\n(check the data's own date field vs the cron hour, SPEC §2.1)."
    )


if __name__ == "__main__":
    main()
