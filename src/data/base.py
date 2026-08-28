"""Canonical schemas for the data layer (SPEC 2.2).

The "single DataProvider interface" is a signature contract, not a class:
chain providers implement
    get_option_chain(symbol, snapshot_date, spot, cfg) -> DataFrame[CHAIN_COLUMNS-input]
and src/run_daily.py is the only place vendors are composed/routed.
"""

CHAIN_COLUMNS = [
    "snapshot_date", "spot", "expiry", "dte", "strike", "kind",
    "bid", "ask", "mid", "close", "volume", "open_interest",
    "vendor_iv", "source",
]

UNDERLYING_COLUMNS = ["date", "close", "adjusted_close", "volume"]

LIVE_SOURCES = ("yfinance",)  # rows with real bid/ask quotes
