"""P4: ATM Greeks tiles + delta/gamma curves at each strike's own IV."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest
import yaml

from src.analytics.chain_iv import compute_chain_iv
from src.analytics.greeks_panel import CURVE_COLUMNS, TILE_COLUMNS, compute_greeks_panel
from src.data.base import CHAIN_COLUMNS
from src.models.black_scholes import bs_price, greeks

TODAY = dt.date(2026, 8, 28)
SPOT, R, Q = 770.0, 0.0415, 0.0098


def cfg():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


def make_chain(expiries=((dt.date(2026, 9, 25), 28), (dt.date(2026, 11, 20), 84)),
               strikes=tuple(np.arange(700.0, 841.0, 10.0)), sigma=0.22):
    rows = []
    for expiry, dte in expiries:
        for strike in strikes:
            for kind in ("call", "put"):
                px = float(bs_price(SPOT, strike, dte / 365.0, R, sigma, Q, kind))
                rows.append({"snapshot_date": TODAY, "spot": SPOT, "expiry": expiry,
                             "dte": dte, "strike": strike, "kind": kind,
                             "bid": px * 0.99, "ask": px * 1.01, "mid": px, "close": px,
                             "volume": 10, "open_interest": 100.0, "vendor_iv": np.nan,
                             "source": "yfinance"})
    out, _ = compute_chain_iv(pd.DataFrame(rows, columns=CHAIN_COLUMNS), R, Q)
    return out


class TestTiles:
    def test_picks_nearest_expiry_and_atm_strike_per_kind(self):
        tiles, _ = compute_greeks_panel(make_chain(), R, Q, cfg())
        assert list(tiles.columns) == TILE_COLUMNS
        assert sorted(tiles["kind"]) == ["call", "put"]
        assert (tiles["dte"] == 28).all() and (tiles["strike"] == SPOT).all()

    def test_values_match_closed_form_and_call_delta_near_half(self):
        tiles, _ = compute_greeks_panel(make_chain(), R, Q, cfg())
        call = tiles[tiles["kind"] == "call"].iloc[0]
        ref = greeks(SPOT, SPOT, 28 / 365.0, R, call["iv"], Q, "call")
        for g in ("delta", "gamma", "vega", "theta", "rho"):
            assert call[g] == pytest.approx(ref[g], rel=1e-9)
        assert call["delta"] == pytest.approx(0.5, abs=0.05)

    def test_put_call_delta_parity(self):
        tiles, _ = compute_greeks_panel(make_chain(), R, Q, cfg())
        t = tiles.set_index("kind")
        assert t.loc["call", "delta"] - t.loc["put", "delta"] == pytest.approx(
            np.exp(-Q * 28 / 365.0), abs=1e-9)
        assert t.loc["call", "gamma"] == pytest.approx(t.loc["put", "gamma"], rel=1e-9)

    def test_missing_kind_yields_one_tile(self):
        chain = make_chain()
        chain = chain[chain["kind"] == "call"]
        tiles, curves = compute_greeks_panel(chain, R, Q, cfg())
        assert list(tiles["kind"]) == ["call"] and (curves["kind"] == "call").all()


class TestCurves:
    def test_gamma_peaks_near_spot_and_delta_is_monotone(self):
        _, curves = compute_greeks_panel(make_chain(), R, Q, cfg())
        assert list(curves.columns) == CURVE_COLUMNS
        calls = curves[curves["kind"] == "call"].sort_values("strike")
        peak = calls.loc[calls["gamma"].idxmax(), "strike"]
        assert abs(peak / SPOT - 1) < 0.02
        assert calls["delta"].is_monotonic_decreasing
        puts = curves[curves["kind"] == "put"].sort_values("strike")
        assert (puts["delta"] <= 0).all() and puts["delta"].is_monotonic_decreasing

    def test_curves_use_each_strikes_own_iv(self):
        chain = make_chain()
        # poison one strike's IV; its delta must move, neighbours must not
        chain.loc[(chain["strike"] == 800.0) & (chain["kind"] == "call") & (chain["dte"] == 28),
                  "iv"] = 0.50
        _, curves = compute_greeks_panel(chain, R, Q, cfg())
        calls = curves[curves["kind"] == "call"].set_index("strike")
        ref_800 = greeks(SPOT, 800.0, 28 / 365.0, R, 0.50, Q, "call")["delta"]
        ref_790 = greeks(SPOT, 790.0, 28 / 365.0, R, 0.22, Q, "call")["delta"]
        assert calls.loc[800.0, "delta"] == pytest.approx(ref_800, rel=1e-6)
        assert calls.loc[790.0, "delta"] == pytest.approx(ref_790, rel=1e-6)

    def test_empty_chain(self):
        tiles, curves = compute_greeks_panel(make_chain().iloc[0:0], R, Q, cfg())
        assert tiles.empty and curves.empty
        assert list(tiles.columns) == TILE_COLUMNS and list(curves.columns) == CURVE_COLUMNS
