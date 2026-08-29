"""Assemble the one static dashboard page (SPEC 2.5).

Self-contained: the vendored plotly.js bundle is inlined; no external
requests at view time. Captions are part of the deliverable -- they carry
the interview-prep story (SPEC 2.5) -- keep them verbatim.
"""
import math
from pathlib import Path

import plotly.io as pio

from src.render.tiles import TILES_CSS

_BUNDLE = Path(__file__).parent / "assets" / "plotly-cartesian-2.35.2.min.js"


def _fmt_pct(val) -> str:
    """Format a fraction as a percent, or 'n/a' when missing/NaN."""
    if val is None or (isinstance(val, (int, float)) and math.isnan(val)):
        return "n/a"
    return f"{val:.1%}"


CAPTIONS = {
    "P1": (
        "Five inputs, one formula. Bumping each by ±20% shows that among the "
        "inputs a trader must estimate, volatility is the only one worth arguing "
        "about — the rate and dividend bars barely register. Spot and time are "
        "shown separately because nobody disagrees about them: a 20% move in "
        "spot is a crash, not a difference of opinion. This is why option desks "
        "quote volatility, not price."
    ),
    "P2": (
        "Black-Scholes assumes one volatility should price every strike. The "
        "market disagrees: implied vol rises steadily for lower strikes — the "
        "skew — because crash protection commands a premium the model does not "
        "anticipate. Each line is one expiry, and every point was solved from a "
        "real quote by this project's own Newton–Brent inverter."
    ),
    "P3": (
        "Each point is the at-the-money implied vol for one expiry — the "
        "market's volatility price over that horizon. An upward slope reads as "
        "calm now, more risk priced later; kinks flag scheduled events such as "
        "Fed meetings. When a previous session is stored, its curve appears "
        "in grey — labeled with its date — showing the day-over-day shift."
    ),
    "P4": (
        "The Greeks are the model's partial derivatives — how the price responds "
        "to spot (delta), how delta itself responds (gamma), and how the price "
        "bleeds with time (theta) or breathes with volatility (vega). Each tile "
        "is computed by this project's own closed-form code for the at-the-money "
        "~30-day option; the curves evaluate every strike at its own market "
        "implied vol, which is what a hedging desk actually uses. Gamma peaking "
        "at the money is the engine behind the hedged-P&L story in Q4."
    ),
    "P5": (
        "Implied vol is the market's price for the next month of movement; "
        "realized vol is what actually happened. The shaded gap is implied minus "
        "trailing realized — the live view — but the honest scorecard is the "
        "dashed forward series: the vol that subsequently occurred after each "
        "quote. When implied sits persistently above forward realized, option "
        "sellers are being paid a premium for bearing variance risk; that "
        "premium, not forecasting skill, is what the gap usually measures."
    ),
    "P6": (
        "The 25-delta skew is the price of crash insurance in one number: the "
        "implied vol of a put that pays off on a roughly one-in-four down-move, "
        "minus that of the mirror-image call, at the ~30-day expiry — interpolated "
        "in delta space from this project's own Greeks. Black-Scholes says it "
        "should be zero. For equity indices it almost never is, and it jumps on "
        "risk-off days; the annotations mark the ones we have noted."
    ),
    "P7": (
        "Put-call parity is the one relationship here that needs no model at all: "
        "a call minus a put must equal a forward, or there is a free lunch. Cells "
        "show the dollar gap between the two sides, measured against each expiry's "
        "own market-implied forward — the median of what every near-the-money "
        "strike's own C − P implies, so a stale close on the underlying cannot "
        "masquerade as an arbitrage. That calibration costs something worth "
        "naming: a pure level error at the money is absorbed into the forward "
        "and can no longer register here, so what is left under test is whether "
        "the rest of the strike ladder is the right straight line through that "
        "point. Anything smaller than the combined bid–ask spread, or on a "
        "strike where one leg has no open interest, is muted because it cannot be "
        "traded. Put-rich gaps often survive that are still not arbitrage — SPY "
        "options are American, and early exercise makes European parity only "
        "approximate for an in-the-money put — so those are muted too, whenever "
        "the gap fits inside K·(1 − e⁻ʳᵀ) − S·(1 − e⁻ᵠᵀ) plus the combined bid–ask: "
        "the interest earned on the strike over the option's remaining life, "
        "less the dividends forgone by giving up the stock. That is tighter than "
        "the rigorous ceiling on early exercise, K·(1 − e⁻ʳᵀ), so it explains away "
        "less than it could. Only what survives every test is left burning. This "
        "model-free check is why desks trust Black-Scholes "
        "as a quoting convention even when they distrust it as a forecast."
    ),
    "P9": (
        "Price every contract with one flat volatility — today's ~30-day "
        "at-the-money implied vol — and compare to the market. The error is not "
        "noise: the wings are systematically rich against the flat-vol model, on "
        "both sides, day after day. A structured error means the world has a "
        "feature the model lacks (fat tails, volatility that moves), which is "
        "exactly why Heston and SABR exist. This is the smile of the panel above, "
        "re-expressed in price."
    ),
}

QUESTIONS = [
    ("Q1", "Everyone has the same formula. Why does anyone disagree about prices?",
     ["P1", "P4"], ""),
    ("Q2", "Where does the market ignore the model, and why?",
     ["P2", "P3", "P6", "P9"], ""),
    ("Q3", "Is implied volatility a prediction, or a price?",
     ["P5"], ""),
    ("Q4", "The model claims you can hedge an option perfectly. Can you?",
     [], "Delta-hedged P&L simulation arrives in Phase 6."),
    ("Q5", "Given all these flaws, why does every desk still use Black-Scholes?",
     ["P7"], ""),
]

_PANEL_TITLES = {
    "P1": "Input sensitivity (tornado)",
    "P2": "IV smile / skew by expiry",
    "P3": "ATM IV term structure",
    "P4": "Greeks — ATM ~30-DTE call & put",
    "P5": "Implied vs realized volatility",
    "P6": "25-delta skew — time series",
    "P7": "Put-call parity checker",
    "P9": "Model-vs-market price heatmap",
}

_CSS = """
body { font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
       max-width: 960px; margin: 0 auto; padding: 0 16px 48px; color: #1a1a2e; }
h1 { margin-top: 28px; } h2 { margin-top: 40px; border-bottom: 2px solid #eee;
     padding-bottom: 6px; } .meta, .caption, .placeholder, footer
     { color: #555; font-size: 0.92rem; } .caption { margin: 4px 0 24px; }
.placeholder { font-style: italic; } footer { margin-top: 48px;
     border-top: 1px solid #eee; padding-top: 12px; }
.figure { width: 100%; overflow-x: auto; }
""" + TILES_CSS + ".stat { font-weight: 600; margin: 4px 0 8px; }\n"


def render_page(figures: dict, status: dict, extras: dict | None = None) -> str:
    bundle = _BUNDLE.read_text()
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>vol-lens — Black-Scholes vs the market, daily</title>",
        f"<style>{_CSS}</style>",
        f"<script>{bundle}</script>",
        "</head><body>",
        "<h1>vol-lens</h1>",
        "<p class='meta'>A daily reality-check of the Black-Scholes model "
        "against the live US options market. Underlying: <b>SPY</b> · "
        f"spot <b>{status['spot']:.2f}</b> · session "
        f"<b>{status['snapshot_date']}</b> · chain source "
        f"<b>{status['source']}</b> · IV solver convergence "
        f"<b>{_fmt_pct(status.get('iv_convergence'))}</b></p>",
    ]
    for qid, question, panel_ids, coming in QUESTIONS:
        parts.append(f"<h2>{qid}. {question}</h2>")
        rendered_any = False
        for pid in panel_ids:
            if pid in figures:
                parts.append(f"<h3>{_PANEL_TITLES[pid]}</h3>")
                parts.append((extras or {}).get(pid, ""))
                parts.append("<div class='figure'>")
                parts.append(pio.to_html(
                    figures[pid], full_html=False, include_plotlyjs=False,
                    config={"responsive": True, "displaylogo": False}))
                parts.append(f"</div><p class='caption'>{CAPTIONS[pid]}</p>")
                rendered_any = True
        if not rendered_any:
            parts.append(f"<p class='placeholder'>{coming}</p>")
    parts.extend([
        "<footer>Educational project — simulated analytics on end-of-day "
        f"data; nothing here is investment advice. Last updated "
        f"{status['last_success_utc']} · {status['rows_stored']} contracts "
        "stored · SPY options are American-style, priced here with a "
        "European model (a measured approximation, discussed in the "
        "repository). Source code: <a href='https://github.com/TK-Chang239/"
        "BlackScholesVolDashboard'>github.com/TK-Chang239/"
        "BlackScholesVolDashboard</a>.</footer>",
        "</body></html>"])
    return "".join(parts)
