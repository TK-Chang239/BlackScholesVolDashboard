"""Assemble the one static dashboard page (SPEC 2.5).

Self-contained: the vendored plotly.js bundle is inlined; no external
requests at view time. Captions are part of the deliverable -- they carry
the interview-prep story (SPEC 2.5) -- keep them verbatim.
"""
from pathlib import Path

import plotly.io as pio

_BUNDLE = Path(__file__).parent / "assets" / "plotly-cartesian-2.35.2.min.js"

CAPTIONS = {
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
        "Fed meetings. Yesterday's curve in grey shows the day-over-day shift."
    ),
}

QUESTIONS = [
    ("Q1", "Everyone has the same formula. Why does anyone disagree about prices?",
     [], "Input-sensitivity panel arrives in Phase 4."),
    ("Q2", "Where does the market ignore the model, and why?",
     ["P2", "P3"], "Model-vs-market heatmap arrives in Phase 5."),
    ("Q3", "Is implied volatility a prediction, or a price?",
     [], "Implied-vs-realized panel arrives in Phase 4."),
    ("Q4", "The model claims you can hedge an option perfectly. Can you?",
     [], "Delta-hedged P&L simulation arrives in Phase 6."),
    ("Q5", "Given all these flaws, why does every desk still use Black-Scholes?",
     [], "Put-call parity checker arrives in Phase 5."),
]

_PANEL_TITLES = {"P2": "IV smile / skew by expiry", "P3": "ATM IV term structure"}

_CSS = """
body { font-family: -apple-system, system-ui, Segoe UI, Roboto, sans-serif;
       max-width: 960px; margin: 0 auto; padding: 0 16px 48px; color: #1a1a2e; }
h1 { margin-top: 28px; } h2 { margin-top: 40px; border-bottom: 2px solid #eee;
     padding-bottom: 6px; } .meta, .caption, .placeholder, footer
     { color: #555; font-size: 0.92rem; } .caption { margin: 4px 0 24px; }
.placeholder { font-style: italic; } footer { margin-top: 48px;
     border-top: 1px solid #eee; padding-top: 12px; }
.figure { width: 100%; overflow-x: auto; }
"""


def render_page(figures: dict, status: dict) -> str:
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
        f"<b>{status.get('iv_convergence', float('nan')):.1%}</b></p>",
    ]
    for qid, question, panel_ids, coming in QUESTIONS:
        parts.append(f"<h2>{qid}. {question}</h2>")
        rendered_any = False
        for pid in panel_ids:
            if pid in figures:
                parts.append(f"<h3>{_PANEL_TITLES[pid]}</h3><div class='figure'>")
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
        "repository). Source code: GitHub (link once public).</footer>",
        "</body></html>"])
    return "".join(parts)
