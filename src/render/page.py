"""Assemble the one static dashboard page (SPEC 2.5).

Self-contained: the vendored plotly.js bundle is inlined; no external
requests at view time. Captions are part of the deliverable -- they carry
the interview-prep story (SPEC 2.5) -- keep them verbatim.
"""
import datetime as dt
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
        "traded. Put-rich gaps often survive that are not evidence of arbitrage "
        "either — SPY options are American, and early exercise makes European "
        "parity only approximate for an in-the-money put — so those are muted too, "
        "whenever the gap fits inside K·(1 − e⁻ʳᵀ) − S·(1 − e⁻ᵠᵀ) plus the combined "
        "bid–ask: the interest earned on the strike over the option's remaining "
        "life, less the dividends forgone by giving up the stock. That is tighter "
        "than the rigorous ceiling on early exercise, K·(1 − e⁻ʳᵀ), so it explains "
        "away less than it could. Muting there is a claim about consistency, not "
        "proof: a gap inside the bound is consistent with early exercise, but a "
        "stale mark of the same size fits the bound just as comfortably, and the "
        "bound cannot tell them apart. Only what survives every test is left "
        "burning. This "
        "model-free check is why desks trust Black-Scholes "
        "as a quoting convention even when they distrust it as a forecast."
    ),
    "P8a": (
        "On the first stored session of each month this simulation sells one "
        "at-the-money straddle on the monthly expiry nearest 30 days out, and "
        "delta-hedges it once a day with this project's own Black-Scholes deltas, "
        "at each leg's own market implied vol. Nearest 30 is not 30: the chain "
        "stores monthly expiries only, so from the first session of a month the "
        "nearest monthly is usually 16–18 days away, and about 45 whenever that "
        "month's own monthly is missing from the ladder — the stat line above "
        "gives the range actually traded, and Q4's second panel says what that "
        "does to the dots. Cash earns the risk-free rate and the hedge earns the "
        "dividend yield, so the line below is the whole position: option marks, "
        "share hedge, carry. A cumulative P&L cannot skip days, so unlike the "
        "histogram below this line also includes every session with no chain "
        "quote: settlement is marked at intrinsic value from the underlying "
        "close, a market fact, and every other one is marked with our own model "
        "at the last quoted vol. Two different things put a session in that last "
        "group, and only one of them is missing data. The archive stores no "
        "expiry closer than its minimum days-to-expiry filter, so over its final "
        "stretch a trade's own expiry is absent from every stored chain by "
        "design — permanent, and no backfill can change it. The rest are "
        "sessions the stored archive holds no usable quote for that straddle "
        "on — the chain file missing, or present without that expiry and strike "
        "on both legs — and those are the only ones held against a trade when "
        "deciding whether it is quoted well enough to earn a dot on the next "
        "panel. Where either kind occurs, the stat line above gives its count. "
        "The thin grey lines are the individual trades; the blue line is their sum."
    ),
    "P8b": (
        "The money chart. Each dot is one round trip: what the straddle was sold "
        "at, minus what the underlying actually went on to realize, against the "
        "dollars the hedged position made or lost. Black-Scholes predicts the "
        "relationship — a delta-hedged short option earns about "
        "½ Γ S² (σ²_implied − σ²_realized) dt per day, so selling vol above what "
        "arrives should pay and selling it below should not. The slope is that "
        "prediction measured on real quotes. The scatter around the line is not "
        "noise to be explained away: it is discrete-hedging error, the price of "
        "re-hedging once a day instead of continuously, plus the vol "
        "mark-to-market the model assumes away. And part of it is not error at "
        "all — the trades are not the same length. That same formula scales the "
        "round trip with √T, so a 45-day trade earns roughly 1.6× the dollars of "
        "a 17-day one at identical edge, and the entry rule delivers both; the "
        "stat line above gives the tenor range these dots were drawn from."
    ),
    "P8c": (
        "The same P&L, one session at a time — counting only sessions marked at "
        "a market price: a chain quote, or settlement at intrinsic value from "
        "the underlying close. Every other session is marked with our own model "
        "at the last quoted vol, so it moves with spot and time decay alone and "
        "is Black-Scholes-conforming by construction; letting those days in "
        "would narrow this histogram towards the answer the model wants. That "
        "holds for both kinds of modelled session — the ones the archive holds "
        "no usable quote for, and the ones in a trade's final stretch where the "
        "stored chain never carries its expiry at all. The panel above has to "
        "tell those two "
        "apart to decide which trades are honestly quoted; this one does not, "
        "because a frozen-vol mark flatters it either way. Each "
        "trade's opening day is dropped too — it is a definitional "
        "zero, not a session. Black-Scholes says a perfectly hedged option "
        "position has no risk left; a histogram with visible width is the direct "
        "measurement of how wrong that is at a daily hedging frequency. The "
        "spread is gamma and vol moves; the centre is the premium being earned "
        "or bled."
    ),
    "P9": (
        "Price every contract with one flat volatility — today's ~30-day "
        "at-the-money implied vol — and compare to the market. The error is not "
        "noise: the wings are systematically rich against the flat-vol model, on "
        "both sides, day after day. A structured error means the world has a "
        "feature the model lacks (fat tails, volatility that moves), which is "
        "exactly why Heston and SABR exist. This is the smile shown earlier in "
        "this section, re-expressed in price."
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
     ["P8a", "P8b", "P8c"], ""),
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
    "P8a": "Cumulative P&L — simulated delta-hedged straddles",
    "P8b": "Per-trade P&L vs (entry IV − realized vol)",
    "P8c": "Daily hedged P&L — distribution",
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
""" + TILES_CSS + ".stat { font-weight: 600; margin: 4px 0 8px; }\n" + (
    ".stale { background: #fff5f5; border: 1px solid #f0c0c0; border-radius: 6px;\n"
    "         padding: 10px 12px; margin: 12px 0; color: #8a2b2b; font-size: 0.92rem; }\n"
) + (
    "@media (max-width: 700px) {\n"
    "  /* Two-column subplot figures compress to ~170px per panel on a phone, which\n"
    "     is not a chart. Hold them at a readable width and let .figure scroll. */\n"
    "  .figure-wide > div { min-width: 660px; }\n"
    "}\n"
)

# Calendar days since the snapshot before the page admits it may be stale. Four
# clears a Friday-session page read on the following Tuesday; anything longer
# means a scheduled run has actually been missed. This is a DISPLAY threshold and
# is deliberately not run_daily's STALENESS_DAYS, which decides whether to run.
STALE_AFTER_DAYS = 4


def staleness_banner(status: dict, today: dt.date | None = None) -> str:
    """A visible warning when the page is older than it should be, else ''.

    SPEC 2.1 asks for staleness to be visible rather than silent. A fresh page
    says nothing -- a banner that is always present is one nobody reads.
    """
    today = today or dt.date.today()
    try:
        snapshot = dt.date.fromisoformat(status["snapshot_date"])
    except (KeyError, TypeError, ValueError):
        return ""
    days = (today - snapshot).days
    if days <= STALE_AFTER_DAYS:
        return ""
    return (f"<p class='stale'>This page has not updated in {days} days — the "
            f"most recent session it shows is <b>{snapshot.isoformat()}</b>. The "
            "daily job has not published since then, so every number below is "
            "that session's, not today's.</p>")


def render_page(figures: dict, status: dict, extras: dict | None = None,
                 today: dt.date | None = None) -> str:
    bundle = _BUNDLE.read_text()
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>vol-lens — Black-Scholes vs the market, daily</title>",
        f"<style>{_CSS}</style>",
        f"<script>{bundle}</script>",
        "</head><body>",
        "<h1>vol-lens</h1>",
        staleness_banner(status, today),
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
                wide = " figure-wide" if (figures[pid].layout.meta or {}).get("stack_narrow") else ""
                parts.append(f"<div class='figure{wide}'>")
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
