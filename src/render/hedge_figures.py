"""P8 figures: cumulative P&L, the daily P&L histogram, and the P&L-vs-edge scatter.

Kept out of figures.py so neither file has to hold every panel on the page.
All money is per share: the simulation sells ONE option on ONE share.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.render.base import LAYOUT, empty_figure

_PNL_AXIS = "P&L ($ per share)"


def build_hedge_pnl_figure(port: pd.DataFrame, daily: pd.DataFrame) -> go.Figure:
    """Portfolio cumulative P&L, with each trade's own running P&L behind it."""
    if port.empty:
        return empty_figure("Cumulative P&L — all simulated trades",
                            "No simulated trade yet — the first opens on the "
                            "first stored session of a month.")
    fig = go.Figure()
    if not daily.empty:
        for entry, g in daily.groupby("entry_date", sort=True):
            g = g.sort_values("date")
            fig.add_trace(go.Scatter(
                x=g["date"], y=g["pv"], mode="lines", name=f"trade {entry.isoformat()}",
                line=dict(width=1, color="#b0b7c3"), legendgroup="trades",
                showlegend=False, hovertemplate="%{x}<br>trade P&L $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=port["date"], y=port["pnl_cum"], mode="lines", name="Cumulative (all trades)",
        line=dict(width=2.5, color="#2b6cb0"),
        hovertemplate="%{x}<br>cumulative $%{y:.2f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(width=1, color="#999", dash="dot"))
    fig.update_layout(title="Cumulative P&L — all simulated trades", **LAYOUT)
    fig.update_yaxes(title_text=_PNL_AXIS)
    fig.update_xaxes(title_text="Session")
    return fig


def build_hedge_histogram_figure(daily: pd.DataFrame) -> go.Figure:
    """Distribution of daily hedged P&L, over market-marked sessions only.

    Takes the PER-TRADE daily frame, not the aggregated portfolio, because both
    exclusions below are properties of a trade rather than of a calendar day.

    Each trade's own opening day is a definitional zero -- no prior mark to
    difference against. Slicing only the portfolio's first row dropped the
    archive's first day and nothing else, so every later entry that did not
    happen to overlap a live trade planted an exact 0.00 at the mode of a panel
    whose entire caption is about its width.

    A MODEL-marked day prices the straddle with `bs_price` at the carried-forward
    IV, so it moves with spot and time decay at frozen vol and has no vol
    mark-to-market in it at all: those days are Black-Scholes-conforming by
    construction. The scatter and the fit already refuse them (that is what
    `MIN_MARKET_MARK_SHARE` is for); admitting them here narrowed the measured
    width towards the answer the model wants.
    """
    empty = empty_figure("Daily hedged P&L — distribution",
                         "Not enough market-quoted sessions yet to show a distribution.")
    if daily.empty:
        return empty
    kept = daily[(daily["date"] != daily["entry_date"])
                 & (daily["mark_source"] != "model")]
    if kept.empty:
        return empty
    daily_pnl = kept.groupby("date")["pnl_day"].sum().dropna()
    if len(daily_pnl) < 1:
        return empty
    fig = go.Figure(go.Histogram(x=daily_pnl.to_numpy(dtype=float), nbinsx=25,
                                 marker=dict(color="#2b6cb0", line=dict(width=0.5, color="white")),
                                 name="Daily P&L",
                                 hovertemplate="$%{x}<br>%{y} sessions<extra></extra>"))
    mean = float(daily_pnl.mean())
    fig.add_vline(x=0, line=dict(width=1, color="#999", dash="dot"))
    fig.add_vline(x=mean, line=dict(width=2, color="#c05621"),
                  annotation_text=f"mean ${mean:.2f}", annotation_position="top right")
    fig.update_layout(title="Daily hedged P&L — distribution", showlegend=False, **LAYOUT)
    fig.update_xaxes(title_text=_PNL_AXIS)
    fig.update_yaxes(title_text="Sessions")
    return fig


def build_hedge_scatter_figure(trades: pd.DataFrame, fit: dict,
                               next_settlement=None) -> go.Figure:
    """The money chart: round-trip P&L against the edge the trade was sold at.

    x is in VOL POINTS (edge * 100) to match P6 and P9; y is dollars per share.
    Only settled trades appear -- an open trade has no round-trip P&L and no
    lifetime realized vol yet.
    """
    title = "Per-trade P&L vs (entry IV − realized vol)"
    settled = (trades[(trades["status"] == "settled") & trades["edge"].notna()
                       & trades["pnl"].notna()]
               if not trades.empty else trades)
    if settled.empty:
        # "reached expiry" and "plottable" are different questions: `sparse` is a
        # sub-kind of settled, so a trade can be finished and still be off this
        # chart. Only promise a FIRST settlement when nothing has finished at all.
        reached = (int((trades["status"].isin(("settled", "sparse"))).sum())
                   if not trades.empty else 0)
        if reached:
            plural = "s have" if reached != 1 else " has"
            carries = "none carries" if reached != 1 else "it does not carry"
            return empty_figure(
                title, f"{reached} simulated trade{plural} reached expiry, but "
                       f"{carries} enough market-quoted marks to plot.")
        when = (f" The first settles {next_settlement.isoformat()}."
                if next_settlement is not None else "")
        return empty_figure(title, "No simulated trade has reached expiry yet." + when)

    x = settled["edge"].to_numpy(dtype=float) * 100.0
    y = settled["pnl"].to_numpy(dtype=float)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text", name="Settled trades",
        text=[d.strftime("%b %Y") for d in settled["entry_date"]],
        textposition="top center", textfont=dict(size=10, color="#666"),
        marker=dict(size=11, color="#2b6cb0", line=dict(width=1, color="white")),
        customdata=np.column_stack([settled["entry_iv"] * 100, settled["lifetime_rv"] * 100]),
        hovertemplate=("%{text}<br>entry IV %{customdata[0]:.1f} vs realized "
                       "%{customdata[1]:.1f} vol pts<br>P&L $%{y:.2f}<extra></extra>")))
    if np.isfinite(fit.get("slope", np.nan)):
        xs = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(
            x=xs, y=fit["slope"] * xs + fit["intercept"], mode="lines",
            name="Least-squares fit", line=dict(width=2, color="#c05621", dash="dash"),
            hoverinfo="skip"))
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98, xanchor="left", yanchor="top",
            showarrow=False, align="left", font=dict(size=12, color="#444"),
            text=(f"slope ${fit['slope']:.2f} per vol point · R² {fit['r2']:.2f} "
                  f"· {fit['n']} trades"))
    fig.add_hline(y=0, line=dict(width=1, color="#999", dash="dot"))
    fig.add_vline(x=0, line=dict(width=1, color="#999", dash="dot"))
    fig.update_layout(title=title, **LAYOUT)
    fig.update_xaxes(title_text="Entry IV − realized vol over the trade's life (vol points)")
    fig.update_yaxes(title_text=_PNL_AXIS)
    return fig
