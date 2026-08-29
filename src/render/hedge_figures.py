"""P8 figures: cumulative P&L, the P&L-vs-edge scatter, the daily P&L histogram.

Kept out of figures.py so neither file has to hold every panel on the page.
All money is per share: the simulation sells ONE option on ONE share.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.render.base import LAYOUT, empty_figure

_PNL_AXIS = "P&L ($ per share)"


def build_hedge_pnl_figure(port: pd.DataFrame, trades: pd.DataFrame,
                           daily: pd.DataFrame) -> go.Figure:
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


def build_hedge_histogram_figure(port: pd.DataFrame) -> go.Figure:
    """Distribution of daily hedged P&L.

    The archive's FIRST session has no prior mark to difference against, so it
    is dropped. Later trades' own opening days are not dropped: they are folded
    into a portfolio total that already carries the older trades' moves.
    """
    daily_pnl = port["pnl_day"].iloc[1:] if len(port) > 1 else pd.Series(dtype=float)
    daily_pnl = daily_pnl.dropna()
    if len(daily_pnl) < 1:
        return empty_figure("Daily hedged P&L — distribution",
                            "Not enough sessions yet to show a distribution.")
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
