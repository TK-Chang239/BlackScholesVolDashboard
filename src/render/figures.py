"""Plotly figure builders for the dashboard panels. Pure: frames in, Figure out."""
from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=50, r=20, t=100, b=45),
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.12),
    font=dict(size=13),
    title_y=0.97,
    title_yanchor="top",
)


def build_smile_figure(smile: pd.DataFrame, spot: float) -> go.Figure:
    if smile.empty:
        return _empty_figure(
            title="Implied volatility smile — solved from market quotes",
            message="No converged OTM quotes for this session",
            xaxis_title="Moneyness K/S", yaxis_title="Implied vol",
            yaxis_tickformat=".0%")
    fig = go.Figure()
    for (expiry, dte), g in smile.groupby(["expiry", "dte"], sort=True):
        g = g.sort_values("strike")
        fig.add_trace(go.Scatter(
            x=g["moneyness"], y=g["iv"], mode="lines+markers",
            name=f"{expiry.isoformat()} ({dte}d)",
            hovertemplate="K/S %{x:.3f}<br>IV %{y:.1%}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[1.0, 1.0], y=[smile["iv"].min(), smile["iv"].max()],
        mode="lines", name="ATM (K = S)",
        line=dict(dash="dot", color="grey", width=1),
        hoverinfo="skip",
    ))
    fig.update_layout(
        title="Implied volatility smile — solved from market quotes",
        xaxis_title="Moneyness K/S", yaxis_title="Implied vol",
        yaxis_tickformat=".0%", **_LAYOUT)
    return fig


def build_term_structure_figure(today: pd.DataFrame,
                                previous: pd.DataFrame | None,
                                previous_label: str = "previous session") -> go.Figure:
    fig = go.Figure()
    if previous is not None and len(previous):
        fig.add_trace(go.Scatter(
            x=previous["dte"], y=previous["atm_iv"], mode="lines+markers",
            name=previous_label, line=dict(color="lightgrey"),
            marker=dict(color="lightgrey"),
            hovertemplate=f"%{{x}}d: %{{y:.1%}}<extra>{previous_label}</extra>"))
    fig.add_trace(go.Scatter(
        x=today["dte"], y=today["atm_iv"], mode="lines+markers",
        name="today",
        hovertemplate="%{x}d: %{y:.1%}<extra>today</extra>"))
    fig.update_layout(
        title="ATM implied vol term structure",
        xaxis_title="Days to expiry", yaxis_title="ATM implied vol",
        yaxis_tickformat=".0%", **_LAYOUT)
    return fig


def _empty_figure(title: str, message: str, **layout) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16))
    fig.update_layout(title=title, **layout, **_LAYOUT)
    return fig


def build_sensitivity_figure(sens: pd.DataFrame, bump: float) -> go.Figure:
    title = f"Which input moves the price? ±{bump:.0%} on each, ~30-DTE ATM call"
    if sens.empty or sens["up_pct"].isna().all():
        return _empty_figure(title, "No ATM implied vol for this session")
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.14,
        subplot_titles=("Inputs the market argues about", "Inputs everyone observes"))
    for col, group in ((1, "argued"), (2, "observed")):
        g = sens[sens["group"] == group].sort_values("span")   # biggest bar on top
        # down bars label INSIDE (near the axis) so long negative bars (e.g. S at
        # -100%) don't push an outside label left into the y-axis category labels;
        # up bars stay OUTSIDE since they never crowd the axis.
        for tag, color, name, textposition, insidetextanchor in (
                ("down_pct", "#c44e52", f"input −{bump:.0%}", "inside", "start"),
                ("up_pct", "#4c72b0", f"input +{bump:.0%}", "outside", None)):
            bar_kwargs = dict(
                y=g["label"], x=g[tag], orientation="h", name=name,
                marker_color=color, showlegend=(col == 1),
                text=g[tag], texttemplate="%{x:+.1%}", textposition=textposition,
                cliponaxis=False,
                hovertemplate="%{y}: %{x:+.1%}<extra>" + name + "</extra>")
            if insidetextanchor:
                bar_kwargs["insidetextanchor"] = insidetextanchor
            fig.add_trace(go.Bar(**bar_kwargs), row=1, col=col)
    fig.update_layout(title=title, barmode="overlay", **_LAYOUT)
    fig.update_xaxes(title_text="Price change", tickformat="+.0%")
    return fig


def build_greeks_curves_figure(curves: pd.DataFrame, spot: float) -> go.Figure:
    title = "Delta and gamma across strikes — ~30-DTE expiry, each at its own market IV"
    if curves.empty:
        return _empty_figure(title, "No converged quotes for the ~30-DTE expiry")
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Delta vs strike", "Gamma vs strike"))
    colors = {"call": "#4c72b0", "put": "#dd8452"}
    for kind, g in curves.groupby("kind"):
        g = g.sort_values("strike")
        for col, y in ((1, "delta"), (2, "gamma")):
            fig.add_trace(go.Scatter(
                x=g["strike"], y=g[y], mode="lines+markers", name=kind,
                legendgroup=kind, showlegend=(col == 1), marker=dict(size=4),
                line=dict(color=colors.get(kind)),
                hovertemplate="K %{x:.0f}: %{y:.4f}<extra>" + kind + "</extra>"),
                row=1, col=col)
    for col in (1, 2):
        fig.add_vline(x=spot, line=dict(dash="dot", color="grey", width=1), row=1, col=col)
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_xaxes(title_text="Strike")
    return fig


def _break_gaps(series: pd.DataFrame, max_gap_days: int = 30) -> pd.DataFrame:
    """Insert an all-NaN row between sessions more than `max_gap_days` apart.

    Scatter traces default to connectgaps=False, so a NaN y-value stops both
    the line and a `tonexty` fill; without this, a long silent stretch in the
    stored history (e.g. a single depth-probe session two years before the
    daily series starts) draws a straight diagonal segment -- and a huge
    shaded wedge -- across the gap instead of leaving it visibly empty.
    """
    if len(series) < 2:
        return series
    dates = list(series["date"])
    rows = [series.iloc[0].to_dict()]
    for i in range(1, len(series)):
        if (dates[i] - dates[i - 1]).days > max_gap_days:
            gap_row = {col: float("nan") for col in series.columns}
            gap_row["date"] = dates[i - 1] + timedelta(days=1)
            rows.append(gap_row)
        rows.append(series.iloc[i].to_dict())
    return pd.DataFrame(rows, columns=series.columns)


def build_iv_rv_figure(series: pd.DataFrame, summary: dict, cfg: dict) -> go.Figure:
    rv_cfg = cfg["realized_vol"]
    title = "Implied vs realized volatility — ~30-DTE ATM IV against what SPY actually did"
    if series.empty:
        return _empty_figure(title, "No sessions with a 30-DTE ATM implied vol yet")
    plotted = _break_gaps(series)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["rv_trailing"], mode="lines+markers",
        name=f"Realized, trailing {rv_cfg['windows'][0]}d", line=dict(color="#55a868"),
        marker=dict(size=4), hovertemplate="%{x}: %{y:.1%}<extra>trailing RV</extra>"))
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["atm_iv"], mode="lines+markers",
        name="Implied, ~30-DTE ATM", line=dict(color="#4c72b0"), marker=dict(size=4),
        fill="tonexty", fillcolor="rgba(76,114,176,0.12)",
        hovertemplate="%{x}: %{y:.1%}<extra>implied</extra>"))
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["fwd_rv"], mode="lines+markers",
        name=f"Realized, forward {rv_cfg['forward_horizon_days']}d (at quote date)",
        line=dict(color="#dd8452", dash="dash"), marker=dict(size=4), connectgaps=False,
        hovertemplate="%{x}: %{y:.1%}<extra>forward RV</extra>"))
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["spread_running_mean"], mode="lines",
        name="Mean spread (IV − RV)", line=dict(color="grey", dash="dot"),
        hovertemplate="%{x}: %{y:+.1%}<extra>mean spread</extra>"), secondary_y=True)
    since = summary["history_since"].isoformat() if summary["history_since"] else "n/a"
    fig.add_annotation(text=f"accumulating since {since} · {summary['n_sessions']} sessions",
                       xref="paper", yref="paper", x=0.0, y=1.02, showarrow=False,
                       font=dict(size=11, color="#555"), xanchor="left")
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_yaxes(title_text="Annualized vol", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Spread", tickformat="+.1%", secondary_y=True, showgrid=False)
    first, last = series["date"].iloc[0], series["date"].iloc[-1]
    if (last - first).days > 365:
        fig.update_xaxes(range=[last - timedelta(days=180), last + timedelta(days=3)])
    return fig
