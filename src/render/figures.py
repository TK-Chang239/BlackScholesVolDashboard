"""Plotly figure builders for the dashboard panels. Pure: frames in, Figure out."""
import pandas as pd
import plotly.graph_objects as go

_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=50, r=20, t=40, b=45),
    height=420,
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    font=dict(size=13),
)


def build_smile_figure(smile: pd.DataFrame, spot: float) -> go.Figure:
    fig = go.Figure()
    if smile.empty:
        fig.add_annotation(
            text="No converged OTM quotes for this session",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16))
        fig.update_layout(
            title="Implied volatility smile — solved from market quotes",
            xaxis_title="Moneyness K/S", yaxis_title="Implied vol",
            yaxis_tickformat=".0%", **_LAYOUT)
        return fig
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
