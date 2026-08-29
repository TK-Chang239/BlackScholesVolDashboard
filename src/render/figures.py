"""Plotly figure builders for the dashboard panels. Pure: frames in, Figure out."""
from datetime import timedelta

import numpy as np
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

    Scatter traces default to connectgaps=False, so a NaN y-value stops the
    line; without this, a long silent stretch in the stored history (e.g. a
    single depth-probe session two years before the daily series starts)
    draws a straight diagonal segment across the gap instead of leaving it
    visibly empty.
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


def _iv_rv_band_traces(plotted: pd.DataFrame) -> list[go.Scatter]:
    """One closed toself polygon per contiguous (IV, trailing-RV) run.

    A single `tonexty` fill on the IV line bridges straight across a NaN gap
    even when the line itself breaks there, so a wide shaded wedge would
    still cross a long silent stretch (e.g. the 2024 depth probe, ~1.9 years
    before the 2026 series) and bleed into the 180-day default view. Building
    one explicit polygon per contiguous run of real data keeps the band
    confined to where both series actually have values.
    """
    traces: list[go.Scatter] = []
    valid = plotted["atm_iv"].notna() & plotted["rv_trailing"].notna()
    run: list[int] = []
    for i, ok in enumerate(valid):
        if ok:
            run.append(i)
            continue
        if run:
            traces.append(_band_trace(plotted, run))
            run = []
    if run:
        traces.append(_band_trace(plotted, run))
    return traces


def _band_trace(plotted: pd.DataFrame, run: list[int]) -> go.Scatter:
    dates = plotted["date"].iloc[run].tolist()
    iv = plotted["atm_iv"].iloc[run].tolist()
    rv_trailing = plotted["rv_trailing"].iloc[run].tolist()
    return go.Scatter(
        x=list(dates) + list(dates[::-1]), y=list(iv) + list(rv_trailing[::-1]),
        fill="toself", fillcolor="rgba(76,114,176,0.12)", line=dict(width=0),
        hoverinfo="skip", showlegend=False, name="IV − RV band")


def build_iv_rv_figure(series: pd.DataFrame, summary: dict, cfg: dict) -> go.Figure:
    rv_cfg = cfg["realized_vol"]
    title = "Implied vs realized volatility — ~30-DTE ATM IV against what SPY actually did"
    if series.empty:
        return _empty_figure(title, "No sessions with a 30-DTE ATM implied vol yet")
    plotted = _break_gaps(series)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for band in _iv_rv_band_traces(plotted):
        fig.add_trace(band)
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["rv_trailing"], mode="lines+markers",
        name=f"Realized, trailing {rv_cfg['windows'][0]}d", line=dict(color="#55a868"),
        marker=dict(size=4), hovertemplate="%{x}: %{y:.1%}<extra>trailing RV</extra>"))
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["atm_iv"], mode="lines+markers",
        name="Implied, ~30-DTE ATM", line=dict(color="#4c72b0"), marker=dict(size=4),
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
    # This panel's four legend entries wrap onto two rows under _LAYOUT's above-plot
    # legend, overlapping the title; put the legend below the plot instead.
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.22, x=0),
                      margin=dict(b=110))
    fig.update_yaxes(title_text="Annualized vol", tickformat=".0%", secondary_y=False)
    fig.update_yaxes(title_text="Spread", tickformat="+.1%", secondary_y=True, showgrid=False)
    _apply_recent_range(fig, series["date"])
    return fig


def _apply_recent_range(fig: go.Figure, dates: pd.Series,
                        recent_days: int = 180, span_days: int = 365) -> None:
    """Default the x-axis to the recent window when the history is long
    (a lone depth-probe session years back would otherwise crush the daily
    series into a sliver). Double-click restores autorange."""
    if len(dates) == 0:
        return
    first, last = dates.iloc[0], dates.iloc[-1]
    if (last - first).days > span_days:
        fig.update_xaxes(range=[last - timedelta(days=recent_days), last + timedelta(days=3)])


def build_skew_figure(metrics: pd.DataFrame, annotations: pd.DataFrame) -> go.Figure:
    title = "25-delta skew — put IV minus call IV at the ~30-DTE expiry"
    series = metrics[["date", "skew_25d", "skew_25d_dte"]].dropna(subset=["skew_25d"])
    series = series.sort_values("date").reset_index(drop=True)
    if series.empty:
        return _empty_figure(title, "No session with a bracketed 25-delta skew yet",
                             yaxis_title="Vol points")
    plotted = _break_gaps(series)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["skew_25d"] * 100.0, mode="lines+markers",
        name="25-delta skew (put − call)", line=dict(color="#8172b2"), marker=dict(size=5),
        customdata=plotted["skew_25d_dte"],
        hovertemplate="%{x}: %{y:+.1f} vol pts (%{customdata:.0f}d expiry)<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="grey", width=1, dash="dot"))
    by_date = dict(zip(series["date"], series["skew_25d"] * 100.0))
    for _, a in annotations.iterrows():
        if a["date"] in by_date:
            fig.add_annotation(x=a["date"], y=by_date[a["date"]], text=str(a["note"]),
                               showarrow=True, arrowhead=2, ax=0, ay=-40,
                               font=dict(size=11))
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_yaxes(title_text="Vol points (put − call)", ticksuffix="")
    _apply_recent_range(fig, series["date"])
    return fig


def _expiry_label(expiry, dte) -> str:
    return f"{expiry.isoformat()} ({int(dte)}d)"


def _muted_heatmap_layers(z_all, z_hot, xcats, y, colorbar_title: str, zmin: float, zmax: float,
                          hovertemplate: str, **subplot) -> list[go.Heatmap]:
    """Two layers: every cell faint, then only the non-muted cells at full
    opacity with the colorbar — SPEC 3 P7/P9's 'within-spread region muted'.

    `xcats` (not `x`) so that `**subplot`'s own `x=` (colorbar x-position,
    used by P9's two-subplot layout) doesn't collide with the heatmap's
    x-axis categories parameter.
    """
    common = dict(x=xcats, y=y, colorscale="RdBu", zmid=0, zmin=zmin, zmax=zmax,
                  hoverongaps=False, hovertemplate=hovertemplate)
    return [
        go.Heatmap(z=z_all, opacity=0.3, showscale=False, name="within spread", **common),
        go.Heatmap(z=z_hot, opacity=1.0, showscale=True, name="beyond spread",
                   colorbar=dict(title=colorbar_title, **subplot), **common),
    ]


def build_parity_figure(parity: pd.DataFrame, spot: float) -> go.Figure:
    raw_title = "Put-call parity — C − P versus S·e⁻ᵠᵀ − K·e⁻ʳᵀ, in dollars"
    if parity.empty:
        return _empty_figure(raw_title, "No strike/expiry pairs with both a call and a put priced")
    p = parity.copy()
    # Headline the forward-calibrated deviation when there is one; fall back to
    # the spot-based one when no expiry brackets spot (or on an old frame).
    calibrated = "deviation_fwd" in p.columns and bool(p["deviation_fwd"].notna().any())
    value_col = "deviation_fwd" if calibrated else "deviation"
    viol_col = "tradeable_violation_fwd" if calibrated else "tradeable_violation"
    title = ("Put-call parity — C − P against the market's own implied forward"
             if calibrated else raw_title)
    liquid = (p["liquid"].astype(bool) if "liquid" in p.columns
              else pd.Series(True, index=p.index))
    # An in-the-money put gap inside the American early-exercise bound is not
    # arbitrage, so it must not read as an alarm: it stays in the faint layer.
    explained = (p["early_exercise_explained"].astype(bool)
                 if "early_exercise_explained" in p.columns
                 else pd.Series(False, index=p.index))
    p["label"] = [_expiry_label(e, d) for e, d in zip(p["expiry"], p["dte"])]
    order = (p[["label", "dte"]].drop_duplicates().sort_values("dte")["label"].tolist())
    z_all = p.pivot(index="strike", columns="label", values=value_col).reindex(columns=order)
    hot = p[value_col].where(p[viol_col].astype(bool) & liquid & ~explained)
    z_hot = p.assign(hot=hot).pivot(index="strike", columns="label", values="hot").reindex(columns=order)
    # Robust colour limit, read off the LIQUID cells only: a strike with no open
    # interest can carry a $160 stale-quote outlier, and letting it into the
    # percentile flattens every tradeable cell to white.
    vals = np.abs(p.loc[liquid, value_col].to_numpy(dtype=float))
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:                    # nothing liquid: scale on what there is
        vals = np.abs(p[value_col].to_numpy(dtype=float))
        finite = vals[np.isfinite(vals)]
    lim = float(np.percentile(finite, 98, method="lower")) if finite.size else 1.0
    lim = max(lim, 0.05)
    fig = go.Figure()
    for layer in _muted_heatmap_layers(
            z_all.to_numpy(dtype=float), z_hot.to_numpy(dtype=float), order, z_all.index.tolist(),
            ("C − P − e⁻ʳᵀ(F − K), $" if calibrated else "C − P − parity, $"), -lim, lim,
            "%{x}<br>K %{y}: %{z:+.2f} $<extra></extra>"):
        fig.add_trace(layer)
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=spot, y1=spot,
                  line=dict(color="grey", width=1, dash="dot"))
    if p["spread"].notna().sum() == 0:
        fig.add_annotation(text="close-based session — spreads unknown, tradeability not assessable",
                           xref="paper", yref="paper", x=0.0, y=1.02, showarrow=False,
                           xanchor="left", font=dict(size=11, color="#555"))
    n_ee = int((explained & liquid).sum())
    if n_ee:
        fig.add_annotation(
            text=(f"{n_ee} put-rich gap{'' if n_ee == 1 else 's'} within the American "
                  f"early-exercise bound {'is' if n_ee == 1 else 'are'} shown muted "
                  "— not arbitrage"),
            xref="paper", yref="paper", x=0.0, y=1.02, showarrow=False,
            xanchor="left", font=dict(size=11, color="#555"))
    fig.update_layout(title=title, **_LAYOUT)
    fig.update_xaxes(title_text="Expiry")
    fig.update_yaxes(title_text="Strike")
    return fig


def build_model_vs_market_figure(mvm: pd.DataFrame, flat_vol: float) -> go.Figure:
    title = "Model vs market — every contract priced at one flat vol"
    if mvm.empty:
        return _empty_figure(title, "No flat ~30-DTE ATM vol to price the chain with")
    title += f" ({flat_vol:.1%})"
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("Calls", "Puts"))
    metrics = [("deviation_pct", "% of market price", 1.0, "%{z:+.0%}"),
               ("deviation_vol", "Vol points", 0.20, "%{z:+.1%}")]
    for mi, (col, cb_title, lim, fmt) in enumerate(metrics):
        for ci, kind in enumerate(("call", "put"), start=1):
            g = mvm[mvm["kind"] == kind].copy()
            g["label"] = [_expiry_label(e, d) for e, d in zip(g["expiry"], g["dte"])]
            order = g[["label", "dte"]].drop_duplicates().sort_values("dte")["label"].tolist()
            z_all = g.pivot(index="strike", columns="label", values=col).reindex(columns=order)
            z_hot = (g.assign(hot=g[col].where(~g["within_spread"]))
                       .pivot(index="strike", columns="label", values="hot").reindex(columns=order))
            layers = _muted_heatmap_layers(
                z_all.to_numpy(dtype=float), z_hot.to_numpy(dtype=float), order,
                z_all.index.tolist(), cb_title, -lim, lim,
                "%{x}<br>K %{y}: " + fmt + "<extra>" + kind + "</extra>",
                x=1.02 if ci == 2 else 0.46, len=0.9)
            for layer in layers:
                layer.visible = (mi == 0)
                fig.add_trace(layer, row=1, col=ci)
    vis_pct = [True] * 4 + [False] * 4
    vis_vol = [False] * 4 + [True] * 4
    fig.update_layout(
        title=title, **_LAYOUT,
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.14, xanchor="left",
                          yanchor="bottom", showactive=True,
                          buttons=[dict(label="% of market price", method="update",
                                        args=[{"visible": vis_pct}]),
                                   dict(label="Vol points", method="update",
                                        args=[{"visible": vis_vol}])])])
    fig.update_xaxes(title_text="Expiry")
    fig.update_yaxes(title_text="Strike", row=1, col=1)
    return fig
