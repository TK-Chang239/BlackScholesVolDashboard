"""Plotly figure builders for the dashboard panels. Pure: frames in, Figure out."""
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.render import theme
from src.render.base import (LAYOUT as _LAYOUT, empty_figure as _empty_figure,
                            subplots as _subplots)

# P7 carries a note above the plot area (close-based, or the early-exercise
# count), so it needs more headroom above the top row of cells than a panel
# whose plot starts right under the card head.
_PARITY_LAYOUT = {**_LAYOUT, "margin": dict(l=54, r=20, t=76, b=45)}

# P1's up bars label outside the bar end with cliponaxis=False, so the label on
# the longest bar is drawn into the right margin rather than inside the plot.
# That bar is the +20% spot bump, whose label runs to eight characters
# (+1252.1% on a typical session) and would be cut off by the shared r=20.
# Sized for twelve, since the number is data-dependent and only grows as the
# option gets further out of the money.
_TORNADO_LAYOUT = {**_LAYOUT, "margin": dict(l=54, r=84, t=34, b=45)}


def build_smile_figure(smile: pd.DataFrame, spot: float) -> go.Figure:
    if smile.empty:
        return _empty_figure(
            title="Implied volatility smile — solved from market quotes",
            message="No converged OTM quotes for this session",
            xaxis_title="Moneyness K/S", yaxis_title="Implied vol",
            yaxis_tickformat=".0%")
    fig = go.Figure()
    # Expiries are ordered, not merely distinct, so they take theme.SEQUENCE --
    # a ramp monotone in lightness, nearest expiry darkest -- rather than six
    # competing hues. Six hues legible against each other AND against warm
    # paper do not exist in this design system; see theme.PALETTE_NOTES.
    groups = list(smile.groupby(["expiry", "dte"], sort=True))
    shades = theme.ramp(len(groups))
    for i, ((expiry, dte), g) in enumerate(groups):
        g = g.sort_values("strike")
        shade = shades[i]
        fig.add_trace(go.Scatter(
            x=g["moneyness"], y=g["iv"], mode="lines+markers",
            name=f"{expiry.isoformat()} ({dte}d)",
            line=dict(color=shade), marker=dict(color=shade, size=5),
            hovertemplate="K/S %{x:.3f}<br>IV %{y:.1%}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[1.0, 1.0], y=[smile["iv"].min(), smile["iv"].max()],
        mode="lines", name="ATM (K = S)",
        line=dict(dash="dot", color=theme.REFERENCE, width=1),
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
            name=previous_label, line=dict(color=theme.MUTED),
            marker=dict(color=theme.MUTED),
            hovertemplate=f"%{{x}}d: %{{y:.1%}}<extra>{previous_label}</extra>"))
    fig.add_trace(go.Scatter(
        x=today["dte"], y=today["atm_iv"], mode="lines+markers",
        name="today", line=dict(color=theme.SERIES_INK),
        marker=dict(color=theme.SERIES_INK),
        hovertemplate="%{x}d: %{y:.1%}<extra>today</extra>"))
    fig.update_layout(
        title="ATM implied vol term structure",
        xaxis_title="Days to expiry", yaxis_title="ATM implied vol",
        yaxis_tickformat=".0%", **_LAYOUT)
    return fig


def build_sensitivity_figure(sens: pd.DataFrame, bump: float) -> go.Figure:
    title = f"Which input moves the price? ±{bump:.0%} on each, ~30-DTE ATM call"
    if sens.empty or sens["up_pct"].isna().all():
        return _empty_figure(title, "No ATM implied vol for this session")
    fig = _subplots(
        rows=1, cols=2, horizontal_spacing=0.14,
        subplot_titles=("Inputs the market argues about", "Inputs everyone observes"))
    for col, group in ((1, "argued"), (2, "observed")):
        g = sens[sens["group"] == group].sort_values("span")   # biggest bar on top
        # down bars label INSIDE (near the axis) so long negative bars (e.g. S at
        # -100%) don't push an outside label left into the y-axis category labels;
        # up bars stay OUTSIDE since they never crowd the axis.
        for tag, color, name, textposition, insidetextanchor in (
                ("down_pct", theme.SERIES_CONTRAST, f"input −{bump:.0%}", "inside", "start"),
                ("up_pct", theme.SERIES_INK, f"input +{bump:.0%}", "outside", None)):
            bar_kwargs = dict(
                y=g["label"], x=g[tag], orientation="h", name=name,
                marker_color=color, showlegend=(col == 1),
                text=g[tag], texttemplate="%{x:+.1%}", textposition=textposition,
                cliponaxis=False,
                hovertemplate="%{y}: %{x:+.1%}<extra>" + name + "</extra>")
            if insidetextanchor:
                bar_kwargs["insidetextanchor"] = insidetextanchor
            fig.add_trace(go.Bar(**bar_kwargs), row=1, col=col)
    fig.update_layout(title=title, barmode="overlay", meta=dict(stack_narrow=True),
                      **_TORNADO_LAYOUT)
    fig.update_xaxes(title_text="Price change", tickformat="+.0%")
    return fig


def build_greeks_curves_figure(curves: pd.DataFrame, spot: float) -> go.Figure:
    title = "Delta and gamma across strikes — ~30-DTE expiry, each at its own market IV"
    if curves.empty:
        return _empty_figure(title, "No converged quotes for the ~30-DTE expiry")
    fig = _subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Delta vs strike", "Gamma vs strike"))
    colors = {"call": theme.SERIES_INK, "put": theme.SERIES_CONTRAST}
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
        fig.add_vline(x=spot, line=dict(dash="dot", color=theme.REFERENCE, width=1), row=1, col=col)
    fig.update_layout(title=title, meta=dict(stack_narrow=True), **_LAYOUT)
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
        fill="toself", fillcolor="rgba(26,26,24,0.07)", line=dict(width=0),
        hoverinfo="skip", showlegend=False, name="IV − RV band")


def build_iv_rv_figure(series: pd.DataFrame, summary: dict, cfg: dict) -> go.Figure:
    rv_cfg = cfg["realized_vol"]
    title = "Implied vs realized volatility — ~30-DTE ATM IV against what SPY actually did"
    if series.empty:
        return _empty_figure(title, "No sessions with a 30-DTE ATM implied vol yet")
    plotted = _break_gaps(series)
    # The mean spread is a DERIVED quantity an order of magnitude smaller than
    # the levels above it (tenths of a vol point against 10-18%). It used to
    # share this frame on a second y-axis, which let its line cross the implied
    # and realized lines at points that meant nothing -- the reader cannot see
    # that a crossing on a dual-axis chart is an artefact of two scales. It gets
    # its own strip instead, sharing the x-axis so dates still line up.
    fig = _subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.74, 0.26], vertical_spacing=0.06)
    for band in _iv_rv_band_traces(plotted):
        fig.add_trace(band, row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["rv_trailing"], mode="lines+markers",
        name=f"Realized, trailing {rv_cfg['windows'][0]}d", line=dict(color=theme.SERIES_CONTRAST),
        marker=dict(size=4), hovertemplate="%{x}: %{y:.1%}<extra>trailing RV</extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["atm_iv"], mode="lines+markers",
        name="Implied, ~30-DTE ATM", line=dict(color=theme.SERIES_INK), marker=dict(size=4),
        hovertemplate="%{x}: %{y:.1%}<extra>implied</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["fwd_rv"], mode="lines+markers",
        name=f"Realized, forward {rv_cfg['forward_horizon_days']}d (at quote date)",
        line=dict(color=theme.SERIES_CONTRAST, dash="dash"), marker=dict(size=4), connectgaps=False,
        hovertemplate="%{x}: %{y:.1%}<extra>forward RV</extra>"), row=1, col=1)
    # Alone in its own titled strip, so a legend entry would only repeat the
    # axis title.
    fig.add_trace(go.Scatter(
        x=plotted["date"], y=plotted["spread_running_mean"], mode="lines",
        name="Mean spread (IV − RV)", showlegend=False,
        line=dict(color=theme.SERIES_INK, dash="dot"),
        hovertemplate="%{x}: %{y:+.1%}<extra>mean spread</extra>"), row=2, col=1)
    fig.add_hline(y=0, line=dict(color=theme.REFERENCE, width=1), row=2, col=1)
    since = summary["history_since"].isoformat() if summary["history_since"] else "n/a"
    fig.add_annotation(text=f"accumulating since {since} · {summary['n_sessions']} sessions",
                       xref="paper", yref="paper", x=0.0, y=1.02, showarrow=False,
                       font=dict(size=11, color=theme.TEXT_SECONDARY), xanchor="left")
    fig.update_layout(title=title, **_LAYOUT)
    # Three legend entries still wrap under _LAYOUT's above-plot legend and
    # overlap the annotation; put the legend below the plot instead.
    fig.update_layout(height=540,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0),
                      margin=dict(b=96))
    fig.update_yaxes(title_text="Annualized vol", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Mean spread", tickformat="+.1%", row=2, col=1)
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
        name="25-delta skew (put − call)", line=dict(color=theme.SERIES_INK), marker=dict(size=5),
        customdata=plotted["skew_25d_dte"],
        hovertemplate="%{x}: %{y:+.1f} vol pts (%{customdata:.0f}d expiry)<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=theme.REFERENCE, width=1, dash="dot"))
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
                          hovertemplate: str, hot_name: str = "beyond spread",
                          all_name: str = "within spread",
                          showscale: bool = True, **subplot) -> list[go.Heatmap]:
    """Two layers: every cell faint, then only the non-muted cells at full
    opacity with the colorbar — SPEC 3 P7/P9's 'within-spread region muted'.

    `xcats` (not `x`) so that `**subplot`'s own `x=` (colorbar x-position,
    used by P9's two-subplot layout) doesn't collide with the heatmap's
    x-axis categories parameter. `showscale=False` keeps the trace (and so
    the trace count the P9 toggle's `visible` arrays depend on) while
    suppressing a second colorbar for a scale that is already shown.

    `all_name` names the faint layer, which carries EVERY cell -- the hot ones
    included -- so what it should be called depends on what the caller mutes.
    """
    common = dict(x=xcats, y=y, colorscale=theme.DIVERGING, zmid=0, zmin=zmin, zmax=zmax,
                  hoverongaps=False, hovertemplate=hovertemplate)
    return [
        go.Heatmap(z=z_all, opacity=0.3, showscale=False, name=all_name, **common),
        go.Heatmap(z=z_hot, opacity=1.0, showscale=showscale, name=hot_name,
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
    # An in-the-money put gap inside the American early-exercise bound is
    # consistent with early exercise -- not proof of it, but enough that it must
    # not read as an alarm: it stays in the faint layer.
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
            "%{x}<br>K %{y}: %{z:+.2f} $<extra></extra>",
            hot_name="unexplained", all_name="all pairs"):
        fig.add_trace(layer)
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=spot, y1=spot,
                  line=dict(color=theme.REFERENCE, width=1, dash="dot"))
    # Both notes sit ABOVE the plot area (paper coords, anchored by their bottom
    # edge), never over the top row of cells or the y-axis labels. They are
    # mutually exclusive: a close-based frame has no tradeable violations to
    # explain. `_PARITY_LAYOUT` gives them the headroom.
    note = dict(xref="paper", yref="paper", x=0.0, y=1.02, showarrow=False,
                xanchor="left", yanchor="bottom", font=dict(size=11, color=theme.TEXT_SECONDARY))
    if p["spread"].notna().sum() == 0:
        fig.add_annotation(text="close-based session — spreads unknown, tradeability not assessable",
                           **note)
    n_ee = int((explained & liquid).sum())
    if n_ee:
        fig.add_annotation(
            text=(f"{n_ee} put-rich gap{'' if n_ee == 1 else 's'} shown muted "
                  "— consistent with the American early-exercise bound"),
            **note)
    fig.update_layout(title=title, **_PARITY_LAYOUT)
    fig.update_xaxes(title_text="Expiry")
    fig.update_yaxes(title_text="Strike")
    return fig


def build_model_vs_market_figure(mvm: pd.DataFrame, flat_vol: float) -> go.Figure:
    title = "Model vs market — every contract priced at one flat vol"
    if mvm.empty:
        return _empty_figure(title, "No flat ~30-DTE ATM vol to price the chain with")
    title += f" ({flat_vol:.1%})"
    fig = _subplots(rows=1, cols=2, horizontal_spacing=0.14,
                        subplot_titles=("Calls", "Puts"))
    # `deviation_vol` is a FRACTION of vol; a vol point is 1/100 of it. P6 one
    # section above already multiplies by 100 and hovers "+4.0 vol pts", so
    # plotting the fraction under a colorbar titled "Vol points" would have two
    # panels of the same page disagreeing by 100x about what a vol point is.
    # Scale here, in the figure, exactly as P6 does. `deviation_pct` is a
    # fraction OF A PRICE and plotly's `%` format is the right reading of it,
    # so it keeps scale 1.0.
    metrics = [("deviation_pct", "% of market price", 1.0, 1.0, "%{z:+.0%}"),
               ("deviation_vol", "Vol points", 100.0, 20.0, "%{z:+.1f} vol pts")]
    for mi, (col, cb_title, scale, lim, fmt) in enumerate(metrics):
        for ci, kind in enumerate(("call", "put"), start=1):
            g = mvm[mvm["kind"] == kind].copy()
            g[col] = g[col].astype(float) * scale
            g["label"] = [_expiry_label(e, d) for e, d in zip(g["expiry"], g["dte"])]
            order = g[["label", "dte"]].drop_duplicates().sort_values("dte")["label"].tolist()
            z_all = g.pivot(index="strike", columns="label", values=col).reindex(columns=order)
            z_hot = (g.assign(hot=g[col].where(~g["within_spread"]))
                       .pivot(index="strike", columns="label", values="hot").reindex(columns=order))
            # Both subplots of a metric share zmin/zmax, so ONE colorbar says
            # everything — and it goes to the right of the whole figure. Drawing
            # a second at x=0.46 put it inside the Puts subplot, on top of its
            # y-axis tick labels. The trace itself stays, so the toggle's
            # `visible` arrays still line up 4-for-4.
            layers = _muted_heatmap_layers(
                z_all.to_numpy(dtype=float), z_hot.to_numpy(dtype=float), order,
                z_all.index.tolist(), cb_title, -lim, lim,
                "%{x}<br>K %{y}: " + fmt + "<extra>" + kind + "</extra>",
                showscale=(ci == 2), x=1.02, len=0.9)
            for layer in layers:
                layer.visible = (mi == 0)
                fig.add_trace(layer, row=1, col=ci)
    vis_pct = [True] * 4 + [False] * 4
    vis_vol = [False] * 4 + [True] * 4
    fig.update_layout(
        title=title, meta=dict(stack_narrow=True), **_LAYOUT,
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=1.14, xanchor="left",
                          yanchor="bottom", showactive=True,
                          buttons=[dict(label="% of market price", method="update",
                                        args=[{"visible": vis_pct}]),
                                   dict(label="Vol points", method="update",
                                        args=[{"visible": vis_vol}])])])
    fig.update_xaxes(title_text="Expiry")
    fig.update_yaxes(title_text="Strike", row=1, col=1)
    return fig
