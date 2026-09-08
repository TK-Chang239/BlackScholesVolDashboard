"""Layout defaults, the empty-state figure, and the line-break rules, shared by
every figure module."""
import datetime as dt

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.render import theme

# The longest the session calendar goes without a session in normal operation:
# a Friday to the following Tuesday, over a holiday Monday. Anything longer is
# a stretch the archive does not cover. An unscheduled multi-day closure would
# read as a hole too, which is the safe direction to be wrong in -- the line
# breaks rather than spanning days on an assumption.
MAX_MARKET_CLOSURE_DAYS = 4

LAYOUT = dict(
    template=theme.TEMPLATE_NAME,
    # No top margin for a title: the panel card's head carries it now, so the
    # figure only has to leave room for the legend that sits above the plot.
    margin=dict(l=54, r=20, t=34, b=45),
    height=440,
    legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
    font=dict(size=13),
    title_y=0.97,
    title_yanchor="top",
)


def _sorted_sessions(sessions) -> list:
    return sorted(set(sessions))


def _nan_row(series: pd.DataFrame, at: dt.date) -> dict:
    row = {col: float("nan") for col in series.columns}
    row["date"] = at
    return row


def _calendar_hole(a, b, sessions: list, max_gap: int) -> bool:
    """Does the session calendar itself go dark between `a` and `b`?

    `a` and `b` bracket the window rather than being looked up in it. They are
    plotted observations, so they are covered by definition -- but they need
    not be IN the calendar: the hedge P&L runs on underlying closes, so a
    point can fall on a session no chain was stored for. Measuring only the
    sessions strictly inside the span would find at most one of them across
    the emptiest possible window, leaving no consecutive pair to compare and
    reading 18 months of nothing as continuous.
    """
    window = [a] + [d for d in sessions if a < d < b] + [b]
    return any((window[i] - window[i - 1]).days > max_gap
               for i in range(1, len(window)))


def _break(series: pd.DataFrame, sessions, should_break) -> pd.DataFrame:
    """Insert an all-NaN row between consecutive points `should_break` flags.

    Scatter traces default to connectgaps=False, so a NaN y-value stops the
    line there instead of drawing a segment across it.
    """
    if sessions is None or len(series) < 2:
        return series
    sess = _sorted_sessions(sessions)
    dates = list(series["date"])
    rows = [series.iloc[0].to_dict()]
    for i in range(1, len(dates)):
        if should_break(dates[i - 1], dates[i], sess):
            rows.append(_nan_row(series, dates[i - 1] + dt.timedelta(days=1)))
        rows.append(series.iloc[i].to_dict())
    return pd.DataFrame(rows, columns=series.columns)


def break_unmeasured(series: pd.DataFrame, sessions,
                     max_gap: int = MAX_MARKET_CLOSURE_DAYS) -> pd.DataFrame:
    """Break a PER-SESSION metric wherever the archive cannot speak.

    Two conditions, and both are ignorance rather than absence of news. A
    stored session between two plotted points that carries no value is a
    session we could not measure -- for the 25-delta skew, one where no expiry
    sat in the target DTE band. And a hole in the session calendar itself is a
    stretch that was never covered at all.

    The first matters more than its size suggests: those holes fall at the same
    point in every expiry cycle, so the span is not merely across unmeasured
    days, it is between two DIFFERENT maturities -- a decaying ~21-day contract
    on the left, a fresh ~39-day one on the right. `skew.py` already refuses to
    splice maturities WITHIN a row for exactly that reason; a line drawn across
    the hole reintroduces the same splice between rows.
    """
    def should_break(a, b, sess):
        return (any(a < d < b for d in sess)
                or _calendar_hole(a, b, sess, max_gap))
    return _break(series, sessions, should_break)


def break_archive_gaps(series: pd.DataFrame, sessions,
                       max_gap: int = MAX_MARKET_CLOSURE_DAYS) -> pd.DataFrame:
    """Break a CUMULATIVE series only where the session calendar goes dark.

    The weaker rule of the two, and deliberately so. A session with no row in a
    running total means no position was open, and a total that does not move
    while nothing is held is measured, not assumed -- breaking there would
    claim an ignorance the archive does not have. Only a stretch the archive
    never covered is unknown, and there the flat line makes a real claim it
    cannot support: that the strategy did nothing, when the truth is that the
    simulation did not exist.
    """
    def should_break(a, b, sess):
        return _calendar_hole(a, b, sess, max_gap)
    return _break(series, sessions, should_break)


def empty_figure(title: str, message: str, **layout) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False,
                       font=dict(family=theme.FONT_MONO, size=13,
                                 color=theme.TEXT_TERTIARY))
    fig.update_layout(title=title, **layout, **LAYOUT)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def subplots(**kwargs) -> go.Figure:
    """`make_subplots`, with its titles restyled as design-system chart labels.

    Plotly hardcodes 16px on the annotations it builds for `subplot_titles`, so
    the template's annotation defaults reach the family but not the size: the
    result is a mono heading larger than the panel title above it. The design
    system's chart label is a small recessive mono micro-label instead.
    """
    fig = make_subplots(**kwargs)
    for annotation in fig.layout.annotations:
        annotation.text = (annotation.text or "").upper()
        annotation.font = dict(family=theme.FONT_MONO, size=10,
                               color=theme.TEXT_SECONDARY)
    return fig
