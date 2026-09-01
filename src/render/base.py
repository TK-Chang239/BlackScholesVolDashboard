"""Layout defaults and the empty-state figure, shared by every figure module."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.render import theme

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
