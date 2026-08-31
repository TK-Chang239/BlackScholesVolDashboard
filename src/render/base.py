"""Layout defaults and the empty-state figure, shared by every figure module."""
import plotly.graph_objects as go

LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=50, r=20, t=100, b=45),
    height=460,
    legend=dict(orientation="h", yanchor="bottom", y=1.12),
    font=dict(size=13),
    title_y=0.97,
    title_yanchor="top",
)


def empty_figure(title: str, message: str, **layout) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(size=16))
    fig.update_layout(title=title, **layout, **LAYOUT)
    return fig
