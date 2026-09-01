"""Greek stat tiles (SPEC 3 P4). This is where SPEC 2.3's display scaling
lives: theta per DAY (/365), vega per VOL POINT (/100), rho per 1% (/100).
delta and gamma are shown raw.

Presentation is the design system's big-number card; the classes it uses are
defined once in `src.render.styles`, not here.
"""
import pandas as pd

_GREEKS = [  # (column, label, unit, divisor)
    ("delta", "Delta", "per $1 spot", 1.0),
    ("gamma", "Gamma", "delta per $1", 1.0),
    ("vega", "Vega", "per vol pt", 100.0),
    ("theta", "Theta", "per day", 365.0),
    ("rho", "Rho", "per 1% rate", 100.0),
]


def greek_tiles_html(tiles: pd.DataFrame, tiles_prev: pd.DataFrame | None,
                     prev_label: str) -> str:
    if tiles is None or tiles.empty:
        return "<p class='placeholder'>No converged ATM quotes to compute Greeks for.</p>"
    parts = []
    prev = tiles_prev.set_index("kind") if tiles_prev is not None and len(tiles_prev) else None
    for _, row in tiles.iterrows():
        parts.append(
            f"<div class='tiles-head'>ATM <strong>{row['kind']}</strong> · K {row['strike']:.0f} · "
            f"{row['expiry'].isoformat()} ({int(row['dte'])}d) · IV {row['iv']:.1%}</div>"
            "<div class='tiles'>")
        for col, label, unit, div in _GREEKS:
            val = row[col] / div
            change = ""
            if prev is not None and row["kind"] in prev.index:
                d = val - prev.loc[row["kind"], col] / div
                change = f"<div class='d'>{d:+.3f} vs {prev_label}</div>"
            parts.append(f"<div class='tile'><div class='k'>{label}</div>"
                         f"<div class='v'>{val:.3f}</div><div class='u'>{unit}</div>{change}</div>")
        parts.append("</div>")
    return "".join(parts)
