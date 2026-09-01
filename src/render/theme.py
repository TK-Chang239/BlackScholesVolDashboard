"""The OpenLia Report design system, as this project consumes it.

One source of truth for every colour, font and rule the page and its figures
draw with. Nothing else in `src/render` should hardcode a hex value: the
figure modules import the names below, and the page imports the CSS blocks.

Two deliberate departures from the design system are documented at their
definitions -- SERIES_INK as a near-achromatic series colour, and the cool
pole of DIVERGING. Both were measured, not guessed; see PALETTE_NOTES.
"""
import base64
import functools
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

_FONT_DIR = Path(__file__).parent / "assets" / "fonts"

# ── Design-system tokens ─────────────────────────────────────────────────
# Values are the OpenLia Report DS's own, copied verbatim from its
# tokens/colors_and_type.css. Names follow the DS's semantic names.

INK = "#1A1A18"             # --color-text-primary
PAPER = "#F2F1E8"           # --color-bg-base
SURFACE = "#FAFAF4"         # --color-bg-elevated -- the plot area
SURFACE_SUBTLE = "#F5F4EF"  # --color-bg-input
BORDER = "#E0DED5"          # --color-border-subtle -- gridlines
BORDER_STRONG = "#B0AEA6"   # --color-border-strong -- axis lines
TEXT_SECONDARY = "#737268"  # --color-text-secondary
TEXT_TERTIARY = "#B0AEA6"   # --color-text-tertiary

ACID = "#D4FF00"            # --color-accent-primary
ACID_ON = "#3D4D00"         # --color-accent-on -- the only text colour on ACID
YELLOW_100 = "#F0FF99"
YELLOW_600 = "#A8CC00"
YELLOW_800 = "#6B8200"
YELLOW_900 = "#3D4D00"

SUCCESS = "#6B8200"         # --color-feedback-success
WARNING = "#DC9614"         # --color-feedback-warning
ERROR = "#E05C30"           # --color-feedback-error

NEUTRAL_200 = "#E0DED5"
NEUTRAL_400 = "#B0AEA6"
NEUTRAL_600 = "#737268"

FONT_DISPLAY = "Geist, system-ui, -apple-system, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, 'Courier New', monospace"

# ── Series palette ───────────────────────────────────────────────────────
# Sized to what the charts actually need. The largest CATEGORICAL demand on
# this page is two series in one frame (implied vs realized, call vs put,
# hedged vs total); the six-way demand -- P2's expiry ladder -- is ORDERED
# data and takes the sequential ramp below, not six competing hues.
#
# Every pair here was checked with the dataviz skill's validator against
# SURFACE. See PALETTE_NOTES for the numbers and the two accepted failures.

SERIES_INK = INK            # slot 1: the series the panel is about
SERIES_CONTRAST = ERROR     # slot 2 when the second series is a rival quantity
SERIES_ALT = SUCCESS        # slot 2 when ERROR would wrongly read as "bad"

REFERENCE = NEUTRAL_600     # zero lines, mean lines, annotations
MUTED = NEUTRAL_400         # previous-session overlays, individual trades
GRID = BORDER

# Ordered ramp for P2's expiry ladder. Monotone in lightness (contrast on
# SURFACE runs 16.6 : 11.7 : 8.9 : 6.9 : 4.7 : 3.3), so the nearest expiry is
# the darkest and the ladder reads as a ladder. Every step clears 3:1, which
# is why it stops at #7A9400 rather than running on into ACID -- ACID is
# 1.05:1 on paper and vanishes as a line.
SEQUENCE = ["#1A1A18", "#2C3A05", "#3D4D00", "#4A5F00", "#5E7A00", "#7A9400"]

# Diverging scale for the P7 and P9 heatmaps, replacing Plotly's RdBu.
# Low = warm (burnt), midpoint = the DS's own warm neutral, high = cool.
# That keeps the sign convention RdBu gave these panels: negative warm,
# positive cool.
DIVERGING = [
    [0.00, "#7E2E12"],
    [0.25, ERROR],
    [0.45, "#F3C9B8"],
    [0.50, PAPER],
    [0.55, "#C3D0DC"],
    [0.75, "#4E6E8E"],
    [1.00, "#2F4B6B"],
]


def ramp(n: int) -> list[str]:
    """`n` colours spread across SEQUENCE, interpolated between its steps.

    Taking the first `n` entries instead would hand a three-expiry chain the
    three DARKEST steps, which are near-indistinguishable -- the ladder only
    reads as a ladder when it uses the ramp's whole range. Chains longer than
    SEQUENCE interpolate rather than repeating a colour.
    """
    if n <= 0:
        return []
    if n == 1:
        return [SEQUENCE[0]]
    stops = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in SEQUENCE]
    out = []
    for i in range(n):
        pos = i * (len(stops) - 1) / (n - 1)
        lo = min(int(pos), len(stops) - 2)
        frac = pos - lo
        rgb = tuple(round(stops[lo][c] + frac * (stops[lo + 1][c] - stops[lo][c]))
                    for c in range(3))
        out.append("#%02X%02X%02X" % rgb)
    return out


PALETTE_NOTES = """\
Measured with the dataviz skill's validate_palette.js against SURFACE #FAFAF4.

PASSES
  SERIES_INK vs SERIES_CONTRAST   CVD dE 35.0 (protan) · normal 45.5 · both >= 3:1
  SERIES_INK vs SERIES_ALT        CVD dE 37.2 (deutan) · normal 37.5 · both >= 3:1
  DIVERGING poles                 CVD dE 20.2 (protan) · normal 32.7 · both >= 3:1
  SEQUENCE                        lightness strictly monotone, every step >= 3:1

ACCEPTED FAILURES, both on SERIES_INK
  Lightness band and chroma floor flag #1A1A18 as near-achromatic. Using ink as
  the lead series colour is the design system's own convention (its price-line
  chart strokes #1A1A18 at 2px) and carries no readability cost: ink separates
  from every other slot at dE 35+.

WHY THE COOL POLE IS NOT A DS COLOUR
  The design system has exactly one hue axis -- yellow-green through burnt
  orange -- and that is the axis protanopia collapses. Every in-system pole
  pair measured dE 2.4-3.3 under protan (#E05C30 vs #6B8200 is 3.3), which is
  below even the 6-8 band that secondary encoding can rescue. These heatmaps
  print no cell values, so colour alone carries sign; an in-system scale would
  have made sign unreadable for red-green colourblind readers, a regression
  against the RdBu it replaces. #2F4B6B is the minimum foreign hue that keeps
  sign legible, desaturated to sit against warm paper.
"""


# ── Fonts, inlined ───────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def font_face_css() -> str:
    """`@font-face` rules with the woff2 payloads inlined as data URIs.

    SPEC 2.5 makes the page self-contained -- no request at view time -- so the
    families cannot be linked from a font CDN. Latin subsets keep the four
    faces to ~78 KB base64 against the 5 MB page budget.
    """
    faces = [
        ("Geist", "100 900", "Geist-variable-latin.woff2"),
        ("IBM Plex Mono", "400", "IBMPlexMono-400-latin.woff2"),
        ("IBM Plex Mono", "500", "IBMPlexMono-500-latin.woff2"),
        ("IBM Plex Mono", "600", "IBMPlexMono-600-latin.woff2"),
    ]
    out = []
    for family, weight, filename in faces:
        payload = base64.b64encode((_FONT_DIR / filename).read_bytes()).decode()
        out.append(
            f"@font-face{{font-family:'{family}';font-style:normal;"
            f"font-weight:{weight};font-display:swap;"
            f"src:url(data:font/woff2;base64,{payload}) format('woff2');}}")
    return "".join(out)


# ── Plotly template ──────────────────────────────────────────────────────

TEMPLATE_NAME = "openlia"


def _build_template() -> go.layout.Template:
    axis = dict(
        gridcolor=GRID, gridwidth=1, zeroline=False,
        linecolor=BORDER_STRONG, linewidth=1,
        ticks="outside", tickcolor=BORDER_STRONG, ticklen=4,
        tickfont=dict(family=FONT_MONO, size=10, color=TEXT_SECONDARY),
        title=dict(font=dict(family=FONT_MONO, size=10, color=TEXT_SECONDARY)),
        automargin=True,
    )
    return go.layout.Template(layout=dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_DISPLAY, size=13, color=INK),
        xaxis=axis, yaxis=axis,
        colorway=[SERIES_INK, SERIES_CONTRAST, SERIES_ALT, REFERENCE],
        colorscale=dict(diverging=DIVERGING,
                        sequential=[[0.0, SURFACE], [1.0, INK]],
                        sequentialminus=[[0.0, SURFACE], [1.0, ERROR]]),
        legend=dict(
            font=dict(family=FONT_MONO, size=10, color=TEXT_SECONDARY),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        title=dict(font=dict(family=FONT_DISPLAY, size=15, color=INK)),
        hoverlabel=dict(
            bgcolor=INK, bordercolor=INK,
            font=dict(family=FONT_MONO, size=11, color=SURFACE),
        ),
        annotationdefaults=dict(
            font=dict(family=FONT_MONO, size=10, color=TEXT_SECONDARY)),
        # P9's layer toggle. Plotly's default control is blue-tinted, and it
        # derives the pressed state from `bgcolor` itself -- there is no
        # active-colour property to set -- so giving it the warm paper the
        # cards already use is what takes the blue out of both states. INK
        # labels it at 16.6:1.
        updatemenudefaults=dict(
            bgcolor=PAPER, bordercolor=BORDER_STRONG, borderwidth=1,
            font=dict(family=FONT_MONO, size=10, color=INK),
        ),
        margin=dict(l=54, r=20, t=16, b=44),
    ))


def register_template() -> str:
    """Register the template with Plotly and return its name.

    Idempotent -- the module-level call below runs it once at import, and
    re-registering the same name is harmless.
    """
    pio.templates[TEMPLATE_NAME] = _build_template()
    return TEMPLATE_NAME


register_template()
