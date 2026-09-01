"""The page's stylesheet, built from the design-system tokens in `theme`.

Kept apart from `theme` so that module stays a readable table of values: this
one is the OpenLia Report chrome -- masthead, metric strip, contents, numbered
sections, panel cards -- expressed against those values.

Light only, by design. The design system ships a dark block, but the Plotly
figures are baked server-side, so a CSS-only toggle would leave every plot
area on warm paper inside a dark page.
"""
from src.render import theme as t


def _tokens() -> str:
    """The DS custom properties this page actually uses."""
    return f""":root{{
--ink:{t.INK};--paper:{t.PAPER};--surface:{t.SURFACE};--surface-subtle:{t.SURFACE_SUBTLE};
--border:{t.BORDER};--border-strong:{t.BORDER_STRONG};
--text-2:{t.TEXT_SECONDARY};--text-3:{t.TEXT_TERTIARY};
--acid:{t.ACID};--acid-on:{t.ACID_ON};--yellow-100:{t.YELLOW_100};--yellow-800:{t.YELLOW_800};
--success:{t.SUCCESS};--warning:{t.WARNING};--error:{t.ERROR};
--font-display:{t.FONT_DISPLAY};--font-mono:{t.FONT_MONO};
}}"""


_CHROME = """
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;padding:0;background:var(--paper);color:var(--ink);
  font-family:var(--font-display);font-size:14px;line-height:1.6;
  -webkit-font-smoothing:antialiased}
a{color:inherit}

.shell{max-width:1120px;margin:0 auto;padding:48px 32px 96px}

/* ── Masthead ─────────────────────────────────────────────── */
.masthead{display:grid;grid-template-columns:1fr auto;gap:24px;align-items:end;
  padding-bottom:24px;border-bottom:2px solid var(--ink);margin-bottom:0}
.eyebrow{font-family:var(--font-mono);font-size:11px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text-2)}
.eyebrow .accent{color:var(--yellow-800);font-weight:600}
.mh-title{font-family:var(--font-display);font-size:56px;font-weight:500;
  line-height:1;letter-spacing:-.03em;margin:10px 0 0}
.mh-title em{font-style:normal;background:var(--acid);color:var(--acid-on);
  padding:0 8px;margin-left:-2px;border-radius:4px}
.mh-sub{font-size:15px;color:var(--text-2);margin:12px 0 0;max-width:520px;
  line-height:1.5}

/* ── Status pill ──────────────────────────────────────────── */
.pill{display:inline-flex;align-items:center;gap:8px;white-space:nowrap;
  padding:6px 12px;border-radius:9999px;border:1px solid var(--border);
  background:var(--surface);font-family:var(--font-mono);font-size:10px;
  letter-spacing:.12em;text-transform:uppercase;color:var(--text-2);font-weight:500}
.pill::before{content:'';width:6px;height:6px;border-radius:50%;
  background:var(--text-3)}
.pill.on::before{background:var(--success);box-shadow:0 0 0 2px rgba(107,130,0,.18)}
.pill.err::before{background:var(--error);box-shadow:0 0 0 2px rgba(224,92,48,.18)}

/* ── Metric strip ─────────────────────────────────────────── */
.metric-strip{display:grid;grid-template-columns:repeat(5,1fr);
  border-bottom:1px solid var(--border);margin-bottom:44px}
.ms-cell{padding:16px 16px 16px 0;border-right:1px solid var(--border);
  display:flex;flex-direction:column;gap:5px}
.ms-cell:not(:first-child){padding-left:16px}
.ms-cell:last-child{border-right:0}
.ms-label{font-family:var(--font-mono);font-size:9px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text-2)}
.ms-value{font-family:var(--font-display);font-size:22px;line-height:1.1;
  letter-spacing:-.02em;font-weight:500;font-variant-numeric:tabular-nums;
  word-break:break-word}

/* ── Contents ─────────────────────────────────────────────── */
/* One column per question, never `auto-fit`: the 1px gap over a border-coloured
   background is what draws the rules between cards, so any column count that
   does not divide the five questions leaves an empty cell showing as a solid
   grey block. Five, or one. */
.toc{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;
  background:var(--border);border:1px solid var(--border);border-radius:8px;
  overflow:hidden;margin-bottom:64px}
.toc a{background:var(--surface);padding:14px 16px;text-decoration:none;
  display:flex;flex-direction:column;gap:6px;transition:background 200ms
  cubic-bezier(.16,1,.3,1)}
.toc a:hover{background:#FFFCE5}
.toc-num{font-family:var(--font-mono);font-size:10px;letter-spacing:.12em;
  color:var(--text-2)}
.toc-name{font-size:13px;font-weight:500;line-height:1.35}
.toc-count{font-family:var(--font-mono);font-size:9px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--text-3);margin-top:auto}

/* ── Sections ─────────────────────────────────────────────── */
section{margin-bottom:72px;scroll-margin-top:24px}
.section-head{display:grid;grid-template-columns:64px 1fr auto;gap:20px;
  align-items:baseline;margin-bottom:24px;padding-bottom:14px;
  border-bottom:1px solid var(--border)}
.section-num{font-family:var(--font-mono);font-size:11px;letter-spacing:.14em;
  color:var(--text-2)}
.section-name{font-family:var(--font-display);font-size:26px;font-weight:500;
  letter-spacing:-.02em;line-height:1.25;margin:0}
.section-count{font-family:var(--font-mono);font-size:9px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--text-3);white-space:nowrap}

/* ── Panel cards ──────────────────────────────────────────── */
.panel{background:var(--surface);border:1px solid var(--border);border-radius:8px;
  overflow:hidden;margin-bottom:20px}
.panel-head{padding:11px 16px;border-bottom:1px solid var(--border);
  background:var(--paper);display:flex;justify-content:space-between;
  align-items:center;gap:12px}
.panel-id{font-family:var(--font-mono);font-size:9.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text-2)}
.panel-id strong{color:var(--ink);font-weight:600}
.tag{font-family:var(--font-mono);font-size:8.5px;letter-spacing:.1em;
  text-transform:uppercase;padding:3px 7px;background:var(--border);
  color:var(--text-2);border-radius:3px;font-weight:500;white-space:nowrap}
.panel-body{padding:18px 20px 20px}

.panel-sub{font-size:13px;line-height:1.45;color:var(--ink);margin:0 0 14px;
  max-width:74ch}
.figure{width:100%;overflow-x:auto}

/* ── Captions and stat lines ──────────────────────────────── */
.caption{color:var(--text-2);font-size:13px;line-height:1.65;
  margin:14px 0 0;padding-top:14px;border-top:1px solid var(--border);
  max-width:74ch}
.stat{font-family:var(--font-mono);font-size:12px;line-height:1.6;
  color:var(--ink);font-variant-numeric:tabular-nums;
  background:var(--paper);border-radius:6px;padding:11px 14px;
  margin:0 0 14px;border-left:3px solid var(--acid)}
.stat + .caption{margin-top:0;padding-top:0;border-top:0}
.placeholder{color:var(--text-3);font-style:italic;font-size:13px;margin:0}

/* ── Staleness callout ────────────────────────────────────── */
.stale{background:var(--surface);border:1px solid var(--border);
  border-left:3px solid var(--error);border-radius:6px;padding:14px 16px;
  margin:24px 0 0;color:var(--ink);font-size:13px;line-height:1.6}
.stale b{font-family:var(--font-mono);font-weight:600}

/* ── Greek tiles ──────────────────────────────────────────── */
.tiles-head{font-family:var(--font-mono);font-size:9.5px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--text-2);margin:0 0 10px;
  padding-bottom:8px;border-bottom:1px dashed var(--border-strong)}
.tiles-head strong{color:var(--ink);font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));
  gap:10px;margin:0 0 22px}
.tile{background:var(--paper);border:1px solid var(--border);border-radius:8px;
  padding:14px;display:flex;flex-direction:column;gap:6px}
.tile .k{font-family:var(--font-mono);font-size:9px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--text-2)}
.tile .v{font-family:var(--font-display);font-size:26px;line-height:1;
  letter-spacing:-.03em;font-weight:500;font-variant-numeric:tabular-nums}
.tile .u{font-size:11.5px;color:var(--text-2)}
.tile .d{font-family:var(--font-mono);font-size:10px;color:var(--text-2);
  font-variant-numeric:tabular-nums;padding-top:8px;margin-top:auto;
  border-top:1px solid var(--border)}

/* ── Footer ───────────────────────────────────────────────── */
footer{margin-top:72px;padding-top:20px;border-top:2px solid var(--ink);
  color:var(--text-2);font-size:12.5px;line-height:1.65;max-width:74ch}
footer a{color:var(--ink);text-decoration-color:var(--border-strong);
  text-underline-offset:2px}

/* ── Narrow screens ───────────────────────────────────────── */
@media (max-width: 820px){
  .shell{padding:32px 18px 64px}
  .masthead{grid-template-columns:1fr;align-items:start}
  /* The pill is a grid item once the masthead stacks, and a stretched
     inline-flex runs the full column width. */
  .masthead .pill{justify-self:start}
  .mh-title{font-size:40px}
  .metric-strip{grid-template-columns:repeat(2,1fr)}
  .ms-cell{border-bottom:1px solid var(--border)}
  .ms-cell:nth-child(2n){border-right:0}
  .ms-cell:nth-child(2n+1){padding-left:0}
  /* Five cells over two columns leaves the fifth alone beside a hole that
     still draws its rules. Let it take the whole row instead. */
  .ms-cell:last-child{grid-column:1/-1;border-right:0;border-bottom:0}
  .toc{grid-template-columns:1fr}
  .section-head{grid-template-columns:1fr;gap:6px}
  .section-count{display:none}
  .section-name{font-size:21px}
  .panel-body{padding:14px}
}
@media (max-width: 700px){
  /* Two-column subplot figures compress to ~170px per panel on a phone, which
     is not a chart. Hold them at a readable width and let .figure scroll. */
  .figure-wide > div{min-width:660px}
}
"""


def page_css() -> str:
    """The full stylesheet, fonts inlined, ready to drop in a <style> tag."""
    return t.font_face_css() + _tokens() + _CHROME
