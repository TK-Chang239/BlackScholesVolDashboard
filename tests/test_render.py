"""Render layer: figures carry the right traces; page is self-contained."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.render.figures import build_smile_figure, build_term_structure_figure
from src.render.page import render_page

SPOT = 770.0


def smile_frame():
    return pd.DataFrame({
        "expiry": [dt.date(2026, 9, 18)] * 4 + [dt.date(2026, 11, 20)] * 4,
        "dte": [21] * 4 + [84] * 4,
        "strike": [700.0, 740.0, 770.0, 800.0] * 2,
        "moneyness": [k / SPOT for k in (700.0, 740.0, 770.0, 800.0)] * 2,
        "kind": ["put", "put", "call", "call"] * 2,
        "iv": [0.28, 0.24, 0.21, 0.20, 0.26, 0.23, 0.215, 0.205],
    })


def term_frame(shift=0.0):
    return pd.DataFrame({
        "expiry": [dt.date(2026, 9, 18), dt.date(2026, 11, 20)],
        "dte": [21, 84],
        "atm_iv": [0.21 + shift, 0.23 + shift],
    })


def fake_status():
    return {
        "last_success_utc": "2026-08-28T21:35:00+00:00",
        "snapshot_date": "2026-08-28", "source": "yfinance",
        "rows_stored": 400, "spot": SPOT,
        "risk_free_rate": 0.0415, "dividend_yield": 0.0098,
        "iv_convergence": 0.985,
    }


class TestFigures:
    def test_smile_one_trace_per_expiry_plus_atm_marker(self):
        fig = build_smile_figure(smile_frame(), SPOT)
        # expiry traces are lines+markers; the ATM reference is lines-only
        expiry_traces = [t for t in fig.data if t.mode and "markers" in t.mode]
        assert len(expiry_traces) == 2  # one per expiry
        assert any("ATM" in (t.name or "") for t in fig.data)

    def test_term_structure_overlay_present_and_absent(self):
        fig2 = build_term_structure_figure(term_frame(), term_frame(shift=0.01))
        assert len(fig2.data) == 2
        fig1 = build_term_structure_figure(term_frame(), None)
        assert len(fig1.data) == 1

    def test_term_overlay_is_muted(self):
        fig = build_term_structure_figure(term_frame(), term_frame(shift=0.01))
        prev = [t for t in fig.data if "previous session" in (t.name or "").lower()]
        assert len(prev) == 1


class TestPage:
    def page(self, **kw):
        figs = {"P2": build_smile_figure(smile_frame(), SPOT),
                "P3": build_term_structure_figure(term_frame(), None)}
        figs.update(kw.pop("figures", {}))
        return render_page(figs, fake_status())

    def test_self_contained_no_external_refs(self):
        from src.render.page import _BUNDLE
        html = self.page()
        own_markup = html.replace(_BUNDLE.read_text(), "")
        assert "https://cdn.plot.ly" not in own_markup
        assert "<script src=" not in own_markup
        assert html.count("<html") == 1

    def test_bundle_inlined(self):
        html = self.page()
        assert len(html) > 500_000                  # the vendored bundle is in there
        assert len(html.encode()) < 5_000_000       # SPEC 2.5 budget

    def test_structure_header_questions_footer(self):
        html = self.page()
        assert "viewport" in html
        for must in ("SPY", "770", "2026-08-28",     # header: underlying, spot, date
                     "Q1", "Q2", "Q3", "Q4", "Q5",   # five question sections
                     "Educational project",           # footer disclaimer
                     "yfinance"):                     # data-source transparency
            assert must in html, must

    def test_missing_panel_renders_placeholder(self):
        html = render_page({}, fake_status())
        assert "Phase" in html                       # 'arrives in Phase N' notes
        assert "<html" in html


class TestSensitivityFigure:
    def test_two_subplots_and_two_bar_traces_each(self):
        from src.analytics.sensitivity import compute_sensitivity
        from src.render.figures import build_sensitivity_figure
        fig = build_sensitivity_figure(compute_sensitivity(770.0, 0.12, 30, 0.04, 0.01, 0.20), 0.20)
        bars = [t for t in fig.data if t.type == "bar"]
        assert len(bars) == 4                        # (down, up) x (argued, observed)
        assert {t.xaxis for t in bars} == {"x", "x2"}
        assert "Volatility" in "".join(str(t.y) for t in bars)

    def test_nan_input_renders_annotation(self):
        from src.analytics.sensitivity import compute_sensitivity
        from src.render.figures import build_sensitivity_figure
        fig = build_sensitivity_figure(compute_sensitivity(770.0, float("nan"), 30, 0.04, 0.01, 0.2), 0.2)
        assert not [t for t in fig.data if t.type == "bar"]
        assert fig.layout.annotations and "No" in fig.layout.annotations[0].text


class TestGreekTiles:
    def _tiles(self, theta=-36.5, vega=95.0, rho=31.0, delta=0.52):
        import datetime as dt
        return pd.DataFrame([{
            "kind": "call", "expiry": dt.date(2026, 9, 25), "dte": 28, "strike": 770.0,
            "iv": 0.12, "delta": delta, "gamma": 0.021, "vega": vega, "theta": theta, "rho": rho,
        }])

    def test_display_scaling(self):
        from src.render.tiles import greek_tiles_html
        html = greek_tiles_html(self._tiles(), None, "previous session")
        assert "-0.100" in html          # theta/365 = -0.1 per day
        assert "0.950" in html           # vega/100 per vol point
        assert "0.310" in html           # rho/100 per 1%
        assert "0.520" in html and "per day" in html and "per vol pt" in html

    def test_one_day_change_line(self):
        from src.render.tiles import greek_tiles_html
        html = greek_tiles_html(self._tiles(delta=0.52), self._tiles(delta=0.50), "prev session 2026-08-27")
        assert "+0.020" in html and "prev session 2026-08-27" in html

    def test_empty_is_placeholder(self):
        from src.render.tiles import greek_tiles_html
        from src.analytics.greeks_panel import TILE_COLUMNS
        html = greek_tiles_html(pd.DataFrame(columns=TILE_COLUMNS), None, "x")
        assert "placeholder" in html


class TestGreeksCurvesFigure:
    def test_two_subplots_per_kind(self):
        from src.render.figures import build_greeks_curves_figure
        curves = pd.DataFrame({
            "kind": ["call", "call", "put", "put"], "strike": [760.0, 780.0, 760.0, 780.0],
            "moneyness": [0.987, 1.013, 0.987, 1.013],
            "delta": [0.6, 0.4, -0.4, -0.6], "gamma": [0.02, 0.02, 0.02, 0.02]})
        fig = build_greeks_curves_figure(curves, 770.0)
        scatters = [t for t in fig.data if t.type == "scatter" and t.name in ("call", "put")]
        assert len(scatters) == 4 and {t.xaxis for t in scatters} == {"x", "x2"}


class TestIvRvFigure:
    def _series(self):
        import datetime as dt
        d = [dt.date(2026, 8, 20) + dt.timedelta(days=i) for i in range(3)]
        return pd.DataFrame({"date": d, "atm_iv": [0.12, 0.13, 0.14],
                             "rv_trailing": [0.10, 0.10, 0.11], "fwd_rv": [0.11, np.nan, np.nan],
                             "spread": [0.02, 0.03, 0.03],
                             "spread_running_mean": [0.02, 0.025, 0.0267]})

    def test_traces_and_since_annotation(self):
        import yaml
        from src.analytics.iv_rv import iv_rv_summary
        from src.render.figures import build_iv_rv_figure
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        s = self._series()
        fig = build_iv_rv_figure(s, iv_rv_summary(s), cfg)
        names = [t.name for t in fig.data]
        assert any("implied" in n.lower() for n in names)
        assert any("trailing" in n.lower() for n in names)
        assert any("forward" in n.lower() for n in names)
        assert any(t.fill == "toself" for t in fig.data)
        assert any("2026-08-20" in (a.text or "") for a in fig.layout.annotations)

    def test_empty_series(self):
        import yaml
        from src.analytics.iv_rv import IV_RV_COLUMNS, iv_rv_summary
        from src.render.figures import build_iv_rv_figure
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        s = pd.DataFrame(columns=IV_RV_COLUMNS)
        fig = build_iv_rv_figure(s, iv_rv_summary(s), cfg)
        assert not fig.data and fig.layout.annotations

    def test_wide_span_defaults_to_last_180_days(self):
        import yaml
        from src.analytics.iv_rv import iv_rv_summary
        from src.render.figures import build_iv_rv_figure
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        dates = [dt.date(2024, 9, 3)] + \
            [dt.date(2026, 8, 20) + dt.timedelta(days=i) for i in range(3)]
        s = pd.DataFrame({"date": dates, "atm_iv": [0.16, 0.12, 0.13, 0.14],
                          "rv_trailing": [0.11, 0.10, 0.10, 0.11],
                          "fwd_rv": [0.11, np.nan, np.nan, np.nan],
                          "spread": [0.05, 0.02, 0.03, 0.03],
                          "spread_running_mean": [0.05, 0.035, 0.0333, 0.0325]})
        fig = build_iv_rv_figure(s, iv_rv_summary(s), cfg)
        rng = fig.layout.xaxis.range
        assert rng is not None
        last = dates[-1]
        lower = pd.Timestamp(rng[0]).date()
        assert (last - lower).days <= 180

    def test_short_span_leaves_autorange(self):
        import yaml
        from src.analytics.iv_rv import iv_rv_summary
        from src.render.figures import build_iv_rv_figure
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        s = self._series()
        fig = build_iv_rv_figure(s, iv_rv_summary(s), cfg)
        assert fig.layout.xaxis.range is None

    def test_large_gap_breaks_line_and_fill(self):
        import yaml
        from src.analytics.iv_rv import iv_rv_summary
        from src.render.figures import build_iv_rv_figure
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        dates = [dt.date(2024, 9, 3)] + \
            [dt.date(2026, 8, 20) + dt.timedelta(days=i) for i in range(3)]
        s = pd.DataFrame({"date": dates, "atm_iv": [0.16, 0.12, 0.13, 0.14],
                          "rv_trailing": [0.11, 0.10, 0.10, 0.11],
                          "fwd_rv": [0.11, np.nan, np.nan, np.nan],
                          "spread": [0.05, 0.02, 0.03, 0.03],
                          "spread_running_mean": [0.05, 0.035, 0.0333, 0.0325]})
        fig = build_iv_rv_figure(s, iv_rv_summary(s), cfg)
        iv_trace = next(t for t in fig.data if "implied" in (t.name or "").lower())
        ys = list(iv_trace.y)
        assert any(y is None or (isinstance(y, float) and np.isnan(y)) for y in ys)

    def test_contiguous_series_has_no_gap_break(self):
        import yaml
        from src.analytics.iv_rv import iv_rv_summary
        from src.render.figures import build_iv_rv_figure
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
        s = self._series()
        fig = build_iv_rv_figure(s, iv_rv_summary(s), cfg)
        iv_trace = next(t for t in fig.data if "implied" in (t.name or "").lower())
        ys = list(iv_trace.y)
        assert not any(y is None or (isinstance(y, float) and np.isnan(y)) for y in ys)


class TestPagePhase4:
    def _figs(self):
        import plotly.graph_objects as go
        return {p: go.Figure() for p in ("P1", "P2", "P3", "P4", "P5")}

    def _status(self):
        return {"spot": 771.1, "snapshot_date": "2026-08-27", "source": "yfinance",
                "iv_convergence": 0.99, "last_success_utc": "2026-08-28T14:38:34+00:00",
                "rows_stored": 1459}

    def test_q1_and_q3_render_panels_not_placeholders(self):
        from src.render.page import render_page
        html = render_page(self._figs(), self._status())
        assert "Input-sensitivity panel arrives" not in html
        assert "Implied-vs-realized panel arrives" not in html
        q1 = html.index("Q1."); q2 = html.index("Q2."); q3 = html.index("Q3.")
        q4 = html.index("<h2>Q4.")  # plain "Q4." also matches inside P4's verbatim caption text
        assert q1 < html.index("Input sensitivity") < q2
        assert q3 < html.index("Implied vs realized") < q4

    def test_extras_are_inserted_under_their_panel(self):
        from src.render.page import render_page
        html = render_page(self._figs(), self._status(),
                           extras={"P4": "<div class='tiles'>TILES-HERE</div>",
                                   "P5": "<p class='stat'>STAT-HERE</p>"})
        assert html.index("Greeks") < html.index("TILES-HERE")
        assert html.index("Implied vs realized") < html.index("STAT-HERE")

    def test_footer_links_repo(self):
        from src.render.page import render_page
        assert "https://github.com/TK-Chang239/BlackScholesVolDashboard" in \
            render_page(self._figs(), self._status())

    def test_every_rendered_panel_has_a_caption(self):
        from src.render.page import CAPTIONS, render_page
        html = render_page(self._figs(), self._status())
        for pid in ("P1", "P2", "P3", "P4", "P5"):
            assert CAPTIONS[pid] in html


class TestSkewFigure:
    def _metrics(self, dates, skews):
        return pd.DataFrame({"date": dates, "skew_25d": skews,
                             "skew_25d_dte": [30.0] * len(dates)})

    def test_vol_points_and_annotation_match(self):
        import datetime as dt
        from src.render.figures import build_skew_figure
        d = [dt.date(2026, 8, 24) + dt.timedelta(days=i) for i in range(4)]
        m = self._metrics(d, [0.030, 0.032, 0.045, 0.031])
        ann = pd.DataFrame({"date": [d[2], dt.date(2026, 1, 1)], "note": ["risk-off", "absent"]})
        fig = build_skew_figure(m, ann)
        trace = [t for t in fig.data if "skew" in (t.name or "").lower()][0]
        assert max(v for v in trace.y if v == v) == pytest.approx(4.5)
        texts = [a.text for a in fig.layout.annotations]
        assert "risk-off" in texts and "absent" not in texts

    def test_gap_breaks_and_recent_range(self):
        import datetime as dt
        from src.render.figures import build_skew_figure
        d = [dt.date(2024, 9, 3)] + [dt.date(2026, 8, 24) + dt.timedelta(days=i) for i in range(3)]
        fig = build_skew_figure(self._metrics(d, [0.02, 0.03, 0.03, 0.03]),
                                pd.DataFrame(columns=["date", "note"]))
        trace = [t for t in fig.data if "skew" in (t.name or "").lower()][0]
        assert any(v is None or v != v for v in trace.y)
        assert fig.layout.xaxis.range is not None

    def test_all_nan_is_empty_figure(self):
        import datetime as dt
        from src.render.figures import build_skew_figure
        m = self._metrics([dt.date(2026, 8, 24)], [float("nan")])
        fig = build_skew_figure(m, pd.DataFrame(columns=["date", "note"]))
        assert not fig.data and fig.layout.annotations
