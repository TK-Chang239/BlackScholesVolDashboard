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

    def test_phase5_panels_and_sections(self):
        import plotly.graph_objects as go
        from src.render.page import CAPTIONS, render_page
        figs = {p: go.Figure() for p in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P9")}
        html = render_page(figs, self._status())
        q2, q3, q5 = html.index("<h2>Q2."), html.index("<h2>Q3."), html.index("<h2>Q5.")
        assert q2 < html.index("25-delta skew") < html.index("Model-vs-market") < q3
        assert q5 < html.index("Put-call parity checker")
        assert "arrives in Phase 5" not in html
        for pid in ("P6", "P7", "P9"):
            assert CAPTIONS[pid] in html

    def test_p7_caption_states_the_bound_and_its_limits_honestly(self):
        from src.render.page import CAPTIONS
        cap = CAPTIONS["P7"]
        # the muted population is put-rich gaps, not "deep in-the-money puts"
        assert "in-the-money puts often show" not in cap
        # the code tests |gap| <= bound + spread, so the caption must say so
        assert "K·(1 − e⁻ʳᵀ) − S·(1 − e⁻ᵠᵀ) plus the combined bid–ask" in cap
        # the rigorous ceiling is K(1 - e^-rT); the bound we use is that less the
        # forgone dividends, so it may not claim to rule early exercise out
        assert "no matter how the option is modelled" not in cap
        assert "interest earned on the strike" in cap
        assert "less the dividends forgone" in cap
        # only an in-the-money put can be exercised early for value (the gate)
        assert "in-the-money put" in cap
        # and the calibration's own blind spot is disclosed
        assert "level error" in cap

    def test_p7_caption_calls_the_muting_consistency_not_proof(self):
        from src.render.page import CAPTIONS
        cap = CAPTIONS["P7"]
        # a gap inside the bound is CONSISTENT WITH early exercise; a stale mark
        # of the same size fits the bound just as well and is not arbitrage either
        assert "consistent with early exercise" in cap
        assert "not proof" in cap
        assert "stale mark" in cap
        assert "are still not arbitrage" not in cap


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


class TestParityFigure:
    def _parity(self, quoted=True, cells=None):
        import datetime as dt
        e1, e2 = dt.date(2026, 9, 25), dt.date(2026, 11, 20)
        cells = cells or ((740.0, 0.02, False), (770.0, -0.01, False), (800.0, 0.9, True))
        rows = []
        for expiry, dte in ((e1, 28), (e2, 84)):
            for k, dev, viol in cells:
                rows.append({"expiry": expiry, "dte": dte, "strike": k, "moneyness": k / 770.0,
                             "call_price": 10.0, "put_price": 9.0,
                             "call_oi": 500.0, "put_oi": 500.0, "liquid": True,
                             "lhs": 1.0, "rhs": 1.0 - dev,
                             "deviation": dev, "forward": 772.0, "deviation_fwd": dev,
                             "spread": 0.3 if quoted else float("nan"),
                             "tradeable_violation": viol if quoted else False,
                             "tradeable_violation_fwd": viol if quoted else False})
        return pd.DataFrame(rows)

    def test_two_layers_and_hot_cells(self):
        from src.render.figures import build_parity_figure
        fig = build_parity_figure(self._parity(), 770.0)
        heat = [t for t in fig.data if t.type == "heatmap"]
        assert len(heat) == 2
        faint, hot = heat
        assert faint.opacity == 0.3 and faint.showscale is False and hot.showscale is True
        # the hot layer is liquid AND beyond spread AND unaccounted for
        assert hot.name == "unexplained"
        # ...and the faint layer carries EVERY cell -- within-spread, illiquid,
        # early-exercise-explained and the hot ones -- so it cannot be called
        # "within spread"
        assert faint.name == "all pairs"
        hot_vals = [v for row in hot.z for v in row if v is not None and v == v]
        assert hot_vals == pytest.approx([0.9, 0.9])
        assert list(faint.x) == ["2026-09-25 (28d)", "2026-11-20 (84d)"]
        assert not any("spreads unknown" in (a.text or "") for a in fig.layout.annotations)

    def test_close_based_is_annotated_and_nothing_hot(self):
        from src.render.figures import build_parity_figure
        fig = build_parity_figure(self._parity(quoted=False), 770.0)
        hot = [t for t in fig.data if t.type == "heatmap"][1]
        assert all(v is None or v != v for row in hot.z for v in row)
        assert any("spreads unknown" in (a.text or "") for a in fig.layout.annotations)

    def test_outlier_does_not_flatten_the_scale(self):
        from src.render.figures import build_parity_figure
        # 60 cells and TWO dead quotes: at n = 6 `method="lower"` discards exactly
        # the maximum whatever the percentile, so a small fixture would pass this
        # without ever exercising the 98th-percentile rule the test is named for.
        cells = [(600.0 + 5.0 * i, 0.02 + 0.03 * (i % 8), (i % 8) == 7) for i in range(30)]
        p = self._parity(cells=cells)
        assert len(p) == 60
        p.loc[p.index[:2], ["deviation", "deviation_fwd"]] = 160.0
        fig = build_parity_figure(p, 770.0)
        faint = [t for t in fig.data if t.type == "heatmap"][0]
        assert faint.zmax < 20.0        # scale set by the bulk, not the outliers
        assert faint.zmax == pytest.approx(0.23)     # the largest genuine cell

    def test_non_liquid_violation_is_not_hot(self):
        from src.render.figures import build_parity_figure
        p = self._parity()
        p["liquid"] = False
        fig = build_parity_figure(p, 770.0)
        hot = [t for t in fig.data if t.type == "heatmap"][1]
        assert all(v is None or v != v for row in hot.z for v in row)

    def test_falls_back_to_spot_based_deviation_when_no_forward(self):
        from src.render.figures import build_parity_figure
        p = self._parity()
        p["deviation_fwd"] = float("nan")
        p["forward"] = float("nan")
        p["tradeable_violation_fwd"] = False
        fig = build_parity_figure(p, 770.0)
        assert "implied forward" not in fig.layout.title.text
        hot = [t for t in fig.data if t.type == "heatmap"][1]
        hot_vals = [v for row in hot.z for v in row if v is not None and v == v]
        assert hot_vals == pytest.approx([0.9, 0.9])

    def test_early_exercise_explained_violation_is_muted(self):
        from src.render.figures import build_parity_figure
        p = self._parity()
        p["early_exercise_bound"] = 5.0
        # the 28-day violation is inside the bound; the 84-day one is not
        p["early_exercise_explained"] = p["tradeable_violation_fwd"] & (p["dte"] == 28)
        fig = build_parity_figure(p, 770.0)
        hot = [t for t in fig.data if t.type == "heatmap"][1]
        hot_vals = [v for row in hot.z for v in row if v is not None and v == v]
        assert hot_vals == pytest.approx([0.9])          # only the unexplained one stays hot
        assert any("early-exercise bound" in (a.text or "") for a in fig.layout.annotations)
        assert any("1 put-rich gap" in (a.text or "") for a in fig.layout.annotations)
        # the bound says the gap is CONSISTENT WITH early exercise, not that it
        # is early exercise -- a stale mark inside the bound is not arbitrage either
        note = [a.text for a in fig.layout.annotations if "early-exercise" in (a.text or "")][0]
        assert "consistent with the American early-exercise bound" in note
        assert "not arbitrage" not in note

    def test_early_exercise_annotation_sits_above_the_plot_area(self):
        from src.render.figures import build_parity_figure
        p = self._parity()
        p["early_exercise_bound"] = 5.0
        p["early_exercise_explained"] = p["tradeable_violation_fwd"] & (p["dte"] == 28)
        fig = build_parity_figure(p, 770.0)
        ann = [a for a in fig.layout.annotations if "early-exercise" in (a.text or "")][0]
        assert ann.xref == "paper" and ann.yref == "paper"
        assert ann.xanchor == "left" and ann.x == 0.0
        # above the top row of cells, not on top of them
        assert ann.y >= 1.02 and ann.yanchor == "bottom"
        assert fig.layout.margin.t >= 110       # room for it under the title

    def test_close_based_annotation_also_sits_above_the_plot_area(self):
        from src.render.figures import build_parity_figure
        fig = build_parity_figure(self._parity(quoted=False), 770.0)
        ann = [a for a in fig.layout.annotations if "spreads unknown" in (a.text or "")][0]
        assert ann.yref == "paper" and ann.y >= 1.02 and ann.yanchor == "bottom"

    def test_no_early_exercise_annotation_when_nothing_is_explained(self):
        from src.render.figures import build_parity_figure
        p = self._parity()
        p["early_exercise_bound"] = 5.0
        p["early_exercise_explained"] = False
        fig = build_parity_figure(p, 770.0)
        hot = [t for t in fig.data if t.type == "heatmap"][1]
        hot_vals = [v for row in hot.z for v in row if v is not None and v == v]
        assert hot_vals == pytest.approx([0.9, 0.9])
        assert not any("early-exercise" in (a.text or "") for a in fig.layout.annotations)

    def test_colour_limit_is_set_by_liquid_cells_only(self):
        from src.render.figures import build_parity_figure
        p = self._parity()
        dead = p.iloc[[0, 0, 0]].copy()          # three strikes nobody holds...
        dead["strike"] = [600.0, 620.0, 640.0]
        dead[["deviation", "deviation_fwd"]] = 160.0     # ...quoted at nonsense
        dead["liquid"] = False
        fig = build_parity_figure(pd.concat([p, dead], ignore_index=True), 770.0)
        faint = [t for t in fig.data if t.type == "heatmap"][0]
        assert faint.zmax == pytest.approx(0.9)          # set by the liquid bulk
        assert faint.zmin == pytest.approx(-0.9)

    def test_empty(self):
        from src.analytics.parity import PARITY_COLUMNS
        from src.render.figures import build_parity_figure
        fig = build_parity_figure(pd.DataFrame(columns=PARITY_COLUMNS), 770.0)
        assert not fig.data and fig.layout.annotations


class TestModelVsMarketFigure:
    def _mvm(self):
        import datetime as dt
        rows = []
        for kind in ("call", "put"):
            for k, pct, vol, muted in ((740.0, 0.02, 0.002, True), (800.0, 0.35, 0.05, False)):
                rows.append({"kind": kind, "expiry": dt.date(2026, 9, 25), "dte": 28, "strike": k,
                             "moneyness": k / 770.0, "market": 10.0, "model": 10.0 * (1 - pct),
                             "deviation_pct": pct, "deviation_vol": vol, "quoted": True,
                             "within_spread": muted})
        return pd.DataFrame(rows)

    def test_eight_layers_toggle_and_visibility(self):
        from src.render.figures import build_model_vs_market_figure
        fig = build_model_vs_market_figure(self._mvm(), 0.12)
        heat = [t for t in fig.data if t.type == "heatmap"]
        assert len(heat) == 8
        assert [t.visible for t in heat] == [True] * 4 + [False] * 4
        assert {t.xaxis for t in heat} == {"x", "x2"}
        buttons = fig.layout.updatemenus[0].buttons
        assert [b.label for b in buttons] == ["% of market price", "Vol points"]
        assert list(buttons[1].args[0]["visible"]) == [False] * 4 + [True] * 4
        hot_pct_calls = heat[1]
        vals = [v for row in hot_pct_calls.z for v in row if v is not None and v == v]
        assert vals == pytest.approx([0.35])
        assert [t.name for t in heat[:2]] == ["within spread", "beyond spread"]

    def test_one_colorbar_per_metric_and_it_clears_the_puts_subplot(self):
        from src.render.figures import build_model_vs_market_figure
        fig = build_model_vs_market_figure(self._mvm(), 0.12)
        heat = [t for t in fig.data if t.type == "heatmap"]
        shown = [t for t in heat if t.showscale]
        # both subplots of a metric share zmin/zmax, so one colorbar suffices --
        # and it lives to the right of the Puts subplot, not inside it
        assert len(shown) == 2
        assert all(t.xaxis == "x2" for t in shown)
        assert all(t.colorbar.x >= 1.0 for t in shown)
        assert [heat.index(t) for t in shown] == [3, 7]
        # the toggle still swaps exactly four traces for four
        assert len(heat) == 8
        assert [t.visible for t in heat] == [True] * 4 + [False] * 4
        buttons = fig.layout.updatemenus[0].buttons
        assert list(buttons[0].args[0]["visible"]) == [True] * 4 + [False] * 4
        assert list(buttons[1].args[0]["visible"]) == [False] * 4 + [True] * 4

    def test_vol_point_layer_is_plotted_in_vol_points(self):
        # `deviation_vol` is a FRACTION. P6 one section above multiplies it by 100
        # and hovers "+4.0 vol pts"; this panel must mean the same thing by a vol
        # point, and its colorbar says "Vol points" -- so the z values, the scale
        # and the hover all have to be in vol points too.
        from src.render.figures import build_model_vs_market_figure
        fig = build_model_vs_market_figure(self._mvm(), 0.12)
        heat = [t for t in fig.data if t.type == "heatmap"]
        vol_faint, vol_hot = heat[4], heat[5]        # calls, vol-point metric
        vals = sorted(v for row in vol_faint.z for v in row if v is not None and v == v)
        assert vals == pytest.approx([0.2, 5.0])     # 0.002 and 0.05 as vol points
        hot_vals = [v for row in vol_hot.z for v in row if v is not None and v == v]
        assert hot_vals == pytest.approx([5.0])
        assert (vol_faint.zmin, vol_faint.zmax) == (-20, 20)
        assert "%{z:+.1f} vol pts" in vol_hot.hovertemplate
        assert "%{z:+.1%}" not in vol_hot.hovertemplate
        assert heat[7].colorbar.title.text == "Vol points"
        # ...and the toggle still swaps exactly four visible traces for four
        assert len(heat) == 8
        assert [t.visible for t in heat] == [True] * 4 + [False] * 4
        buttons = fig.layout.updatemenus[0].buttons
        assert list(buttons[1].args[0]["visible"]) == [False] * 4 + [True] * 4

    def test_percent_metric_is_untouched_by_the_vol_point_scaling(self):
        from src.render.figures import build_model_vs_market_figure
        fig = build_model_vs_market_figure(self._mvm(), 0.12)
        heat = [t for t in fig.data if t.type == "heatmap"]
        vals = sorted(v for row in heat[0].z for v in row if v is not None and v == v)
        assert vals == pytest.approx([0.02, 0.35])   # still fractions of market price
        assert (heat[0].zmin, heat[0].zmax) == (-1.0, 1.0)
        assert "%{z:+.0%}" in heat[1].hovertemplate

    def test_empty(self):
        from src.analytics.model_vs_market import MVM_COLUMNS
        from src.render.figures import build_model_vs_market_figure
        fig = build_model_vs_market_figure(pd.DataFrame(columns=MVM_COLUMNS), 0.12)
        assert not fig.data and fig.layout.annotations


class TestParityStat:
    def _summary(self, **over):
        s = {"n_pairs": 420, "n_quoted": 420, "n_liquid": 400,
             "n_tradeable_violations": 132, "n_tradeable_violations_fwd": 3,
             "n_violations_early_exercise": 2, "n_violations_unexplained": 1,
             "share_within_spread": 0.6857, "share_within_spread_fwd": 0.9929,
             "max_abs_deviation": 160.28, "max_abs_deviation_fwd": 1.42}
        s.update(over)
        return s

    def _carry(self, **over):
        c = {"implied_carry": 0.02864, "dte_min": 84.0, "dte_max": 203.0,
             "carry_lo": 0.02827, "carry_hi": 0.03232, "n_expiries": 4}
        c.update(over)
        return c

    def _one_expiry_carry(self, value=0.0281, dte=84.0):
        return {"implied_carry": value, "dte_min": dte, "dte_max": dte,
                "carry_lo": value, "carry_hi": value, "n_expiries": 1}

    def _no_carry(self):
        return {"implied_carry": float("nan"), "dte_min": float("nan"),
                "dte_max": float("nan"), "carry_lo": float("nan"),
                "carry_hi": float("nan"), "n_expiries": 0}

    def test_quoted_sentence_leads_with_the_unexplained_count(self):
        from src.render.stats import parity_summary_html
        html = parity_summary_html(self._summary(), self._one_expiry_carry(), 0.0383, 0.0098)
        assert "400 liquid" in html
        assert "1 unexplained tradeable violation" in html
        assert "2 more put-rich gaps" in html and "early-exercise bound" in html
        assert "not arbitrage" in html
        assert "in-the-money" not in html          # the ITM gate makes the count exact,
                                                   # but the page still need not claim it
        # the calibrated total and the raw spot-based number stay visible
        assert "3 tradeable violations" in html
        assert "99.3%" in html and "forward is calibrated" in html
        assert "132" in html and "420" in html
        assert "2.81%" in html and "2.85%" in html and "84-day" in html

    def test_lead_publishes_the_unexplained_share_of_liquid_pairs(self):
        # the reader should not have to divide 4 by 400 themselves
        from src.render.stats import parity_summary_html
        html = parity_summary_html(self._summary(n_violations_unexplained=4),
                                   self._one_expiry_carry(), 0.0383, 0.0098)
        assert "4 unexplained tradeable violations (1.0% of liquid pairs)" in html
        html2 = parity_summary_html(self._summary(n_liquid=200, n_violations_unexplained=1),
                                    self._one_expiry_carry(), 0.0383, 0.0098)
        assert "1 unexplained tradeable violation (0.5% of liquid pairs)" in html2

    def test_no_early_exercise_clause_when_none_are_explained(self):
        from src.render.stats import parity_summary_html
        html = parity_summary_html(
            self._summary(n_violations_early_exercise=0, n_violations_unexplained=3),
            self._one_expiry_carry(), 0.0383, 0.0098)
        assert "3 unexplained tradeable violations" in html
        assert "early-exercise" not in html and "0 more" not in html

    def test_singular_violation_and_missing_share(self):
        from src.render.stats import parity_summary_html
        html = parity_summary_html(
            self._summary(n_tradeable_violations_fwd=1, n_tradeable_violations=1,
                          n_violations_early_exercise=0, n_violations_unexplained=1,
                          n_liquid=1, share_within_spread_fwd=float("nan")),
            self._no_carry(), 0.0383, 0.0098)
        assert "1 tradeable violation " in html and "nan" not in html.lower()
        assert "carry" not in html.lower()

    def test_no_liquid_pairs_refuses_to_publish_a_zero_all_clear(self):
        # A1: with no open interest anywhere on the frame, every liquidity-gated
        # count is zero by construction. Printing them is a false all-clear.
        from src.render.stats import parity_summary_html
        html = parity_summary_html(
            self._summary(n_liquid=0, n_tradeable_violations_fwd=0,
                          n_violations_early_exercise=0, n_violations_unexplained=0,
                          share_within_spread_fwd=float("nan")),
            self._one_expiry_carry(), 0.0383, 0.0098)
        assert "0 liquid" not in html and "0 unexplained" not in html
        assert "0 tradeable violations" not in html
        assert "open interest" in html
        assert "liquidity could not be assessed" in html
        # the spot-derived comparison is not liquidity-gated, so it survives
        assert "132 of 420 quoted pairs exceed the spread" in html
        assert "2.81%" in html                     # and so does the carry
        assert "nan" not in html.lower()

    def _close_based(self, **over):
        s = self._summary(n_quoted=0, n_tradeable_violations=0, n_tradeable_violations_fwd=0,
                          n_violations_early_exercise=0, n_violations_unexplained=0,
                          share_within_spread=float("nan"),
                          share_within_spread_fwd=float("nan"))
        s.update(over)
        return s

    def test_close_based_sentence_describes_what_the_figure_actually_plots(self):
        # `build_parity_figure` titles a close-based frame "against the market's
        # own implied forward" too, so the stat line must not say "raw".
        from src.render.stats import parity_summary_html
        html = parity_summary_html(self._close_based(), self._no_carry(), 0.0383, 0.0098)
        assert "spreads unknown, so tradeability cannot be assessed" in html
        assert "market-implied forward" in html
        assert "raw deviations only" not in html
        assert "carry" not in html.lower()

    def test_close_based_without_a_calibrated_forward_says_raw(self):
        from src.render.stats import parity_summary_html
        html = parity_summary_html(self._close_based(max_abs_deviation_fwd=float("nan")),
                                   self._no_carry(), 0.0383, 0.0098)
        assert "raw deviations only" in html and "nan" not in html.lower()

    def test_stale_spot_claim_only_when_the_calibration_removed_most_of_them(self):
        from src.render.stats import parity_summary_html
        # 132 -> 3 is a collapse, and the causal story is earned
        assert "stale-spot artefact" in parity_summary_html(
            self._summary(), self._one_expiry_carry(), 0.0383, 0.0098)
        # 132 -> 100 is not
        html = parity_summary_html(self._summary(n_tradeable_violations_fwd=100),
                                   self._one_expiry_carry(), 0.0383, 0.0098)
        assert "stale-spot" not in html and "fifteen minutes" not in html
        assert "132 of 420 quoted pairs exceed the spread" in html
        # and neither is a handful of raw violations
        html = parity_summary_html(self._summary(n_tradeable_violations=4),
                                   self._one_expiry_carry(), 0.0383, 0.0098)
        assert "stale-spot" not in html
        assert "4 of 420 quoted pairs exceed the spread" in html

    def test_carry_sentence_prints_the_median_the_range_and_the_span(self):
        # A3: showing the whole set is what makes this honest rather than
        # flattering -- the median alone would hide a 40 bp disagreement.
        from src.render.stats import parity_summary_html
        html = parity_summary_html(self._summary(), self._carry(), 0.0383, 0.0098)
        assert "ATM implied carry across the 84\u2013203-day expiries" in html
        assert "median 2.86%" in html
        assert "range 2.83\u20133.23%" in html
        assert "vs our r \u2212 q = 2.85%" in html

    def test_carry_sentence_names_the_expiry_when_only_one_qualifies(self):
        from src.render.stats import parity_summary_html
        html = parity_summary_html(self._summary(),
                                   self._one_expiry_carry(0.0281, 28.0), 0.0383, 0.0098)
        assert "ATM implied carry at the 28-day expiry: 2.81%" in html
        assert "range" not in html and "median" not in html

    def test_no_pairs(self):
        from src.render.stats import parity_summary_html
        s = self._summary(n_pairs=0, n_quoted=0, n_liquid=0)
        assert "No strike" in parity_summary_html(s, self._no_carry(), 0.04, 0.01)
