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
        # Phase 6 retired the last "arrives in Phase N" placeholder (Q4's), so
        # every question's `coming` string is now "" -- but the placeholder
        # markup itself must still render (empty) rather than raise or vanish,
        # since a degraded run could still hand render_page an incomplete
        # figures dict.
        html = render_page({}, fake_status())
        assert html.count("class='placeholder'") == 5
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

    def test_p8_captions_do_not_claim_a_missing_session_they_cannot_prove(self):
        # R2: a `model_gap` day is not necessarily an absent session. `_quote`
        # also returns None on a chain that IS stored but whose (expiry, strike)
        # pair lost a leg to an unconverged IV or to filter_chain's moneyness
        # gate -- the case test_chain_present_but_missing_the_quoted_pair_is_
        # model_marked covers. "The archive simply does not carry that session"
        # asserts a cause the data can contradict: the F2 mistake, re-entering
        # through a static caption twenty lines under a stat line that hedges
        # the identical claim correctly.
        from src.render.page import CAPTIONS
        for pid in ("P8a", "P8c"):
            cap = CAPTIONS[pid]
            assert "the archive simply does not carry" not in cap
            assert "the ones the archive is missing" not in cap
            assert "no usable quote" in cap
        assert "for that straddle" in CAPTIONS["P8a"]

    def test_p8a_caption_promises_only_the_counts_the_stat_line_gives(self):
        # R1: the promise must stay scoped to the kinds that actually occur --
        # `_mark_source_clause` names a count for each kind present and none for
        # one that is absent. The three shapes are asserted behaviourally in
        # TestHedgeSummaryHtml; this pins the sentence they back.
        from src.render.page import CAPTIONS
        assert ("Where either kind occurs, the stat line above gives its count"
                in CAPTIONS["P8a"])


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


class TestHedgeFigures:
    def _port(self):
        import datetime as dt
        return pd.DataFrame({
            "date": [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(5)],
            "pnl_day": [0.0, 1.0, -0.5, 2.0, -1.0],
            "pnl_cum": [0.0, 1.0, 0.5, 2.5, 1.5],
            "n_open": [1, 1, 1, 2, 2],
        })

    def _trades(self):
        import datetime as dt
        return pd.DataFrame({
            "entry_date": [dt.date(2026, 6, 1)], "expiry": [dt.date(2026, 7, 17)],
            "strike": [100.0], "dte_at_entry": [30], "entry_iv": [0.20],
            "entry_straddle": [6.0], "exit_date": [dt.date(2026, 6, 5)],
            "exit_value": [4.0], "pnl": [1.5], "n_days": [4],
            "n_market_marks": [4], "n_model_marks": [0], "n_structural_marks": [0],
            "market_mark_share": [1.0], "quotable_mark_share": [1.0],
            "lifetime_rv": [0.12], "edge": [0.08], "status": ["open"],
        })

    def _daily_overlapping(self):
        # Two trades, both live on 2026-06-02 and 2026-06-03: entry_date A opens
        # first and is still open when entry_date B opens. Row order is
        # deliberately NOT date-sorted within a group, so the test also covers
        # the implementation's own `g.sort_values("date")` before plotting.
        import datetime as dt
        return pd.DataFrame({
            "date": [dt.date(2026, 6, 3), dt.date(2026, 6, 1), dt.date(2026, 6, 2),
                     dt.date(2026, 6, 4), dt.date(2026, 6, 2), dt.date(2026, 6, 3)],
            "entry_date": [dt.date(2026, 6, 1)] * 3 + [dt.date(2026, 6, 2)] * 3,
            "pv": [1.0, 0.0, 0.5, 0.2, 0.0, -0.3],
        })

    def test_pnl_figure_plots_the_cumulative_line(self):
        from src.render.hedge_figures import build_hedge_pnl_figure
        port = self._port()
        fig = build_hedge_pnl_figure(port, pd.DataFrame())
        cum = [t for t in fig.data if "cumulative" in (t.name or "").lower()]
        assert len(cum) == 1
        assert list(cum[0].y) == port["pnl_cum"].tolist()
        assert "$" in fig.layout.yaxis.title.text

    def test_pnl_figure_is_empty_state_with_no_data(self):
        from src.render.hedge_figures import build_hedge_pnl_figure
        fig = build_hedge_pnl_figure(
            pd.DataFrame(columns=["date", "pnl_day", "pnl_cum", "n_open"]),
            pd.DataFrame())
        assert len(fig.data) == 0
        assert fig.layout.annotations[0].text

    def test_pnl_figure_overlays_one_line_per_overlapping_trade(self):
        from src.render.hedge_figures import build_hedge_pnl_figure
        fig = build_hedge_pnl_figure(self._port(), self._daily_overlapping())
        overlay = [t for t in fig.data if (t.name or "").startswith("trade ")]
        assert len(overlay) == 2
        by_name = {t.name: list(t.y) for t in overlay}
        assert set(by_name) == {"trade 2026-06-01", "trade 2026-06-02"}
        # date-sorted within each group, not the input row order
        assert by_name["trade 2026-06-01"] == [0.0, 0.5, 1.0]
        assert by_name["trade 2026-06-02"] == [0.0, -0.3, 0.2]

    def _daily_sequential(self):
        """Two trades that do NOT overlap -- the shape F4 was found on.

        Trade B opens on 2026-06-10, a day no other trade is live, so its
        definitional day-0 zero is the WHOLE of that session's portfolio total.
        Slicing only the portfolio's first row left it in the sample. One of
        trade A's days is model-marked (F6).
        """
        import datetime as dt
        return pd.DataFrame({
            "date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2), dt.date(2026, 6, 3),
                     dt.date(2026, 6, 10), dt.date(2026, 6, 11), dt.date(2026, 6, 12)],
            "entry_date": [dt.date(2026, 6, 1)] * 3 + [dt.date(2026, 6, 10)] * 3,
            "pnl_day": [0.0, 1.0, -0.5, 0.0, 2.0, -1.0],
            "mark_source": ["market", "market", "model_gap",
                            "market", "market", "settlement"],
        })

    def _hist_x(self, daily):
        from src.render.hedge_figures import build_hedge_histogram_figure
        fig = build_hedge_histogram_figure(daily)
        bars = [t for t in fig.data if t.type == "histogram"]
        assert len(bars) == 1
        return list(bars[0].x), fig

    def test_histogram_drops_every_trades_own_entry_day_not_only_the_first(self):
        # F4: 2026-06-10 is trade B's entry, an exact 0.00 at the mode of a
        # panel whose whole caption is about its width.
        x, fig = self._hist_x(self._daily_sequential())
        assert 0.0 not in x
        assert any("mean" in (a.text or "").lower() for a in fig.layout.annotations)

    def test_histogram_counts_only_market_marked_sessions(self):
        # F6: a model mark prices the straddle at the carried-forward vol, so it
        # is Black-Scholes-conforming by construction and narrows the measured
        # width. -0.5 is the model-marked day; the settlement day (-1.0) stays.
        x, _ = self._hist_x(self._daily_sequential())
        assert x == [1.0, 2.0, -1.0]

    def test_histogram_sums_concurrent_trades_within_one_session(self):
        import datetime as dt
        d1, d2, d3 = (dt.date(2026, 6, i) for i in (1, 2, 3))
        daily = pd.DataFrame({
            "date": [d1, d2, d3, d2, d3],
            "entry_date": [d1, d1, d1, d2, d2],
            "pnl_day": [0.0, 1.0, 0.5, 0.0, 0.25],
            "mark_source": ["market"] * 5,
        })
        x, _ = self._hist_x(daily)
        assert x == pytest.approx([1.0, 0.75])   # d3 carries both live trades

    def test_histogram_is_empty_state_when_nothing_is_market_marked(self):
        import datetime as dt
        from src.render.hedge_figures import build_hedge_histogram_figure
        only_entries_and_models = pd.DataFrame({
            "date": [dt.date(2026, 6, 1), dt.date(2026, 6, 2)],
            "entry_date": [dt.date(2026, 6, 1)] * 2,
            "pnl_day": [0.0, 1.0], "mark_source": ["market", "model_gap"],
        })
        fig = build_hedge_histogram_figure(only_entries_and_models)
        assert len(fig.data) == 0
        assert fig.layout.annotations[0].text

    def test_histogram_excludes_both_kinds_of_model_day(self):
        # The structural/gap split exists to decide which TRADES are quoted well
        # enough to plot as a dot. It changes nothing here: a structural day is
        # still a frozen-vol model price, and it would flatter this histogram
        # exactly as much as a gap day would. Both stay out.
        import datetime as dt
        days = [dt.date(2026, 6, i) for i in range(1, 5)]
        daily = pd.DataFrame({
            "date": days, "entry_date": [days[0]] * 4,
            "pnl_day": [0.0, 1.0, -0.5, -0.25],
            "mark_source": ["market", "market", "model_gap", "model_structural"],
        })
        x, _ = self._hist_x(daily)
        assert x == [1.0]

    def test_histogram_is_empty_state_with_no_trades(self):
        from src.analytics.hedge_sim import DAILY_COLUMNS
        from src.render.hedge_figures import build_hedge_histogram_figure
        fig = build_hedge_histogram_figure(pd.DataFrame(columns=DAILY_COLUMNS))
        assert len(fig.data) == 0


class TestHedgeScatterAndStat:
    def _settled(self, n=4):
        import datetime as dt
        return pd.DataFrame({
            "entry_date": [dt.date(2026, m, 1) for m in range(3, 3 + n)],
            "expiry": [dt.date(2026, m + 1, 17) for m in range(3, 3 + n)],
            "strike": [100.0] * n, "dte_at_entry": [30] * n,
            "entry_iv": [0.20] * n, "entry_straddle": [6.0] * n,
            "exit_date": [dt.date(2026, m + 1, 17) for m in range(3, 3 + n)],
            "exit_value": [4.0] * n, "pnl": [1.0, 2.0, 3.0, 4.0][:n],
            "n_days": [30] * n, "n_market_marks": [27] * n, "n_model_marks": [3] * n,
            "n_structural_marks": [2] * n,
            "market_mark_share": [0.9] * n, "quotable_mark_share": [27 / 28] * n,
            "lifetime_rv": [0.19, 0.18, 0.17, 0.16][:n],
            "edge": [0.01, 0.02, 0.03, 0.04][:n], "status": ["settled"] * n,
        })

    def test_scatter_is_in_vol_points_and_draws_the_fit(self):
        from src.analytics.hedge_sim import fit_pnl_vs_edge
        from src.render.hedge_figures import build_hedge_scatter_figure
        trades = self._settled()
        fit = fit_pnl_vs_edge(trades)
        fig = build_hedge_scatter_figure(trades, fit)
        pts = [t for t in fig.data if t.mode and "markers" in t.mode]
        assert len(pts) == 1
        assert list(pts[0].x) == pytest.approx([1.0, 2.0, 3.0, 4.0])   # vol POINTS
        assert list(pts[0].y) == pytest.approx([1.0, 2.0, 3.0, 4.0])
        assert any(t.mode == "lines" for t in fig.data)     # the regression line
        text = " ".join(a.text for a in fig.layout.annotations)
        assert "slope" in text.lower() and "vol point" in text.lower()
        assert "vol point" in fig.layout.xaxis.title.text.lower()

    def test_scatter_omits_open_trades(self):
        import datetime as dt
        from src.render.hedge_figures import build_hedge_scatter_figure
        trades = self._settled()
        trades.loc[3, "status"] = "open"
        fig = build_hedge_scatter_figure(trades, {"n": 3, "slope": 1.0,
                                                  "intercept": 0.0, "r2": 1.0})
        pts = [t for t in fig.data if t.mode and "markers" in t.mode][0]
        assert len(pts.x) == 3

    def test_scatter_omits_settled_trades_with_no_pnl(self):
        # Symmetric with fit_pnl_vs_edge's own filter: a settled trade missing
        # its P&L should not appear on the scatter even though it has an edge.
        from src.render.hedge_figures import build_hedge_scatter_figure
        trades = self._settled()
        trades.loc[3, "pnl"] = float("nan")
        fig = build_hedge_scatter_figure(trades, {"n": 3, "slope": 1.0,
                                                  "intercept": 0.0, "r2": 1.0})
        pts = [t for t in fig.data if t.mode and "markers" in t.mode][0]
        assert len(pts.x) == 3

    def test_scatter_empty_state_names_the_next_settlement(self):
        import datetime as dt
        from src.render.hedge_figures import build_hedge_scatter_figure
        fig = build_hedge_scatter_figure(
            pd.DataFrame(columns=["edge", "pnl", "status", "entry_date"]),
            {"n": 0, "slope": float("nan"), "intercept": float("nan"), "r2": float("nan")},
            next_settlement=dt.date(2026, 9, 18))
        assert len(fig.data) == 0
        assert "2026-09-18" in fig.layout.annotations[0].text

    def test_scatter_empty_state_does_not_promise_a_first_settlement_once_one_has_settled(self):
        # F1: a `sparse` trade HAS reached expiry -- it is only unplottable.
        # "No simulated trade has reached expiry yet. The first settles ..." was
        # false on the real archive, where a 2024 trade settled two years ago.
        import datetime as dt
        from src.render.hedge_figures import build_hedge_scatter_figure
        trades = self._settled(n=1)
        trades.loc[0, "status"] = "sparse"
        fig = build_hedge_scatter_figure(
            trades, {"n": 0, "slope": float("nan"), "intercept": float("nan"),
                     "r2": float("nan")},
            next_settlement=dt.date(2026, 9, 18))
        text = fig.layout.annotations[0].text
        assert len(fig.data) == 0
        assert "has reached expiry" in text and "reached expiry yet" not in text
        assert "2026-09-18" not in text
        assert "market-quoted marks" in text

    def test_scatter_with_one_point_draws_no_line(self):
        from src.render.hedge_figures import build_hedge_scatter_figure
        trades = self._settled(n=1)
        fig = build_hedge_scatter_figure(trades, {"n": 1, "slope": float("nan"),
                                                  "intercept": float("nan"),
                                                  "r2": float("nan")})
        assert not any(t.mode == "lines" for t in fig.data)


class TestHedgeSummaryHtml:
    def _summary(self, **over):
        import datetime as dt
        base = {"n_trades": 1, "n_settled": 0, "n_open": 1, "n_sparse": 0,
                "n_reached_expiry": 0, "n_months_skipped": 0,
                "first_entry": dt.date(2026, 8, 14),
                "cum_pnl": 1.25, "n_days": 11, "market_mark_share": 1.0,
                "quotable_mark_share": 1.0, "n_model_marks": 0,
                "n_structural_marks": 0, "n_gap_marks": 0, "dte_min": 7,
                "dte_at_entry_min": 35, "dte_at_entry_max": 35,
                "next_settlement": dt.date(2026, 9, 18),
                "slope": float("nan"), "r2": float("nan")}
        return {**base, **over}

    def test_labels_the_panel_as_a_simulation(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary())
        assert "simulation" in html.lower()
        assert "class='stat'" in html

    def test_no_settled_trade_says_so_and_names_the_date(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary())
        assert "2026-09-18" in html
        assert "slope" not in html.lower()

    def test_one_settled_trade_is_not_called_a_relationship(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_settled=1, n_open=0, slope=3.1, r2=1.0))
        assert "one" in html.lower() or "1 " in html
        assert "slope" not in html.lower()

    def test_enough_trades_reports_the_slope(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=6, n_settled=6, n_open=0,
                                                slope=2.4, r2=0.71))
        assert "2.4" in html and "vol point" in html.lower()
        assert "0.71" in html or "71" in html

    def test_a_few_settled_trades_draws_but_does_not_claim_the_fit(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=3, n_settled=3, n_open=0,
                                                slope=1.5, r2=0.5))
        assert "too few" in html.lower()
        assert "1.5" not in html and "0.5" not in html

    def test_degenerate_fit_does_not_print_a_fabricated_slope(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=6, n_settled=6, n_open=0,
                                                slope=float("nan"), r2=float("nan")))
        assert "nan" not in html.lower()

    def test_no_trades_at_all(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=0, n_open=0, n_days=0,
                                                first_entry=None, next_settlement=None,
                                                cum_pnl=float("nan")))
        assert "no simulated trade" in html.lower()
        assert "produced no trade" not in html    # n_months_skipped is 0 -- nothing to disclose

    def test_no_trades_but_every_month_was_skipped_still_discloses_the_count(self):
        # M1: the n_trades == 0 early return used to print ONLY "No simulated
        # trade yet -- the first opens on the first stored session of a month",
        # which contradicts status.json's hedge_months_skipped on an archive
        # where every month was skipped -- it reads as "still pending" when
        # several months already came and went without seeding a trade.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=0, n_open=0, n_days=0,
                                                first_entry=None, next_settlement=None,
                                                cum_pnl=float("nan"), n_months_skipped=3))
        assert "no simulated trade" in html.lower()
        assert "3 months in the archive produced no trade" in html

    def test_model_marked_share_is_disclosed_when_material(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(market_mark_share=0.82,
                                                quotable_mark_share=0.82,
                                                n_model_marks=9, n_gap_marks=9))
        assert "18%" in html or "82%" in html

    def test_model_marked_share_does_not_assert_a_cause_the_data_contradicts(self):
        # F2: the clause blamed "an expiry drops out of the stored chain in its
        # final week" unconditionally. On the real archive 12 of 13 model marks
        # come from a month holding NO chain at all, on any day of the trade.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(market_mark_share=0.48,
                                                quotable_mark_share=0.48,
                                                n_model_marks=13, n_gap_marks=13))
        assert "52%" in html
        assert "final week" not in html.lower()
        assert "modelled by design" not in html

    def test_structural_model_marks_are_named_as_by_design_not_as_missing_data(self):
        # The defect a real run exposed: a 16-day trade's final week can never
        # carry a quote, because chain_filter.dte_min keeps its expiry out of
        # every stored chain. The page must still report how much of the P&L is
        # our own model AT ALL (38%), say plainly why most of it is unavoidable,
        # and give the share the status is actually judged on.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(
            market_mark_share=0.625, quotable_mark_share=1.0, n_model_marks=6,
            n_structural_marks=6, n_gap_marks=0, dte_min=7))
        assert "38% of daily marks are our own model" in html      # modelled at all
        assert "All 6 of them fall inside 7 days of expiry" in html
        assert "modelled by design" in html and "no backfill can change that" in html
        # nothing avoidable: quotable is EXACTLY 1.0, so 100% is not a rounding
        assert "100% of marks come from the market" in html
        assert "measured per trade" in html   # 100% here is a blend, the gate is not

    def test_the_structural_window_named_is_the_configured_dte_min(self):
        # config is the single source of tunables (SPEC 6): 7 must not be baked
        # into the sentence any more than into the engine.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(
            market_mark_share=0.5, quotable_mark_share=0.8, n_model_marks=10,
            n_structural_marks=4, n_gap_marks=6, dte_min=14))
        assert "inside 14 days of expiry" in html
        assert "7 days" not in html
        assert "80% of marks come from the market" in html

    def test_a_sample_with_no_structural_marks_claims_no_structural_split(self):
        # F2 still holds: where every model mark is a real gap -- the 2024 trade,
        # whose whole life has no chain files -- the sentence must not blame a
        # by-design final week for it.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(
            market_mark_share=0.48, quotable_mark_share=0.48, n_model_marks=13,
            n_structural_marks=0, n_gap_marks=13, dte_min=7))
        assert "52% of daily marks are our own model" in html
        assert "modelled by design" not in html
        assert "fall inside 7 days of expiry" not in html
        assert "holds no usable quote for that straddle" in html

    def test_every_kind_of_model_mark_present_is_given_its_own_count(self):
        # R1: the P8a caption promises "where either kind occurs, the stat line
        # above gives its count". Reporting the blend and the structural count
        # only left the reader to subtract for the gap count.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(
            market_mark_share=0.6, quotable_mark_share=0.75, n_model_marks=16,
            n_structural_marks=6, n_gap_marks=10, dte_min=7))
        assert "6 of those 16 modelled sessions" in html    # structural and total
        assert "the other 10 are" in html                   # and the gap count

    def test_a_gap_only_sample_still_gives_the_gap_count(self):
        # R1: this branch used to return a bare percentage -- gap marks present,
        # their count nowhere on the page. It is the shape of any sample whose
        # trades are all mid-life, and the shape this archive had in July.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(
            market_mark_share=0.5, quotable_mark_share=0.5, n_model_marks=12,
            n_structural_marks=0, n_gap_marks=12))
        assert "All 12 of those modelled sessions" in html

    def test_a_lone_model_mark_is_disclosed_though_the_share_rounds_to_all(self):
        # R1/R4: the clause used to be gated on `market_mark_share < 0.999`, so a
        # model mark inside that band vanished from the page entirely -- the
        # disclosure suppressed by a rounding threshold rather than by an actual
        # absence. The gate is now the exact count, and neither percentage rounds
        # INTO a totality: 0.05% modelled is "under 1%", never "0%", and the
        # quotable share beside it is "over 99%", never "100%".
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(
            market_mark_share=0.9995, quotable_mark_share=0.9995,
            n_model_marks=1, n_structural_marks=0, n_gap_marks=1))
        assert "under 1% of daily marks are our own model" in html
        assert "0% of daily marks" not in html
        assert "All 1 of those modelled sessions" in html

    def test_a_near_total_quotable_share_is_not_rounded_up_to_all_of_them(self):
        # R4: at 0.9972 `{:.0%}` printed "100% of marks come from the market"
        # while a gap mark existed. ~200 quotable sessions with one gap in them
        # reaches that, which this archive does inside a year.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(
            market_mark_share=0.95, quotable_mark_share=0.9972, n_model_marks=20,
            n_structural_marks=19, n_gap_marks=1))
        assert "over 99% of marks come from the market" in html
        assert "100%" not in html

    def test_that_share_always_has_a_percentage_to_point_at(self):
        # R3: the fully-quoted branch read "... is marked at a market price; it
        # is that share ..." with no share anywhere in the sentence.
        from src.render.stats import hedge_summary_html
        for quotable in (1.0, 0.75):
            html = hedge_summary_html(self._summary(
                market_mark_share=0.6, quotable_mark_share=quotable,
                n_model_marks=16, n_structural_marks=6, n_gap_marks=10))
            assert "of marks come from the market" in html
            pct = html.index("of marks come from the market")
            assert "it is that share" in html
            assert html.index("it is that share", pct) > pct   # antecedent first

    def test_nothing_plottable_yet_names_the_quotable_rule_not_a_bare_one(self):
        # R5: this branch fires in exactly the state the structural split exists
        # to resolve, and named a different rule than the two sentences beside it.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=2, n_settled=0, n_sparse=1,
                                                n_reached_expiry=1, n_open=1))
        assert "enough market marks on the sessions the archive could have quoted" in html

    def test_a_settled_but_sparse_trade_is_not_reported_as_nothing_finished(self):
        # F1: the stat line contradicted itself -- "none has reached expiry yet"
        # two sentences before "1 trade(s) are excluded from the scatter".
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=2, n_settled=0, n_sparse=1,
                                                n_reached_expiry=1, n_open=1))
        assert "none has reached expiry yet" not in html
        assert "1 trade has reached expiry" in html
        assert "2026-09-18" not in html      # not "the FIRST settles ..."

    def test_nothing_finished_at_all_still_names_the_first_settlement(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_reached_expiry=0))
        assert "none has reached expiry yet" in html
        assert "2026-09-18" in html

    def test_the_tenor_actually_traded_is_printed(self):
        # F5: the caption promised "about 30 days out"; the rule delivers ~17.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_trades=2, dte_at_entry_min=17,
                                                dte_at_entry_max=35))
        assert "17–35 days from expiry" in html

    def test_a_single_tenor_is_printed_without_a_range(self):
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(dte_at_entry_min=17, dte_at_entry_max=17))
        assert "sold 17 days from expiry" in html

    def test_skipped_months_are_disclosed(self):
        # F11: June 2026 is skipped on the real archive (holiday-shifted monthly),
        # and a silently dropped month is a selection the reader cannot see.
        from src.render.stats import hedge_summary_html
        html = hedge_summary_html(self._summary(n_months_skipped=1))
        assert "1 month in the archive produced no trade" in html

    def test_no_skipped_months_says_nothing(self):
        from src.render.stats import hedge_summary_html
        assert "produced no trade" not in hedge_summary_html(self._summary())

    def test_the_disclosure_line_does_not_claim_a_mid_it_never_paid(self):
        # F3: `compute_chain_iv` routes price_used to the mid only for LIVE rows;
        # 27 of 28 stored chains are backfill, so every trade to date was entered
        # and marked at the close. Mirror the "(close-based)" convention.
        from src.render.stats import _sim_label, hedge_summary_html
        label = _sim_label(0.0)
        assert "sold at the mid" not in label
        assert "close-based" in label and "close" in label
        for html in (hedge_summary_html(self._summary()),
                     hedge_summary_html(self._summary(n_trades=0, n_open=0, n_days=0,
                                                      first_entry=None,
                                                      next_settlement=None,
                                                      cum_pnl=float("nan")))):
            assert "sold at the mid" not in html
            assert "close-based" in html

    def test_sim_label_states_no_spread_is_charged_at_zero_cost(self):
        # M3: "No bid-ask spread is ever crossed" was written as a permanent
        # fact but is only true because transaction_cost_bps is 0 in config;
        # config's own comment says v2 adds a toggle. Assert the zero-cost
        # wording is conditioned on the config, not stated as eternal.
        from src.render.stats import _sim_label
        label = _sim_label(0.0)
        assert "no bid–ask spread is ever crossed" not in label.lower()
        assert "transaction_cost_bps is 0" in label
        assert "not a real position" in label.lower()

    def test_sim_label_states_the_configured_cost_when_nonzero(self):
        # M3: a config flip to a non-zero transaction_cost_bps must not leave
        # "no bid-ask spread is ever crossed" standing on the page --
        # `simulate_trade` (src/analytics/hedge_sim.py) already charges this
        # cost on every hedge trade once it is non-zero.
        from src.render.stats import _sim_label
        label = _sim_label(5.0)
        assert "5 bps" in label
        assert "no bid–ask spread is ever crossed" not in label.lower()
        assert "transaction_cost_bps is 0" not in label
        assert "not a real position" in label.lower()

    def test_hedge_summary_html_threads_the_configured_cost_through(self):
        # M3: `hedge_summary_html` receives only the summary dict, so the
        # configured cost has to reach it via `summary["cost_bps"]` rather than
        # an import of config into the render layer.
        from src.render.stats import hedge_summary_html
        html_default = hedge_summary_html(self._summary())          # no cost_bps key
        html_zero = hedge_summary_html(self._summary(cost_bps=0.0))
        html_nonzero = hedge_summary_html(self._summary(cost_bps=7.5))
        assert "transaction_cost_bps is 0" in html_default
        assert "transaction_cost_bps is 0" in html_zero
        assert "7.5 bps" in html_nonzero
        assert "transaction_cost_bps is 0" not in html_nonzero


class TestP8Page:
    def test_q4_lists_the_three_p8_panels(self):
        from src.render.page import QUESTIONS
        q4 = [q for q in QUESTIONS if q[0] == "Q4"][0]
        assert q4[2] == ["P8a", "P8b", "P8c"]

    def test_every_p8_panel_has_a_title_and_a_caption(self):
        from src.render.page import CAPTIONS, _PANEL_TITLES
        for pid in ("P8a", "P8b", "P8c"):
            assert _PANEL_TITLES[pid]
            assert CAPTIONS[pid]

    def test_p8a_caption_states_the_entry_rule_the_code_actually_runs(self):
        # F5: "about 30 days out" is not what the code does. From the first
        # session of a month the monthly ladder's nearest expiry is ~16-18 days.
        from src.render.page import CAPTIONS
        cap = CAPTIONS["P8a"]
        assert "about 30 days out" not in cap
        assert "monthly expiry nearest 30 days out" in cap
        assert "16–18 days" in cap

    def test_p8a_caption_admits_the_cumulative_line_keeps_model_marked_days(self):
        # F6: P8a may keep every day -- a cumulative P&L that skipped days would
        # be wrong -- but it must not claim more than it measures.
        from src.render.page import CAPTIONS
        assert "marked with our own model" in CAPTIONS["P8a"]

    def test_p8b_caption_does_not_blame_the_vertical_spread_on_hedging_alone(self):
        # F5: a 45-day trade earns ~1.6x the dollars of a 17-day one at the same
        # edge, and the entry rule delivers both.
        from src.render.page import CAPTIONS
        cap = CAPTIONS["P8b"]
        assert "√T" in cap and "45-day" in cap and "17-day" in cap

    def test_p8c_caption_says_the_histogram_counts_only_market_marked_sessions(self):
        # F6: the caption calls the width "the direct measurement", so it has to
        # say which days were measured.
        from src.render.page import CAPTIONS
        cap = CAPTIONS["P8c"]
        assert "counting only sessions marked at a market price" in cap
        assert "opening day is dropped" in cap
        assert "direct measurement" in cap      # the original claim is still made

    def test_p8a_and_p8c_captions_admit_the_three_way_split(self):
        # M2: settlement is neither a chain quote nor a model mark -- it prices
        # the straddle at intrinsic from the underlying close, and (unlike a
        # model mark) counts as a market fact in the P8c histogram. The old
        # wording described only a quoted/model dichotomy; on the real archive
        # the 2024-09-20 settlement is exactly the missing third case (no chain
        # file exists for that date at all).
        from src.render.page import CAPTIONS
        assert "settlement is marked at intrinsic value" in CAPTIONS["P8a"]
        assert "settlement at intrinsic value" in CAPTIONS["P8c"]
        # neither caption may still claim a plain quoted/model binary
        assert "the archive actually quotes" not in CAPTIONS["P8c"]
        assert "sessions the archive holds no quote for" not in CAPTIONS["P8a"]

    def test_scatter_caption_states_the_gamma_pnl_result_and_the_hedging_error(self):
        from src.render.page import CAPTIONS
        cap = CAPTIONS["P8b"].lower()
        assert "gamma" in cap or "Γ" in CAPTIONS["P8b"]
        assert "discrete" in cap and "hedg" in cap

    def test_page_renders_the_p8_panels_and_the_simulation_label(self):
        from src.render.page import _PANEL_TITLES, render_page
        from src.render.base import empty_figure
        figures = {pid: empty_figure(pid, "x") for pid in ("P8a", "P8b", "P8c")}
        status = {"spot": 100.0, "snapshot_date": "2026-08-28", "source": "yfinance",
                  "iv_convergence": 0.99, "last_success_utc": "2026-08-29T00:00:00+00:00",
                  "rows_stored": 10}
        html = render_page(figures, status, extras={"P8a": "<p class='stat'>SIMLABEL</p>"})
        assert "SIMLABEL" in html
        assert "Phase 6" not in html          # the Q4 placeholder is gone
        for pid in ("P8a", "P8b", "P8c"):
            assert _PANEL_TITLES[pid] in html
