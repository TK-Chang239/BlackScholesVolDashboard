"""Stat sentences that sit above a panel (the P5 one lives in analytics.iv_rv)."""
import math


def _carry_clause(carry: dict | None, r: float, q: float) -> str:
    """The implied-carry sentence: the median, its range, and the tenor span.

    `carry` is `analytics.parity.carry_reference`'s dict. It reduces several
    expiries to one number, so the sentence has to show the set it reduced --
    printing the median alone would flatter it by hiding a disagreement the
    reader is entitled to see. When only one expiry qualified there is no set
    to show, so the sentence names that expiry instead.
    """
    if not carry or carry.get("implied_carry") is None:
        return ""
    value = carry["implied_carry"]
    if value is None or math.isnan(value):
        return ""
    lo, hi = carry["carry_lo"], carry["carry_hi"]
    dte_min, dte_max = carry["dte_min"], carry["dte_max"]
    if carry["n_expiries"] <= 1 or dte_min == dte_max:
        return (f" ATM implied carry at the {dte_max:.0f}-day expiry: "
                f"{value:.2%} vs our r − q = {r - q:.2%}.")
    return (f" ATM implied carry across the {dte_min:.0f}–{dte_max:.0f}-day expiries: "
            f"median {value:.2%} (range {lo * 100:.2f}–{hi:.2%}) "
            f"vs our r − q = {r - q:.2%}.")


def _raw_comparison(summary: dict) -> str:
    return (f" Measured instead against our spot-derived forward, "
            f"{summary['n_tradeable_violations']} of {summary['n_quoted']} quoted "
            f"pairs exceed the spread")


def parity_summary_html(summary: dict, carry: dict | None,
                        r: float, q: float) -> str:
    if summary["n_pairs"] == 0:
        return "<p class='stat'>No strike/expiry pairs with both legs priced this session.</p>"
    if summary["n_quoted"] == 0:
        # The figure calibrates the forward on close-based frames too (they carry
        # no open interest, so every pair counts as liquid and every expiry gets
        # a forward). Say what is actually plotted -- "raw deviations" belongs
        # only to the case where no expiry bracketed spot and there is no forward.
        max_fwd = summary.get("max_abs_deviation_fwd", float("nan"))
        plotted = ("the heatmap shows raw deviations only"
                   if max_fwd is None or math.isnan(max_fwd)
                   else "the heatmap shows the gap against each expiry's own "
                        "market-implied forward")
        return (f"<p class='stat'>{summary['n_pairs']} strike/expiry pairs — close-based session, "
                f"spreads unknown, so tradeability cannot be assessed; {plotted}.</p>")
    if summary["n_liquid"] == 0:
        # Every count below the lead is computed over the LIQUID pairs, so with
        # none of them they are all zero by construction -- a silent all-clear
        # rather than a result. Say that the measurement is missing instead of
        # printing its absence as a reassuring zero. (`compute_parity` already
        # treats a wholly absent or wholly zero open-interest column as unknown
        # and calls every pair liquid, so reaching here means open interest was
        # present on the frame but never on both legs of the same pair.)
        text = (f"<p class='stat'>{summary['n_pairs']} strike/expiry pairs — no pair carries "
                f"open interest on both legs, so liquidity could not be assessed and the "
                f"forward-calibrated violation counts are withheld rather than published "
                f"as zeros.")
        text += _raw_comparison(summary) + "."
        return text + _carry_clause(carry, r, q) + "</p>"
    n_fwd = summary["n_tradeable_violations_fwd"]
    n_unexplained = summary["n_violations_unexplained"]
    n_ee = summary["n_violations_early_exercise"]
    # Lead with the number that is actually a puzzle: what neither the forward
    # calibration nor the American early-exercise bound can account for -- and
    # give it as a share too, so the reader is not left to divide 6 by 585.
    share_unexplained = n_unexplained / summary["n_liquid"]
    text = (f"<p class='stat'>{summary['n_liquid']} liquid strike/expiry pairs · "
            f"{n_unexplained} unexplained tradeable "
            f"violation{'' if n_unexplained == 1 else 's'} "
            f"({share_unexplained:.1%} of liquid pairs)")
    # Deliberately not "in-the-money put gaps", even though the ITM gate in
    # `early_exercise_explained` now guarantees every one of them is: the count
    # is a classification result, and naming the gate's own precondition as if
    # it were an observation would make the sentence circular. The gate exists
    # because the bound grows with K·rT on strikes where the American premium is
    # zero, so without it the classifier explains away out-of-the-money wing
    # puts that early exercise cannot have produced.
    if n_ee:
        text += (f" · {n_ee} more put-rich gap{'' if n_ee == 1 else 's'} "
                 f"{'sits' if n_ee == 1 else 'sit'} inside the American early-exercise "
                 f"bound — the extra value an early-exercisable put has over the "
                 f"European one parity is written for, which is not arbitrage")
    text += (f". Once each expiry's forward is calibrated to the market's own ATM "
             f"C − P there {'is' if n_fwd == 1 else 'are'} "
             f"{n_fwd} tradeable violation{'' if n_fwd == 1 else 's'} in all")
    share_fwd = summary.get("share_within_spread_fwd", float("nan"))
    if share_fwd is not None and not math.isnan(share_fwd):
        text += f" · {share_fwd:.1%} within spread"
    n_raw = summary["n_tradeable_violations"]
    text += "." + _raw_comparison(summary)
    # The stale-spot story is a causal claim, so it is only earned on a session
    # where the calibration actually removed most of them. On a day with a
    # handful of raw violations, or where calibrating changed little, print the
    # comparison and let the reader draw their own conclusion.
    if n_raw >= 10 and n_fwd <= n_raw / 2:
        text += (" — largely a stale-spot artefact: near the money, where "
                 "SPY's combined bid–ask is about a dollar, the options close (16:15 ET) is "
                 "fifteen minutes after the stock's")
    text += "."
    return text + _carry_clause(carry, r, q) + "</p>"
