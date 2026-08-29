"""Stat sentences that sit above a panel (the P5 one lives in analytics.iv_rv)."""
import math


def parity_summary_html(summary: dict, carry_value: float, carry_dte: float,
                        r: float, q: float) -> str:
    if summary["n_pairs"] == 0:
        return "<p class='stat'>No strike/expiry pairs with both legs priced this session.</p>"
    if summary["n_quoted"] == 0:
        return (f"<p class='stat'>{summary['n_pairs']} strike/expiry pairs — close-based session, "
                "spreads unknown, so tradeability cannot be assessed; the heatmap shows raw "
                "deviations only.</p>")
    n_fwd = summary["n_tradeable_violations_fwd"]
    n_unexplained = summary["n_violations_unexplained"]
    n_ee = summary["n_violations_early_exercise"]
    # Lead with the number that is actually a puzzle: what neither the forward
    # calibration nor the American early-exercise bound can account for.
    text = (f"<p class='stat'>{summary['n_liquid']} liquid strike/expiry pairs · "
            f"{n_unexplained} unexplained tradeable "
            f"violation{'' if n_unexplained == 1 else 's'}")
    # Deliberately not "in-the-money put gaps": on 2026-08-28, 38 of the 40 are,
    # but two are out-of-the-money puts, and the page must not claim otherwise.
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
    text += ("."
             f" Measured instead against our spot-derived forward, "
             f"{summary['n_tradeable_violations']} of {summary['n_quoted']} quoted pairs "
             f"exceed the spread — largely a stale-spot artefact: near the money, where "
             f"SPY's combined bid–ask is about a dollar, the options close (16:15 ET) is "
             f"fifteen minutes after the stock's.")
    if carry_value is not None and not math.isnan(carry_value):
        text += (f" ATM implied carry at the {carry_dte:.0f}-day expiry: "
                 f"{carry_value:.2%} vs our r − q = {r - q:.2%}.")
    return text + "</p>"
