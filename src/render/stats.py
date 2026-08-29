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
    text = (f"<p class='stat'>{summary['n_pairs']} strike/expiry pairs · "
            f"{summary['n_tradeable_violations']} tradeable violation"
            f"{'' if summary['n_tradeable_violations'] == 1 else 's'} "
            f"(|deviation| beyond the combined bid–ask spread) · "
            f"{summary['share_within_spread']:.1%} within spread.")
    if carry_value is not None and not math.isnan(carry_value):
        text += (f" ATM implied carry at the {carry_dte:.0f}-day expiry: "
                 f"{carry_value:.2%} vs our r − q = {r - q:.2%}.")
    return text + "</p>"
