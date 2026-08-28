"""Hand-built Black-Scholes-Merton model layer (SPEC §2.3).

No pricing libraries: NumPy + scipy.stats.norm + scipy.optimize.brentq only.
All functions are pure and vectorized via NumPy broadcasting.

Contracts:
- NaN-never-raise: invalid *numeric* inputs (negative T or sigma, nonpositive
  S or K, non-finite values) yield NaN elementwise; callers count NaNs into
  status.json. Programming errors (bad `kind`) raise ValueError.
- Edge behavior: as T→0 or sigma→0 the price converges to discounted
  intrinsic on the forward, max(S e^{-qT} - K e^{-rT}, 0) for calls; this
  emerges from the clamped d1/d2 limit rather than special-case branches.
- Greeks (Task 4) are raw calculus derivatives: theta per year, vega per
  unit vol, rho per unit rate. Display scaling is the render layer's job.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_MIN_VOL = 1e-4
_MAX_VOL = 5.0
_EPS = 1e-12


def _validate_kind(kind: str) -> None:
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")


def _as_arrays(*vals):
    return [np.asarray(v, dtype=np.float64) for v in vals]


def _invalid_mask(S, K, T, r, sigma, q):
    return (
        (S <= 0) | (K <= 0) | (T < 0) | (sigma < 0)
        | ~np.isfinite(S + K + T + r + sigma + q)
    )


def _d1_d2(S, K, T, sigma, r, q):
    sqrt_t = np.sqrt(T)
    denom = np.maximum(sigma * sqrt_t, _EPS)  # sigma*sqrt(T) underflow guard
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / denom
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def bs_price(S, K, T, r, sigma, q=0.0, kind="call"):
    """European option price with continuous dividend yield q.

    Scalars in -> float out; arrays in -> broadcast ndarray out.
    """
    _validate_kind(kind)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    with np.errstate(all="ignore"):
        invalid = _invalid_mask(S, K, T, r, sigma, q)
        d1, d2 = _d1_d2(S, K, T, sigma, r, q)
        disc_s = S * np.exp(-q * T)
        disc_k = K * np.exp(-r * T)
        if kind == "call":
            price = disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2)
        else:
            price = disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1)
        price = np.where(invalid, np.nan, price)
    return price if price.ndim else float(price)


def greeks(S, K, T, r, sigma, q=0.0, kind="call"):
    """Closed-form BSM Greeks as raw calculus derivatives.

    theta is per YEAR, vega per UNIT vol, rho per UNIT rate — display
    scaling (per-day, per-vol-point, per-1%) belongs to the render layer.
    """
    _validate_kind(kind)
    S, K, T, r, sigma, q = _as_arrays(S, K, T, r, sigma, q)
    with np.errstate(all="ignore"):
        invalid = _invalid_mask(S, K, T, r, sigma, q)
        d1, d2 = _d1_d2(S, K, T, sigma, r, q)
        sqrt_t = np.sqrt(T)
        pdf1 = norm.pdf(d1)
        disc_q = np.exp(-q * T)
        disc_r = np.exp(-r * T)
        gamma = disc_q * pdf1 / np.maximum(S * sigma * sqrt_t, _EPS)
        vega = S * disc_q * pdf1 * sqrt_t
        common_theta = -S * disc_q * pdf1 * sigma / np.maximum(2 * sqrt_t, _EPS)
        if kind == "call":
            delta = disc_q * norm.cdf(d1)
            theta = common_theta - r * K * disc_r * norm.cdf(d2) + q * S * disc_q * norm.cdf(d1)
            rho = K * T * disc_r * norm.cdf(d2)
        else:
            delta = disc_q * (norm.cdf(d1) - 1.0)
            theta = common_theta + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)
            rho = -K * T * disc_r * norm.cdf(-d2)
        out = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
        out = {k: np.where(invalid, np.nan, v) for k, v in out.items()}
    return {k: (v if v.ndim else float(v)) for k, v in out.items()}
