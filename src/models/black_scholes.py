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
from scipy.optimize import brentq
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


def implied_vol(price, S, K, T, r, q=0.0, kind="call", tol=1e-10, max_iter=20):
    """Invert bs_price for sigma. SPEC §2.3 algorithm:

    1. No-arbitrage pre-check; violations -> NaN, no iteration.
    2. Brenner-Subrahmanyam initial guess, clamped to [_MIN_VOL, _MAX_VOL].
    3. Vectorized Newton-Raphson on our own vega (tol in price, max_iter).
    4. Scalar brentq fallback for Newton's failures (tiny vega, out of
       bounds, no convergence). No sign change on the bracket -> NaN.

    tol is in price space: resulting sigma accuracy is ~ tol/vega, so a
    tight price tolerance is what buys 1e-6 sigma recovery off-ATM.
    """
    _validate_kind(kind)
    arrs = np.broadcast_arrays(*_as_arrays(price, S, K, T, r, q))
    price, S, K, T, r, q = (a.copy() for a in arrs)
    out = np.full(price.shape, np.nan)

    with np.errstate(all="ignore"):
        disc_s = S * np.exp(-q * T)
        disc_k = K * np.exp(-r * T)
        if kind == "call":
            lower, upper = np.maximum(disc_s - disc_k, 0.0), disc_s
        else:
            lower, upper = np.maximum(disc_k - disc_s, 0.0), disc_k
        solvable = (
            np.isfinite(price) & (price > 0) & (price >= lower) & (price <= upper)
            & (S > 0) & (K > 0) & (T > 0) & np.isfinite(S + K + T + r + q)
        )

        # Brenner-Subrahmanyam: near-exact ATM, decent start elsewhere.
        sigma = np.clip(
            np.sqrt(2.0 * np.pi / np.maximum(T, _EPS)) * price / S, _MIN_VOL, _MAX_VOL
        )
        active = solvable.copy()

        # sigma-space error of a price-space-tol stop is ~ tol/vega (see
        # docstring); a raw "vega < 1e-10" gate lets vega ~ 1e-9 points
        # "converge" with 0.1+ sigma error, well outside the SPEC's 1e-6
        # corner-contract promise. Require vega large enough that tol/vega
        # is itself within the 1e-6 target before trusting a diff-only stop.
        vega_floor = tol / 1e-6
        for _ in range(max_iter):
            if not active.any():
                break
            diff = bs_price(S, K, T, r, sigma, q, kind) - price
            d1, _ = _d1_d2(S, K, T, sigma, r, q)
            vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

            # Tiny vega or non-finite diff: the diff can be spuriously tiny
            # (price and its neighbors are all ~0 deep OTM) while sigma is
            # nowhere near the root. Release to Brent *before* checking
            # convergence, or a near-zero vega point can be falsely accepted
            # as "converged" simply because it has a near-zero price.
            active &= ~(active & ((vega < vega_floor) | ~np.isfinite(diff)))

            converged = active & (np.abs(diff) < tol)
            out = np.where(converged, sigma, out)
            active &= ~converged

            step = np.where(active, diff / np.maximum(vega, 1e-300), 0.0)
            sigma = sigma - step
            # Stepped out of bounds: hand off to Brent, don't clamp-and-lie.
            active &= ~((sigma < _MIN_VOL) | (sigma > _MAX_VOL) | ~np.isfinite(sigma))

        need_brent = solvable & np.isnan(out)

    it = np.nditer(need_brent, flags=["multi_index"])
    for flag in it:
        if not flag:
            continue
        idx = it.multi_index

        def f(s, idx=idx):
            return bs_price(S[idx], K[idx], T[idx], r[idx], s, q[idx], kind) - price[idx]

        try:
            root = brentq(f, _MIN_VOL, _MAX_VOL, xtol=1e-10)
        except ValueError:
            continue  # no sign change on [MIN_VOL, MAX_VOL]: price unreachable -> NaN

        # Deep ITM/short-T corners can be flat to float64 precision across a
        # wide sigma range (bs_price(sigma) rounds to the same float as the
        # target price for many sigma, sometimes exactly at _MIN_VOL) so
        # brentq can return a numerically "valid" root that is nowhere near
        # the true sigma. Vega at the root measures whether that root is
        # actually distinguishable from its neighbors; if not, the price
        # carries no recoverable vol information there -> NaN, not a guess.
        d1_root, _ = _d1_d2(S[idx], K[idx], T[idx], root, r[idx], q[idx])
        vega_root = S[idx] * np.exp(-q[idx] * T[idx]) * norm.pdf(d1_root) * np.sqrt(T[idx])
        if vega_root >= vega_floor:
            out[idx] = root

    return out if out.ndim else float(out)
