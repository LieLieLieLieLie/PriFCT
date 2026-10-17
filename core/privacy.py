"""
core/privacy.py — (ε, δ)-Differential Privacy via Gaussian mechanism.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from typing import Tuple


def compute_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """Analytically calibrate the Gaussian mechanism (Balle & Wang, 2018).

    For mu = sensitivity / sigma, the exact Gaussian privacy profile is

      delta(eps, mu) = Phi(-eps/mu + mu/2)
                       - exp(eps) Phi(-eps/mu - mu/2).

    Solving this scalar monotone equation avoids applying the older
    sqrt(2 log(1.25/delta))/epsilon sufficient bound outside its usual
    epsilon < 1 statement.  ``sensitivity`` may be an ndarray.
    """
    if epsilon == float('inf'):
        return np.zeros_like(sensitivity, dtype=np.float64)
    if not np.isfinite(epsilon) or epsilon <= 0 or not (0 < delta < 1):
        raise ValueError("epsilon must be positive and delta must lie in (0, 1)")

    def privacy_profile(mu):
        return (norm.cdf(-epsilon / mu + mu / 2.0)
                - np.exp(epsilon) * norm.cdf(-epsilon / mu - mu / 2.0)
                - delta)

    lo = 1e-12
    hi = max(1.0, np.sqrt(2.0 * epsilon) + 2.0)
    while privacy_profile(hi) < 0:
        hi *= 2.0
    mu_star = brentq(privacy_profile, lo, hi, xtol=1e-13, rtol=1e-12)
    return np.asarray(sensitivity, dtype=np.float64) / mu_star


def clip_prototypes(prototypes: np.ndarray, max_norm: float) -> np.ndarray:
    norms = np.linalg.norm(prototypes, axis=1, keepdims=True)
    return prototypes * np.minimum(1.0, max_norm / (norms + 1e-8))


def add_dp_noise(
    prototypes: np.ndarray,
    epsilon: float,
    delta: float,
    max_norm: float,
    rng: np.random.Generator = None,
    counts: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    if rng is None:
        rng = np.random.default_rng()
    # `prototypes` must be means of individually L2-clipped records. Clipping
    # the released mean here is harmless post-processing but is not a
    # substitute for record-level clipping before aggregation.
    clipped = clip_prototypes(prototypes, max_norm)
    if counts is None:
        sensitivity = np.full((len(clipped), 1), 2.0 * max_norm, dtype=np.float32)
    else:
        counts = np.asarray(counts, dtype=np.float32).reshape(-1, 1)
        sensitivity = 2.0 * max_norm / np.maximum(counts, 1.0)
    sigma = compute_sigma(sensitivity, epsilon, delta)
    # For the vector-valued Gaussian mechanism, sigma is the per-coordinate
    # standard deviation. Dividing by sqrt(d) would invalidate the stated DP
    # guarantee. High-dimensional utility is handled by the public projection
    # performed before prototype aggregation (see core/prototype.py).
    noisy = clipped + rng.normal(0, sigma, clipped.shape).astype(np.float32)
    return noisy, sigma


def privacy_report(epsilon, delta, max_norm, sigma) -> dict:
    return {
        "epsilon":     epsilon,
        "delta":       delta,
        "sensitivity": 2.0 * max_norm,
        "sigma":       float(np.max(sigma)) if np.ndim(sigma) else sigma,
        "guarantee":   f"({epsilon:.2f}, {delta:.2e})-DP",
    }
