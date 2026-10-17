"""One-message DP linear-softmax control on frozen, unit-bounded features.

This is a matched privacy/resource control, not a reproduction of a recent
paper. It reads private source records on every full-batch optimisation step;
the source compute differs from the proposed one-pass prototype aggregation.
Each complete feature-label record contributes ONE jointly clipped gradient
for all K(d+1) head parameters (including bias). Under replace-one adjacency,
the average gradient sensitivity is 2*clip_norm/N. T adaptive Gaussian steps
compose by zCDP; no subsampling amplification is assumed.

Only the final head and public configuration may be released. Training loss,
clipping fractions and other private training statistics are deliberately not
returned. Public benchmark seeds are reproducibility aids and must never be
published for a genuinely protected deployment. The default uses hidden
SystemRandom Gaussian draws, never a global NumPy/Torch seed.
"""
from dataclasses import dataclass
import itertools
import math
import random
from typing import Optional

import numpy as np


PUBLIC_GRID = [
    {"steps": steps, "learning_rate": lr, "clip_norm": 1.0, "l2": l2}
    for steps, lr, l2 in itertools.product([25, 100], [0.5, 1.0], [0.001, 0.01])
]


def gaussian_composition(epsilon, delta, steps, n_public, clip_norm):
    """Conservative full-record replace-one accounting for all source passes.

    rho_total = T/(2*m^2); epsilon = rho+2*sqrt(rho*log(1/delta)).
    The rationalised inverse avoids cancellation at small epsilon.
    sigma is the std of each coordinate of the AVERAGED clipped gradient.
    """
    epsilon = float(epsilon)
    if epsilon <= 0 or math.isnan(epsilon):
        raise ValueError("epsilon must be positive, or infinity for a labelled nonprivate control")
    if not 0 < float(delta) < 1:
        raise ValueError("delta must be in (0, 1)")
    if isinstance(steps, bool) or int(steps) != steps or steps <= 0:
        raise ValueError("steps must be a positive public integer")
    if isinstance(n_public, bool) or int(n_public) != n_public or n_public <= 0:
        raise ValueError("N must be a fixed positive public integer")
    if not math.isfinite(float(clip_norm)) or clip_norm <= 0:
        raise ValueError("clip_norm must be a positive public constant")
    sensitivity = 2.0 * float(clip_norm) / int(n_public)
    if math.isinf(epsilon):
        return {"epsilon": epsilon, "delta": float(delta), "steps": int(steps),
                "n_public": int(n_public), "clip_norm": float(clip_norm),
                "sensitivity": sensitivity, "rho_total": math.inf,
                "noise_multiplier": 0.0, "sigma_gradient": 0.0,
                "guarantee": "nonprivate", "accountant": "no privacy claimed"}
    log_delta = math.log(1.0 / float(delta))
    sqrt_rho = epsilon / (math.sqrt(log_delta + epsilon) + math.sqrt(log_delta))
    rho = sqrt_rho * sqrt_rho
    multiplier = math.sqrt(int(steps) / (2.0 * rho))
    return {"epsilon": epsilon, "delta": float(delta), "steps": int(steps),
            "n_public": int(n_public), "clip_norm": float(clip_norm),
            "sensitivity": sensitivity, "rho_total": rho,
            "noise_multiplier": multiplier, "sigma_gradient": multiplier * sensitivity,
            "guarantee": "full feature-label record replace-one (epsilon,delta)-DP",
            "accountant": "adaptive Gaussian zCDP composition; no subsampling"}


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def clipped_record_gradients(x, y, head, clip_norm):
    """Small CPU audit helper; NEVER release its output on protected data.

    Returns actual per-record joint gradients [N,K,d+1], useful for testing
    the record-level norm bound and replacement sensitivity. Training avoids
    allocating this tensor by using an algebraically identical factorisation.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    head = np.asarray(head, dtype=np.float64)
    z = np.column_stack([x, np.ones(len(x))])
    residual = _softmax(z @ head.T)
    residual[np.arange(len(x)), y] -= 1.0
    norms = np.linalg.norm(residual, axis=1) * np.linalg.norm(z, axis=1)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-30))
    return (residual * scale[:, None])[:, :, None] * z[:, None, :]


@dataclass(frozen=True)
class PrivateLinearHead:
    """A single final K(d+1) float64 message, plus public provenance."""
    parameters: np.ndarray
    accounting: dict
    public_config: dict

    @property
    def query_bytes(self):
        return int(self.parameters.nbytes)

    def predict_proba(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] + 1 != self.parameters.shape[1]:
            raise ValueError("prediction feature dimension must match the released head")
        if not np.isfinite(x).all():
            raise ValueError("prediction features must be finite")
        return _softmax(x @ self.parameters[:, :-1].T + self.parameters[:, -1])


def fit_private_head(x_source, y_source, k, epsilon, delta=1e-5, steps=50,
                     learning_rate=1.0, clip_norm=1.0, l2=0.01,
                     seed: Optional[int] = None, device="auto", n_public=None):
    """Fit a source-only head; target observations or labels are not accepted.

    Public initial head is exactly zero, with no nonprivate pretrained-head
    shortcut. L2 regularisation is public and applied after clipping/noise;
    it is data-independent and does not alter the stated sensitivity. The
    default learning rate and stopping count are public, not privately tuned.
    Invalid inputs raise errors, never silently fall back to nonprivate fit.
    """
    x = np.asarray(x_source, dtype=np.float64)
    y = np.asarray(y_source)
    if x.ndim != 2 or len(x) == 0 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("source features must be a nonempty finite matrix")
    if np.max(np.linalg.norm(x, axis=1)) > 1.0 + 1e-9:
        raise ValueError("use the frozen unit-bounded source map before fitting")
    if isinstance(k, bool) or int(k) != k or k < 2:
        raise ValueError("k must be the fixed public class universe, at least two")
    k = int(k)
    if y.shape != (len(x),) or not np.isfinite(y).all() or np.any(y != y.astype(int)) or np.any((y < 0) | (y >= k)):
        raise ValueError("source labels must belong to the public class universe")
    if n_public is None:
        n_public = len(x)
    if int(n_public) != len(x):
        raise ValueError("replace-one accounting requires fixed public N equal to cohort size")
    if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and public")
    if not math.isfinite(float(l2)) or l2 < 0:
        raise ValueError("l2 must be nonnegative and public")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    accounting = gaussian_composition(epsilon, delta, steps, n_public, clip_norm)
    shape = (k, x.shape[1] + 1)
    sigma = accounting["sigma_gradient"]
    rng = np.random.default_rng(seed) if seed is not None else None
    secure = random.SystemRandom() if seed is None else None

    def noise():
        if sigma == 0.0:
            return np.zeros(shape, dtype=np.float64)
        if rng is not None:
            return rng.normal(0.0, sigma, size=shape)
        return np.array([secure.gauss(0.0, sigma) for _ in range(math.prod(shape))]).reshape(shape)

    use_cuda = False
    if device in ("auto", "cuda"):
        import torch
        use_cuda = torch.cuda.is_available()
        if device == "cuda" and not use_cuda:
            raise RuntimeError("CUDA explicitly requested but unavailable")
    z = np.column_stack([x, np.ones(len(x))])
    if use_cuda:
        import torch
        zt = torch.as_tensor(z, dtype=torch.float64, device="cuda")
        yt = torch.as_tensor(y.astype(int), dtype=torch.long, device="cuda")
        head = torch.zeros(shape, dtype=torch.float64, device="cuda")
        index = torch.arange(len(x), device="cuda")
        znorm = torch.linalg.vector_norm(zt, dim=1)
        for _ in range(int(steps)):
            residual = torch.softmax(zt @ head.T, dim=1)
            residual[index, yt] -= 1.0
            norms = torch.linalg.vector_norm(residual, dim=1) * znorm
            scale = torch.clamp(clip_norm / torch.clamp(norms, min=1e-30), max=1.0)
            gradient = (residual * scale[:, None]).T @ zt / n_public
            gradient[:, :-1] += l2 * head[:, :-1]
            head -= learning_rate * (gradient + torch.as_tensor(noise(), dtype=torch.float64, device="cuda"))
        parameters = head.cpu().numpy()
    else:
        head = np.zeros(shape, dtype=np.float64)
        znorm = np.linalg.norm(z, axis=1)
        index = np.arange(len(x))
        for _ in range(int(steps)):
            residual = _softmax(z @ head.T)
            residual[index, y.astype(int)] -= 1.0
            norms = np.linalg.norm(residual, axis=1) * znorm
            scale = np.minimum(1.0, clip_norm / np.maximum(norms, 1e-30))
            gradient = (residual * scale[:, None]).T @ z / n_public
            gradient[:, :-1] += l2 * head[:, :-1]
            head -= learning_rate * (gradient + noise())
        parameters = head
    if not np.isfinite(parameters).all():
        raise FloatingPointError("private optimisation diverged; do not use a nonprivate fallback")
    public_config = {"steps": int(steps), "learning_rate": float(learning_rate),
                     "clip_norm": float(clip_norm), "l2": float(l2),
                     "k_public": k, "d_public": x.shape[1], "n_public": int(n_public),
                     "source_passes": int(steps), "initialisation": "public zero head",
                     "device": "cuda" if use_cuda else "cpu",
                     "message": "one final K(d+1) float64 head; metadata excluded",
                     "randomness": "hidden SystemRandom" if seed is None else "seeded public-benchmark simulation; not a deployable private release"}
    return PrivateLinearHead(parameters.copy(), accounting, public_config)
