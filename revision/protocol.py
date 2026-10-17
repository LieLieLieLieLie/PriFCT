"""Audited one-message release, fixed-cost transport, and local stability.

Source arrays never enter target functions. Seeded releases are for reproducible
research only; a deployment must use fresh secret entropy and release once.
"""
import warnings
from dataclasses import dataclass
import numpy as np
import ot
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from core.privacy import compute_sigma


@dataclass(frozen=True)
class Message:
    prototypes: np.ndarray
    counts: np.ndarray
    sigma: np.ndarray
    epsilon: float
    delta: float = 1e-5
    radius: float = 1.0


def clip(x, radius=1.0):
    return x * np.minimum(1., radius / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-15))


def release(x, y, k, epsilon, seed, radius=1., delta=1e-5):
    x = clip(np.asarray(x, float), radius)
    counts = np.bincount(y, minlength=k)
    if np.any(counts == 0):
        raise ValueError("Empty public class partition")
    means = np.stack([x[y == c].mean(0) for c in range(k)])
    sigma = np.asarray(compute_sigma(2 * radius / counts, epsilon, delta))[:, None]
    noise = np.random.default_rng(seed).normal(size=means.shape) * sigma
    # Projection back onto the public ball is DP post-processing.
    return Message(clip(means + noise, radius), counts, sigma, epsilon, delta, radius)


def signatures(p, length=32):
    # Fixed scale 2C, not a data-dependent maximum; all input norms <= C=1.
    d = cdist(p, p) / 2.
    return np.stack([np.quantile(np.delete(row, i), np.linspace(0, 1, length))
                     for i, row in enumerate(d)])


def costs(target, source, rho=0., length=32, declared_coordinates=False):
    if rho and (not declared_coordinates or target.shape[1] != source.shape[1]):
        raise ValueError("Direct cost requires declared comparable coordinates")
    a, b = signatures(target, length), signatures(source, length)
    relational = np.mean((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return ((1-rho) * relational + rho * cdist(target, source)**2 / 4.
            if rho else relational)


def transport(g, mode="partial", mass=.9, tau=.01):
    a, b = np.ones(len(g))/len(g), np.ones(g.shape[1])/g.shape[1]
    with warnings.catch_warnings(record=True) as ws:
        if mode == "partial":
            t = ot.partial.entropic_partial_wasserstein(a, b, g, reg=tau, m=mass,
                    numItermax=3000, stopThr=1e-12)
        elif mode == "full":
            t = ot.sinkhorn(a, b, g, reg=tau, numItermax=1000, stopThr=1e-9)
        elif mode == "uot":
            t = ot.unbalanced.sinkhorn_knopp_unbalanced(a, b, g, reg=tau, reg_m=1.,
                    numItermax=1000, stopThr=1e-9)
        else:
            raise ValueError(mode)
    if not np.isfinite(t).all() or np.min(t) < -1e-10:
        raise RuntimeError("Non-finite or negative transport plan")
    if mode == "partial" and (abs(t.sum()-mass)>1e-5 or np.any(t.sum(1)>a+1e-5)
                               or np.any(t.sum(0)>b+1e-5)):
        # Tight convergence and an explicit feasibility check; no unregularised fallback.
        t = ot.partial.entropic_partial_wasserstein(a, b, g, reg=tau, m=mass,
                    numItermax=30000, stopThr=1e-14)
        if abs(t.sum()-mass)>1e-5 or np.any(t.sum(1)>a+1e-5) or np.any(t.sum(0)>b+1e-5):
            raise RuntimeError(f"Partial transport feasibility violation: mass={t.sum()}, row={t.sum(1)}, col={t.sum(0)}")
    if mode == "full" and (np.max(abs(t.sum(1)-a))>1e-5 or np.max(abs(t.sum(0)-b))>1e-5):
        # Log-domain retry solves the same entropic objective, not a new baseline.
        t = ot.sinkhorn(a,b,g,reg=tau,method='sinkhorn_log',numItermax=30000,stopThr=1e-12)
        if np.max(abs(t.sum(1)-a))>1e-5 or np.max(abs(t.sum(0)-b))>1e-5:
            raise RuntimeError("Full transport marginal violation")
    if np.min(t.sum(1)) <= 1e-14:
        raise RuntimeError("Undefined zero-mass row")
    return t / t.sum(1, keepdims=True), len(ws)


def assignment(s):
    if s.shape[0] != s.shape[1]:
        return s.argmax(1), None
    rows, cols = linear_sum_assignment(-s)
    mapping = cols[np.argsort(rows)]
    best = s[np.arange(len(s)), mapping].sum()
    second = -np.inf
    for row, col in enumerate(mapping):
        competitor = s.copy()
        competitor[row, col] = -1e9
        rr, cc = linear_sum_assignment(-competitor)
        second = max(second, s[rr, cc].sum())
    return mapping, float(max(0., best-second))


def align(target, message, rho=0., mode="partial", mass=.9, tau=.01,
          length=32, declared_coordinates=False):
    g = costs(target, message.prototypes, rho, length, declared_coordinates)
    s, nw = transport(g, mode, mass, tau)
    mapping, gap = assignment(s)
    return s, mapping, gap, nw


def screen(target, message, rho, s, mapping, gap, seed, multipliers=(1.,),
           counts=(100,), mass=.9, tau=.01, declared_coordinates=False):
    if gap is None:
        return [{"status":"not-screened", "accepted":None}]
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(max(counts), *message.prototypes.shape)) * message.sigma
    records = []
    for multiplier in multipliers:
        deviations, same = [], []
        for z in noise:
            p = clip(message.prototypes + multiplier * z)
            perturbed = Message(p, message.counts, message.sigma, message.epsilon)
            sp, mp, _, _ = align(target, perturbed, rho, mass=mass, tau=tau,
                                declared_coordinates=declared_coordinates)
            deviations.append(float(np.max(abs(sp-s))))
            same.append(bool(np.array_equal(mp, mapping)))
        for b in counts:
            eta = float(np.quantile(deviations[:b], .95))
            radius = gap/(2*len(mapping))
            gap_accepted = bool(gap>1e-12 and eta<radius)
            successes = int(np.sum(same[:b]))
            lower = float(beta.ppf(.05,successes,b-successes+1)) if successes else 0.
            # Exact one-sided binomial confidence for this released state's
            # declared local perturbation law. It is NOT confidence in semantics.
            accepted = bool(gap>1e-12 and lower>=.9)
            records.append({"multiplier":multiplier,"B":b,"gap":gap,
                "radius":radius,"eta95":eta,"stability":float(np.mean(same[:b])),
                "gap_accepted":gap_accepted,"successes":successes,"stability_lower95":lower,
                "accepted":accepted,"status":"stable" if accepted else "abstain"})
    return records


def fit_predict(xtr, xte, clusters, mapping, s, message, variant="core"):
    pseudo = mapping[clusters]
    if variant == "core":
        conditions = s @ message.prototypes
        # Test-cluster features enter only through target training centres.
        return conditions, pseudo
    return pseudo


def classify(xtr, xte, ytr, yte):
    if len(np.unique(ytr)) < 2:
        pred = np.full(len(xte), int(ytr[0]))
        prob = np.eye(len(np.unique(yte)))[pred]
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.,max_iter=1000,
                    multi_class="multinomial",solver="lbfgs"))
        model.fit(xtr, ytr)
        prob = model.predict_proba(xte)
        pred = model.predict(xte)
    auc = None
    if prob.shape[1] == len(np.unique(yte)):
        auc = float(roc_auc_score(yte, prob[:,1] if prob.shape[1]==2 else prob,
                      multi_class="ovr",average="macro"))
    return {"accuracy":float(accuracy_score(yte,pred)),
            "macro_f1":float(f1_score(yte,pred,average="macro",zero_division=0)),
            "macro_auc":auc,"n_test":len(yte),"n_correct":int((pred==yte).sum()),
            "predictions":pred.tolist()}
