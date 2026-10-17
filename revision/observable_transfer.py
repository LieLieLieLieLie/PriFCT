"""Public observability calibration and simultaneous propagated noise intervals.

An interval certifies agreement with the clean SAME pipeline, never truth.
The public nonlinear map is frozen before the sole labelled-record query.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax, logsumexp
from revision.prototype_certificate import prototype_radii
from revision.uncertain_moments import target_graph


def probability_intervals(x, joint, temperature=.1, alpha=.05):
    p, radius, details = prototype_radii(joint, alpha)
    distance = cdist(x, p)
    log_lower = -(distance + radius[None, :])**2 / temperature
    log_upper = -np.maximum(distance - radius[None, :], 0)**2 / temperature
    lower, upper = np.empty_like(distance), np.empty_like(distance)
    for c in range(len(p)):
        denominator = log_upper.copy(); denominator[:, c] = log_lower[:, c]
        lower[:, c] = np.exp(log_lower[:, c] - logsumexp(denominator, axis=1))
        denominator = log_lower.copy(); denominator[:, c] = log_upper[:, c]
        upper[:, c] = np.exp(log_upper[:, c] - logsumexp(denominator, axis=1))
    return lower, upper, radius, details


def predict_graph(xtr, xte, joint, alpha=.05, beta=.2, steps=50,
                  temperature=.1, certify=True, graph=None, hard_teacher=False):
    graph = target_graph(xtr) if graph is None else graph
    prototype, _ = joint.decode()
    initial = softmax(-cdist(xtr, prototype.prototypes)**2 / temperature, axis=1)
    if hard_teacher:initial=np.eye(initial.shape[1])[initial.argmax(1)]
    probability = initial.copy()
    for _ in range(steps):
        probability = (1-beta)*initial + beta*graph.transition@probability
    prediction_probability = graph.interpolate(xte, probability)
    prediction = prediction_probability.argmax(1)
    output = {'probabilities': prediction_probability, 'predictions': prediction}
    if certify:
        lo0, hi0, radii, details = probability_intervals(xtr, joint, temperature, alpha)
        if hard_teacher:
            lower_hard=np.zeros_like(lo0);upper_hard=np.ones_like(hi0)
            for c in range(lo0.shape[1]):
                other=np.arange(lo0.shape[1])!=c
                lower_hard[:,c]=lo0[:,c]>hi0[:,other].max(1)
                upper_hard[:,c]=hi0[:,c]>=lo0[:,other].max(1)
            lo0,hi0=lower_hard,upper_hard
        lo, hi = lo0.copy(), hi0.copy()
        for _ in range(steps):
            lo = (1-beta)*lo0 + beta*graph.transition@lo
            hi = (1-beta)*hi0 + beta*graph.transition@hi
        lo, hi = graph.interpolate(xte, lo), graph.interpolate(xte, hi)
        competing = hi.copy(); competing[np.arange(len(prediction)), prediction] = -np.inf
        gap = lo[np.arange(len(prediction)), prediction] - competing.max(1)
        output.update(lower=lo, upper=hi, accepted=gap>1e-12,
                      certificate_gap=gap, prototype_radii=radii, certificate_details=details)
    return output
