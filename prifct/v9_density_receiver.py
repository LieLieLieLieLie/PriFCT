"""Development hypothesis: noisy-prior density ratios and samplewise transport.

Only public geometry, an existing DP class-moment/count packet and unlabelled
adaptation functions are inputs. Class balancing is an explicit objective,
not a claim that a deployment's target label frequencies are known.
"""
import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from scipy.linalg import solve
from prifct.v7_anchored_checked import transport
from prifct.v7_spectral_contrast import reconstruct

def decoded_weight(raw, covariance, sigma, ridge, n_public, normalization=True):
    weight, _ = reconstruct(raw[:, :-1], covariance, sigma, ridge, no_denoising=True)
    q = np.maximum(raw[:, -1] * n_public, 1.)
    q /= q.sum()
    if normalization:
        weight = weight / q[None, :]
    return weight, q

def fit_offset(adaptation_scores, temperature, balance_strength):
    scores = np.asarray(adaptation_scores, float)
    n, k = scores.shape
    a = np.full(n, 1. / n)
    q = np.full(k, 1. / k)
    if balance_strength == 0:
        return np.zeros(k), {'gradient_infinity': 0., 'iterations': 0}
    beta = balance_strength * temperature
    def state(v, hessian=False):
        posterior = softmax((scores-v)/temperature, axis=1)
        prior = q*np.exp(v/beta)
        mass = posterior.mean(0)
        value = beta*np.sum(q*np.expm1(v/beta)) + temperature*np.mean(logsumexp((scores-v)/temperature, axis=1))
        gradient = prior-mass
        if not hessian:
            return value, gradient
        second = np.diag(prior/beta+mass/temperature) - (posterior.T@posterior)/(n*temperature)
        return value, gradient, second
    fit = minimize(state, np.zeros(k), jac=True, method='L-BFGS-B', options={'ftol':1e-14,'gtol':1e-10,'maxiter':500,'maxls':60})
    v = fit.x.copy()
    for _ in range(40):
        value, gradient, second = state(v, True)
        if np.max(abs(gradient)) <= 1e-10:
            break
        direction = solve(second, -gradient, assume_a='pos')
        scale = 1.
        for _ in range(50):
            candidate = v+scale*direction
            if np.max(candidate/beta)>700:
                scale *= .5
                continue
            val, grad = state(candidate)
            if val <= value+1e-14 and np.max(abs(grad)) < np.max(abs(gradient)):
                v = candidate
                break
            scale *= .5
        else:
            break
    _, gradient = state(v)
    error = float(np.max(abs(gradient)))
    if not np.isfinite(v).all() or error > 2e-9:
        raise RuntimeError('Density transport dual did not converge: '+str(error))
    return -v, {'gradient_infinity': error, 'iterations': int(fit.nit)}

def probability(scores, offset, temperature):
    p = softmax((np.asarray(scores, float) + offset) / temperature, axis=1)
    p = np.maximum(p, 1e-12)
    p /= p.sum(1, keepdims=True)
    p = p.astype(np.float32).astype(np.float64)
    return p / p.sum(1, keepdims=True)
