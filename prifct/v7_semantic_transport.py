"""Development candidate: distribution messages and posterior-compatible graph.

Public maps and frequencies are fixed before the source query. The complete
feature has norm <=1; joint feature/count sensitivity is exactly bounded by
2/N under replacement of a full labelled record. No semantic recovery or
universal superiority is claimed. Kernel means and label propagation are
known tools; the complete coupled design still requires contribution evidence.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.special import softmax
from prifct.method import fixed_transport
from prifct.v6_public_relation import squared_distance


@dataclass(frozen=True)
class MultiScaleSignature:
    frequencies: tuple
    dimension: int

    @classmethod
    def public_fit(cls, public, seed=76011, width=64):
        public = np.asarray(public, float)
        if public.ndim != 2 or len(public) < 3 or width < 2 or width % 2:
            raise ValueError('Finite independent public coordinates and even width required')
        if not np.isfinite(public).all() or np.max(np.linalg.norm(public, axis=1)) > 1 + 1e-9:
            raise ValueError('The public common map must be bounded')
        distance = squared_distance(public, public)
        values = distance[np.triu_indices(len(public), 1)]
        bandwidth = max(float(np.median(values)), 1e-4)
        rng = np.random.default_rng(seed)
        frequencies = tuple(rng.normal(size=(public.shape[1], width // 2)) * np.sqrt(2 / (bandwidth * factor)) for factor in [.25, 1.])
        return cls(frequencies, public.shape[1])

    def transform(self, x):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError('Finite same-map coordinates required')
        if np.max(np.linalg.norm(x, axis=1), initial=0) > 1 + 1e-9:
            raise ValueError('Source input must already obey public-map bound')
        pieces = [x]
        for frequency in self.frequencies:
            phase = x @ frequency
            pieces.append(np.c_[np.cos(phase), np.sin(phase)] / np.sqrt(frequency.shape[1]))
        result = np.concatenate(pieces, axis=1) / np.sqrt(len(pieces))
        # Each Fourier block has exact norm one, and the linear block norm <=1.
        assert np.max(np.linalg.norm(result, axis=1), initial=0) <= 1 + 1e-9
        return result


def posterior(features, message, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive temperature required')
    decoded, _ = message.decode()
    # Average kernel similarity; no clean source counts or source records.
    return softmax(np.asarray(features) @ decoded.prototypes.T / temperature, axis=1)


def compatible_transition(transition, initial, bandwidth=.1):
    """Symmetric, nonnegative affinity based only on existing DP posteriors.

    Hellinger compatibility reduces cross-posterior diffusion. It does not
    establish that a confident initial class is semantically correct.
    """
    if bandwidth <= 0:
        raise ValueError('Positive compatibility bandwidth required')
    graph = transition.tocoo()
    root = np.sqrt(np.maximum(initial, 0))
    divergence = np.sum((root[graph.row] - root[graph.col]) ** 2, axis=1) / 2
    values = graph.data * np.exp(-divergence / bandwidth)
    symmetric = sparse.csr_matrix((values, (graph.row, graph.col)), shape=graph.shape)
    symmetric = (symmetric + symmetric.T) / 2
    symmetric.setdiag(0)
    symmetric.eliminate_zeros()
    degree = np.asarray(symmetric.sum(axis=1)).ravel()
    # Add a self loop only for disconnected vertices; never divide by zero.
    isolated = degree <= 1e-15
    symmetric += sparse.diags(isolated.astype(float))
    degree = np.asarray(symmetric.sum(axis=1)).ravel()
    return sparse.diags(1 / degree) @ symmetric


def anchored_diffusion(transition, initial, mixing, tolerance=1e-9, max_steps=300):
    """Unique solution Q=(1-a)P+a WQ; W stochastic, 0<=a<1.

    Contraction implies a posterior perturbation bound ||dQ||_infty <=
    ||dP||_infty for FIXED W. W depends on P in the full candidate, so this
    fixed-graph result must not be advertised as end-to-end robustness.
    """
    if not 0 <= mixing < 1:
        raise ValueError('Contraction parameter must be in [0,1)')
    result = initial.copy()
    residual = 0.
    for step in range(max_steps):
        update = (1 - mixing) * initial + mixing * (transition @ result)
        residual = float(np.max(np.abs(update - result)))
        result = update
        if residual <= tolerance:
            break
    error_bound = residual / (1 - mixing)
    return result, {'iterations': step + 1, 'fixed_graph_infinity_error_bound': error_bound}


def predict(problem, signature, message, *, temperature=.02,
            transport_mix=.25, graph_mix=.5, compatible=True, hard=False):
    """Receiver-only predictor: no source rows or target labels accepted."""
    train_features = signature.transform(problem.x)
    test_features = signature.transform(problem.xtest)
    initial = posterior(train_features, message, temperature)
    test = posterior(test_features, message, temperature)
    diagnostics = {'target_training_labels': 0}
    if transport_mix:
        center = posterior(signature.transform(problem.centers), message, temperature)
        cost = -np.log(np.maximum(center, 1e-12))
        plan, conditional = fixed_transport(cost, mass=1., tau=1., target_mass=problem.target_mass)
        teacher = conditional[:, :problem.k]
        if hard:
            teacher = np.eye(problem.k)[teacher.argmax(1)]
        initial = (1 - transport_mix) * initial + transport_mix * (problem.responsibilities @ teacher)
        test = (1 - transport_mix) * test + transport_mix * (problem.test_responsibilities @ teacher)
        diagnostics['plan'] = plan
        diagnostics['teacher'] = teacher
    if graph_mix:
        graph = compatible_transition(problem.transition, initial) if compatible else problem.transition
        smooth, convergence = anchored_diffusion(graph, initial, graph_mix)
        edges = problem.test_transition.tocoo()
        if compatible:
            divergence = np.sum((np.sqrt(test[edges.row]) - np.sqrt(initial[edges.col])) ** 2, axis=1) / 2
            values = edges.data * np.exp(-divergence / .1)
            test_graph = sparse.csr_matrix((values, (edges.row, edges.col)), shape=problem.test_transition.shape)
            degree = np.asarray(test_graph.sum(axis=1)).ravel()
            safe = degree > 1e-15
            propagated = test.copy()
            propagated[safe] = (test_graph @ smooth)[safe] / degree[safe, None]
        else:
            propagated = problem.test_transition @ smooth
        test = (1 - graph_mix) * test + graph_mix * propagated
        diagnostics.update(convergence)
    test = np.maximum(test, 1e-12)
    test /= test.sum(axis=1, keepdims=True)
    return test, diagnostics
