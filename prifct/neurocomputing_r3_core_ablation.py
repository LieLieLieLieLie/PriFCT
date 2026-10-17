"""R4 same-packet component controls for streamlined PriFCT.

The controls reuse every registered UCI HAR private packet.  They alter only
receiver-side public geometry, public-pair correspondence, or the unlabelled
prior correction.  No source query, cohort, label role, or tuning grid is
changed.  The shuffled-pair permutation is fixed before reading outcomes.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/models/eswa_fedkapt/semantic_v47_uci_har"
OUT = SOURCE / "r4_same_packet_components"
PAPER_GENERATED = ROOT.parent / "paper/prifct/generated"
VARIANTS = [
    "Full PriFCT",
    "Trace-scaled identity",
    "Identity geometry",
    "Shuffled public pairs",
    "No prior correction",
]
METRICS = ["subject_balanced_accuracy", "macro_f1", "nll", "brier", "ece"]
OPERATOR_OPTIONS = [
    {"bandwidth_factor": bandwidth, "ridge": ridge}
    for bandwidth in (0.25, 1.0)
    for ridge in (0.01, 0.1, 1.0)
]


def unit_clip(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0)


@dataclass(frozen=True)
class PublicLogScale:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x):
        x = np.asarray(x, float)
        return (np.sign(x) * np.log1p(abs(x)) - self.mean) / self.std


@dataclass(frozen=True)
class PublicRegressionBridge:
    source_scaler: object
    target_scaler: object
    source_basis: np.ndarray
    source_scale: float
    target_landmarks: np.ndarray
    target_bandwidth: float
    target_coefficients: np.ndarray

    def source(self, x):
        return unit_clip(
            self.source_scaler.transform(x) @ self.source_basis / self.source_scale
        )

    def target(self, x):
        raw = self.target_scaler.transform(x)
        kernel = np.exp(
            -squared_distance(raw, self.target_landmarks) / self.target_bandwidth
        )
        return unit_clip(kernel @ self.target_coefficients)


@dataclass(frozen=True)
class MultiScaleSignature:
    frequencies: tuple
    dimension: int

    def transform(self, x):
        x = np.asarray(x, float)
        pieces = [x]
        for frequency in self.frequencies:
            phase = x @ frequency
            pieces.append(
                np.c_[np.cos(phase), np.sin(phase)]
                / np.sqrt(frequency.shape[1])
            )
        return np.concatenate(pieces, axis=1) / np.sqrt(len(pieces))


@dataclass(frozen=True)
class ProjectedSignature:
    original: object
    basis: np.ndarray

    def transform(self, x):
        return self.original.transform(x) @ self.basis


@dataclass(frozen=True)
class FunctionalBridge:
    point_bridge: object
    signature: object
    bandwidth: float
    coefficients: np.ndarray

    def source(self, raw):
        return self.signature.transform(self.point_bridge.source(raw))

    def target(self, raw):
        standardized = self.point_bridge.target_scaler.transform(raw)
        kernel = np.exp(
            -squared_distance(
                standardized, self.point_bridge.target_landmarks
            )
            / self.bandwidth
        )
        return unit_clip(kernel @ self.coefficients)


def install_pickle_shims() -> None:
    """Expose saved public-map classes without importing legacy Torch code."""
    modules = {
        "prifct.v7_public_bridge": {
            "PublicLogScale": PublicLogScale,
            "PublicRegressionBridge": PublicRegressionBridge,
        },
        "prifct.v7_functional_bridge": {"FunctionalBridge": FunctionalBridge},
        "prifct.v7_anchored_distribution": {"ProjectedSignature": ProjectedSignature},
        "prifct.v7_semantic_transport": {"MultiScaleSignature": MultiScaleSignature},
    }
    for module_name, members in modules.items():
        module = types.ModuleType(module_name)
        for name, value in members.items():
            setattr(module, name, value)
        sys.modules[module_name] = module


def kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        np.sum(a * (np.log(np.maximum(a, 1e-300)) - np.log(b)) - a + b)
    )


def transport(cost, mass, prior, temperature, prior_strength):
    """Checked convex dual used by the frozen prior-correction stage."""
    mass = np.asarray(mass, float)
    prior = np.asarray(prior, float)
    mass /= mass.sum()
    prior /= prior.sum()
    log_prior = np.log(prior)
    tau = temperature
    beta = prior_strength

    if beta == 0:
        conditional = softmax(log_prior[None, :] - cost / tau, axis=1)
        return mass[:, None] * conditional, {
            "dual_gradient_infinity": 0.0,
            "duality_gap": 0.0,
            "iterations": 0,
        }

    def state(vector, hessian=False):
        posterior = softmax(
            log_prior[None, :] - (cost + vector[None, :]) / tau, axis=1
        )
        dual_prior = prior * np.exp(vector / beta)
        column_mass = (mass[:, None] * posterior).sum(0)
        value = beta * np.sum(prior * np.expm1(vector / beta)) + tau * np.sum(
            mass
            * logsumexp(
                log_prior[None, :] - (cost + vector[None, :]) / tau,
                axis=1,
            )
        )
        gradient = dual_prior - column_mass
        if not hessian:
            return value, gradient
        second = (
            np.diag(dual_prior / beta + column_mass / tau)
            - (posterior.T * mass) @ posterior / tau
        )
        return value, gradient, second, posterior

    fit = minimize(
        state,
        np.zeros(len(prior)),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 3000, "maxls": 60},
    )
    vector = fit.x
    refinements = 0
    for _ in range(40):
        value, gradient, hessian, posterior = state(vector, True)
        if np.max(abs(gradient)) <= 1e-10:
            break
        direction = solve(hessian, -gradient, assume_a="pos")
        scale = 1.0
        for _ in range(50):
            candidate = vector + scale * direction
            if np.max(candidate / beta) > 700:
                scale *= 0.5
                continue
            new_value, new_gradient = state(candidate)
            if new_value <= value + 1e-14 and np.max(abs(new_gradient)) < np.max(
                abs(gradient)
            ):
                vector = candidate
                refinements += 1
                break
            scale *= 0.5
        else:
            break
    value, gradient, _, posterior = state(vector, True)
    plan = mass[:, None] * posterior
    primal = (
        float(np.sum(plan * cost))
        + tau * kl_divergence(plan, mass[:, None] * prior[None, :])
        + beta * kl_divergence(plan.sum(0), prior)
    )
    gap = float(primal + value)
    error = float(np.max(abs(gradient)))
    if not np.isfinite(plan).all() or error > 2e-9 or gap > 1e-10 or gap < -1e-8:
        raise RuntimeError(f"Checked dual solver failed {(error, gap)}")
    return plan, {
        "dual_gradient_infinity": error,
        "duality_gap": max(gap, 0.0),
        "iterations": int(fit.nit) + refinements,
        "newton_refinements": refinements,
    }


def squared_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NumPy equivalent of the frozen public distance routine (no Torch import)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return np.maximum(
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * x @ y.T,
        0.0,
    )


def simplex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    ordered = -np.sort(-x, axis=1)
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    active = ordered - cumulative / np.arange(1, x.shape[1] + 1) > 0
    count = active.sum(1)
    threshold = cumulative[np.arange(len(x)), count - 1] / count
    return np.maximum(x - threshold[:, None], 0.0)


def public_operators(point, target_fit, target_held, adaptation, validation):
    inputs = [target_fit, target_held, adaptation, validation]
    distance = [
        squared_distance(point.target_scaler.transform(x), point.target_landmarks)
        for x in inputs
    ]
    result = []
    for option in OPERATOR_OPTIONS:
        kernels = [
            np.exp(-value / (point.target_bandwidth * option["bandwidth_factor"]))
            for value in distance
        ]
        design = kernels[0]
        regression = solve(
            design.T @ design + option["ridge"] * np.eye(design.shape[1]),
            design.T,
            assume_a="pos",
        )
        result.append({
            "option": option,
            "regression": regression,
            "held": kernels[1],
            "adaptation": kernels[2],
            "validation": kernels[3],
        })
    return result


def calibrated(operators, source_fit, source_held, weight, teacher_temperature):
    fit = softmax(source_fit @ weight / teacher_temperature, axis=1)
    held = softmax(source_held @ weight / teacher_temperature, axis=1)
    choices = []
    for operator in operators:
        coefficients = operator["regression"] @ fit
        prediction = simplex(operator["held"] @ coefficients)
        error = float(np.mean(np.sum((prediction - held) ** 2, axis=1)))
        choices.append((error, coefficients))
    selected = min(range(len(choices)), key=lambda index: choices[index][0])
    coefficients = choices[selected][1]
    operator = operators[selected]
    return {
        "adaptation": simplex(operator["adaptation"] @ coefficients),
        "validation": simplex(operator["validation"] @ coefficients),
        "selected": selected,
        "public_errors": [value[0] for value in choices],
    }


def log_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.maximum(np.asarray(probability, float), 1e-12)
    probability /= probability.sum(1, keepdims=True)
    return np.log(probability)


def offset(logscores, temperature, balance_strength, prior, mass=None):
    n, k = logscores.shape
    mass = np.ones(n) / n if mass is None else np.asarray(mass, float)
    prior = np.maximum(np.asarray(prior, float), 1e-12)
    prior /= prior.sum()
    if balance_strength == 0:
        return np.zeros(k), {
            "dual_gradient_infinity": 0.0,
            "duality_gap": 0.0,
            "iterations": 0,
        }
    cost = -logscores + temperature * np.log(prior)[None, :]
    plan, solver = transport(
        cost, mass, prior, temperature, balance_strength * temperature
    )
    conditional = plan / (mass / mass.sum())[:, None]
    correction = (
        temperature * np.log(conditional[0])
        + cost[0]
        - temperature * np.log(prior)
    )
    correction -= correction.mean()
    return correction, solver


def predict(logscores, correction, temperature):
    probability = softmax(
        (np.asarray(logscores, float) + correction) / temperature, axis=1
    )
    probability = np.maximum(probability, 1e-12)
    probability /= probability.sum(1, keepdims=True)
    probability = probability.astype(np.float32).astype(np.float64)
    return probability / probability.sum(1, keepdims=True)


def features(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return np.c_[np.asarray(x, float) - mean, np.ones(len(x))]


def public_covariance(functions: np.ndarray) -> np.ndarray:
    functions = np.asarray(functions, float)
    return functions.T @ functions / len(functions)


def reconstruct(first, covariance, sigma, ridge, *, no_denoising=False):
    """Closed-form ridge reconstruction used by the registered receiver."""
    first = np.asarray(first, float)
    covariance = np.asarray(covariance, float)
    k, dimension = first.shape
    values, basis = np.linalg.eigh((covariance + covariance.T) / 2.0)
    values = np.maximum(values, 0.0)
    coordinates = first @ basis
    common = coordinates.mean(0, keepdims=True)
    contrast = coordinates - common
    radius_squared = np.sum(contrast**2, axis=0)
    threshold = max(k - 3, 0) * sigma * sigma
    shrinkage = (
        np.ones(dimension)
        if no_denoising or k < 4 or sigma == 0
        else np.maximum(1 - threshold / np.maximum(radius_squared, 1e-300), 0.0)
    )
    filtered = common + contrast * shrinkage
    weight = basis @ (filtered.T / (values[:, None] + ridge))
    return weight, {
        "public_spectrum": values,
        "shrink": shrinkage,
        "contrast_radius_squared": radius_squared,
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_probability_calibration(p: np.ndarray, option: dict) -> np.ndarray:
    p = np.asarray(p, float)
    value = softmax(np.log(np.maximum(p, 1e-30)) / option["temperature"], axis=1)
    value = (1.0 - option["uniform_mass"]) * value + option["uniform_mass"] / p.shape[1]
    value /= value.sum(1, keepdims=True)
    if not np.array_equal(value.argmax(1), p.argmax(1)):
        raise RuntimeError("Decision-preserving calibration changed an argmax")
    return value


def base_metrics(y: np.ndarray, p: np.ndarray, k: int) -> dict:
    p = np.maximum(np.asarray(p, float), 1e-15)
    p /= p.sum(1, keepdims=True)
    pred = p.argmax(1)
    confidence = p.max(1)
    correct = pred == y
    ece = 0.0
    for lo in np.arange(0.0, 1.0, 0.1):
        mask = (confidence >= lo) & (confidence < lo + 0.1 if lo < 0.9 else confidence <= 1.0)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {
        "macro_f1": float(f1_score(y, pred, labels=np.arange(k), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "nll": float(log_loss(y, p, labels=np.arange(k))),
        "brier": float(np.mean(np.sum((p - np.eye(k)[y]) ** 2, axis=1))),
        "ece": float(ece),
    }


def score(y: np.ndarray, p: np.ndarray, subjects: np.ndarray, k: int) -> dict:
    result = base_metrics(y, p, k)
    per_subject = {}
    pred = p.argmax(1)
    for subject in np.unique(subjects):
        mask = subjects == subject
        available = np.unique(y[mask])
        per_subject[str(int(subject))] = {
            "balanced_accuracy": float(np.mean([
                (pred[mask][y[mask] == c] == c).mean() for c in available
            ]))
        }
    result["per_subject_balanced"] = per_subject
    result["subject_balanced_accuracy"] = float(np.mean([
        item["balanced_accuracy"] for item in per_subject.values()
    ]))
    return result


def subject_metrics(y: np.ndarray, p: np.ndarray, subjects: np.ndarray, k: int) -> dict:
    result = {}
    for subject in np.unique(subjects):
        mask = subjects == subject
        item = base_metrics(y[mask], p[mask], k)
        pred = p[mask].argmax(1)
        available = np.unique(y[mask])
        item["subject_balanced_accuracy"] = float(np.mean([
            (pred[y[mask] == c] == c).mean() for c in available
        ]))
        result[str(int(subject))] = {m: float(item[m]) for m in METRICS}
    return result


def bootstrap_delta(full: np.ndarray, control: np.ndarray, higher: bool, seed: int) -> dict:
    # Positive means that the full method is better for every endpoint.
    delta = full - control if higher else control - full
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(delta), size=(10000, len(delta)))
    draws = delta[index].mean(1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "n_people": int(len(delta)),
    }


def main() -> None:
    install_pickle_shims()
    protocol = json.loads((SOURCE / "protocol.json").read_text(encoding="utf-8"))
    with np.load(SOURCE / "descriptors.npz") as saved:
        arrays = {name: saved[name] for name in saved.files}
    arrays["roles"] = {
        role: np.flatnonzero(np.isin(arrays["subjects"], people))
        for role, people in protocol["roles"].items()
    }
    rng_public = np.random.default_rng(91311)
    fit, held = [], []
    for person in protocol["roles"]["public"]:
        eligible = np.flatnonzero(arrays["subjects"] == person)
        order = rng_public.permutation(eligible)
        fit.extend(order[:96])
        held.extend(order[96:120])
    arrays["fit"] = np.asarray(fit)
    arrays["held"] = np.asarray(held)
    options = protocol["options"]["Ridge source teacher"]
    calibration_path = (
        ROOT
        / "results/models/eswa_fedkapt/semantic_probability_calibration"
        / "maxent_v19_development/summary.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["settings"][
        "transfer"
    ]["Ridge source teacher"]["selected"]["options"]
    OUT.mkdir(parents=True, exist_ok=True)
    predictions = OUT / "predictions"
    predictions.mkdir(exist_ok=True)
    protocol_out = {
        "scope": "R4 same-packet receiver component controls on the registered UCI HAR split",
        "private_packets": "Exact existing epsilon=2/10, three-seed packets reused",
        "variants": VARIANTS,
        "trace_scaled_control": "Replace C_p by tr(C_p)/d times I_d; retain the registered development-selected ridge lambda so the quadratic trace scale and tuning budget are matched",
        "identity_control": "Replace public covariance C_p by I_d; retain the registered ridge lambda",
        "pair_control": "Apply fixed independent permutations to source-side public fit and held rows only; target public rows and all marginals are unchanged",
        "prior_control": "Set balance_strength to zero; retain the same noisy count packet but do not use it for the adaptation offset",
        "permutation_seeds": {"accel_to_gyro": 91731, "gyro_to_accel": 91732},
        "selection": "No variant-specific tuning and no outcome-dependent permutation",
        "probability_calibration": calibration,
        "code_sha256": sha(Path(__file__)),
        "parent_protocol_sha256": sha(SOURCE / "protocol.json"),
    }
    protocol_path = OUT / "protocol.json"
    protocol_path.write_text(json.dumps(protocol_out, indent=2), encoding="utf-8")

    rows = []
    source_index = arrays["roles"]["source"]
    adaptation_index = arrays["roles"]["adaptation"]
    test_index = arrays["roles"]["test"]
    for direction_index, (source_view, target_view) in enumerate(protocol["directions"]):
        with (SOURCE / "public_maps" / f"{source_view}_{target_view}.pkl").open("rb") as stream:
            public = pickle.load(stream)
        mapping, point = public["full"], public["point"]
        fit, held = arrays["fit"], arrays["held"]
        source_fit = mapping.source(arrays[source_view][fit])
        source_held = mapping.source(arrays[source_view][held])
        source_private = mapping.source(arrays[source_view][source_index])
        adaptation = mapping.target(arrays[target_view][adaptation_index])
        test = mapping.target(arrays[target_view][test_index])
        covariance = public_covariance(source_fit)
        operators = public_operators(
            point,
            arrays[target_view][fit],
            arrays[target_view][held],
            arrays[target_view][adaptation_index],
            arrays[target_view][test_index],
        )
        rng = np.random.default_rng(91731 + direction_index)
        fit_permutation = rng.permutation(len(source_fit))
        held_permutation = rng.permutation(len(source_held))

        for epsilon in protocol["budgets"]:
            for seed in protocol["seeds"]:
                packet_path = SOURCE / "queries" / f"{source_view}_{target_view}_e{epsilon}_s{seed}.npz"
                with np.load(packet_path) as saved:
                    raw = saved["raw"]
                    sigma = float(saved["sigma"])
                noisy_prior = np.maximum(raw[:, -1] * len(source_private), 1.0)
                noisy_prior /= noisy_prior.sum()
                for variant in VARIANTS:
                    if variant == "Trace-scaled identity":
                        decoder_covariance = (
                            np.trace(covariance) / covariance.shape[0]
                        ) * np.eye(covariance.shape[0])
                    elif variant == "Identity geometry":
                        decoder_covariance = np.eye(covariance.shape[0])
                    else:
                        decoder_covariance = covariance
                    weight, diagnostic = reconstruct(
                        raw[:, :-1], decoder_covariance, sigma, options["ridge"], no_denoising=True
                    )
                    weight = np.r_[weight, np.zeros((1, raw.shape[0]))]
                    mean = np.zeros(source_fit.shape[1])
                    fit_features = features(source_fit, mean)
                    held_features = features(source_held, mean)
                    if variant == "Shuffled public pairs":
                        fit_features = fit_features[fit_permutation]
                        held_features = held_features[held_permutation]
                    state = calibrated(
                        operators,
                        fit_features,
                        held_features,
                        weight,
                        options["teacher_temperature"],
                    )
                    strength = 0.0 if variant == "No prior correction" else options["balance_strength"]
                    correction, solver = offset(
                        log_probability(state["adaptation"]),
                        options["receiver_temperature"],
                        strength,
                        noisy_prior,
                    )
                    probability = predict(
                        log_probability(state["validation"]),
                        correction,
                        options["receiver_temperature"],
                    )
                    probability = apply_probability_calibration(probability, calibration)
                    result = score(
                        arrays["y"][test_index], probability, arrays["subjects"][test_index], 6
                    )
                    by_subject = subject_metrics(
                        arrays["y"][test_index], probability, arrays["subjects"][test_index], 6
                    )
                    identifier = (
                        f"{source_view}_{target_view}_e{epsilon}_s{seed}_"
                        + variant.lower().replace(" ", "_")
                    )
                    output = predictions / f"{identifier}.npz"
                    np.savez_compressed(
                        output,
                        y=arrays["y"][test_index],
                        subjects=arrays["subjects"][test_index],
                        probability=probability.astype(np.float32),
                    )
                    rows.append({
                        "id": identifier,
                        "variant": variant,
                        "direction": f"{source_view}_to_{target_view}",
                        "epsilon": float(epsilon),
                        "seed": int(seed),
                        "metrics": {m: float(result[m]) for m in METRICS},
                        "per_subject": by_subject,
                        "operator_selected": int(state["selected"]),
                        "public_errors": [float(x) for x in state["public_errors"]],
                        "decoder_spectrum": [float(x) for x in diagnostic["public_spectrum"]],
                        "solver": solver,
                        "prediction_sha256": sha(output),
                    })

    (OUT / "results.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    by_variant = {}
    person_values = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {
            metric: float(np.mean([row["metrics"][metric] for row in selected]))
            for metric in METRICS
        }
        people = sorted(selected[0]["per_subject"], key=int)
        person_values[variant] = {
            person: {
                metric: float(np.mean([
                    row["per_subject"][person][metric] for row in selected
                ]))
                for metric in METRICS
            }
            for person in people
        }

    comparisons = {}
    for control_index, control in enumerate(VARIANTS[1:]):
        comparisons[control] = {}
        people = sorted(person_values[VARIANTS[0]], key=int)
        for metric_index, metric in enumerate(METRICS):
            full = np.array([person_values[VARIANTS[0]][p][metric] for p in people])
            other = np.array([person_values[control][p][metric] for p in people])
            comparisons[control][metric] = bootstrap_delta(
                full, other, metric in ("subject_balanced_accuracy", "macro_f1"),
                91800 + 10 * control_index + metric_index,
            )

    summary = {
        "complete": True,
        "outcomes": len(rows),
        "failures": 0,
        "by_variant": by_variant,
        "person_values": person_values,
        "paired_full_minus_control_utility": comparisons,
        "protocol_sha256": sha(protocol_path),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    PAPER_GENERATED.mkdir(parents=True, exist_ok=True)
    table_rows = []
    best_values = [
        max(100 * by_variant[v]["subject_balanced_accuracy"] for v in VARIANTS),
        max(100 * by_variant[v]["macro_f1"] for v in VARIANTS),
        min(by_variant[v]["nll"] for v in VARIANTS),
        min(by_variant[v]["brier"] for v in VARIANTS),
        min(by_variant[v]["ece"] for v in VARIANTS),
    ]
    for variant in VARIANTS:
        item = by_variant[variant]
        values = [
            100 * item["subject_balanced_accuracy"],
            100 * item["macro_f1"],
            item["nll"], item["brier"], item["ece"],
        ]
        cells = [f"{values[0]:.2f}", f"{values[1]:.2f}", f"{values[2]:.3f}", f"{values[3]:.3f}", f"{values[4]:.3f}"]
        cells = [
            f"\\textbf{{{cell}}}" if abs(value - best) < 1e-12 else cell
            for cell, value, best in zip(cells, values, best_values)
        ]
        table_rows.append(variant + " & " + " & ".join(cells) + r" \\")
    (PAPER_GENERATED / "r4_component_rows.tex").write_text(
        "\n".join(table_rows) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["by_variant"], indent=2))


if __name__ == "__main__":
    main()
