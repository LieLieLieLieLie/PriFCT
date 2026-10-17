"""Untouched PAMAP2 evaluation of the fully frozen v127 PriFCT adapter."""
from __future__ import annotations

import os
os.urandom = lambda n: bytes(n)  # deterministic workaround for local Windows CNG failure

import hashlib
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.linalg import svd, solve
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.metrics import f1_score, log_loss
from threadpoolctl import threadpool_limits

from core.privacy import compute_sigma
from prifct.v7_public_bridge import fit_public_bridge
from prifct.v7_functional_bridge import fit_functional_bridge
from prifct.v6_public_relation import squared_distance
from prifct.v55_public_mixture_teacher import fit as mixture_fit


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/models/eswa_fedkapt"
OUT = BASE / "semantic_v132_pamap2_frozen_prifct"
DATA = ROOT / "data/pamap2/descriptors"
DESIGN = BASE / "semantic_v128_pamap2_preregistration/design_before_acquisition.json"
AMEND = BASE / "semantic_v130_pamap2_preutility_feasibility/availability_amendment_before_utility.json"
DESCRIPTORS = BASE / "semantic_v131_pamap2_descriptors/descriptor_manifest.json"
PUBLIC_PROTOCOL = BASE / "semantic_v115_realworld_cuda_public_preflight/protocol.json"
FROZEN = BASE / "semantic_v127_three_panel_reliability_gate/protocol.json"
ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
LABEL = {activity: index for index, activity in enumerate(ACTIVITIES)}
ENDPOINTS = ["subject_balanced_accuracy", "macro_f1", "nll", "brier", "ece"]
OPERATOR_OPTIONS = [dict(bandwidth_factor=b, ridge=r) for b in [.25, 1.] for r in [.01, .1, 1.]]


class MultiScaleSignature:
    def __init__(self, frequencies, dimension):
        self.frequencies = frequencies
        self.dimension = dimension

    @classmethod
    def public_fit(cls, public, seed=76011, width=64):
        public = np.asarray(public, float)
        distance = squared_distance(public, public)
        values = distance[np.triu_indices(len(public), 1)]
        bandwidth = max(float(np.median(values)), 1e-4)
        rng = np.random.default_rng(seed)
        frequencies = tuple(rng.normal(size=(public.shape[1], width // 2)) *
                            np.sqrt(2 / (bandwidth * factor)) for factor in [.25, 1.])
        return cls(frequencies, public.shape[1])

    def transform(self, x):
        x = np.asarray(x, float)
        pieces = [x]
        for frequency in self.frequencies:
            phase = x @ frequency
            pieces.append(np.c_[np.cos(phase), np.sin(phase)] / np.sqrt(frequency.shape[1]))
        return np.concatenate(pieces, axis=1) / np.sqrt(len(pieces))


class ProjectedSignature:
    def __init__(self, original, basis):
        self.original = original
        self.basis = basis

    @classmethod
    def public_fit(cls, original, public, rank=32):
        functions = original.transform(public)
        _, _, vectors = svd(functions, full_matrices=False)
        return cls(original, vectors[:min(rank, len(vectors))].T)

    def transform(self, x):
        return self.original.transform(x) @ self.basis


def fit_maps(source_fit, target_fit, source_held, target_held, protocol):
    point = fit_public_bridge(source_fit, target_fit, **protocol["point"])
    point_source_fit = point.source(source_fit)
    signature = MultiScaleSignature.public_fit(point_source_fit, seed=protocol["signature"]["seed"],
                                               width=protocol["signature"]["width"])
    projected = ProjectedSignature.public_fit(signature, point_source_fit, rank=protocol["signature"]["rank"])
    candidates = []
    for option in protocol["function_options"]:
        model = fit_functional_bridge(point, projected, source_fit, target_fit,
                                      ridge=option["ridge"], bandwidth_factor=option["bandwidth_factor"])
        error = float(np.mean(np.sum((model.target(target_held) - model.source(source_held)) ** 2, axis=1)))
        candidates.append((error, option, model))
    error, option, full = min(candidates, key=lambda item: item[0])
    return {"point": point, "full": full, "selected_option": option, "public_held_function_mse": error}


def public_operators(point, target_fit, target_held, adaptation, validation):
    inputs = [target_fit, target_held, adaptation, validation]
    distance = [squared_distance(point.target_scaler.transform(x), point.target_landmarks) for x in inputs]
    result = []
    for option in OPERATOR_OPTIONS:
        kernels = [np.exp(-value / (point.target_bandwidth * option["bandwidth_factor"])) for value in distance]
        a = kernels[0]
        regression = solve(a.T @ a + option["ridge"] * np.eye(a.shape[1]), a.T, assume_a="pos")
        result.append({"option": option, "regression": regression, "held": kernels[1],
                       "adaptation": kernels[2], "validation": kernels[3]})
    return result


def log_probability(probability):
    probability = np.maximum(np.asarray(probability, float), 1e-12)
    return np.log(probability / probability.sum(axis=1, keepdims=True))


def _transport(cost, mass, q, temperature, prior_strength):
    mass = np.asarray(mass, float); mass /= mass.sum()
    q = np.asarray(q, float); q /= q.sum()
    def objective(value):
        logits = np.log(q)[None, :] - (cost + value[None, :]) / temperature
        posterior = softmax(logits, axis=1)
        columns = (mass[:, None] * posterior).sum(axis=0)
        loss = prior_strength * np.sum(q * np.expm1(value / prior_strength))
        loss += temperature * np.sum(mass * logsumexp(logits, axis=1))
        gradient = q * np.exp(value / prior_strength) - columns
        return loss, gradient
    fitted = minimize(objective, np.zeros(len(q)), jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 3000, "maxls": 60})
    conditional = softmax(np.log(q)[None, :] - (cost + fitted.x[None, :]) / temperature, axis=1)
    return mass[:, None] * conditional, {"iterations": int(fitted.nit),
                                         "dual_gradient_infinity": float(np.max(np.abs(objective(fitted.x)[1])))}


def offset(logscores, temperature, balance_strength, q, mass=None):
    n, classes = logscores.shape
    mass = np.ones(n) / n if mass is None else np.asarray(mass, float)
    q = np.maximum(np.asarray(q, float), 1e-12); q /= q.sum()
    if balance_strength == 0:
        return np.zeros(classes), {"iterations": 0, "dual_gradient_infinity": 0.0}
    cost = -logscores + temperature * np.log(q)[None, :]
    plan, diagnostics = _transport(cost, mass, q, temperature, balance_strength * temperature)
    conditional = plan / (mass / mass.sum())[:, None]
    correction = temperature * np.log(conditional[0]) + cost[0] - temperature * np.log(q)
    correction -= correction.mean()
    return correction, diagnostics


def predict(logscores, correction, temperature):
    return softmax((logscores + correction) / temperature, axis=1)


def entropic_calibrated(operators, source_fit, source_held, weight, teacher_temperature):
    fit = softmax(source_fit @ weight / teacher_temperature, axis=1)
    held = softmax(source_held @ weight / teacher_temperature, axis=1)
    choices = []
    for index, operator in enumerate(operators):
        coefficients = operator["regression"] @ fit
        prediction = operator["held"] @ coefficients
        for temperature in [.01, .03, .1, .3]:
            probability = softmax(prediction / temperature, axis=1)
            error = float(np.mean(np.sum((probability - held) ** 2, axis=1)))
            choices.append((error, index, temperature, coefficients))
    error, index, temperature, coefficients = min(choices, key=lambda value: value[0])
    operator = operators[index]
    stable = lambda value: value / temperature - logsumexp(value / temperature, axis=1, keepdims=True)
    return {"adaptation_log": stable(operator["adaptation"] @ coefficients),
            "validation_log": stable(operator["validation"] @ coefficients),
            "projection_temperature": temperature, "public_error": error}


def public_anchors(x, count):
    x = np.asarray(x, float)
    first = int(np.argmax(np.sum((x - x.mean(0)) ** 2, axis=1)))
    chosen = [first]; nearest = squared_distance(x, x[chosen]).ravel()
    while len(chosen) < count:
        candidate = int(np.argmax(nearest)); chosen.append(candidate)
        nearest = np.minimum(nearest, squared_distance(x, x[[candidate]]).ravel())
    return x[chosen]


def anchor_release(x, y, classes, anchors, epsilon, seed, delta=1e-5):
    x = np.asarray(x, float); y = np.asarray(y, int); n, dimension = x.shape
    regions = np.argmin(squared_distance(x, anchors), axis=1)
    query = np.zeros((classes, len(anchors), dimension + 1))
    for row in range(n):
        query[y[row], regions[row], :-1] += x[row] / n
        query[y[row], regions[row], -1] += 1.0 / n
    sigma = float(compute_sigma(2.0 / n, epsilon, delta))
    return query + np.random.default_rng(seed).normal(size=query.shape) * sigma, sigma


def anchor_decode(raw, n, anchors, sigma, uncertainty_scale=0.0):
    counts = np.maximum(raw[:, :, -1] * n, 1.0)
    means = raw[:, :, :-1] * n / counts[:, :, None]
    means /= np.maximum(np.linalg.norm(means, axis=2, keepdims=True), 1.0)
    within = counts / counts.sum(axis=1, keepdims=True)
    prior = counts.sum(axis=1); prior /= prior.sum()
    return means, within, prior


def teacher_probability(x, means, within, prior, temperature, norm_correction=0.0):
    similarity = np.einsum("nd,kad->nka", np.asarray(x, float), means)
    similarity -= 0.5 * float(norm_correction) * np.sum(means * means, axis=2)[None, :, :]
    logits = logsumexp(similarity / temperature + np.log(within)[None, :, :], axis=2)
    return softmax(logits + np.log(prior)[None, :], axis=1)


def normalized_covariance_distance(fit, held):
    first, second = np.cov(fit, rowvar=False), np.cov(held, rowvar=False)
    first /= max(float(np.linalg.norm(first)), 1e-15)
    second /= max(float(np.linalg.norm(second)), 1e-15)
    return float(np.linalg.norm(first - second))


def joint_release(x, y, classes, epsilon, seed, delta=1e-5):
    x = np.asarray(x, float); y = np.asarray(y, int); n = len(x)
    clean = np.zeros((classes, x.shape[1] + 1))
    np.add.at(clean[:, :-1], y, x / n)
    np.add.at(clean[:, -1], y, 1.0 / n)
    sigma = float(compute_sigma(2.0 / n, epsilon, delta))
    raw = clean + np.random.default_rng(seed).normal(size=clean.shape) * sigma
    return raw, sigma


def evaluate_metrics(y, probability, subjects, classes):
    probability = np.maximum(np.asarray(probability, float), 1e-15)
    probability /= probability.sum(axis=1, keepdims=True)
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1); correct = prediction == y
    ece = 0.0
    for lower in np.arange(0.0, 1.0, 0.1):
        upper = lower + 0.1
        mask = (confidence >= lower) & ((confidence < upper) if lower < 0.9 else (confidence <= 1.0))
        if np.any(mask):
            ece += float(mask.mean() * abs(confidence[mask].mean() - correct[mask].mean()))
    per_subject = {}
    for participant in np.unique(subjects):
        mask = subjects == participant; available = np.unique(y[mask])
        per_subject[str(int(participant))] = float(np.mean([
            np.mean(prediction[mask][y[mask] == label] == label) for label in available]))
    return {"subject_balanced_accuracy": float(np.mean(list(per_subject.values()))),
            "macro_f1": float(f1_score(y, prediction, labels=np.arange(classes), average="macro", zero_division=0)),
            "nll": float(log_loss(y, probability, labels=np.arange(classes))),
            "brier": float(np.mean(np.sum((probability - np.eye(classes)[y]) ** 2, axis=1))),
            "ece": float(ece), "per_subject_balanced": per_subject}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def cohort(people, location, modality, include_labels=False):
    pieces = []
    for participant in people:
        with np.load(DATA / f"subject{participant:02d}_{location}.npz") as data:
            piece = {
                "x": data[modality].astype(np.float64),
                "subjects": np.full(len(data["y"]), participant, dtype=np.int16),
                "bins": data["bin"].astype(np.int64),
            }
            if include_labels:
                piece["y"] = np.asarray([LABEL[int(value)] for value in data["y"]], dtype=np.int16)
            pieces.append(piece)
    arrays = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    order = np.lexsort((arrays["bins"], arrays["subjects"]))
    return {key: value[order] for key, value in arrays.items()}


def split_public(subjects, seed, fit_per_person, held_per_person):
    fit, held = [], []
    for participant in np.unique(subjects):
        rows = np.flatnonzero(subjects == participant)
        draw = np.random.default_rng(seed + int(participant)).permutation(rows)
        required = fit_per_person + held_per_person
        if len(draw) < required:
            raise RuntimeError(f"public participant {participant} has {len(draw)} < {required} windows")
        fit.extend(draw[:fit_per_person])
        held.extend(draw[fit_per_person:required])
    return np.asarray(fit), np.asarray(held)


def cap(n, size, seed):
    if n < size:
        raise RuntimeError(f"cohort {n} smaller than frozen cap {size}")
    return np.sort(np.random.default_rng(seed).permutation(n)[:size])


def temperature_scale(probability, temperature):
    powered = np.maximum(probability, 1e-300) ** (1.0 / temperature)
    return powered / powered.sum(axis=1, keepdims=True)


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    design, amendment, frozen = read(DESIGN), read(AMEND), read(FROZEN)
    protocol = {
        "scope": "First untouched PAMAP2 classifier evaluation of the byte-frozen v127 two-branch PriFCT adapter",
        "roles": {key: amendment[key] for key in ["public_people", "source_people", "adaptation_people", "test_people"]},
        "classes": len(ACTIVITIES), "activities": ACTIVITIES,
        "directions": design["directions"], "budgets": design["budgets"], "seeds": design["release_seeds"],
        "public_fit_per_person": design["public_fit_per_person"], "public_held_per_person": design["public_held_per_person"],
        "public_seed": design["public_seed"], "source_cap": design["source_cap"], "adaptation_cap": design["adaptation_cap"],
        "source_seed": design["source_seed"], "adaptation_seed": design["adaptation_seed"],
        "gate": frozen["gate"], "stable_branch": frozen["stable_branch"], "unstable_branch": frozen["unstable_branch"],
        "routing": "compute every direction's unlabelled public covariance statistic, then apply the registered maximum-across-directions gate before any private query",
        "endpoints": ENDPOINTS,
        "source_design_sha256": sha(DESIGN), "availability_amendment_sha256": sha(AMEND),
        "descriptor_manifest_sha256": sha(DESCRIPTORS), "public_map_protocol_sha256": sha(PUBLIC_PROTOCOL),
        "frozen_adapter_sha256": sha(FROZEN), "no_pamap2_outcome_selection": True,
        "implementation_sha256": sha(Path(__file__)),
    }
    path = OUT / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise RuntimeError("Executed PAMAP2 PriFCT protocol changed")
    if not path.exists():
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def prepare(protocol):
    roles = protocol["roles"]
    public_protocol = read(PUBLIC_PROTOCOL)
    maps_dir = OUT / "public_maps"; maps_dir.mkdir(exist_ok=True)
    prepared = []
    for location, source_modality, target_modality in protocol["directions"]:
        case = f"{location}_{source_modality}__{location}_{target_modality}"
        source_public = cohort(roles["public_people"], location, source_modality)
        target_public = cohort(roles["public_people"], location, target_modality)
        if not (np.array_equal(source_public["subjects"], target_public["subjects"])
                and np.array_equal(source_public["bins"], target_public["bins"])):
            raise RuntimeError("public pairing mismatch")
        fit, held = split_public(source_public["subjects"], protocol["public_seed"],
                                 protocol["public_fit_per_person"], protocol["public_held_per_person"])
        map_path = maps_dir / f"{case}.pkl"
        if map_path.exists():
            with map_path.open("rb") as stream:
                mapping = pickle.load(stream)
        else:
            mapping = fit_maps(source_public["x"][fit], target_public["x"][fit],
                               source_public["x"][held], target_public["x"][held], public_protocol)
            with map_path.open("wb") as stream:
                pickle.dump(mapping, stream, protocol=4)
        source_fit = mapping["full"].source(source_public["x"][fit])
        source_held = mapping["full"].source(source_public["x"][held])
        gate = normalized_covariance_distance(source_fit, source_held)
        prepared.append({"case": case, "location": location, "source_modality": source_modality,
                         "target_modality": target_modality, "map": mapping, "map_path": map_path,
                         "source_fit": source_fit, "source_held": source_held, "gate": gate,
                         "source_public": source_public, "target_public": target_public,
                         "fit": fit, "held": held})
        print("PAMAP2 public map", case, "gate", round(gate, 6), flush=True)
    route = "stable" if max(item["gate"] for item in prepared) < protocol["gate"]["threshold"] else "unstable"
    (OUT / "prequery_route.json").write_text(json.dumps({
        "direction_statistics": {item["case"]: item["gate"] for item in prepared},
        "maximum": max(item["gate"] for item in prepared), "threshold": protocol["gate"]["threshold"],
        "route": route, "private_or_target_labels_used": False,
    }, indent=2), encoding="utf-8")
    return prepared, route


def run(protocol):
    prepared, route = prepare(protocol)
    ledger = OUT / "results.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()] if ledger.exists() else []
    done = {row["id"] for row in rows}
    predictions = OUT / "predictions"; predictions.mkdir(exist_ok=True)
    stable, unstable = protocol["stable_branch"], protocol["unstable_branch"]
    roles = protocol["roles"]
    for item in prepared:
        location, sm, tm = item["location"], item["source_modality"], item["target_modality"]
        mapping, point = item["map"]["full"], item["map"]["point"]
        source = cohort(roles["source_people"], location, sm, include_labels=True)
        adaptation = cohort(roles["adaptation_people"], location, tm)
        test = cohort(roles["test_people"], location, tm, include_labels=True)
        source_ids = cap(len(source["x"]), protocol["source_cap"], protocol["source_seed"])
        adaptation_ids = cap(len(adaptation["x"]), protocol["adaptation_cap"], protocol["adaptation_seed"])
        private = mapping.source(source["x"][source_ids])
        adaptation_x = mapping.target(adaptation["x"][adaptation_ids])
        test_x = mapping.target(test["x"])
        if route == "stable":
            operators = public_operators(point, item["target_public"]["x"][item["fit"]],
                                         item["target_public"]["x"][item["held"]],
                                         adaptation["x"][adaptation_ids], test["x"])
        else:
            anchors = public_anchors(item["source_fit"], unstable["anchors"])
        for epsilon, seed in itertools.product(protocol["budgets"], protocol["seeds"]):
            ident = f"{item['case']}_e{epsilon}_s{seed}"
            if ident in done:
                continue
            if route == "stable":
                raw, sigma = joint_release(private, source["y"][source_ids], protocol["classes"], epsilon, seed)
                weight, mean, teacher_diagnostic = mixture_fit(item["source_fit"], raw, len(private), stable["ridge"], 1.0)
                state = entropic_calibrated(operators, item["source_fit"] - mean,
                                             item["source_held"] - mean, weight,
                                             stable["teacher_temperature"])
                prior = np.maximum(raw[:, -1] * len(private), 1.0); prior /= prior.sum()
                correction, transport = offset(state["adaptation_log"], stable["receiver_temperature"],
                                               stable["balance_strength"], prior)
                probability = predict(state["validation_log"], correction, stable["receiver_temperature"])
                probability = temperature_scale(probability, stable["final_temperature"])
                diagnostic = {"teacher": teacher_diagnostic, "transport": transport}
            else:
                raw, sigma = anchor_release(private, source["y"][source_ids], protocol["classes"], anchors, epsilon, seed)
                means, within, prior = anchor_decode(raw, len(private), anchors, sigma, unstable["uncertainty_scale"])
                adaptation_log = log_probability(teacher_probability(
                    adaptation_x, means, within, prior, unstable["teacher_temperature"], unstable["norm_correction"]))
                test_log = log_probability(teacher_probability(
                    test_x, means, within, prior, unstable["teacher_temperature"], unstable["norm_correction"]))
                correction, transport = offset(adaptation_log, unstable["receiver_temperature"],
                                               unstable["balance_strength"], prior)
                probability = predict(test_log, correction, unstable["receiver_temperature"])
                probability = temperature_scale(probability, unstable["final_temperature"])
                diagnostic = {"transport": transport}
            result = evaluate_metrics(test["y"], probability, test["subjects"], protocol["classes"])
            output = predictions / f"{ident}.npz"
            np.savez_compressed(output, probability=probability, y=test["y"], subjects=test["subjects"], bins=test["bins"])
            row = {"id": ident, "case_id": item["case"], "method": "PriFCT-v127", "route": route,
                   "epsilon": epsilon, "seed": seed, "sigma": sigma, "public_covariance_distance": item["gate"],
                   "results": result, "diagnostic": diagnostic, "status": "success",
                   "protocol_sha256": sha(OUT / "protocol.json"), "map_sha256": sha(item["map_path"]),
                   "prediction_sha256": sha(output)}
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            rows.append(row); done.add(ident)
            print("PAMAP2 PriFCT", ident, round(result["subject_balanced_accuracy"], 6), flush=True)
    expected = len(protocol["directions"]) * len(protocol["budgets"]) * len(protocol["seeds"])
    if len(rows) != expected:
        raise RuntimeError(f"incomplete PriFCT ledger {len(rows)}/{expected}")
    summary = {"complete": True, "outcomes": len(rows), "failures": 0, "route": route,
               "metrics": {endpoint: float(np.mean([row["results"][endpoint] for row in rows])) for endpoint in ENDPOINTS},
               "no_pamap2_outcome_selection": True,
               "claim_status": "Independent PriFCT outcomes exposed; comparator phase must use the already frozen settings."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(1):
        run(register())
