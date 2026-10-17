"""End-to-end public-map sensitivity for streamlined UCI HAR PriFCT.

Unlike the frozen-map receiver audit, every scenario refits both the point and
functional public maps using only that scenario's unlabelled paired records.
Because the source query changes with the refitted map, each scenario receives
its own epsilon=2 Gaussian packet.  Results therefore measure the complete
fixed algorithm under public-resource perturbations; they are not a same-packet
postprocessing comparison and are never combined as an ensemble.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np

try:
    import cvxpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cvxpy"] = types.ModuleType("cvxpy")

from prifct import v47_uci_har as base
from prifct.calibrate_semantic_probabilities import apply as apply_probability_calibration
from prifct.v14_semantic_functions import calibrated, log_probability, offset, predict, public_operators
from prifct.v15_maxent_teacher import features
from prifct.v6_development_pilot import score
from prifct.v7_anchored_distribution import ProjectedSignature
from prifct.v7_functional_bridge import fit_functional_bridge
from prifct.v7_public_bridge import fit_public_bridge
from prifct.v7_public_quadratic import public_covariance
from prifct.v7_semantic_transport import MultiScaleSignature
from prifct.v7_spectral_contrast import reconstruct
from revision.joint_release import release_joint


EPSILON = 2.0
SEEDS = [91331, 91332, 91333]
PER_PERSON_SIZES = [12, 24, 48, 96]
OUT_DIR = base.OUT / "fullmap_public_sensitivity"
LEDGER = OUT_DIR / "results.jsonl"
SUMMARY = base.OUT / "streamlined_fullmap_public_sensitivity.json"
CALIBRATION = (
    base.BASE
    / "semantic_probability_calibration/maxent_v19_development/summary.json"
)
METRICS = ["subject_balanced_accuracy", "macro_f1", "nll", "brier", "ece"]


def stable_token(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def balanced_subset(arrays, per_person: int, seed: int) -> np.ndarray:
    chosen = []
    rng = np.random.default_rng(seed)
    fit_set = set(arrays["fit"].tolist())
    for person in arrays["roles_by_name"]["public"]:
        eligible = np.array(sorted(fit_set.intersection(np.flatnonzero(arrays["subjects"] == person))))
        chosen.extend(rng.permutation(eligible)[:per_person])
    return np.asarray(chosen, dtype=int)


def fit_maps(arrays, source_view: str, target_view: str, fit: np.ndarray,
             held: np.ndarray, scenario: str):
    folder = OUT_DIR / "public_maps"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source_view}_{target_view}_{scenario}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)
    sx, tx = arrays[source_view][fit], arrays[target_view][fit]
    sh, th = arrays[source_view][held], arrays[target_view][held]
    point_choices = []
    for options in base.OPTIONS:
        point = fit_public_bridge(sx, tx, rank=32, landmarks=256, seed=91312, **options)
        error = float(np.mean(np.sum((point.target(th) - point.source(sh)) ** 2, axis=1)))
        point_choices.append((error, point))
    point = min(point_choices, key=lambda item: item[0])[1]
    original = MultiScaleSignature.public_fit(point.source(sx), seed=91313)
    signature = ProjectedSignature.public_fit(original, point.source(sx), rank=32)
    full_choices = []
    for options in base.OPTIONS:
        full = fit_functional_bridge(point, signature, sx, tx, **options)
        error = float(np.mean(np.sum((full.target(th) - full.source(sh)) ** 2, axis=1)))
        full_choices.append((error, full))
    result = {"point": point, "full": min(full_choices, key=lambda item: item[0])[1]}
    with path.open("wb") as handle:
        pickle.dump(result, handle, protocol=4)
    return result


def read_rows():
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def append(row):
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    protocol = base.register()
    arrays = base.data(protocol)
    arrays["roles_by_name"] = protocol["roles"]
    source = arrays["roles"]["source"]
    adaptation = arrays["roles"]["adaptation"]
    test = arrays["roles"]["test"]
    options = protocol["options"]["Ridge source teacher"]
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))["settings"]["transfer"]["Ridge source teacher"]["selected"]["options"]
    existing = {row["id"] for row in read_rows()}

    scenarios = []
    for per_person in PER_PERSON_SIZES:
        scenarios.append((f"balanced_{6 * per_person}", balanced_subset(arrays, per_person, 91600 + per_person), arrays["held"], "pair_count"))
    # Preserve the originally registered ordering as the full-resource anchor.
    # balanced_576 above deliberately remains in the ledger as a same-set
    # order-perturbation audit rather than being overwritten or discarded.
    scenarios.append(("canonical_576", arrays["fit"], arrays["held"], "pair_count_anchor"))
    for excluded in protocol["roles"]["public"]:
        fit = arrays["fit"][arrays["subjects"][arrays["fit"]] != excluded]
        held = arrays["held"][arrays["subjects"][arrays["held"]] != excluded]
        scenarios.append((f"leave_public_person_{excluded}", fit, held, "public_person_composition"))

    for source_view, target_view in protocol["directions"]:
        for scenario, fit, held, scenario_type in scenarios:
            bridges = fit_maps(arrays, source_view, target_view, fit, held, scenario)
            mapping, point = bridges["full"], bridges["point"]
            public_source = mapping.source(arrays[source_view][fit])
            public_held = mapping.source(arrays[source_view][held])
            private_source = mapping.source(arrays[source_view][source])
            covariance = public_covariance(public_source)
            operators = public_operators(point, arrays[target_view][fit], arrays[target_view][held], arrays[target_view][adaptation], arrays[target_view][test])
            for seed in SEEDS:
                ident = f"{source_view}_{target_view}_{scenario}_s{seed}"
                if ident in existing:
                    continue
                try:
                    packet = release_joint(private_source, arrays["y"][source], 6, EPSILON, seed=seed)
                    prior = np.maximum(packet.raw[:, -1] * len(source), 1.0)
                    prior /= prior.sum()
                    weight, _ = reconstruct(packet.raw[:, :-1], covariance, packet.sigma_query, options["ridge"], no_denoising=True)
                    weight = np.r_[weight, np.zeros((1, 6))]
                    state = calibrated(operators, features(public_source, np.zeros(public_source.shape[1])), features(public_held, np.zeros(public_source.shape[1])), weight, options["teacher_temperature"])
                    receiver_offset, diagnostic = offset(log_probability(state["adaptation"]), options["receiver_temperature"], options["balance_strength"], prior, None)
                    probability = predict(log_probability(state["validation"]), receiver_offset, options["receiver_temperature"])
                    probability = apply_probability_calibration(probability, calibration)
                    metrics = score(arrays["y"][test], probability, arrays["subjects"][test], 6)
                    query_path = OUT_DIR / f"query_{ident}.npz"
                    np.savez_compressed(query_path, raw=packet.raw, sigma=packet.sigma_query, n_public=len(source))
                    row = {"id": ident, "status": "success", "direction": f"{source_view}_{target_view}", "seed": seed, "scenario": scenario, "scenario_type": scenario_type, "fit_pairs": int(len(fit)), "held_pairs": int(len(held)), "excluded_public_person": int(scenario.rsplit("_", 1)[-1]) if scenario_type == "public_person_composition" else None, "metrics": metrics, "offset_diagnostic": diagnostic, "query_sha256": hashlib.sha256(query_path.read_bytes()).hexdigest()}
                except Exception as error:  # retained in the ledger; never success-filtered silently
                    row = {"id": ident, "status": "failure", "direction": f"{source_view}_{target_view}", "seed": seed, "scenario": scenario, "scenario_type": scenario_type, "fit_pairs": int(len(fit)), "held_pairs": int(len(held)), "error": repr(error)}
                append(row)
                existing.add(ident)
                print(ident, row.get("metrics", {}).get("subject_balanced_accuracy", row["status"]), flush=True)

    rows = read_rows()
    summary = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        selected = [row for row in rows if row["scenario"] == scenario]
        successful = [row for row in selected if row["status"] == "success"]
        summary[scenario] = {"fit_pairs": selected[0]["fit_pairs"], "scenario_type": selected[0]["scenario_type"], "outcomes": len(selected), "successes": len(successful), "failures": len(selected) - len(successful)}
        if successful:
            summary[scenario].update({metric: float(np.mean([row["metrics"][metric] for row in successful])) for metric in METRICS})
    payload = {"scope": "Full public-map refitting sensitivity on fixed UCI HAR cohorts", "privacy": "Every scenario changes the query function and therefore uses its own epsilon=2 packet; scenarios are not jointly disclosed or ensembled", "ordering_note": "canonical_576 preserves the registered fit-array order; balanced_576 retains a complete-set order perturbation and is not discarded", "public_labels": 0, "expected_outcomes": 2 * len(scenarios) * len(SEEDS), "failures": sum(row["status"] != "success" for row in rows), "rows": rows, "summary": summary}
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
