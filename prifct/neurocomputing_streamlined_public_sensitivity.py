"""Receiver-stage public-pair sensitivity for streamlined UCI HAR PriFCT.

The frozen public feature map is held fixed.  Each direction/seed uses one
epsilon=2 packet for every public-pair scenario, so pair-count and public-person
composition changes are postprocessing comparisons rather than extra private
queries.  This audit isolates the covariance/probability-transfer stage; it is
not a full public-map retraining study.
"""
from __future__ import annotations

import json
import sys
import types

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
from prifct.v7_public_quadratic import public_covariance
from prifct.v7_spectral_contrast import reconstruct
from revision.joint_release import release_joint


EPSILON = 2.0
SEEDS = [91331, 91332, 91333]
PER_PERSON_SIZES = [12, 24, 48, 96]
OUT = base.OUT / "streamlined_public_sensitivity.json"
CALIBRATION = (
    base.BASE
    / "semantic_probability_calibration/maxent_v19_development/summary.json"
)


def balanced_subset(arrays, per_person: int, seed: int) -> np.ndarray:
    chosen = []
    rng = np.random.default_rng(seed)
    fit_set = set(arrays["fit"].tolist())
    for person in arrays["roles_by_name"]["public"]:
        eligible = np.array(
            sorted(fit_set.intersection(np.flatnonzero(arrays["subjects"] == person)))
        )
        chosen.extend(rng.permutation(eligible)[:per_person])
    return np.asarray(chosen, dtype=int)


def main() -> None:
    protocol = base.register()
    arrays = base.data(protocol)
    arrays["roles_by_name"] = protocol["roles"]
    source = arrays["roles"]["source"]
    adaptation = arrays["roles"]["adaptation"]
    test = arrays["roles"]["test"]
    options = protocol["options"]["Ridge source teacher"]
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))["settings"][
        "transfer"
    ]["Ridge source teacher"]["selected"]["options"]
    rows = []

    for source_view, target_view in protocol["directions"]:
        bridges = base.maps(arrays, source_view, target_view)
        mapping, point = bridges["full"], bridges["point"]
        private_source = mapping.source(arrays[source_view][source])
        target_adaptation = arrays[target_view][adaptation]
        target_test = arrays[target_view][test]
        scenarios = []
        for per_person in PER_PERSON_SIZES:
            scenarios.append(
                (
                    f"balanced_{6 * per_person}",
                    balanced_subset(arrays, per_person, 91600 + per_person),
                    arrays["held"],
                    "pair_count",
                )
            )
        for excluded in protocol["roles"]["public"]:
            fit = arrays["fit"][arrays["subjects"][arrays["fit"]] != excluded]
            held = arrays["held"][arrays["subjects"][arrays["held"]] != excluded]
            scenarios.append(
                (
                    f"leave_public_person_{excluded}",
                    fit,
                    held,
                    "public_person_composition",
                )
            )

        for seed in SEEDS:
            packet = release_joint(
                private_source, arrays["y"][source], 6, EPSILON, seed=seed
            )
            prior = np.maximum(packet.raw[:, -1] * len(source), 1.0)
            prior /= prior.sum()
            for name, fit, held, scenario_type in scenarios:
                public_source = mapping.source(arrays[source_view][fit])
                public_held = mapping.source(arrays[source_view][held])
                covariance = public_covariance(public_source)
                weight, _ = reconstruct(
                    packet.raw[:, :-1],
                    covariance,
                    packet.sigma_query,
                    options["ridge"],
                    no_denoising=True,
                )
                weight = np.r_[weight, np.zeros((1, 6))]
                mean = np.zeros(public_source.shape[1])
                operators = public_operators(
                    point,
                    arrays[target_view][fit],
                    arrays[target_view][held],
                    target_adaptation,
                    target_test,
                )
                state = calibrated(
                    operators,
                    features(public_source, mean),
                    features(public_held, mean),
                    weight,
                    options["teacher_temperature"],
                )
                receiver_offset, diagnostic = offset(
                    log_probability(state["adaptation"]),
                    options["receiver_temperature"],
                    options["balance_strength"],
                    prior,
                    None,
                )
                probability = predict(
                    log_probability(state["validation"]),
                    receiver_offset,
                    options["receiver_temperature"],
                )
                probability = apply_probability_calibration(probability, calibration)
                metrics = score(
                    arrays["y"][test], probability, arrays["subjects"][test], 6
                )
                rows.append(
                    {
                        "direction": f"{source_view}_{target_view}",
                        "seed": seed,
                        "scenario": name,
                        "scenario_type": scenario_type,
                        "fit_pairs": int(len(fit)),
                        "held_pairs": int(len(held)),
                        "excluded_public_person": int(name.rsplit("_", 1)[-1])
                        if scenario_type == "public_person_composition"
                        else None,
                        "metrics": metrics,
                        "offset_diagnostic": diagnostic,
                    }
                )
                print(
                    source_view,
                    target_view,
                    seed,
                    name,
                    metrics["subject_balanced_accuracy"],
                    flush=True,
                )

    summary = {}
    for name in sorted({row["scenario"] for row in rows}):
        selected = [row for row in rows if row["scenario"] == name]
        summary[name] = {
            "fit_pairs": selected[0]["fit_pairs"],
            "scenario_type": selected[0]["scenario_type"],
            **{
                key: float(np.mean([row["metrics"][key] for row in selected]))
                for key in [
                    "subject_balanced_accuracy",
                    "macro_f1",
                    "nll",
                    "brier",
                    "ece",
                ]
            },
        }
    payload = {
        "scope": "Frozen-map UCI HAR receiver-stage public-pair sensitivity",
        "privacy": "One epsilon=2 packet per direction/seed is reused across all public scenarios",
        "limitation": "Public feature map is not refit; this isolates covariance and probability-transfer resources",
        "public_labels": 0,
        "rows": rows,
        "summary": summary,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
