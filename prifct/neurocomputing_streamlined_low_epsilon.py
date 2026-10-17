"""Low-epsilon stress test for the streamlined UCI HAR PriFCT receiver.

This is an alternative-release research audit, not a jointly released
ensemble.  It reuses the frozen v47 cohorts, public maps, decoder settings and
target receiver.  No configuration is selected from the results.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import numpy as np

# ``v47_uci_har`` imports a legacy score helper whose unused training module
# mentions CVXPY.  The streamlined audit never calls that solver, so permit a
# minimal import-only shim when CVXPY is absent from the execution environment.
try:
    import cvxpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cvxpy"] = types.ModuleType("cvxpy")

from prifct import v47_uci_har as base
from prifct.v14_semantic_functions import calibrated, log_probability, offset, predict, public_operators
from prifct.v15_maxent_teacher import features
from prifct.v7_core_target import CoreTarget
from prifct.v7_public_quadratic import public_covariance
from prifct.v7_spectral_contrast import reconstruct
from revision.joint_release import release_joint
from prifct.v6_development_pilot import score
from prifct.calibrate_semantic_probabilities import apply as apply_probability_calibration


BUDGETS = [0.1, 0.25, 0.5, 1.0, 2.0, 10.0]
SEEDS = [91331, 91332, 91333]
OUT = base.OUT / "streamlined_low_epsilon.json"
CALIBRATION = (
    base.BASE
    / "semantic_probability_calibration/maxent_v19_development/summary.json"
)


def main() -> None:
    protocol = base.register()
    arrays = base.data(protocol)
    source = arrays["roles"]["source"]
    adaptation = arrays["roles"]["adaptation"]
    test = arrays["roles"]["test"]
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))["settings"][
        "transfer"
    ]["Ridge source teacher"]["selected"]["options"]
    rows = []

    for source_view, target_view in protocol["directions"]:
        bridges = base.maps(arrays, source_view, target_view)
        mapping = bridges["full"]
        point = bridges["point"]
        fit = arrays["fit"]
        held = arrays["held"]
        public_source = mapping.source(arrays[source_view][fit])
        public_held = mapping.source(arrays[source_view][held])
        private_source = mapping.source(arrays[source_view][source])
        target_adaptation = mapping.target(arrays[target_view][adaptation])
        target_test = mapping.target(arrays[target_view][test])
        covariance = public_covariance(public_source)
        geometry = CoreTarget.build(target_adaptation, target_test, 6, 91321)
        operators = public_operators(
            point,
            arrays[target_view][fit],
            arrays[target_view][held],
            arrays[target_view][adaptation],
            arrays[target_view][test],
        )
        options = protocol["options"]["Ridge source teacher"]

        for epsilon in BUDGETS:
            for seed in SEEDS:
                packet = release_joint(
                    private_source,
                    arrays["y"][source],
                    6,
                    epsilon,
                    seed=seed,
                )
                raw = packet.raw
                prior = np.maximum(raw[:, -1] * len(source), 1.0)
                prior /= prior.sum()
                weight, _ = reconstruct(
                    raw[:, :-1],
                    covariance,
                    packet.sigma_query,
                    options["ridge"],
                    no_denoising=True,
                )
                weight = np.r_[weight, np.zeros((1, 6))]
                mean = np.zeros(public_source.shape[1])
                state = calibrated(
                    operators,
                    features(public_source, mean),
                    features(public_held, mean),
                    weight,
                    options["teacher_temperature"],
                )
                adaptation_probability = state["adaptation"]
                test_probability = state["validation"]
                receiver_scores = log_probability(adaptation_probability)
                receiver_offset, diagnostic = offset(
                    receiver_scores,
                    options["receiver_temperature"],
                    options["balance_strength"],
                    prior,
                    None,
                )
                probability = predict(
                    log_probability(test_probability),
                    receiver_offset,
                    options["receiver_temperature"],
                )
                raw_metrics = score(
                    arrays["y"][test], probability, arrays["subjects"][test], 6
                )
                calibrated_probability = apply_probability_calibration(
                    probability, calibration
                )
                calibrated_metrics = score(
                    arrays["y"][test],
                    calibrated_probability,
                    arrays["subjects"][test],
                    6,
                )
                rows.append(
                    {
                        "direction": f"{source_view}_{target_view}",
                        "epsilon": epsilon,
                        "seed": seed,
                        "sigma_query": float(packet.sigma_query),
                        "raw": raw_metrics,
                        "calibrated": calibrated_metrics,
                        "offset_diagnostic": diagnostic,
                    }
                )
                print(
                    source_view,
                    target_view,
                    epsilon,
                    seed,
                    calibrated_metrics["subject_balanced_accuracy"],
                    flush=True,
                )

    by_budget = {}
    for epsilon in BUDGETS:
        selected = [row for row in rows if row["epsilon"] == epsilon]
        by_budget[str(epsilon)] = {
            key: float(np.mean([row["calibrated"][key] for row in selected]))
            for key in ["subject_balanced_accuracy", "macro_f1", "nll", "brier", "ece"]
        }
    payload = {
        "scope": "Fixed UCI HAR low-epsilon stress test of streamlined PriFCT; alternative releases, not an ensemble",
        "selection": "No result-dependent configuration selection; v47 ridge-receiver settings reused unchanged",
        "budgets": BUDGETS,
        "seeds": SEEDS,
        "probability_calibration": calibration,
        "rows": rows,
        "by_budget": by_budget,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(by_budget, indent=2), flush=True)


if __name__ == "__main__":
    main()
