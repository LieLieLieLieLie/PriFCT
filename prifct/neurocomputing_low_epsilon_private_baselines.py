"""Low-epsilon UCI HAR audit for the three privacy-matched baselines.

All settings are inherited unchanged from the frozen v47 protocol.  Failures
are retained in the ledger; no fallback prediction or success-only mean is
allowed.  Each row is one direction, privacy budget and independent release.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
import traceback
import types
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    import cvxpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cvxpy"] = types.ModuleType("cvxpy")

from prifct import v47_uci_har as base
from prifct.calibrate_semantic_probabilities import apply as apply_probability_calibration
from prifct.v6_development_pilot import score
from prifct.v6_private_generation import PrivateFeatureMap
from prifct.v7_dp_kip import predict as predict_kip
from prifct.v7_dp_ntk import NTKFeatureMap
from prifct.v7_generator_weighted import fit_receiver
from prifct.v7_large_cohort_kip import fit as fit_kip
from prifct.v7_streamed_private_features import release


BUDGETS = [0.1, 0.25, 0.5, 1.0]
SEEDS = [91331, 91332, 91333]
METHODS = ["DP-HP", "DP-NTK", "DP-KIP"]
OUT = base.OUT / "low_epsilon_private_baselines"
LEDGER = OUT / "results.jsonl"
SUMMARY = base.OUT / "low_epsilon_private_baselines.json"
CALIBRATION = (
    base.BASE
    / "semantic_probability_calibration/maxent_v19_development/summary.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append(row: dict) -> None:
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    protocol = base.register()
    arrays = base.data(protocol)
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))["settings"][
        "transfer"
    ]
    rows = [json.loads(line) for line in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    done = {row["id"] for row in rows}
    source, test = arrays["roles"]["source"], arrays["roles"]["test"]

    for source_view, target_view in protocol["directions"]:
        bridges = base.maps(arrays, source_view, target_view)
        for method in METHODS:
            selection = protocol["private_baselines"][method]
            options = selection["options"]
            mapping = bridges["full" if options["representation"] == "function" else "point"]
            x = mapping.source(arrays[source_view][source])
            xt = mapping.target(arrays[target_view][test])
            feature = None
            if method != "DP-KIP":
                public = mapping.source(arrays[source_view][arrays["fit"]])
                chosen = np.random.default_rng(91351).permutation(len(public))[:256]
                distances = np.sum(
                    (public[chosen, None, :] - public[None, chosen, :]) ** 2, axis=2
                )
                width = max(
                    float(np.median(distances[np.triu_indices(256, 1)])), 1e-4
                )
                feature = (
                    PrivateFeatureMap.hermite(
                        x.shape[1],
                        width,
                        order=options["order"],
                        product_order=options["product_order"],
                    )
                    if method == "DP-HP"
                    else NTKFeatureMap(
                        x.shape[1], 6, hidden=options["hidden"], seed=77271
                    )
                )

            for epsilon in BUDGETS:
                for seed in SEEDS:
                    ident = f"{source_view}_{target_view}_{method}_e{epsilon}_s{seed}"
                    if ident in done:
                        continue
                    row = {
                        "id": ident,
                        "method": method,
                        "direction": f"{source_view}_{target_view}",
                        "epsilon": epsilon,
                        "seed": seed,
                        "options": options,
                        "status": "running",
                    }
                    try:
                        model_path = OUT / "models" / f"{ident}.pkl"
                        if method == "DP-KIP":
                            settings = {
                                key: value
                                for key, value in options.items()
                                if key
                                not in [
                                    "representation",
                                    "initialization",
                                    "temperature",
                                    "calibration",
                                ]
                            }
                            scale = (
                                1.0
                                if options["initialization"] == "native_gaussian"
                                else 1 / math.sqrt(x.shape[1])
                            )
                            model = fit_kip(
                                x,
                                arrays["y"][source],
                                6,
                                epsilon=epsilon,
                                seed=seed,
                                initialization_scale=scale,
                                accountant="gdp",
                                **settings,
                            )
                            probability = predict_kip(
                                model, xt, options["temperature"]
                            )
                            with model_path.open("wb") as handle:
                                pickle.dump(model, handle, protocol=4)
                        else:
                            packet = release(
                                x,
                                arrays["y"][source],
                                6,
                                feature,
                                epsilon,
                                seed=seed,
                                device="cuda",
                            )
                            weights = np.ones(packet.raw.shape[1] - 1)
                            if method == "DP-HP":
                                weights[x.shape[1] * (options["order"] + 1) :] = options[
                                    "gamma"
                                ]
                            generator = fit_receiver(
                                packet,
                                feature,
                                steps=options["steps"],
                                hidden=128,
                                samples_per_class=16,
                                seed=77281,
                                device="cuda",
                                loss_weights=weights,
                            )
                            synthetic, labels = generator.sample(
                                n_per_class=128, seed=77291
                            )
                            classifier = LogisticRegression(
                                C=1.0, multi_class="multinomial", max_iter=1000
                            ).fit(synthetic, labels)
                            probability = classifier.predict_proba(xt)
                            with model_path.open("wb") as handle:
                                pickle.dump(
                                    {
                                        "classifier": classifier,
                                        "synthetic": synthetic,
                                        "labels": labels,
                                    },
                                    handle,
                                    protocol=4,
                                )
                        probability = np.asarray(probability, dtype=np.float64)
                        probability /= probability.sum(axis=1, keepdims=True)
                        calibrated_probability = apply_probability_calibration(
                            probability,
                            calibration[method]["selected"]["options"],
                        )
                        run_path = OUT / "runs" / f"{ident}.npz"
                        np.savez_compressed(
                            run_path,
                            probability=probability,
                            calibrated_probability=calibrated_probability,
                            y=arrays["y"][test],
                            subjects=arrays["subjects"][test],
                        )
                        row.update(
                            status="success",
                            raw=score(
                                arrays["y"][test],
                                probability,
                                arrays["subjects"][test],
                                6,
                            ),
                            calibrated=score(
                                arrays["y"][test],
                                calibrated_probability,
                                arrays["subjects"][test],
                                6,
                            ),
                            model_sha256=sha(model_path),
                            run_sha256=sha(run_path),
                        )
                        print(
                            ident,
                            row["calibrated"]["subject_balanced_accuracy"],
                            flush=True,
                        )
                    except Exception as error:  # retain every failure
                        row.update(
                            status="failure",
                            error=f"{type(error).__name__}: {error}",
                            traceback=traceback.format_exc(),
                        )
                        print(ident, row["error"], flush=True)
                    append(row)
                    rows.append(row)
                    done.add(ident)

    expected = len(protocol["directions"]) * len(METHODS) * len(BUDGETS) * len(SEEDS)
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} rows, found {len(rows)}")
    by_method_budget = {}
    for method in METHODS:
        by_method_budget[method] = {}
        for epsilon in BUDGETS:
            selected = [
                row
                for row in rows
                if row["method"] == method and row["epsilon"] == epsilon
            ]
            successful = [row for row in selected if row["status"] == "success"]
            item = {
                "outcomes": len(selected),
                "successes": len(successful),
                "failures": len(selected) - len(successful),
            }
            if len(successful) == len(selected):
                item.update(
                    {
                        metric: float(
                            np.mean([row["calibrated"][metric] for row in successful])
                        )
                        for metric in [
                            "subject_balanced_accuracy",
                            "macro_f1",
                            "nll",
                            "brier",
                            "ece",
                        ]
                    }
                )
            by_method_budget[method][str(epsilon)] = item
    payload = {
        "scope": "Fixed-setting UCI HAR low-epsilon privacy-matched baseline audit",
        "budgets": BUDGETS,
        "seeds": SEEDS,
        "expected_outcomes": expected,
        "failures": sum(row["status"] != "success" for row in rows),
        "by_method_budget": by_method_budget,
        "rows": rows,
    }
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(by_method_budget, indent=2), flush=True)


if __name__ == "__main__":
    main()
