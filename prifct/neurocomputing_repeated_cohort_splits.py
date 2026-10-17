"""Repeated source/public-person split audit on fixed UCI HAR test roles.

The registered adaptation and test people stay fixed.  The original sixteen
public/source people are repartitioned into six public and ten source people
under four prespecified seeds, plus the registered split.  PriFCT, the strongest
private main-table baseline (DP-NTK), and the strongest BA baseline (KPG-RL)
are rerun without split-specific tuning.  Every outcome, including failures,
is retained.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import traceback
import types

import numpy as np
from sklearn.linear_model import LogisticRegression

try:
    import cvxpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cvxpy"] = types.ModuleType("cvxpy")

from prifct import v47_uci_har as base
from prifct.calibrate_semantic_probabilities import apply as apply_probability_calibration
from prifct.neurocomputing_streamlined_fullmap_sensitivity import fit_maps
from prifct.v14_semantic_functions import calibrated, log_probability, offset, predict, public_operators
from prifct.v15_maxent_teacher import features
from prifct.v6_development_pilot import score
from prifct.v7_core_checkpoints import digest
from prifct.v7_dp_ntk import NTKFeatureMap
from prifct.v7_generator_weighted import fit_receiver
from prifct.v7_native_structural import barycentric_classifier, kpg
from prifct.v7_public_quadratic import public_covariance
from prifct.v7_spectral_contrast import reconstruct
from prifct.v7_streamed_private_features import release
from revision.joint_release import release_joint


EPSILON = 2.0
RELEASE_SEEDS = [91331, 91332, 91333]
REDRAW_SEEDS = [91401, 91402, 91403, 91404]
OUT_DIR = base.OUT / "repeated_cohort_splits"
LEDGER = OUT_DIR / "results.jsonl"
SUMMARY = base.OUT / "streamlined_repeated_cohort_splits.json"
CALIBRATION = base.BASE / "semantic_probability_calibration/maxent_v19_development/summary.json"
METRICS = ["subject_balanced_accuracy", "macro_f1", "nll", "brier", "ece"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append(row):
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def public_indices(arrays, people, split_seed):
    fit, held = [], []
    rng = np.random.default_rng(split_seed)
    for person in people:
        eligible = np.flatnonzero(arrays["subjects"] == person)
        chosen = rng.permutation(eligible)
        fit.extend(chosen[:96]); held.extend(chosen[96:120])
    return np.asarray(fit, dtype=int), np.asarray(held, dtype=int)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "models").mkdir(exist_ok=True)
    (OUT_DIR / "runs").mkdir(exist_ok=True)
    protocol = base.register()
    arrays = base.data(protocol)
    adaptation, test = arrays["roles"]["adaptation"], arrays["roles"]["test"]
    cohort_pool = np.asarray(protocol["roles"]["public"] + protocol["roles"]["source"], dtype=int)
    splits = [("registered", protocol["roles"]["public"], protocol["roles"]["source"], 91311)]
    for seed in REDRAW_SEEDS:
        order = np.random.default_rng(seed).permutation(cohort_pool).tolist()
        splits.append((f"redraw_{seed}", order[:6], order[6:], seed + 10))
    calibration = json.loads(CALIBRATION.read_text(encoding="utf-8"))["settings"]["transfer"]
    rows = [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines()] if LEDGER.exists() else []
    done = {row["id"] for row in rows}

    for split_name, public_people, source_people, public_seed in splits:
        fit, held = public_indices(arrays, public_people, public_seed)
        source = np.flatnonzero(np.isin(arrays["subjects"], source_people))
        for source_view, target_view in protocol["directions"]:
            bridges = fit_maps(arrays, source_view, target_view, fit, held, f"cohort_{split_name}")

            # Streamlined PriFCT: one scenario-specific private packet per seed.
            mapping, point = bridges["full"], bridges["point"]
            sx, sh = mapping.source(arrays[source_view][fit]), mapping.source(arrays[source_view][held])
            x = mapping.source(arrays[source_view][source])
            options = protocol["options"]["Ridge source teacher"]
            covariance = public_covariance(sx)
            operators = public_operators(point, arrays[target_view][fit], arrays[target_view][held], arrays[target_view][adaptation], arrays[target_view][test])
            for seed in RELEASE_SEEDS:
                ident = f"{split_name}_{source_view}_{target_view}_PriFCT_s{seed}"
                if ident not in done:
                    row = {"id": ident, "split": split_name, "public_people": public_people, "source_people": source_people, "direction": f"{source_view}_{target_view}", "method": "PriFCT", "epsilon": EPSILON, "seed": seed}
                    try:
                        packet = release_joint(x, arrays["y"][source], 6, EPSILON, seed=seed)
                        prior = np.maximum(packet.raw[:, -1] * len(source), 1.0); prior /= prior.sum()
                        weight, _ = reconstruct(packet.raw[:, :-1], covariance, packet.sigma_query, options["ridge"], no_denoising=True)
                        weight = np.r_[weight, np.zeros((1, 6))]
                        state = calibrated(operators, features(sx, np.zeros(sx.shape[1])), features(sh, np.zeros(sx.shape[1])), weight, options["teacher_temperature"])
                        receiver_offset, diagnostic = offset(log_probability(state["adaptation"]), options["receiver_temperature"], options["balance_strength"], prior, None)
                        probability = predict(log_probability(state["validation"]), receiver_offset, options["receiver_temperature"])
                        probability = apply_probability_calibration(probability, calibration["Ridge source teacher"]["selected"]["options"])
                        row.update(status="success", metrics=score(arrays["y"][test], probability, arrays["subjects"][test], 6), probability_sha256=digest(probability), offset_diagnostic=diagnostic)
                    except Exception as error:
                        row.update(status="failure", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                    append(row); rows.append(row); done.add(ident)
                    print(ident, row.get("metrics", {}).get("subject_balanced_accuracy", row["status"]), flush=True)

            # DP-NTK under the identical split and epsilon.
            selection = protocol["private_baselines"]["DP-NTK"]
            ntk_options = selection["options"]
            ntk_mapping = bridges["full" if ntk_options["representation"] == "function" else "point"]
            ntk_x = ntk_mapping.source(arrays[source_view][source])
            ntk_test = ntk_mapping.target(arrays[target_view][test])
            public = ntk_mapping.source(arrays[source_view][fit])
            feature = NTKFeatureMap(ntk_x.shape[1], 6, hidden=ntk_options["hidden"], seed=77271)
            for seed in RELEASE_SEEDS:
                ident = f"{split_name}_{source_view}_{target_view}_DP-NTK_s{seed}"
                if ident not in done:
                    row = {"id": ident, "split": split_name, "public_people": public_people, "source_people": source_people, "direction": f"{source_view}_{target_view}", "method": "DP-NTK", "epsilon": EPSILON, "seed": seed}
                    try:
                        packet = release(ntk_x, arrays["y"][source], 6, feature, EPSILON, seed=seed, device="cuda")
                        generator = fit_receiver(packet, feature, steps=ntk_options["steps"], hidden=128, samples_per_class=16, seed=77281, device="cuda", loss_weights=np.ones(packet.raw.shape[1] - 1))
                        synthetic, labels = generator.sample(n_per_class=128, seed=77291)
                        classifier = LogisticRegression(C=1.0, multi_class="multinomial", max_iter=1000).fit(synthetic, labels)
                        probability = classifier.predict_proba(ntk_test)
                        probability = apply_probability_calibration(probability, calibration["DP-NTK"]["selected"]["options"])
                        row.update(status="success", metrics=score(arrays["y"][test], probability, arrays["subjects"][test], 6), probability_sha256=digest(probability))
                        del generator
                    except Exception as error:
                        row.update(status="failure", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                    append(row); rows.append(row); done.add(ident)
                    print(ident, row.get("metrics", {}).get("subject_balanced_accuracy", row["status"]), flush=True)

            # KPG-RL, one deterministic nonprivate fit per split/direction.
            ident = f"{split_name}_{source_view}_{target_view}_KPG-RL"
            if ident not in done:
                selection = protocol["context_baselines"]["KPG-RL"]
                kpg_options = selection["options"]
                kpg_mapping = bridges["full" if kpg_options["representation"] == "function" else "point"]
                row = {"id": ident, "split": split_name, "public_people": public_people, "source_people": source_people, "direction": f"{source_view}_{target_view}", "method": "KPG-RL", "epsilon": None, "seed": 91361}
                try:
                    kx = kpg_mapping.source(arrays[source_view][source])
                    ka = kpg_mapping.target(arrays[target_view][adaptation])
                    kt = kpg_mapping.target(arrays[target_view][test])
                    settings = {key: value for key, value in kpg_options.items() if key not in ["representation", "classifier_C"]}
                    indices = np.random.default_rng(91363).permutation(fit)[:32]
                    plan, diagnostic = kpg(kx, ka, kpg_mapping.source(arrays[source_view][indices]), kpg_mapping.target(arrays[target_view][indices]), **settings)
                    classifier, head = barycentric_classifier(plan, ka, arrays["y"][source], classes=6, C=kpg_options["classifier_C"], seed=91364)
                    probability = classifier.predict_proba(kt)
                    probability = apply_probability_calibration(probability, calibration["KPG-RL"]["selected"]["options"])
                    row.update(status="success", metrics=score(arrays["y"][test], probability, arrays["subjects"][test], 6), probability_sha256=digest(probability), diagnostic=diagnostic)
                    del plan
                except Exception as error:
                    row.update(status="failure", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
                append(row); rows.append(row); done.add(ident)
                print(ident, row.get("metrics", {}).get("subject_balanced_accuracy", row["status"]), flush=True)

    expected = len(splits) * len(protocol["directions"]) * (2 * len(RELEASE_SEEDS) + 1)
    by_split_method = {}
    for split_name, *_ in splits:
        by_split_method[split_name] = {}
        for method in ["PriFCT", "DP-NTK", "KPG-RL"]:
            selected = [row for row in rows if row["split"] == split_name and row["method"] == method]
            successful = [row for row in selected if row["status"] == "success"]
            item = {"outcomes": len(selected), "successes": len(successful), "failures": len(selected) - len(successful)}
            if successful:
                item.update({metric: float(np.mean([row["metrics"][metric] for row in successful])) for metric in METRICS})
            by_split_method[split_name][method] = item
    payload = {"scope": "Repeated public/source participant repartitions with fixed adaptation/test people", "split_count": len(splits), "redraw_seeds": REDRAW_SEEDS, "epsilon_private_methods": EPSILON, "expected_outcomes": expected, "failures": sum(row["status"] != "success" for row in rows), "rows": rows, "by_split_method": by_split_method}
    SUMMARY.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(by_split_method, indent=2), flush=True)


if __name__ == "__main__":
    main()
