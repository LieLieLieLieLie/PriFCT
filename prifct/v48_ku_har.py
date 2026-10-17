"""Prospectively frozen KU-HAR participant-disjoint validation."""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import pickle
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

# The reused v47 module imports a legacy development-only robust solver solely
# to obtain its metric helper.  CVXPY is not used anywhere in this validation,
# so keep that optional dependency isolated rather than changing the frozen
# environment or implementation.
if "cvxpy" not in sys.modules:
    sys.modules["cvxpy"] = types.ModuleType("cvxpy")

import prifct.v47_uci_har as uci
from prifct.v7_anchored_distribution import ProjectedSignature
from prifct.v7_functional_bridge import fit_functional_bridge
from prifct.v7_public_bridge import fit_public_bridge
from prifct.v7_semantic_transport import MultiScaleSignature
from prifct.v14_semantic_functions import OPTIONS


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/models/eswa_fedkapt"
OUT = BASE / "semantic_v48_ku_har"
ZIP = ROOT / "data/kuhar/2.Trimmed_interpolated_data.zip"
ROSTER = ["PriFCT", "DP-HP", "DP-NTK", "DP-KIP", "KPG-RL"]
SEEDS = [91731, 91732, 91733]
ACTIVITIES = {0: 0, 1: 1, 5: 2, 11: 3, 15: 4, 16: 5}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def member_catalog():
    pattern = re.compile(r"^(\d+)\.[^/]+/(\d+)_.*\.csv$")
    with zipfile.ZipFile(ZIP) as archive:
        members = []
        for info in archive.infolist():
            match = pattern.match(info.filename)
            if match:
                members.append((info.filename, int(match.group(1)), int(match.group(2))))
    return sorted(members)


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    members = member_catalog()
    all_people = sorted({person for _, _, person in members})
    people = sorted({person for _, label, person in members if label in ACTIVITIES})
    if len(all_people) != 89 or len(people) != 80:
        raise RuntimeError(
            f"schema amendment expects 89 archive and 80 shared-activity participants, "
            f"found {len(all_people)} and {len(people)}"
        )
    ordered = np.random.default_rng(91701).permutation(people).tolist()
    roles = {
        "public": ordered[:16],
        "source": ordered[16:43],
        "adaptation": ordered[43:54],
        "test": ordered[54:],
    }
    selections = read(BASE / "original_v7_research/confirmation_v7/protocol.json")["selections"]
    full_options = read(BASE / "maxent_v19_development/DSA/summary.json")["rankings"][uci.DESIGNS[0]][0]["options"]
    code_paths = [
        "prifct/v48_ku_har.py",
        "prifct/v47_uci_har.py",
        "prifct/windows_provider_bootstrap.py",
        "prifct/v15_maxent_teacher.py",
        "prifct/v14_semantic_functions.py",
        "prifct/v7_public_bridge.py",
        "prifct/v7_functional_bridge.py",
        "prifct/v7_streamed_private_features.py",
        "prifct/v6_private_generation.py",
        "prifct/v7_generator_weighted.py",
        "prifct/v7_dp_ntk.py",
        "prifct/v7_large_cohort_kip.py",
        "prifct/v7_dp_kip.py",
        "revision/joint_release.py",
    ]
    protocol = {
        "scope": "Prospective untouched KU-HAR validation; no outcome-based tuning or selection",
        "dataset": "KU-HAR v5 official trimmed/interpolated raw archive",
        "dataset_sha256": sha(ZIP),
        "schema": {
            "recordings": len(members),
            "archive_participants": len(all_people),
            "shared_activity_participants": len(people),
        },
        "roles": roles,
        "classes": 6,
        "activity_catalog": ["stand", "sit", "lay", "walk", "stair-up", "stair-down"],
        "directions": [["accel", "gyro"], ["gyro", "accel"]],
        "methods": ROSTER,
        "public": "All finite public windows, SHA256 ordered per person; first 80% fit and final 20% held; labels unavailable to mapping and calibration",
        "source": "Every finite source-person window; no class or utility filtering",
        "windows": "Nonoverlapping 300-sample (3 second) windows at the released 100 Hz rate; incomplete tails dropped",
        "features": "Three axes and magnitude; mean, population SD, RMS, range and low/high positive-frequency power fractions = 24 coordinates per view",
        "options": {uci.DESIGNS[0]: full_options},
        "private_baselines": {name: selections[name] for name in ["DP-HP", "DP-NTK", "DP-KIP"]},
        "context_baselines": {"KPG-RL": selections["KPG-RL"]},
        "selection": "All settings frozen before KU-HAR outcome computation from the existing DSA/MF development records",
        "budgets": [2.0, 10.0],
        "seeds": SEEDS,
        "primary": "Equal-weight mean of supported-class balanced accuracy within each of 26 test participants",
        "secondary": ["macro_f1", "nll", "brier", "ece"],
        "privacy": "Complete labelled descriptor-record replacement DP at fixed N/K and delta 1e-5; not person/raw-stream privacy",
        "missing": "All failures remain in the ledger; no success-only replacement",
        "code": {path: sha(ROOT / path) for path in code_paths},
    }
    path = OUT / "protocol.json"
    if path.exists():
        if read(path) != protocol:
            raise RuntimeError("frozen KU-HAR protocol changed")
    else:
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def data(protocol):
    cache = OUT / "descriptors.npz"
    if not cache.exists():
        groups = {name: [] for name in ["accel", "gyro", "y", "subjects", "identities", "records"]}
        dropped_tails = 0
        nonfinite = 0
        members = member_catalog()
        with zipfile.ZipFile(ZIP) as archive:
            for recording_index, (name, label, person) in enumerate(members):
                if label not in ACTIVITIES:
                    continue
                matrix = np.loadtxt(io.BytesIO(archive.read(name)), delimiter=",")
                if matrix.ndim == 1:
                    matrix = matrix[None, :]
                if matrix.shape[1] != 8:
                    raise RuntimeError(f"unexpected KU-HAR columns in {name}: {matrix.shape}")
                count = len(matrix) // 300
                dropped_tails += len(matrix) - 300 * count
                for window_index in range(count):
                    block = matrix[300 * window_index:300 * (window_index + 1)]
                    if not np.isfinite(block).all():
                        nonfinite += 1
                        continue
                    groups["accel"].append(uci.descriptors(block[:, 1:4]))
                    groups["gyro"].append(uci.descriptors(block[:, 5:8]))
                    groups["y"].append(ACTIVITIES[label])
                    groups["subjects"].append(person)
                    groups["identities"].append([person, label, recording_index, window_index])
                    groups["records"].append(name)
        arrays = {name: np.asarray(values) for name, values in groups.items()}
        np.savez_compressed(cache, **arrays)
        schema = {
            "archive_sha256": sha(ZIP),
            "windows": int(len(arrays["y"])),
            "participants": int(len(np.unique(arrays["subjects"]))),
            "class_counts": np.bincount(arrays["y"], minlength=6).tolist(),
            "nonfinite_windows": nonfinite,
            "dropped_tail_samples": dropped_tails,
            "protocol_sha256": sha(OUT / "protocol.json"),
        }
        (OUT / "extraction.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
        print("Extracted KU-HAR descriptor windows", schema, flush=True)
    with np.load(cache) as stored:
        arrays = {name: stored[name] for name in stored.files}
    arrays["roles"] = {
        role: np.flatnonzero(np.isin(arrays["subjects"], people))
        for role, people in protocol["roles"].items()
    }
    fit, held = [], []
    for person in protocol["roles"]["public"]:
        eligible = np.flatnonzero(arrays["subjects"] == person)
        keys = []
        for index in eligible:
            identity = arrays["identities"][index]
            token = f"{person}|{arrays['records'][index]}|{identity[3]}|91711".encode()
            keys.append(hashlib.sha256(token).digest())
        ordered = eligible[np.argsort(np.asarray(keys, dtype="S32"))]
        boundary = max(1, min(len(ordered) - 1, int(np.floor(0.8 * len(ordered)))))
        fit.extend(ordered[:boundary]); held.extend(ordered[boundary:])
    arrays["fit"] = np.asarray(fit, dtype=int)
    arrays["held"] = np.asarray(held, dtype=int)
    if any(len(arrays["roles"][role]) == 0 for role in protocol["roles"]):
        raise RuntimeError("a frozen KU-HAR role has no records")
    return arrays


def maps(arrays, source_name, target_name):
    folder = OUT / "public_maps"; folder.mkdir(exist_ok=True)
    path = folder / f"{source_name}_{target_name}.pkl"
    if path.exists():
        with path.open("rb") as stream:
            return pickle.load(stream)
    fit, held = arrays["fit"], arrays["held"]
    source_fit, target_fit = arrays[source_name][fit], arrays[target_name][fit]
    source_held, target_held = arrays[source_name][held], arrays[target_name][held]
    candidates = []
    for options in OPTIONS:
        point = fit_public_bridge(source_fit, target_fit, rank=32, landmarks=256, seed=91312, **options)
        error = float(np.mean(np.sum((point.target(target_held) - point.source(source_held)) ** 2, axis=1)))
        candidates.append((error, point))
    point = min(candidates, key=lambda item: item[0])[1]
    original = MultiScaleSignature.public_fit(point.source(source_fit), seed=91313)
    signature = ProjectedSignature.public_fit(original, point.source(source_fit), rank=32)
    candidates = []
    for options in OPTIONS:
        full = fit_functional_bridge(point, signature, source_fit, target_fit, **options)
        error = float(np.mean(np.sum((full.target(target_held) - full.source(source_held)) ** 2, axis=1)))
        candidates.append((error, full))
    result = {"point": point, "full": min(candidates, key=lambda item: item[0])[1]}
    with path.open("wb") as stream:
        pickle.dump(result, stream, protocol=4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["extract", "proposed", "private", "context"])
    mode = parser.parse_args().mode
    uci.OUT = OUT; uci.ZIP = ZIP; uci.SEEDS = SEEDS; uci.ROSTER = ROSTER
    uci.maps = maps
    protocol = register()
    arrays = data(protocol)
    if mode == "extract":
        return
    if mode == "proposed":
        uci.proposed(protocol, arrays)
    elif mode == "private":
        uci.private_baselines(protocol, arrays)
    else:
        uci.context(protocol, arrays)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
