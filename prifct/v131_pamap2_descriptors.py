"""Build the frozen PAMAP2 paired-window descriptors from protocol recordings."""
from __future__ import annotations

import json
import math
import os
import zipfile
from pathlib import Path

# This workstation intermittently reports a Windows CNG entropy-provider error.
# The extraction is fully deterministic and uses no cryptography; a fixed byte
# source only lets NumPy initialize its unused module-level RNG.
os.urandom = lambda n: bytes(n)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/models/eswa_fedkapt"
DESIGN_PATH = BASE / "semantic_v128_pamap2_preregistration/design_before_acquisition.json"
AMEND_PATH = BASE / "semantic_v130_pamap2_preutility_feasibility/availability_amendment_before_utility.json"
OUT = BASE / "semantic_v131_pamap2_descriptors"
DATA_OUT = ROOT / "data/pamap2/descriptors"
INNER = Path(r"E:\Downloads\PAMAP2_Dataset.zip")

ACTIVITIES = np.array([1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24], dtype=np.int16)
LOCATIONS = {
    "hand": ((4, 5, 6), (10, 11, 12)),
    "chest": ((21, 22, 23), (27, 28, 29)),
    "ankle": ((38, 39, 40), (44, 45, 46)),
}
USECOLS = [0, 1] + sorted({column for pair in LOCATIONS.values() for columns in pair for column in columns})
COLPOS = {original: position for position, original in enumerate(USECOLS)}


def descriptor(values: np.ndarray) -> np.ndarray:
    magnitude = np.linalg.norm(values, axis=1, keepdims=True)
    signals = np.concatenate([values, magnitude], axis=1)
    rows = []
    for signal in signals.T:
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        rms = float(np.sqrt(np.mean(signal * signal)))
        span = float(np.max(signal) - np.min(signal))
        centered = signal - mean
        power = np.abs(np.fft.rfft(centered)) ** 2
        frequencies = np.fft.rfftfreq(len(signal), d=0.01)
        positive = frequencies > 0
        total = float(np.sum(power[positive]))
        if total <= 1e-20:
            low = high = 0.0
        else:
            low = float(np.sum(power[(frequencies > 0) & (frequencies < 5.0)]) / total)
            high = float(np.sum(power[(frequencies >= 5.0) & (frequencies <= 50.0)]) / total)
        rows.extend([mean, std, rms, span, low, high])
    return np.asarray(rows, dtype=np.float32)


def complete(values: np.ndarray, times: np.ndarray, start: float) -> bool:
    finite = np.all(np.isfinite(values), axis=1)
    supported = times[finite]
    return bool(
        supported.size >= 180
        and supported[0] <= start + 0.1 + 1e-9
        and supported[-1] >= start + 1.9 - 1e-9
        and (supported.size < 2 or np.max(np.diff(supported)) <= 0.05 + 1e-9)
    )


def subject_descriptors(archive: zipfile.ZipFile, participant: int) -> dict[str, dict[str, np.ndarray]]:
    member = f"PAMAP2_Dataset/Protocol/subject10{participant}.dat"
    with archive.open(member) as stream:
        raw = np.loadtxt(stream, dtype=np.float64, usecols=USECOLS)
    times = raw[:, COLPOS[0]]
    labels = raw[:, COLPOS[1]].astype(np.int16)
    registered = np.isin(labels, ACTIVITIES)
    times = times[registered]
    labels = labels[registered]
    raw = raw[registered]
    bins = np.floor(times / 2.0).astype(np.int64)
    keys = labels.astype(np.int64) * 10_000_000 + bins
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    boundaries = np.r_[0, 1 + np.flatnonzero(np.diff(keys_sorted)), len(order)]

    records = {location: {"acc": [], "gyr": [], "y": [], "bin": []} for location in LOCATIONS}
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        label = int(labels[indices[0]])
        bin_id = int(bins[indices[0]])
        start = 2.0 * bin_id
        group_times = times[indices]
        time_order = np.argsort(group_times, kind="stable")
        indices = indices[time_order]
        group_times = group_times[time_order]
        for location, (acc_cols, gyr_cols) in LOCATIONS.items():
            acc = raw[indices][:, [COLPOS[c] for c in acc_cols]]
            gyr = raw[indices][:, [COLPOS[c] for c in gyr_cols]]
            if not complete(acc, group_times, start) or not complete(gyr, group_times, start):
                continue
            acc_finite = np.all(np.isfinite(acc), axis=1)
            gyr_finite = np.all(np.isfinite(gyr), axis=1)
            records[location]["acc"].append(descriptor(acc[acc_finite]))
            records[location]["gyr"].append(descriptor(gyr[gyr_finite]))
            records[location]["y"].append(label)
            records[location]["bin"].append(bin_id)

    result = {}
    for location, values in records.items():
        result[location] = {
            "acc": np.asarray(values["acc"], dtype=np.float32).reshape(-1, 24),
            "gyr": np.asarray(values["gyr"], dtype=np.float32).reshape(-1, 24),
            "y": np.asarray(values["y"], dtype=np.int16),
            "bin": np.asarray(values["bin"], dtype=np.int64),
        }
    return result


def main() -> None:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMEND_PATH.read_text(encoding="utf-8"))
    people = amendment["participant_permutation"]
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "scope": "Frozen PAMAP2 protocol-only paired-window descriptor extraction",
        "inner_archive": str(INNER),
        "design": str(DESIGN_PATH.relative_to(ROOT)),
        "availability_amendment": str(AMEND_PATH.relative_to(ROOT)),
        "participants": people,
        "activities": design["activities"],
        "usecols_zero_based": USECOLS,
        "counts": {},
        "utility_accessed": False,
    }
    with zipfile.ZipFile(INNER) as archive:
        for participant in people:
            print(f"extract participant {participant}", flush=True)
            extracted = subject_descriptors(archive, participant)
            report["counts"][str(participant)] = {}
            for location, arrays in extracted.items():
                path = DATA_OUT / f"subject{participant:02d}_{location}.npz"
                np.savez_compressed(path, **arrays)
                supported = {str(int(label)): int(np.sum(arrays["y"] == label)) for label in ACTIVITIES}
                report["counts"][str(participant)][location] = {
                    "windows": int(len(arrays["y"])),
                    "class_counts": supported,
                    "file": str(path.relative_to(ROOT)),
                }
    report_path = OUT / "descriptor_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
