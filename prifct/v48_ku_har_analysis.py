"""Frozen summary for the prospective KU-HAR validation."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/models/eswa_fedkapt/semantic_v48_ku_har"
ROWS = [json.loads(line) for line in (OUT / "results.jsonl").read_text().splitlines() if line.strip()]
ENDPOINTS = ["subject_balanced_accuracy", "macro_f1", "nll", "brier", "ece"]


def group_rows(method, epsilon=None):
    rows = [row for row in ROWS if row["method"] == method and row["status"] == "success"]
    if epsilon is not None:
        rows = [row for row in rows if float(row["epsilon"]) == float(epsilon)]
    return rows


def subject_vector(rows):
    values = defaultdict(list)
    for row in rows:
        for person, item in row["results"]["per_subject_balanced"].items():
            values[person].append(float(item["balanced_accuracy"]))
    return {person: float(np.mean(items)) for person, items in values.items()}


summary = {}
for method in ["PriFCT", "DP-HP", "DP-NTK", "DP-KIP"]:
    summary[method] = {}
    for epsilon in [2.0, 10.0]:
        rows = group_rows(method, epsilon)
        summary[method][str(epsilon)] = {
            "runs": len(rows),
            **{key: float(np.mean([row["results"][key] for row in rows])) for key in ENDPOINTS},
        }
rows = group_rows("KPG-RL")
summary["KPG-RL"] = {"context": {"runs": len(rows), **{
    key: float(np.mean([row["results"][key] for row in rows])) for key in ENDPOINTS
}}}

rng = np.random.default_rng(91791)
comparisons = {}
ours = subject_vector(group_rows("PriFCT", 2.0))
for method in ["DP-HP", "DP-NTK", "DP-KIP", "KPG-RL"]:
    other = subject_vector(group_rows(method, 2.0) if method != "KPG-RL" else group_rows(method))
    people = sorted(set(ours) & set(other))
    differences = np.asarray([ours[p] - other[p] for p in people])
    observed = float(differences.mean())
    exceed = 0
    total = 200000
    for start in range(0, total, 10000):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(min(10000, total - start), len(differences)))
        exceed += int(np.sum(np.abs((signs * differences).mean(axis=1)) >= abs(observed)))
    pvalue = (exceed + 1) / (total + 1)
    boot = rng.choice(differences, size=(20000, len(differences)), replace=True).mean(axis=1)
    comparisons[method] = {
        "participants": len(people),
        "mean_difference": observed,
        "bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "randomization_p": pvalue,
    }

ordered = sorted(comparisons, key=lambda name: comparisons[name]["randomization_p"])
running = 0.0
for rank, name in enumerate(ordered):
    adjusted = min(1.0, (len(ordered) - rank) * comparisons[name]["randomization_p"])
    running = max(running, adjusted)
    comparisons[name]["holm_p"] = running

gate = (
    all(item["mean_difference"] > 0 and item["holm_p"] < 0.05 for item in comparisons.values())
    and all(summary["PriFCT"]["10.0"]["subject_balanced_accuracy"] >= summary[m]["10.0"]["subject_balanced_accuracy"] for m in ["DP-HP", "DP-NTK", "DP-KIP"])
)
report = {"successful_rows": len(ROWS), "failed_rows": sum(row.get("status") != "success" for row in ROWS), "summary": summary, "epsilon2_comparisons": comparisons, "passes_preregistered_gate": gate}
(OUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

lines = ["# Prospective KU-HAR validation", "", f"Frozen runs: {len(ROWS)} successful, {report['failed_rows']} failed.", "", "| Method | epsilon | Person-BA | F1 | NLL | Brier | ECE |", "|---|---:|---:|---:|---:|---:|---:|"]
for method in ["PriFCT", "DP-HP", "DP-NTK", "DP-KIP"]:
    for epsilon in ["2.0", "10.0"]:
        item = summary[method][epsilon]
        lines.append(f"| {method} | {epsilon} | {item['subject_balanced_accuracy']:.4f} | {item['macro_f1']:.4f} | {item['nll']:.4f} | {item['brier']:.4f} | {item['ece']:.4f} |")
item = summary["KPG-RL"]["context"]
lines.append(f"| KPG-RL | non-private | {item['subject_balanced_accuracy']:.4f} | {item['macro_f1']:.4f} | {item['nll']:.4f} | {item['brier']:.4f} | {item['ece']:.4f} |")
lines += ["", "## Preregistered epsilon=2 person-level comparisons", "", "| Comparator | BA difference | 95% bootstrap CI | randomization p | Holm p |", "|---|---:|---:|---:|---:|"]
for method, item in comparisons.items():
    lo, hi = item["bootstrap_95ci"]
    lines.append(f"| {method} | {item['mean_difference']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {item['randomization_p']:.6f} | {item['holm_p']:.6f} |")
lines += ["", f"**Preregistered P0 gate: {'PASS' if gate else 'FAIL'}.**", "", "A failed gate is retained as adverse evidence and is not converted into a superiority claim."]
(OUT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
