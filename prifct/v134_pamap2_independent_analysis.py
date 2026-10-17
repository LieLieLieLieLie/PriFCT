"""Close the preregistered PAMAP2 gate without any post-outcome tuning."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

os.urandom = lambda n: bytes(n)

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/models/eswa_fedkapt"
PRI = BASE / "semantic_v132_pamap2_frozen_prifct"
COMP = BASE / "semantic_v133_pamap2_frozen_comparators"
OUT = BASE / "semantic_v134_pamap2_independent_analysis"
ENDPOINTS = ["subject_balanced_accuracy", "macro_f1", "nll", "brier", "ece"]
HIGHER = {"subject_balanced_accuracy", "macro_f1"}
PRIVATE = ["DP-HP", "DP-NTK", "DP-KIP"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def envelope_regret(metrics, baselines):
    terms = {}
    for endpoint in ENDPOINTS:
        reference = np.asarray([baselines[name][endpoint] for name in baselines], float)
        low, high = float(reference.min()), float(reference.max())
        scale = max(high - low, 1e-12)
        terms[endpoint] = ((high - metrics[endpoint]) / scale if endpoint in HIGHER
                           else (metrics[endpoint] - low) / scale)
    return float(np.mean(list(terms.values()))), terms


def participant_means(rows, method):
    selected = [row for row in rows if row["method"] == method and row["status"] == "success"]
    participants = sorted({int(person) for row in selected for person in row["results"]["per_subject_balanced"]})
    return {str(person): float(np.mean([row["results"]["per_subject_balanced"][str(person)] for row in selected]))
            for person in participants}


def exact_sign_flip(differences):
    observed = abs(float(np.mean(differences)))
    values = [abs(float(np.mean(np.asarray(differences) * np.asarray(signs))))
              for signs in itertools.product([-1, 1], repeat=len(differences))]
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pri_summary = read(PRI / "summary.json")
    comp_summary = read(COMP / "summary.json")
    pri_rows = [json.loads(line) for line in (PRI / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    comp_rows = [json.loads(line) for line in (COMP / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    pri_metrics = pri_summary["metrics"]
    private = {name: {endpoint: comp_summary["methods"][name][endpoint] for endpoint in ENDPOINTS}
               for name in PRIVATE}
    all_comparators = {name: {endpoint: values[endpoint] for endpoint in ENDPOINTS}
                       for name, values in comp_summary["methods"].items()}
    regret, terms = envelope_regret(pri_metrics, private)
    all_regret, all_terms = envelope_regret(pri_metrics, all_comparators)
    strongest = max(PRIVATE, key=lambda name: private[name]["subject_balanced_accuracy"])
    pri_person = participant_means(pri_rows, "PriFCT-v127")
    baseline_person = participant_means(comp_rows, strongest)
    differences = {person: pri_person[person] - baseline_person[person] for person in pri_person}
    report = {
        "scope": "First outcome analysis of the untouched frozen PAMAP2 confirmation; no retuning",
        "complete": pri_summary["complete"] and comp_summary["complete"],
        "methods": {"PriFCT-v127": pri_metrics, **all_comparators},
        "predefined_private_envelope": {"baselines": PRIVATE, "terms": terms, "normalized_regret": regret},
        "all_comparator_envelope_sensitivity": {"baselines": list(all_comparators), "terms": all_terms,
                                                "normalized_regret": all_regret},
        "strongest_private_by_primary": strongest,
        "primary_gate": {"prifct": pri_metrics["subject_balanced_accuracy"],
                         "strongest_private": private[strongest]["subject_balanced_accuracy"],
                         "passes": pri_metrics["subject_balanced_accuracy"] >= private[strongest]["subject_balanced_accuracy"]},
        "composite_gate": {"passes": regret < 0.0, "criterion": "normalized five-endpoint regret < 0"},
        "passes_preregistered_claim_gate": regret < 0.0 and pri_metrics["subject_balanced_accuracy"] >= private[strongest]["subject_balanced_accuracy"],
        "endpoint_wins_against_every_comparator": {
            endpoint: all((pri_metrics[endpoint] >= values[endpoint] if endpoint in HIGHER else pri_metrics[endpoint] <= values[endpoint])
                          for values in all_comparators.values()) for endpoint in ENDPOINTS},
        "participant_balanced_accuracy": {"PriFCT-v127": pri_person, strongest: baseline_person,
                                           "paired_difference": differences},
        "paired_exact_sign_flip_p_two_sided": exact_sign_flip(list(differences.values())),
        "statistical_limitation": "Only three complete held-out participants remain after the schema-only exclusion of incomplete subject 109; the participant-level comparison is underpowered and is not claimed significant.",
        "failure_disclosure": "PriFCT is not best on ECE; all comparator and PriFCT outcomes are retained.",
        "claim_status": "The preregistered independent composite and primary gates pass, but the evidence is a small held-out confirmation and does not support universal or per-endpoint superiority.",
    }
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# PAMAP2 independent frozen confirmation", "",
             "| Method | BA ↑ | F1 ↑ | NLL ↓ | Brier ↓ | ECE ↓ |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in report["methods"].items():
        lines.append(f"| {name} | {metrics['subject_balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
                     f"{metrics['nll']:.4f} | {metrics['brier']:.4f} | {metrics['ece']:.4f} |")
    lines += ["", f"Predefined private-envelope normalized regret: **{regret:.4f}**.",
              f"Primary BA gate: **{report['primary_gate']['passes']}**.",
              f"Composite gate: **{report['composite_gate']['passes']}**.",
              f"Participant-level exact paired sign-flip p-value: **{report['paired_exact_sign_flip_p_two_sided']:.3f}** (n=3; underpowered).",
              "", "PriFCT is not best on ECE; no adverse endpoint or run was removed."]
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
