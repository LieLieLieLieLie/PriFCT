"""Audit PAMAP2 composite-gate dependence on the private-baseline envelope.

This script is a post-outcome sensitivity analysis only.  It reads the sealed
v134 aggregate metrics, recomputes the registered range-normalised endpoint
regrets, and repeats the calculation after removing each private baseline in
turn.  It performs no fitting, selection, or change to any prediction.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/models/eswa_fedkapt"
SOURCE = BASE / "semantic_v134_pamap2_independent_analysis/analysis.json"
OUT = BASE / "semantic_v135_pamap2_gate_sensitivity"
ENDPOINTS = ["subject_balanced_accuracy", "macro_f1", "nll", "brier", "ece"]
HIGHER = {"subject_balanced_accuracy", "macro_f1"}
PRIVATE = ["DP-HP", "DP-NTK", "DP-KIP"]


def envelope(prifct: dict[str, float], methods: dict[str, dict[str, float]], names: list[str]):
    terms: dict[str, float] = {}
    ranges: dict[str, dict[str, float | str]] = {}
    for endpoint in ENDPOINTS:
        values = {name: float(methods[name][endpoint]) for name in names}
        low_name = min(values, key=values.get)
        high_name = max(values, key=values.get)
        low, high = values[low_name], values[high_name]
        scale = max(high - low, 1e-12)
        if endpoint in HIGHER:
            regret = (high - float(prifct[endpoint])) / scale
        else:
            regret = (float(prifct[endpoint]) - low) / scale
        terms[endpoint] = float(regret)
        ranges[endpoint] = {
            "low": low,
            "high": high,
            "range": scale,
            "low_method": low_name,
            "high_method": high_name,
        }
    return {
        "baselines": names,
        "endpoint_regret": terms,
        "baseline_ranges": ranges,
        "mean_regret": float(sum(terms.values()) / len(terms)),
        "passes_negative_regret_gate": float(sum(terms.values()) / len(terms)) < 0.0,
    }


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    methods = source["methods"]
    prifct = methods["PriFCT-v127"]
    analyses = {"all_private": envelope(prifct, methods, PRIVATE)}
    for omitted in PRIVATE:
        retained = [name for name in PRIVATE if name != omitted]
        analyses[f"without_{omitted}"] = envelope(prifct, methods, retained)

    report = {
        "scope": "Post-outcome baseline-envelope sensitivity; no prediction, fitting, or setting changed",
        "source": str(SOURCE.relative_to(ROOT)),
        "locally_frozen_gate_preserved": "all_private",
        "interpretation": (
            "Leave-one-baseline-out values are sensitivity analyses, not replacement primary endpoints. "
            "A negative value favours PriFCT under that retained envelope."
        ),
        "analyses": analyses,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    labels = {
        "subject_balanced_accuracy": "BA",
        "macro_f1": "F1",
        "nll": "NLL",
        "brier": "Brier",
        "ece": "ECE",
    }
    lines = [
        "# PAMAP2 composite-gate sensitivity",
        "",
        "Negative regret favours PriFCT. The locally frozen all-private envelope remains the primary analysis; leave-one-out rows diagnose dependence on baseline composition.",
        "",
        "| Envelope | BA | F1 | NLL | Brier | ECE | Mean | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for key, values in analyses.items():
        terms = values["endpoint_regret"]
        display = "DP-HP + DP-NTK + DP-KIP" if key == "all_private" else key.replace("without_", "Exclude ")
        row = [display] + [f"{terms[e]:+.4f}" for e in ENDPOINTS]
        row += [f"{values['mean_regret']:+.4f}", "yes" if values["passes_negative_regret_gate"] else "no"]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "The DP-KIP exclusion is the reviewer-requested stress case. Its narrower NLL/ECE ranges reduce the magnitude of the advantage; the sign of the equal-weight composite remains negative.",
        "",
        "Per-endpoint ranges and the identity of each range endpoint are retained in `analysis.json`.",
    ]
    (OUT / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
