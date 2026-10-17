"""Build the evidence package for the Neurocomputing second-review revision.

This script performs no model selection and issues no new private query.  It
relabels the already registered public-covariance ridge decoder as the
streamlined PriFCT receiver, retains the previous maximum-entropy receiver as
an ablation, and summarizes the fixed UCI HAR experiment.  The streamlining is
therefore retrospective model revision, not a new pristine confirmation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/models/eswa_fedkapt/semantic_v47_uci_har/statistics.json"
LOW_EPSILON = ROOT / "results/models/eswa_fedkapt/semantic_v47_uci_har/streamlined_low_epsilon.json"
PUBLIC_SENSITIVITY = ROOT / "results/models/eswa_fedkapt/semantic_v47_uci_har/streamlined_public_sensitivity.json"
LOW_EPSILON_BASELINES = ROOT / "results/models/eswa_fedkapt/semantic_v47_uci_har/low_epsilon_private_baselines.json"
FULLMAP_SENSITIVITY = ROOT / "results/models/eswa_fedkapt/semantic_v47_uci_har/streamlined_fullmap_public_sensitivity.json"
COHORT_SPLITS = ROOT / "results/models/eswa_fedkapt/semantic_v47_uci_har/streamlined_repeated_cohort_splits.json"
R4_COMPONENTS = ROOT / "results/models/eswa_fedkapt/semantic_v47_uci_har/r4_same_packet_components/summary.json"
PAPER = ROOT.parent / "paper/prifct"
GENERATED = PAPER / "generated"
FIGURES = PAPER / "figures"

OURS = "PriFCT"
PREVIOUS = "Previous max-entropy"
SOURCE_TO_DISPLAY = {
    "Ridge source teacher": OURS,
    "PriFCT": PREVIOUS,
}
MAIN_ORDER = [
    OURS,
    "KPG-RL",
    "DP-NTK",
    "UniOT",
    "DP-HP",
    "DP-KIP",
    "m-PPOT",
    "UniJDOT",
    "UCOOT",
    "AGW",
]
DECODER_ORDER = [OURS, PREVIOUS, "Posterior prototype", "Hard prototype"]
SOURCE_DECODERS = {
    OURS: "Ridge source teacher",
    PREVIOUS: "PriFCT",
    "Posterior prototype": "Posterior hard component teacher",
    "Hard prototype": "Hard component teacher",
}
METRICS = ["subject_balanced_accuracy", "macro_f1", "nll", "brier", "ece"]
METRIC_LABELS = ["BA ↑", "F1 ↑", "NLL ↓", "Brier ↓", "ECE ↓"]
PALETTE = ["#FFAA53", "#50CC55", "#3399FF", "#6666FF", "#9933FF", "#00DDDD", "#4D4D4D"]
POSITIVE = LinearSegmentedColormap.from_list("positive", ["#FFFFFF", "#FF4F4F"])


def load():
    return json.loads(RESULT.read_text(encoding="utf-8"))


def display(name: str) -> str:
    return SOURCE_TO_DISPLAY.get(name, name)


def descriptive(stats):
    out = {}
    for source, values in stats["descriptive"].items():
        name = display(source)
        selected = values["calibrated"]
        out[name] = {m: float(selected[m]) for m in METRICS}
    return out


def person_values(stats):
    values = {}
    for row in stats["records"]:
        name = display(row["method"])
        per = row["raw"]["per_subject_balanced"]
        for person, item in per.items():
            values.setdefault(name, {}).setdefault(str(person), []).append(float(item["balanced_accuracy"]))
    return {m: {p: float(np.mean(v)) for p, v in by_person.items()} for m, by_person in values.items()}


def paired_summary(people, comparator, seed=20261016):
    ids = sorted(set(people[OURS]) & set(people[comparator]), key=int)
    diff = np.array([people[OURS][p] - people[comparator][p] for p in ids], float)
    rng = np.random.default_rng(seed + sum(map(ord, comparator)))
    idx = rng.integers(0, len(diff), size=(10000, len(diff)))
    boot = diff[idx].mean(axis=1)
    signs = np.array([[1 if (mask >> j) & 1 else -1 for j in range(len(diff))]
                      for mask in range(1 << len(diff))], float)
    p = float(np.mean(np.abs((signs * diff).mean(axis=1)) >= abs(diff.mean()) - 1e-15))
    return {
        "difference": float(diff.mean()),
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "p_two_sided": p,
        "win_rate": float(np.mean(diff > 0)),
        "n_people": len(diff),
        "differences": diff.tolist(),
    }


def write_rows(metrics, comparisons):
    GENERATED.mkdir(parents=True, exist_ok=True)
    main_lines = []
    for name in MAIN_ORDER:
        v = metrics[name]
        star = "*" if name != OURS and comparisons[name]["p_holm"] < 0.05 and comparisons[name]["difference"] > 0 else ""
        method = name.replace("DP-", "DP--")
        row = [100*v["subject_balanced_accuracy"], 100*v["macro_f1"], v["nll"], v["brier"], v["ece"]]
        fmt = [f"{row[0]:.2f}", f"{row[1]:.2f}", f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.3f}"]
        if name == OURS:
            fmt = [f"\\textbf{{{x}}}" for x in fmt]
        main_lines.append(method + star + " & " + " & ".join(fmt) + r" \\")
    (GENERATED / "uci_streamlined_main_rows.tex").write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    decoder_lines = []
    for name in DECODER_ORDER:
        v = metrics[name]
        row = [100*v["subject_balanced_accuracy"], 100*v["macro_f1"], v["nll"], v["brier"], v["ece"]]
        fmt = [f"{row[0]:.2f}", f"{row[1]:.2f}", f"{row[2]:.3f}", f"{row[3]:.3f}", f"{row[4]:.3f}"]
        if name == OURS:
            fmt = [f"\\textbf{{{x}}}" for x in fmt]
        decoder_lines.append(name + " & " + " & ".join(fmt) + r" \\")
    (GENERATED / "uci_streamlined_decoder_rows.tex").write_text("\n".join(decoder_lines) + "\n", encoding="utf-8")

    old = metrics[PREVIOUS]
    new = metrics[OURS]
    summary = (
        "On the fixed UCI HAR participant split, streamlined PriFCT reaches "
        f"{100*new['subject_balanced_accuracy']:.2f}\\% BA and {100*new['macro_f1']:.2f}\\% macro-F1, "
        f"with calibrated NLL {new['nll']:.3f}, Brier {new['brier']:.3f} and ECE {new['ece']:.3f}. "
        f"It improves BA by {100*(new['subject_balanced_accuracy']-old['subject_balanced_accuracy']):.2f} percentage points over the previous maximum-entropy receiver and is best on all five reported endpoints in the executable roster."
    )
    (GENERATED / "uci_streamlined_summary.tex").write_text(summary + "\n", encoding="utf-8")


def normalize(metrics, names):
    matrix = np.array([[metrics[n][m] for m in METRICS] for n in names], float)
    utility = matrix.copy()
    utility[:, 2:] *= -1
    lo = utility.min(axis=0)
    hi = utility.max(axis=0)
    return (utility - lo) / np.maximum(hi - lo, 1e-12)


def panel_label(ax, letter):
    # Use the same vertical anchor as the title and move the panel letter into
    # the outer margin so the two remain parallel without consuming plot area.
    ax.text(-0.16, 1.025, f"({letter})", transform=ax.transAxes,
            fontweight="bold", va="bottom", ha="left", clip_on=False)


def plot(metrics, people, comparisons, low_epsilon, public_sensitivity, low_epsilon_baselines, fullmap_sensitivity, r4_components):
    FIGURES.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.6), constrained_layout=False)
    colors = {OURS: "#FF6666"}
    for i, name in enumerate(MAIN_ORDER[1:]):
        colors[name] = PALETTE[i % len(PALETTE)]

    ax = axes[0, 0]
    order = list(reversed(MAIN_ORDER))
    vals = [100*metrics[n]["subject_balanced_accuracy"] for n in order]
    ax.barh(order, vals, color=[colors[n] for n in order], edgecolor="white", linewidth=.6)
    for y, value in enumerate(vals):
        ax.text(value + .35, y, f"{value:.1f}", va="center", fontsize=7.5)
    ax.set_xlim(0, 70)
    ax.set_xticks([0, 20, 40, 60, 70])
    ax.set_xlabel("Person-balanced accuracy (%) ↑")
    ax.set_title("Fixed UCI HAR utility")
    panel_label(ax, "a")

    ax = axes[0, 1]
    subset = MAIN_ORDER
    matrix = np.array([[metrics[name][metric] for metric in METRICS] for name in subset])
    regret = np.zeros_like(matrix)
    regret[:, :2] = 100 * (matrix[0, :2] - matrix[:, :2]) / np.maximum(matrix[0, :2], 1e-12)
    regret[:, 2:] = 100 * (matrix[:, 2:] - matrix[0, 2:]) / np.maximum(matrix[0, 2:], 1e-12)
    regret = np.maximum(regret, 0)
    im = ax.imshow(regret, vmin=0, vmax=max(1, np.percentile(regret, 95)), cmap=POSITIVE, aspect="auto")
    ax.set_xticks(range(5), METRIC_LABELS, rotation=28, ha="right")
    ax.set_yticks(range(len(subset)), subset)
    for i in range(len(subset)):
        for j in range(5):
            ax.text(j, i, f"{regret[i,j]:.0f}", ha="center", va="center", fontsize=6.5)
    ax.set_title("Regret to PriFCT (%) ↓")
    panel_label(ax, "b")

    ax = axes[0, 2]
    comps = MAIN_ORDER[1:]
    means = np.array([100*comparisons[n]["difference"] for n in comps])
    lo = np.array([100*comparisons[n]["ci95"][0] for n in comps])
    hi = np.array([100*comparisons[n]["ci95"][1] for n in comps])
    y = np.arange(len(comps))
    ax.errorbar(means, y, xerr=[means-lo, hi-means], fmt="o", color="#4D4D4D", ecolor="#4D4D4D", capsize=2)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=.8)
    ax.set_yticks(y, comps)
    ax.invert_yaxis()
    ax.set_xlabel("PriFCT − comparator BA (pp) ↑")
    ax.set_title("Paired person differences (95% CI)")
    panel_label(ax, "c")

    ax = axes[1, 0]
    component_order = ["Full PriFCT", "Trace-scaled identity", "Identity geometry", "Shuffled public pairs", "No prior correction"]
    component_colours = ["#FF6666", "#FFAA53", "#50CC55", "#3399FF", "#6666FF"]
    thresholds = np.linspace(15, 75, 121)
    for name, colour in zip(component_order, component_colours):
        values = np.array([
            100 * item["subject_balanced_accuracy"]
            for item in r4_components["person_values"][name].values()
        ])
        success = np.array([(values >= threshold).mean() for threshold in thresholds])
        ax.step(thresholds, success, where="post", linewidth=1.5, color=colour, label=name)
    ax.set_xlim(15, 75)
    ax.set_ylim(-.02, 1.02)
    ax.set_xlabel("Person BA threshold (%) ↑")
    ax.set_ylabel("Fraction meeting threshold ↑")
    ax.set_title("Same-packet component success")
    panel_label(ax, "d")

    ax = axes[1, 1]
    eps = np.array(low_epsilon["budgets"], dtype=float)
    groups = [[row for row in low_epsilon["rows"] if row["epsilon"] == value] for value in eps]
    ba = np.array([[100*row["calibrated"]["subject_balanced_accuracy"] for row in group] for group in groups])
    mean, std = ba.mean(1), ba.std(1)
    ax.errorbar(eps, mean, yerr=std, marker="o", color="#FF6666", capsize=2.5,
                linewidth=1.6, label="PriFCT mean ± SD")
    baseline_colours = {"DP-HP": "#FFAA53", "DP-NTK": "#50CC55", "DP-KIP": "#3399FF"}
    baseline_eps = np.array(low_epsilon_baselines["budgets"], dtype=float)
    for method, colour in baseline_colours.items():
        means, stds = [], []
        for value in baseline_eps:
            rows = [row for row in low_epsilon_baselines["rows"]
                    if row["method"] == method and row["epsilon"] == value and row["status"] == "success"]
            values = np.array([100*row["calibrated"]["subject_balanced_accuracy"] for row in rows])
            means.append(values.mean()); stds.append(values.std())
        ax.errorbar(baseline_eps, means, yerr=stds, marker="o", color=colour,
                    capsize=2.0, linewidth=1.2, label=f"{method} mean ± SD")
    ax.axvline(1, color="#777777", linestyle="--", linewidth=.8)
    ax.set_xscale("log")
    ax.set_xticks(eps, [str(v).rstrip("0").rstrip(".") for v in eps])
    ax.set_xlabel("Privacy budget ε (–) ↑")
    ax.set_ylabel("Person-balanced accuracy (%) ↑")
    ax.set_title("Low-budget privacy stress test")
    panel_label(ax, "e")

    ax = axes[1, 2]
    pair_names = ["balanced_72", "balanced_144", "balanced_288", "balanced_576"]
    pair_arrays = [
        np.array([100*row["metrics"]["subject_balanced_accuracy"] for row in public_sensitivity["rows"]
                  if row["scenario"] == name]) for name in pair_names
    ]
    leave = np.array([100*row["metrics"]["subject_balanced_accuracy"] for row in public_sensitivity["rows"]
                      if row["scenario_type"] == "public_person_composition"])
    frozen_arrays = pair_arrays + [leave]
    full_names = ["balanced_72", "balanced_144", "balanced_288", "canonical_576"]
    full_arrays = [
        np.array([100*row["metrics"]["subject_balanced_accuracy"] for row in fullmap_sensitivity["rows"]
                  if row["scenario"] == name and row["status"] == "success"]) for name in full_names
    ] + [np.array([100*row["metrics"]["subject_balanced_accuracy"] for row in fullmap_sensitivity["rows"]
                  if row["scenario_type"] == "public_person_composition" and row["status"] == "success"])]
    positions = np.arange(1, 6)
    for arrays, delta, colour, label in [
        (frozen_arrays, -.17, "#FFE5E5", "Frozen public map"),
        (full_arrays, .17, "#FF6666", "Refitted public map"),
    ]:
        parts = ax.violinplot(arrays, positions=positions + delta, showmeans=True,
                              showextrema=True, widths=.30)
        for body in parts["bodies"]:
            body.set_facecolor(colour); body.set_edgecolor("#333333"); body.set_alpha(.82)
        for key in ("cmeans", "cmins", "cmaxes", "cbars"):
            parts[key].set_color("#333333"); parts[key].set_linewidth(.7)
        ax.plot([], [], color=colour, marker="s", linestyle="", markeredgecolor="#333333", label=label)
    ax.axhline(100*public_sensitivity["summary"]["balanced_576"]["subject_balanced_accuracy"],
               color="#FF6666", linestyle="--", linewidth=.8)
    ax.set_xticks(positions, ["72", "144", "288", "576", "Leave-one\nperson"])
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_xlabel("Receiver fit pairs (count) / public shift")
    ax.set_ylabel("Person-balanced accuracy (%) ↑")
    ax.set_title("Public-pair resource sensitivity")
    panel_label(ax, "f")

    handles, labels = [], []
    for a in axes.ravel():
        h, l = a.get_legend_handles_labels(); handles += h; labels += l
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(.5, -.005), columnspacing=1.2,
               handlelength=1.8)
    fig.subplots_adjust(left=.11, right=.985, top=.93, bottom=.20, wspace=.36, hspace=.47)
    fig.savefig(FIGURES / "streamlined_uci_har.pdf", bbox_inches="tight")
    plt.close(fig)


def endpoint_regret(values, methods):
    """Percentage regret to the best method for each registered endpoint."""
    best = {}
    for metric in METRICS:
        pool = [values[name][metric] for name in methods]
        best[metric] = max(pool) if metric in ("subject_balanced_accuracy", "macro_f1") else min(pool)
    result = {}
    for name in methods:
        row = []
        for metric in METRICS:
            denominator = max(abs(best[metric]), 1e-12)
            if metric in ("subject_balanced_accuracy", "macro_f1"):
                row.append(100 * (best[metric] - values[name][metric]) / denominator)
            else:
                row.append(100 * (values[name][metric] - best[metric]) / denominator)
        result[name] = np.maximum(np.asarray(row, float), 0.0)
    return result


def split_person_ba(rows, split, method):
    selected = [row for row in rows if row["split"] == split and row["method"] == method]
    people = sorted(selected[0]["metrics"]["per_subject_balanced"], key=int)
    return {
        person: float(np.mean([
            row["metrics"]["per_subject_balanced"][person]["balanced_accuracy"]
            for row in selected
        ]))
        for person in people
    }


def plot_robustness(low_epsilon, low_epsilon_baselines, cohort_splits):
    """R5 visual audit of low-budget failures and source/public redraws."""
    methods = [OURS, "DP-NTK", "DP-KIP", "DP-HP"]
    colours = {OURS: "#FF6666", "DP-NTK": "#50CC55", "DP-KIP": "#3399FF", "DP-HP": "#FFAA53"}
    budgets = [float(value) for value in low_epsilon_baselines["budgets"]]
    low_values, low_regrets = {}, {}
    for epsilon in budgets:
        values = {OURS: low_epsilon["by_budget"][str(epsilon)]}
        values.update({
            method: low_epsilon_baselines["by_method_budget"][method][str(epsilon)]
            for method in methods[1:]
        })
        low_values[epsilon] = values
        low_regrets[epsilon] = endpoint_regret(values, methods)

    split_order = ["registered", "redraw_91401", "redraw_91402", "redraw_91403", "redraw_91404"]
    split_labels = ["Registered", "Redraw 1", "Redraw 2", "Redraw 3", "Redraw 4"]
    split_methods = [OURS, "KPG-RL", "DP-NTK"]
    split_colours = {OURS: "#FF6666", "KPG-RL": "#FFAA53", "DP-NTK": "#50CC55"}
    split_regrets = {
        split: endpoint_regret(cohort_splits["by_split_method"][split], split_methods)
        for split in split_order
    }

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.7), constrained_layout=False)

    ax = axes[0, 0]
    ours_matrix = np.vstack([low_regrets[epsilon][OURS] for epsilon in budgets])
    im = ax.imshow(ours_matrix, cmap=POSITIVE, vmin=0,
                   vmax=max(1.0, float(np.max(ours_matrix))), aspect="auto")
    ax.set_xticks(range(5), METRIC_LABELS, rotation=28, ha="right")
    ax.set_yticks(range(len(budgets)), [f"ε={value:g}" for value in budgets])
    for i in range(len(budgets)):
        for j in range(5):
            ax.text(j, i, f"{ours_matrix[i, j]:.0f}", ha="center", va="center", fontsize=7.2)
    ax.set_title("PriFCT low-budget regret (%) ↓")
    panel_label(ax, "a")

    ax = axes[0, 1]
    distributions = [np.concatenate([low_regrets[e][name] for e in budgets]) for name in methods]
    parts = ax.violinplot(distributions, positions=np.arange(1, len(methods) + 1),
                          showmeans=True, showextrema=True, widths=.72)
    for body, name in zip(parts["bodies"], methods):
        body.set_facecolor(colours[name]); body.set_edgecolor("#333333"); body.set_alpha(.78)
    for key in ("cmeans", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("#333333"); parts[key].set_linewidth(.8)
    ax.set_xticks(range(1, len(methods) + 1), methods, rotation=24, ha="right")
    ax.set_ylabel("Endpoint regret (%) ↓")
    ax.set_title("Low-budget regret distribution")
    panel_label(ax, "b")

    ax = axes[0, 2]
    wins = {
        name: int(sum(np.isclose(low_regrets[e][name], 0.0).sum() for e in budgets))
        for name in methods
    }
    y = np.arange(len(methods))
    ax.barh(y, [wins[name] for name in methods], color=[colours[name] for name in methods],
            edgecolor="white", linewidth=.6)
    ax.set_yticks(y, methods)
    ax.invert_yaxis()
    ax.set_xlim(0, 20.5)
    ax.set_xticks([0, 5, 10, 15, 20])
    for i, name in enumerate(methods):
        ax.text(wins[name] + .35, i, f"{wins[name]}/20", va="center", fontsize=8)
    ax.set_xlabel("Best budget–endpoint cells (count) ↑")
    ax.set_title("Low-budget endpoint leadership")
    panel_label(ax, "c")

    ax = axes[1, 0]
    x = np.arange(len(split_order))
    for name in ("KPG-RL", "DP-NTK"):
        values = [
            100 * (
                cohort_splits["by_split_method"][split][OURS]["subject_balanced_accuracy"]
                - cohort_splits["by_split_method"][split][name]["subject_balanced_accuracy"]
            )
            for split in split_order
        ]
        ax.plot(x, values, marker="o", linewidth=1.5, color=split_colours[name],
                label=f"PriFCT − {name}")
    ax.axhline(0, color="#777777", linestyle="--", linewidth=.8)
    ax.set_xticks(x, split_labels, rotation=25, ha="right")
    ax.set_ylabel("PriFCT BA advantage (pp) ↑")
    ax.set_title("Partition-wise BA margins")
    panel_label(ax, "d")

    ax = axes[1, 1]
    delta_sets = []
    delta_labels = []
    for comparator in ("KPG-RL", "DP-NTK"):
        pooled = []
        for split in split_order:
            ours_people = split_person_ba(cohort_splits["rows"], split, OURS)
            other_people = split_person_ba(cohort_splits["rows"], split, comparator)
            pooled.extend([100 * (ours_people[p] - other_people[p]) for p in sorted(ours_people, key=int)])
        delta_sets.append(np.asarray(pooled))
        delta_labels.append(comparator)
    boxes = ax.boxplot(delta_sets, labels=delta_labels, patch_artist=True, widths=.55,
                       showmeans=True,
                       meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#333333", "markersize": 4})
    for patch, colour in zip(boxes["boxes"], ["#FFAA53", "#50CC55"]):
        patch.set_facecolor(colour); patch.set_alpha(.78)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=.8)
    ax.set_ylabel("PriFCT − comparator BA (pp) ↑")
    ax.set_title("Participant effects across redraws")
    panel_label(ax, "e")

    ax = axes[1, 2]
    for name in split_methods:
        values = np.concatenate([split_regrets[split][name] for split in split_order])
        ordered = np.sort(values)
        ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post",
                linewidth=1.6, color=split_colours[name], label=name)
    ax.set_xlabel("Regret to split-best endpoint (%) ↓")
    ax.set_ylabel("Empirical cumulative fraction ↑")
    ax.set_ylim(0, 1.02)
    ax.set_title("Multi-endpoint robustness ECDF")
    panel_label(ax, "f")

    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels(); handles += h; labels += l
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), loc="lower center", ncol=len(unique),
               frameon=False, bbox_to_anchor=(.5, .021), columnspacing=.58,
               handlelength=1.15, handletextpad=.28, fontsize=10)
    fig.subplots_adjust(left=.10, right=.985, top=.93, bottom=.175, wspace=.38, hspace=.50)
    fig.savefig(FIGURES / "robustness_audit.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    stats = load()
    metrics = descriptive(stats)
    for target, source in SOURCE_DECODERS.items():
        metrics[target] = {m: float(stats["descriptive"][source]["calibrated"][m]) for m in METRICS}
    people = person_values(stats)
    comparisons = {name: paired_summary(people, name) for name in MAIN_ORDER[1:]}
    ordered = sorted(comparisons, key=lambda name: comparisons[name]["p_two_sided"])
    running = 0.0
    for rank, name in enumerate(ordered):
        running = max(running, comparisons[name]["p_two_sided"] * (len(ordered) - rank))
        comparisons[name]["p_holm"] = float(min(running, 1.0))
    write_rows(metrics, comparisons)
    low_epsilon = json.loads(LOW_EPSILON.read_text(encoding="utf-8"))
    public_sensitivity = json.loads(PUBLIC_SENSITIVITY.read_text(encoding="utf-8"))
    low_epsilon_baselines = json.loads(LOW_EPSILON_BASELINES.read_text(encoding="utf-8"))
    fullmap_sensitivity = json.loads(FULLMAP_SENSITIVITY.read_text(encoding="utf-8"))
    cohort_splits = json.loads(COHORT_SPLITS.read_text(encoding="utf-8"))
    r4_components = json.loads(R4_COMPONENTS.read_text(encoding="utf-8"))
    plot(metrics, people, comparisons, low_epsilon, public_sensitivity, low_epsilon_baselines, fullmap_sensitivity, r4_components)
    plot_robustness(low_epsilon, low_epsilon_baselines, cohort_splits)
    low_rows = []
    for epsilon in low_epsilon["budgets"]:
        row = low_epsilon["by_budget"][str(epsilon)]
        low_rows.append(
            f"{epsilon:g} & {100*row['subject_balanced_accuracy']:.2f} & "
            f"{100*row['macro_f1']:.2f} & {row['nll']:.3f} & "
            f"{row['brier']:.3f} & {row['ece']:.3f} \\\\"
        )
    (GENERATED / "uci_streamlined_low_epsilon_rows.tex").write_text(
        "\n".join(low_rows) + "\n", encoding="utf-8"
    )
    matched_rows = []
    method_order = [OURS, "DP-NTK", "DP-KIP", "DP-HP"]
    for epsilon in low_epsilon_baselines["budgets"]:
        values = {OURS: low_epsilon["by_budget"][str(epsilon)]}
        values.update({method: low_epsilon_baselines["by_method_budget"][method][str(epsilon)]
                       for method in method_order[1:]})
        best = {
            metric: (max(values[name][metric] for name in method_order)
                     if metric in ("subject_balanced_accuracy", "macro_f1")
                     else min(values[name][metric] for name in method_order))
            for metric in METRICS
        }
        for index, method in enumerate(method_order):
            row = values[method]
            cells = [100*row["subject_balanced_accuracy"], 100*row["macro_f1"],
                     row["nll"], row["brier"], row["ece"]]
            formatted = [f"{cells[0]:.2f}", f"{cells[1]:.2f}", f"{cells[2]:.3f}",
                         f"{cells[3]:.3f}", f"{cells[4]:.3f}"]
            for j, metric in enumerate(METRICS):
                if np.isclose(row[metric], best[metric]):
                    formatted[j] = f"\\textbf{{{formatted[j]}}}"
            epsilon_cell = f"{epsilon:g}" if index == 0 else ""
            matched_rows.append(epsilon_cell + " & " + method.replace("DP-", "DP--") +
                                " & " + " & ".join(formatted) + r" \\")
        if epsilon != low_epsilon_baselines["budgets"][-1]:
            matched_rows.append(r"\addlinespace[2pt]")
    (GENERATED / "uci_streamlined_low_epsilon_baseline_rows.tex").write_text(
        "\n".join(matched_rows) + "\n", encoding="utf-8"
    )
    public_rows = []
    for name in ["balanced_72", "balanced_144", "balanced_288", "balanced_576"]:
        row = public_sensitivity["summary"][name]
        public_rows.append(
            f"{row['fit_pairs']} & {100*row['subject_balanced_accuracy']:.2f} & "
            f"{100*row['macro_f1']:.2f} & {row['nll']:.3f} & "
            f"{row['brier']:.3f} & {row['ece']:.3f} \\\\"
        )
    (GENERATED / "uci_streamlined_public_pair_rows.tex").write_text(
        "\n".join(public_rows) + "\n", encoding="utf-8"
    )
    fullmap_rows = []
    for name in ["balanced_72", "balanced_144", "balanced_288", "canonical_576"]:
        row = fullmap_sensitivity["summary"][name]
        fullmap_rows.append(
            f"{row['fit_pairs']} & {100*row['subject_balanced_accuracy']:.2f} & "
            f"{100*row['macro_f1']:.2f} & {row['nll']:.3f} & "
            f"{row['brier']:.3f} & {row['ece']:.3f} \\\\"
        )
    (GENERATED / "uci_streamlined_fullmap_public_pair_rows.tex").write_text(
        "\n".join(fullmap_rows) + "\n", encoding="utf-8"
    )
    cohort_rows = []
    cohort_order = ["registered", "redraw_91401", "redraw_91402", "redraw_91403", "redraw_91404"]
    cohort_methods = [OURS, "KPG-RL", "DP-NTK"]
    for split_index, split_name in enumerate(cohort_order):
        values = cohort_splits["by_split_method"][split_name]
        best = {
            metric: (max(values[name][metric] for name in cohort_methods)
                     if metric in ("subject_balanced_accuracy", "macro_f1")
                     else min(values[name][metric] for name in cohort_methods))
            for metric in METRICS
        }
        for method_index, method in enumerate(cohort_methods):
            row = values[method]
            cells = [100*row["subject_balanced_accuracy"], 100*row["macro_f1"],
                     row["nll"], row["brier"], row["ece"]]
            formatted = [f"{cells[0]:.2f}", f"{cells[1]:.2f}", f"{cells[2]:.3f}",
                         f"{cells[3]:.3f}", f"{cells[4]:.3f}"]
            for j, metric in enumerate(METRICS):
                if np.isclose(row[metric], best[metric]):
                    formatted[j] = f"\\textbf{{{formatted[j]}}}"
            split_label = ("Registered" if split_name == "registered" else f"Redraw {split_index}") if method_index == 0 else ""
            cohort_rows.append(split_label + " & " + method.replace("DP-", "DP--") + " & " + " & ".join(formatted) + r" \\")
        if split_index != len(cohort_order) - 1:
            cohort_rows.append(r"\addlinespace[2pt]")
    (GENERATED / "uci_streamlined_cohort_split_rows.tex").write_text(
        "\n".join(cohort_rows) + "\n", encoding="utf-8"
    )
    evidence = {
        "complete": True,
        "source": str(RESULT),
        "scope": "Retrospective streamlining on an already executed fixed UCI HAR experiment; not pristine prospective confirmation",
        "streamlined_full": OURS,
        "former_label": "Ridge source teacher",
        "previous_full": PREVIOUS,
        "metrics": {name: metrics[name] for name in sorted(set(MAIN_ORDER + DECODER_ORDER))},
        "paired_comparisons": comparisons,
        "best_on_all_five_executable_endpoints": all(
            metrics[OURS][m] == (max(metrics[n][m] for n in MAIN_ORDER) if i < 2 else min(metrics[n][m] for n in MAIN_ORDER))
            for i, m in enumerate(METRICS)
        ),
        "new_private_queries": 0,
        "figure_1_and_2_changed": False,
        "prospective_confirmation_still_required": True,
        "low_epsilon_stress": low_epsilon["by_budget"],
        "low_epsilon_private_baselines": low_epsilon_baselines["by_method_budget"],
        "low_epsilon_private_baseline_failures": low_epsilon_baselines["failures"],
        "public_pair_sensitivity": public_sensitivity["summary"],
        "fullmap_public_pair_sensitivity": fullmap_sensitivity["summary"],
        "fullmap_public_pair_failures": fullmap_sensitivity["failures"],
        "repeated_cohort_splits": cohort_splits["by_split_method"],
        "repeated_cohort_split_failures": cohort_splits["failures"],
        "r4_same_packet_components": r4_components["by_variant"],
        "r4_component_paired_intervals": r4_components["paired_full_minus_control_utility"],
        "public_labels": 0,
    }
    (GENERATED / "neurocomputing_review2_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps({
        "full_ba": metrics[OURS]["subject_balanced_accuracy"],
        "previous_ba": metrics[PREVIOUS]["subject_balanced_accuracy"],
        "best_all_five": evidence["best_on_all_five_executable_endpoints"],
        "figure": str(FIGURES / "streamlined_uci_har.pdf"),
    }, indent=2))


if __name__ == "__main__":
    main()
