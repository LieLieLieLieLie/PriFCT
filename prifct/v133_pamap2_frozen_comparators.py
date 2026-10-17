"""Frozen same-resource PAMAP2 comparator phase after PriFCT exposure."""
from __future__ import annotations

import os
import sys
import types
import inspect

os.urandom = lambda n: bytes(n)
_asyncio = types.ModuleType("asyncio")
_asyncio.iscoroutinefunction = inspect.iscoroutinefunction
_asyncio.coroutines = types.SimpleNamespace(_is_coroutine=object())
sys.modules.setdefault("asyncio", _asyncio)

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import prifct.v132_pamap2_frozen_prifct as pamap
from prifct import v126_cuda_private_count_precision as private
from prifct import v111_cuda_semantic_core as gpu
from prifct.v6_private_generation import PrivateFeatureMap
from prifct.v7_dp_ntk import NTKFeatureMap
from prifct.v7_large_cohort_kip import fit as fit_kip


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/models/eswa_fedkapt"
OUT = BASE / "semantic_v133_pamap2_frozen_comparators"
SELECTIONS = BASE / "original_v7_research/confirmation_v7/protocol.json"
METHODS = ["DP-HP", "DP-NTK", "DP-KIP", "KPG-RL"]

# v132 was executed with ``python -m`` and its cached public-map pickle records
# the two lightweight signature classes under ``__main__``.  Expose identical
# aliases in this runner before loading those label-blind map artifacts.
ProjectedSignature = pamap.ProjectedSignature
MultiScaleSignature = pamap.MultiScaleSignature


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def register():
    gpu.require_cuda(); OUT.mkdir(parents=True, exist_ok=True)
    primary = pamap.register(); choices = read(SELECTIONS)["selections"]
    protocol = {
        "scope": "PAMAP2 comparator phase with PriFCT outcomes already sealed; no comparator or PriFCT tuning",
        "methods": METHODS, "classes": primary["classes"], "directions": primary["directions"],
        "budgets": primary["budgets"], "seeds": primary["seeds"], "roles": primary["roles"],
        "source_cap": primary["source_cap"], "adaptation_cap": primary["adaptation_cap"],
        "source_seed": primary["source_seed"], "adaptation_seed": primary["adaptation_seed"],
        "selections": {method: choices[method] for method in METHODS},
        "matched_prifct_protocol_sha256": sha(pamap.OUT / "protocol.json"),
        "matched_prifct_summary_sha256": sha(pamap.OUT / "summary.json"),
        "frozen_selection_registry_sha256": sha(SELECTIONS),
        "private_contract": "same 2048 labelled source descriptors, same maps, epsilon/delta and three release seeds; complete-record replacement DP",
        "context_contract": "KPG-RL is non-private context using the same source and unlabelled adaptation caps and the frozen development setting",
        "test_labels": "used only by the final evaluator",
        "implementation_sha256": sha(Path(__file__)),
    }
    path = OUT / "protocol.json"
    if path.exists() and read(path) != protocol:
        raise RuntimeError("Executed PAMAP2 comparator protocol changed")
    if not path.exists():
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol


def data_for_case(primary, item):
    roles = primary["roles"]
    location, sm, tm = item["location"], item["source_modality"], item["target_modality"]
    source = pamap.cohort(roles["source_people"], location, sm, include_labels=True)
    adaptation = pamap.cohort(roles["adaptation_people"], location, tm)
    test = pamap.cohort(roles["test_people"], location, tm, include_labels=True)
    source_ids = pamap.cap(len(source["x"]), primary["source_cap"], primary["source_seed"])
    adaptation_ids = pamap.cap(len(adaptation["x"]), primary["adaptation_cap"], primary["adaptation_seed"])
    return source, adaptation, test, source_ids, adaptation_ids


def append(row, probability, test, protocol):
    probability = gpu.host(private.export(probability)) if torch.is_tensor(probability) else np.asarray(probability, float)
    probability = np.maximum(probability, 1e-15); probability /= probability.sum(axis=1, keepdims=True)
    row.update(status="success", results=pamap.evaluate_metrics(test["y"], probability, test["subjects"], protocol["classes"]),
               protocol_sha256=sha(OUT / "protocol.json"))
    output_dir = OUT / "predictions"; output_dir.mkdir(exist_ok=True)
    output = output_dir / f"{row['id']}.npz"
    np.savez_compressed(output, probability=probability, y=test["y"], subjects=test["subjects"], bins=test["bins"])
    row["prediction_sha256"] = sha(output)
    with (OUT / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
    print("PAMAP2 comparator", row["id"], round(row["results"]["subject_balanced_accuracy"], 6), flush=True)


def run_private(method, protocol, primary, prepared):
    selection = protocol["selections"][method]; options = selection["options"]
    ledger = OUT / "results.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()] if ledger.exists() else []
    done = {row["id"] for row in rows}
    for item in prepared:
        source, adaptation, test, source_ids, adaptation_ids = data_for_case(primary, item)
        mapping = item["map"]["full" if options["representation"] == "function" else "point"]
        x = gpu.tensor(mapping.source(source["x"][source_ids])); xt = gpu.tensor(mapping.target(test["x"]))
        y = source["y"][source_ids]
        if method != "DP-KIP":
            public = gpu.tensor(mapping.source(item["source_public"]["x"][item["fit"]]))
            chosen = np.random.default_rng(128061).permutation(len(public))[:256]
            chosen = torch.as_tensor(chosen, device="cuda")
            distance = gpu.squared_distance(public[chosen], public[chosen])
            triangle = torch.triu_indices(len(chosen), len(chosen), offset=1, device="cuda")
            width = max(torch.quantile(distance[triangle[0], triangle[1]], .5).item(), 1e-4)
            feature = (PrivateFeatureMap.hermite(x.shape[1], width, order=options["order"],
                                                 product_order=options["product_order"])
                       if method == "DP-HP" else
                       NTKFeatureMap(x.shape[1], protocol["classes"], hidden=options["hidden"], seed=77271))
        for epsilon, seed in itertools.product(protocol["budgets"], protocol["seeds"]):
            ident = f"{item['case']}_{method}_e{epsilon}_s{seed}"
            if ident in done:
                continue
            row = {"id": ident, "case_id": item["case"], "method": method,
                   "epsilon": epsilon, "seed": seed, "options": options}
            try:
                if method == "DP-KIP":
                    settings = {key: value for key, value in options.items()
                                if key not in ["representation", "initialization", "temperature", "calibration"]}
                    scale = 1.0 if options.get("initialization", "native_gaussian") == "native_gaussian" else 1 / math.sqrt(x.shape[1])
                    model = fit_kip(x, y, protocol["classes"], epsilon=epsilon, seed=seed,
                                    accountant="gdp", initialization_scale=scale, **settings)
                    probability = private.predict_kip(model, xt, options.get("temperature", options.get("calibration")))
                    row["diagnostics"] = model["diagnostics"]
                else:
                    packet = private.release(x, y, protocol["classes"], feature, epsilon, seed)
                    weights = torch.ones(packet.raw.shape[1] - 1, device="cuda", dtype=torch.float32)
                    if method == "DP-HP":
                        weights[x.shape[1] * (options["order"] + 1):] = options["gamma"]
                    generator = private.fit_receiver(packet, feature, steps=options["steps"], hidden=128,
                                                     samples_per_class=16, seed=77281, loss_weights=weights)
                    synthetic, labels = generator.sample(n_per_class=128, seed=77291)
                    C = 1.0 if options["calibration"].endswith("C1") else 10.0
                    model = private.fit_lr(synthetic, labels, protocol["classes"], C)
                    probability = private.predict_lr(model, xt)
                    row["diagnostics"] = {"generator": generator.diagnostics,
                                          "classifier": model["diagnostics"], "sigma": packet.sigma_query}
                append(row, probability, test, protocol)
            except Exception as error:
                row.update(status="failed", error=repr(error), protocol_sha256=sha(OUT / "protocol.json"))
                with ledger.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print("PAMAP2 comparator failure", ident, repr(error), flush=True)
            rows.append(row); done.add(ident)


def run_kpg(protocol, primary, prepared):
    from prifct.v7_native_structural import kpg, barycentric_classifier
    options = protocol["selections"]["KPG-RL"]["options"]
    ledger = OUT / "results.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()] if ledger.exists() else []
    done = {row["id"] for row in rows}
    for item in prepared:
        ident = f"{item['case']}_KPG-RL"
        if ident in done:
            continue
        source, adaptation, test, source_ids, adaptation_ids = data_for_case(primary, item)
        mapping = item["map"]["full" if options["representation"] == "function" else "point"]
        x = mapping.source(source["x"][source_ids]); tx = mapping.target(adaptation["x"][adaptation_ids]); xt = mapping.target(test["x"])
        public_ids = np.random.default_rng(128063).permutation(len(item["fit"]))[:32]
        public_rows = item["fit"][public_ids]
        settings = {key: value for key, value in options.items() if key not in ["representation", "classifier_C"]}
        row = {"id": ident, "case_id": item["case"], "method": "KPG-RL", "epsilon": "none",
               "seed": 128063, "options": options}
        try:
            plan, diagnostics = kpg(x, tx, mapping.source(item["source_public"]["x"][public_rows]),
                                    mapping.target(item["target_public"]["x"][public_rows]), **settings)
            classifier, head = barycentric_classifier(plan, tx, source["y"][source_ids],
                                                      classes=protocol["classes"], C=options["classifier_C"], seed=128064)
            row["diagnostics"] = diagnostics
            append(row, classifier.predict_proba(xt), test, protocol)
        except Exception as error:
            row.update(status="failed", error=repr(error), protocol_sha256=sha(OUT / "protocol.json"))
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print("PAMAP2 KPG failure", ident, repr(error), flush=True)
        rows.append(row); done.add(ident)


def summarize(protocol):
    rows = [json.loads(line) for line in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = {"methods": {}, "failures": sum(row["status"] != "success" for row in rows)}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        successful = [row for row in selected if row["status"] == "success"]
        if successful:
            summary["methods"][method] = {"outcomes": len(selected), "successes": len(successful),
                **{endpoint: float(np.mean([row["results"][endpoint] for row in successful])) for endpoint in pamap.ENDPOINTS}}
    expected = {"DP-HP": 36, "DP-NTK": 36, "DP-KIP": 36, "KPG-RL": 6}
    summary["complete"] = all(summary["methods"].get(method, {}).get("successes") == count for method, count in expected.items())
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("method", choices=METHODS + ["summary"])
    args = parser.parse_args(); protocol = register(); primary = pamap.register(); prepared, _ = pamap.prepare(primary)
    if args.method == "KPG-RL": run_kpg(protocol, primary, prepared)
    elif args.method != "summary": run_private(args.method, protocol, primary, prepared)
    summarize(protocol)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
