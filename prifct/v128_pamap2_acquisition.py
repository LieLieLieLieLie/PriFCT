"""Pre-register and acquire the official PAMAP2 archive without utility access."""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/models/eswa_fedkapt"
OUT = BASE / "semantic_v128_pamap2_preregistration"
DATA = ROOT / "data/pamap2"
PAGE = "https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring"
ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register():
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    design = {
        "scope": "Prospective untouched PAMAP2 confirmation of the byte-frozen v127 observable two-branch adapter",
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": PAGE, "archive_url": ARCHIVE_URL, "doi": "10.24432/C5NW2H",
        "no_archive_or_utility_accessed_before_registration": True,
        "participants": list(range(1, 10)),
        "participant_permutation_seed": 126701,
        "participant_permutation": [9, 2, 6, 8, 1, 4, 3, 7, 5],
        "public_people": [9, 2], "source_people": [6, 8],
        "adaptation_people": [1], "test_people": [4, 3, 7, 5],
        "activities": {"1": "lying", "2": "sitting", "3": "standing", "4": "walking",
                       "5": "running", "6": "cycling", "7": "Nordic walking",
                       "12": "ascending stairs", "13": "descending stairs",
                       "16": "vacuum cleaning", "17": "ironing", "24": "rope jumping"},
        "activity_policy": "Protocol recordings and the twelve registered activities only; discard transition label 0 and do not use optional sessions to repair support",
        "locations": ["hand", "chest", "ankle"],
        "views": "First documented 16g three-axis accelerometer versus three-axis gyroscope at the same IMU location; do not use the second acceleration range, magnetometer, heart rate or invalid orientation",
        "directions": [[location, source, target] for location in ["hand", "chest", "ankle"]
                       for source, target in [("acc", "gyr"), ("gyr", "acc")]],
        "window": "Nonoverlapping absolute two-second timestamp bins within participant and activity; both modalities must have at least 180 finite samples, first/last support within 0.1 s of bin edges, and no internal gap above 0.05 s",
        "descriptor": "Per axis plus magnitude: mean, population SD, RMS, range, below-5-Hz and 5-to-50-Hz positive-frequency power fractions; 24 coordinates per view",
        "missingness": "No long-gap interpolation; modality-specific nonfinite rows are dropped before the registered paired-window completeness test",
        "public_fit_per_person": 192, "public_held_per_person": 48,
        "public_seed": 128031, "source_cap": 2048, "adaptation_cap": 1024,
        "source_seed": 128041, "adaptation_seed": 128042,
        "budgets": [2.0, 10.0], "release_seeds": [128051, 128052, 128053],
        "methods": ["PriFCT-v127", "DP-HP", "DP-NTK", "DP-KIP", "KPG-RL"],
        "primary_endpoint": "Equal-person mean supported-class balanced accuracy across all six directions, both budgets and three release seeds",
        "secondary_endpoints": ["Macro-F1", "NLL", "Brier", "ECE"],
        "claim_gate": "PriFCT fixed-private-envelope normalized five-endpoint regret below zero and balanced accuracy no lower than the strongest private comparator; report every endpoint, person, direction, budget and failure",
        "no_retuning_after_utility": True,
        "frozen_adapter_sha256": sha(BASE / "semantic_v127_three_panel_reliability_gate/protocol.json"),
        "implementation_sha256": sha(Path(__file__)),
    }
    path = OUT / "design_before_acquisition.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8")); design["registered_utc"] = previous["registered_utc"]
        if previous != design: raise RuntimeError("PAMAP2 pre-acquisition design changed")
    else: path.write_text(json.dumps(design, indent=2), encoding="utf-8")
    return design


def acquire():
    design = register(); archive = DATA / "official.zip"; page = OUT / "official_page.html"
    session = requests.Session(); session.headers["User-Agent"] = "Academic reproducibility acquisition"
    if not page.exists():
        response = session.get(PAGE, timeout=(20, 60)); response.raise_for_status(); page.write_bytes(response.content)
    if not archive.exists():
        if shutil.disk_usage(DATA).free < 10 * 2**30: raise RuntimeError("Preserve at least 10 GiB free space")
        partial = archive.with_suffix(".zip.partial")
        with session.get(ARCHIVE_URL, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with partial.open("wb") as stream:
                for chunk in response.iter_content(2**20):
                    if chunk: stream.write(chunk)
        partial.replace(archive)
    manifest = {"complete": True, "archive": str(archive.relative_to(ROOT)),
                "archive_bytes": archive.stat().st_size, "archive_sha256": sha(archive),
                "official_page_sha256": sha(page),
                "design_sha256": sha(OUT / "design_before_acquisition.json"),
                "no_classifier_utility": True}
    (OUT / "acquisition.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__": acquire()
