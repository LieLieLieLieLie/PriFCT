"""Freeze a schema-only PAMAP2 feasibility amendment before any utility is read.

The official archive was acquired after v128 registration.  This script reads
ZIP directory metadata only.  It does not parse labels, sensor values, or run a
classifier.  PAMAP2 subject 109 is a known incomplete recording; the amendment
removes any participant whose protocol member is below 25% of the median member
size and then shifts the already registered permutation left without reshuffling.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/models/eswa_fedkapt"
PRE = BASE / "semantic_v128_pamap2_preregistration"
OUT = BASE / "semantic_v130_pamap2_preutility_feasibility"
OUTER = Path(r"E:\Downloads\pamap2+physical+activity+monitoring.zip")
INNER = Path(r"E:\Downloads\PAMAP2_Dataset.zip")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    design_path = PRE / "design_before_acquisition.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if not OUTER.is_file() or not INNER.is_file():
        raise FileNotFoundError("The acquired official PAMAP2 outer and inner ZIPs are required")

    with zipfile.ZipFile(OUTER) as archive:
        outer_members = {i.filename: i.file_size for i in archive.infolist()}
    with zipfile.ZipFile(INNER) as archive:
        member_sizes = {}
        for participant in design["participants"]:
            name = f"PAMAP2_Dataset/Protocol/subject10{participant}.dat"
            member_sizes[str(participant)] = archive.getinfo(name).file_size

    median_bytes = float(statistics.median(member_sizes.values()))
    threshold_bytes = 0.25 * median_bytes
    excluded = [int(p) for p, size in member_sizes.items() if size < threshold_bytes]
    amended_permutation = [p for p in design["participant_permutation"] if p not in excluded]
    if len(amended_permutation) < 8:
        raise RuntimeError("Fewer than eight complete protocol participants remain")

    amendment = {
        "scope": "PAMAP2 schema-only feasibility amendment before descriptors or utility",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_design": str(design_path.relative_to(ROOT)),
        "original_design_sha256": file_sha256(design_path),
        "archive_url_registered": design["archive_url"],
        "acquisition_path": str(OUTER),
        "outer_archive_bytes": OUTER.stat().st_size,
        "outer_archive_sha256": file_sha256(OUTER),
        "inner_archive_bytes": INNER.stat().st_size,
        "inner_archive_sha256": file_sha256(INNER),
        "outer_members": outer_members,
        "protocol_member_bytes": member_sizes,
        "availability_rule": "exclude protocol members smaller than 25% of the median protocol-member byte size",
        "median_protocol_member_bytes": median_bytes,
        "threshold_bytes": threshold_bytes,
        "excluded_participants": excluded,
        "reason": "subject 109 is a markedly incomplete protocol recording; no labels, sensor values, descriptors, predictions, or utility were inspected",
        "amendment_rule": "delete excluded participants from the frozen permutation and shift roles left without reshuffling",
        "participant_permutation": amended_permutation,
        "public_people": amended_permutation[0:2],
        "source_people": amended_permutation[2:4],
        "adaptation_people": amended_permutation[4:5],
        "test_people": amended_permutation[5:8],
        "all_other_v128_fields_frozen": True,
        "utility_accessed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "availability_amendment_before_utility.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        amendment["created_utc"] = previous["created_utc"]
        if previous != amendment:
            raise RuntimeError("The pre-utility availability amendment changed")
    else:
        path.write_text(json.dumps(amendment, indent=2), encoding="utf-8")
    print(json.dumps(amendment, indent=2))


if __name__ == "__main__":
    main()
