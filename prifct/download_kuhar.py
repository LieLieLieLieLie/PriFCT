"""Resume-safe acquisition of the preregistered KU-HAR v5 archive."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "kuhar"
DESTINATION = DATA / "2.Trimmed_interpolated_data.zip"
PARTIAL = DATA / "2.Trimmed_interpolated_data.zip.part"
URL = (
    "https://data.mendeley.com/public-files/datasets/45f952y38r/files/"
    "49c6120b-59fd-466c-97da-35d53a4be595/file_downloaded"
)
EXPECTED_BYTES = 161_492_605


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    # This Windows session has a broken DNS provider while direct HTTPS remains
    # usable.  The addresses were resolved immediately before acquisition with
    # Google's public DNS-over-HTTPS endpoint; TLS still verifies the original
    # host names, so this changes only name resolution.
    static_addresses = {
        "data.mendeley.com": "162.159.133.86",
        "prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com": "52.92.2.178",
    }
    original_getaddrinfo = socket.getaddrinfo

    def fixed_getaddrinfo(host, port, *args, **kwargs):
        return original_getaddrinfo(static_addresses.get(host, host), port, *args, **kwargs)

    socket.getaddrinfo = fixed_getaddrinfo
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = PARTIAL.stat().st_size if PARTIAL.exists() else 0
    if start > EXPECTED_BYTES:
        raise RuntimeError(f"partial file is unexpectedly large: {start}")
    request = urllib.request.Request(
        URL,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Range": f"bytes={start}-",
        },
    )
    began = time.time()
    with opener.open(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        if start and status != 206:
            raise RuntimeError(f"server ignored resume range at {start}: HTTP {status}")
        mode = "ab" if start else "wb"
        with PARTIAL.open(mode) as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                output.flush()
                os.fsync(output.fileno())
                size = output.tell()
                elapsed = max(time.time() - began, 1e-9)
                print(
                    json.dumps(
                        {
                            "bytes": size,
                            "percent": round(100 * size / EXPECTED_BYTES, 2),
                            "MiB_per_s": round((size - start) / elapsed / 2**20, 2),
                        }
                    ),
                    flush=True,
                )
    size = PARTIAL.stat().st_size
    if size != EXPECTED_BYTES:
        raise RuntimeError(f"incomplete download: {size} != {EXPECTED_BYTES}")
    PARTIAL.replace(DESTINATION)
    manifest = {
        "dataset": "KU-HAR",
        "doi": "10.17632/45f952y38r.5",
        "source_url": URL,
        "bytes": size,
        "sha256": sha256(DESTINATION),
    }
    (DATA / "acquisition.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
