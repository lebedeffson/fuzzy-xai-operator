#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--source-root", type=Path, default=Path("/tmp/h10c5_sources"))
    args = parser.parse_args()
    root = args.root.resolve()
    source = args.source_root.resolve()
    swe = source / "swebench_lite_test.parquet"
    bugs = source / "BugsInPy"
    zenodo = source / "zenodo_8376824.json"
    records = [
        {
            "source_id": "SWE-bench_Lite_test",
            "role": "executed_primary_screening_source",
            "path": str(swe),
            "available": swe.exists(),
            "sha256": _sha256(swe) if swe.exists() else None,
            "expected_sha256": "7a21f37b8bc179c7db5beeb14e88ac538ba283455c776e6b2535bbfb6e3551b4",
            "revision": "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2",
        },
        {
            "source_id": "BugsInPy",
            "role": "registered_secondary_repository_source_not_scored",
            "path": str(bugs),
            "available": (bugs / ".git").exists(),
            "revision": (
                subprocess.check_output(["git", "-C", str(bugs), "rev-parse", "HEAD"], text=True).strip()
                if (bugs / ".git").exists()
                else None
            ),
        },
        {
            "source_id": "defect4ML",
            "role": "registered_domain_source_not_downloaded_or_scored",
            "metadata_path": str(zenodo),
            "metadata_available": zenodo.exists(),
            "archive_size_bytes": 2_939_196_386,
            "archive_md5": "369726af86ef206b701cce3af6ab88ed",
            "limitation": "The 2.94 GB archive was registered but not silently substituted by another source.",
        },
    ]
    expected = records[0]["expected_sha256"]
    status = "PASS" if records[0]["available"] and records[0]["sha256"] == expected else "BLOCKED_PRIMARY_SOURCE"
    payload = {"status": status, "sources": records}
    output = root / "reports/h10_c5"
    output.mkdir(parents=True, exist_ok=True)
    (output / "SOURCE_REGISTRY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "SOURCE_REGISTRY.md").write_text(
        "# H10-C5 Source Registry\n\n"
        + "\n".join(
            f"- {item['source_id']}: `{item['role']}`, available=`{item.get('available', item.get('metadata_available'))}`"
            for item in records
        )
        + f"\n\nStatus: `{status}`\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise SystemExit(status)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
