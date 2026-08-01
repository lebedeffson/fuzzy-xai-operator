#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_JSON = ROOT / "reports/chapter_revision/LEGACY_EVIDENCE_IMMUTABILITY.json"
OUTPUT_MD = ROOT / "reports/chapter_revision/LEGACY_EVIDENCE_IMMUTABILITY.md"


def _git_last_modified(path: str) -> str:
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%cI", "--", path],
        cwd=ROOT,
        text=True,
    ).strip()


def _manifest_paths(path: Path) -> list[str]:
    paths = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        _digest, relative = line.split("  ", 1)
        paths.append(relative)
    return paths


def build() -> dict[str, object]:
    baseline = (
        json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        if OUTPUT_JSON.exists()
        else None
    )
    h10_c3 = _manifest_paths(
        ROOT / "protocol/h10_c4/H10_C3_BASELINE_SHA256SUMS"
    )
    h10_c4 = _manifest_paths(ROOT / "results/h10_c4/SHA256SUMS")
    selected = sorted(set(h10_c3 + h10_c4))
    files = []
    for relative in selected:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        data = path.read_bytes()
        if relative.startswith("artifacts/h10_c3_r4/"):
            protocol_id = "FXAI-H10-C3-R4-CONFIRMATORY-READINESS"
            result_id = "H10-C3-R4"
        else:
            protocol_id = "h10-c4-operational-utility-v1"
            result_id = "H10-C4"
        files.append(
            {
                "relative_path": relative,
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "protocol_id": protocol_id,
                "result_id": result_id,
                "modification_time": _git_last_modified(relative),
            }
        )
    aggregate = hashlib.sha256(
        "\n".join(
            f"{item['sha256']}  {item['relative_path']}" for item in files
        ).encode()
    ).hexdigest()
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    unchanged = (
        baseline is None
        or (
            baseline.get("aggregate_sha256") == aggregate
            and baseline.get("file_count") == len(files)
        )
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "baseline_commit": (
            baseline.get("baseline_commit")
            if baseline
            else current_commit
        ),
        "verification_commit": current_commit,
        "legacy_evidence_integrity": "PASS" if unchanged else "FAIL",
        "file_count": len(files),
        "aggregate_sha256": aggregate,
        "files": files,
    }
    if not unchanged:
        raise RuntimeError("legacy H10-C3/H10-C4 evidence changed")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(
        "\n".join(
            [
                "# Legacy Evidence Immutability",
                "",
                f"- Baseline commit: `{payload['baseline_commit']}`",
                f"- Checked files: `{len(files)}`",
                f"- Aggregate SHA256: `{aggregate}`",
                "- H10-C3 sealed evidence: immutable",
                "- H10-C4 held-out evidence: immutable",
                "- Status: `PASS`",
                "",
                "Re-run this script after all chapter-expansion work. Any changed",
                "size or SHA256 must set `LEGACY_EVIDENCE_INTEGRITY: FAIL`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


if __name__ == "__main__":
    result = build()
    print(json.dumps({k: result[k] for k in ("file_count", "aggregate_sha256")}))
