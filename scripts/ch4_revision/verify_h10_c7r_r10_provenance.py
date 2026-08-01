#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = ROOT / "results/h10_c7r_r10/R10_RELEASE_PROVENANCE.json"
IMPLEMENTATION_STATUS = (
    ROOT / "results/h10_c7r_r10/R10_IMPLEMENTATION_STATUS.json"
)
VERIFICATION_STATUS = (
    ROOT / "results/h10_c7r_r10/R10_VERIFICATION_STATUS.json"
)


def _git(*arguments: str) -> bytes:
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def main() -> int:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    implementation = str(provenance["implementation_commit"])
    release = str(provenance["release_commit"])
    changed = _git("diff", "--name-only", f"{implementation}..{release}")
    changed_files = tuple(changed.decode().splitlines())
    binary_diff = _git("diff", "--binary", f"{implementation}..{release}")
    diff_sha256 = hashlib.sha256(binary_diff).hexdigest()

    expected_files = tuple(provenance["changed_files"])
    if changed_files != expected_files:
        raise RuntimeError(
            f"release diff files changed: {changed_files!r} != {expected_files!r}"
        )
    if diff_sha256 != provenance["diff_sha256"]:
        raise RuntimeError("release diff SHA256 changed")
    if any(
        not path.startswith(("reports/h10_c7r_r10/", "results/h10_c7r_r10/"))
        for path in changed_files
    ):
        raise RuntimeError("release diff contains implementation paths")
    if bool(provenance["code_changed_between_implementation_and_release"]):
        raise RuntimeError("provenance incorrectly classifies a report-only diff")

    implementation_status = json.loads(
        IMPLEMENTATION_STATUS.read_text(encoding="utf-8")
    )
    verification_status = json.loads(
        VERIFICATION_STATUS.read_text(encoding="utf-8")
    )
    checks = {
        "implementation_commit": (
            implementation_status["implementation_commit"] == implementation
        ),
        "release_commit": implementation_status["release_commit"] == release,
        "implementation_ci": (
            implementation_status["implementation_ci_run"]
            == provenance["implementation_ci"]["run_id"]
            == verification_status["implementation_ci"]["run_id"]
        ),
        "release_ci": (
            implementation_status["release_ci_run"]
            == provenance["release_ci"]["run_id"]
            == verification_status["release_ci"]["run_id"]
        ),
        "release_ci_head": (
            verification_status["release_ci"]["head_sha"] == release
        ),
        "implementation_ci_head": (
            verification_status["implementation_ci"]["head_sha"]
            == implementation
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"R10 provenance mismatch: {checks}")
    print(
        json.dumps(
            {
                "status": "R10_RELEASE_PROVENANCE_PASS",
                "checks": checks,
                "implementation_commit": implementation,
                "release_commit": release,
                "changed_file_count": len(changed_files),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
