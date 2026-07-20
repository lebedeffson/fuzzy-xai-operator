#!/usr/bin/env python3
"""Build a deterministic formative evidence bundle from the verified files."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports/strong_confirmatory"
INCLUDE = (
    "framework/fuzzyxai/fuzzyxai/strong_confirmatory",
    "scripts/strong_confirmatory",
    "scripts/chapter4",
    "tests/strong_confirmatory",
    "study/strong_confirmatory",
    "release_evidence/strong_confirmatory/formative",
    "dissertation_artifacts/strong_confirmatory/chapter4_formative",
    "reports/strong_confirmatory/FORMATIVE_HANDOFF.md",
    "PROJECT_MEMORY.md",
    "RELEASE_STATUS.md",
    "Makefile",
)


def main() -> None:
    subprocess.run([str(_python()), "scripts/strong_confirmatory/verify_formative.py"], cwd=ROOT, check=True)
    implementation = json.loads((ROOT / "study/strong_confirmatory/manifest.json").read_text(encoding="utf-8"))["implementation_commit"]
    files = _files()
    manifest = {
        "schema_version": "1.0",
        "bundle_type": "formative_evidence_only",
        "implementation_commit": implementation,
        "confirmatory_test_opened": False,
        "confirmatory_claim_allowed": False,
        "stable_release_allowed": False,
        "files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path), "size": path.stat().st_size} for path in files],
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    archive = REPORTS / f"fuzzyxai-strong-confirmatory-formative-{implementation[:12]}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("BUNDLE_MANIFEST.json", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        bundle.writestr(info, json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    digest = _sha(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"PASS: strong_formative_bundle files={len(files)} sha256={digest} path={archive.relative_to(ROOT)}")


def _files() -> list[Path]:
    files: set[Path] = set()
    for item in INCLUDE:
        path = ROOT / item
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*") if candidate.is_file() and "__pycache__" not in candidate.parts)
        else:
            raise SystemExit(f"FAIL: bundle input missing: {item}")
    lock = ROOT / "study/strong_confirmatory/confirmatory_protocol_lock.json"
    files.discard(lock)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _python() -> Path:
    import sys

    return Path(sys.executable)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
