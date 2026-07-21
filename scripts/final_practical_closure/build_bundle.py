#!/usr/bin/env python3
"""Build a deterministic practical-closure evidence bundle."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile

from common import ROOT, STUDY, sha256


INCLUDE = (
    "framework/fuzzyxai/fuzzyxai/practical_controller",
    "framework/fuzzyxai/fuzzyxai/strong_confirmatory/route.py",
    "scripts/final_practical_closure",
    "tests/practical_controller",
    "study/final_practical_closure",
    "release_evidence/final_practical_closure/formative",
    "release_evidence/final_practical_closure/claim_registry.json",
    "dissertation_artifacts/final_practical_closure/chapter4_formative",
    "reports/final_practical_closure/FORMATIVE_HANDOFF.md",
    "PROJECT_MEMORY.md",
    "Makefile",
)


def main() -> None:
    subprocess.run([sys.executable, "scripts/final_practical_closure/verify_formative.py"], cwd=ROOT, check=True)
    files = _files()
    study_manifest = json.loads((STUDY / "manifest.json").read_text(encoding="utf-8"))
    implementation = str(study_manifest["implementation_commit"])
    bundle_source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()
    manifest = {
        "schema_version": "1.0",
        "bundle_type": "formative_practical_closure_evidence",
        "implementation_commit": implementation,
        "bundle_source_commit": bundle_source_commit,
        "confirmatory_test_opened": False,
        "confirmatory_claim_allowed": False,
        "stable_release_allowed": False,
        "files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "size": path.stat().st_size} for path in files],
    }
    output = ROOT / "reports/final_practical_closure"
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"fuzzyxai-final-practical-formative-{implementation[:12]}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            _write(bundle, path.relative_to(ROOT).as_posix(), path.read_bytes())
        _write(bundle, "BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    digest = sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"PASS: practical_formative_bundle files={len(files)} sha256={digest} path={archive.relative_to(ROOT)}")


def _files():
    files = set()
    for name in INCLUDE:
        path = ROOT / name
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(candidate for candidate in path.rglob("*") if candidate.is_file() and "__pycache__" not in candidate.parts)
        else:
            raise SystemExit(f"FAIL: missing bundle input {name}")
    files.discard(STUDY / "confirmatory_protocol_lock.json")
    files.discard(STUDY / "confirmatory_opening_record.json")
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def _write(bundle, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    bundle.writestr(info, data)


if __name__ == "__main__":
    main()
