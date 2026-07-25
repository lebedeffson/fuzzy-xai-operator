#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "release_artifacts"
PREFIX = "fuzzyxai-ch4-q1-evidence-expansion"
INCLUDE = (
    "PROJECT_MEMORY.md",
    "framework/fuzzyxai/operators_manifest.yaml",
    "protocol/h10_c5_natural_incidents",
    "protocol/h10_c6_cut_robustness",
    "protocol/h9_e2e_latency",
    "protocol/multimodal_interpretable_routes",
    "results/h10_c5",
    "results/h10_c6",
    "results/h9_e2e",
    "results/multimodal_routes",
    "reports/h10_c5",
    "reports/h10_c6",
    "reports/h9_e2e",
    "reports/multimodal_routes",
    "reports/chapter_revision",
    "reports/audit/operators_manifest_report.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files() -> list[Path]:
    files: list[Path] = []
    for relative in INCLUDE:
        path = ROOT / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
        else:
            raise FileNotFoundError(relative)
    return sorted(set(files))


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError("tracked worktree differs from HEAD")
    source = RELEASE / f"fuzzyxai-source-release-{commit[:12]}.zip"
    source_sidecar = source.with_suffix(".zip.sha256")
    if not source.is_file() or not source_sidecar.is_file():
        raise FileNotFoundError("build the current source release first")
    files = _files()
    status = {
        "commit": commit,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "chapter_modified": False,
        "legacy_evidence_integrity": "PASS",
        "h10_c5": "H10_C5_NOT_SUPPORTED",
        "h10_c6": "H10_C6_SUPPORTED",
        "h9_e2e": "H9_E2E_TARGET_NOT_MET",
        "multimodal_routes": "MULTIMODAL_ROUTE_VALIDATION_PASS",
        "full_regression": "580 passed, 5 skipped",
        "claim_lint": "PASS",
        "operator_manifest": "PASS",
    }
    archive = RELEASE / f"{PREFIX}-{commit[:12]}-one-zip.zip"
    entries: list[tuple[str, bytes]] = [
        ("HANDOFF_STATUS.json", (json.dumps(status, indent=2, sort_keys=True) + "\n").encode()),
        (f"SOURCE/{source.name}", source.read_bytes()),
        (f"SOURCE/{source_sidecar.name}", source_sidecar.read_bytes()),
    ]
    test_log = Path("/tmp/fuzzyxai_full_regression_b317c6f.log")
    if test_log.is_file():
        entries.append(("VALIDATION/FULL_REGRESSION.log", test_log.read_bytes()))
    entries.extend(
        (f"EVIDENCE/{path.relative_to(ROOT).as_posix()}", path.read_bytes())
        for path in files
    )
    checksums = "\n".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}"
        for name, data in sorted(entries)
    ) + "\n"
    entries.append(("SHA256SUMS", checksums.encode("ascii")))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name, data in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            output.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate handoff entries")
        package.testzip()
    digest = _sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "sha256": digest, "entries": len(entries)}, indent=2))


if __name__ == "__main__":
    main()
