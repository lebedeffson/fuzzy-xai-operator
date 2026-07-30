#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVENANCE_PATH = (
    ROOT / "results/h10_c7r_r10/R10_RELEASE_PROVENANCE.json"
)
PACKAGE_FILES = (
    Path("PROJECT_MEMORY.md"),
    Path("protocol/h10_c7r_r10/PARENT_IMMUTABILITY.json"),
    Path("protocol/h10_c7r_r10/R10_DEVELOPMENT_PROTOCOL_LOCK.json"),
    Path(
        "protocol/h10_c7r_r10/recollection/"
        "R10_TECHNICAL_BARRIER_IMAGE_LOCK.json"
    ),
    Path(
        "protocol/h10_c7r_r10/recollection/"
        "R10_TECHNICAL_BARRIER_LOCK.json"
    ),
    Path(
        "protocol/h10_c7r_r10/recollection/"
        "R10_TECHNICAL_BARRIER_RUNTIME_REGISTRY.jsonl"
    ),
    Path(
        "protocol/h10_c7r_r10/recollection/"
        "R10_TECHNICAL_BARRIER_SELECTION.jsonl"
    ),
    Path("reports/h10_c7r_r10/R10_CLAIM_LINT.json"),
    Path("reports/h10_c7r_r10/R10_IMPLEMENTATION_REPORT.md"),
    Path("reports/h10_c7r_r10/R10_RECOLLECTION_RUN_1_REPORT.md"),
    Path("reports/h10_c7r_r10/R10_REPRODUCTION.md"),
    Path("results/h10_c7r_r10/R10_IMPLEMENTATION_STATUS.json"),
    Path(
        "results/h10_c7r_r10/recollection/"
        "R10_RECOLLECTION_INPUT_AUDIT.json"
    ),
    Path(
        "results/h10_c7r_r10/recollection/"
        "R10_RECOLLECTION_RUN_1_STATUS.json"
    ),
    Path("results/h10_c7r_r10/R10_RELEASE_PROVENANCE.json"),
    Path("results/h10_c7r_r10/R10_VERIFICATION_STATUS.json"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_clean() -> str:
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
    )
    if status:
        raise RuntimeError("build requires a clean worktree")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def _zip_tree(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build compact R10 provenance correction evidence"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-release", type=Path, required=True)
    args = parser.parse_args()

    package_commit = _assert_clean()
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    source_release = args.source_release.resolve()
    if not source_release.is_file():
        raise FileNotFoundError(source_release)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    short = package_commit[:12]
    archive_path = (
        output
        / f"fuzzyxai-h10-c7r-r10-provenance-evidence-{short}.zip"
    )

    with tempfile.TemporaryDirectory(prefix="h10-c7r-r10-provenance-") as temp:
        package = Path(temp) / f"fuzzyxai-h10-c7r-r10-provenance-{short}"
        package.mkdir()
        for relative in PACKAGE_FILES:
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)

        _write_json(
            package / "PACKAGE_IDENTITY.json",
            {
                "package_commit": package_commit,
                "implementation_commit": provenance["implementation_commit"],
                "release_commit": provenance["release_commit"],
                "implementation_ci_run": provenance["implementation_ci"][
                    "run_id"
                ],
                "release_ci_run": provenance["release_ci"]["run_id"],
                "scientific_result": "NOT_EVALUATED",
            },
        )
        _write_json(
            package / "SOURCE_RELEASE_REFERENCE.json",
            {
                "embedded": False,
                "file_name": source_release.name,
                "sha256": _sha256(source_release),
                "release_commit": provenance["release_commit"],
            },
        )
        checksums = []
        for path in sorted(item for item in package.rglob("*") if item.is_file()):
            checksums.append(
                f"{_sha256(path)}  {path.relative_to(package).as_posix()}"
            )
        (package / "SHA256SUMS").write_text(
            "\n".join(checksums) + "\n",
            encoding="utf-8",
        )
        _zip_tree(package.parent, archive_path)

    sidecar = archive_path.with_suffix(".zip.sha256")
    sidecar.write_text(
        f"{_sha256(archive_path)}  {archive_path.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "sha256": _sha256(archive_path),
                "source_release_embedded": False,
                "source_release_sha256": _sha256(source_release),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
