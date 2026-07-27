#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "release_artifacts"
PREFIXES = (
    "data/h10_c6_noise/",
    "framework/fuzzyxai/operators_manifest_final_practical_addendum.yaml",
    "protocol/h10_c5c_evidence_retrieval/",
    "reports/chapter_updates/",
    "reports/final_practical/",
    "reports/h10_c5_pilot/",
    "reports/h10_c5c_posthoc/",
    "reports/h10_c6_noise/",
    "reports/integrations/",
    "results/final_practical/",
    "results/h10_c5_pilot/",
    "results/h10_c5c_posthoc/",
    "results/h10_c6_noise/",
    "results/integrations/",
)
EXACT = {
    "PROJECT_MEMORY.md",
    "RELEASE_STATUS.md",
    "reports/chapter_revision/CLAIM_LINT.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def _tracked_inputs() -> list[Path]:
    names = _git("ls-files").splitlines()
    selected = [
        ROOT / name
        for name in names
        if name in EXACT or name.startswith(PREFIXES)
    ]
    missing = [path for path in selected if not path.is_file()]
    if missing:
        raise RuntimeError(f"tracked evidence is missing: {missing}")
    return sorted(selected)


def _verify_checksums(root: Path) -> int:
    lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    for line in lines:
        expected, name = line.split(maxsplit=1)
        path = root / name.lstrip("*")
        if _sha256(path) != expected:
            raise RuntimeError(f"evidence checksum mismatch: {name}")
    return len(lines)


def main() -> None:
    commit = _git("rev-parse", "HEAD")
    if _git("write-tree") != _git("rev-parse", "HEAD^{tree}"):
        raise RuntimeError("index differs from HEAD")
    short = commit[:12]
    source = OUTPUT / f"fuzzyxai-source-release-{short}.zip"
    source_sidecar = source.with_suffix(".zip.sha256")
    if not source.is_file() or not source_sidecar.is_file():
        raise RuntimeError("build the clean source release first")
    expected_source = source_sidecar.read_text(encoding="ascii").split()[0]
    if _sha256(source) != expected_source:
        raise RuntimeError("source release sidecar mismatch")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = OUTPUT / f"fuzzyxai-ch4-final-practical-evidence-{short}.zip"
    with tempfile.TemporaryDirectory() as temporary:
        package = Path(temporary) / "fuzzyxai-ch4-final-practical-evidence"
        package.mkdir()
        for source_path in _tracked_inputs():
            relative = source_path.relative_to(ROOT)
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
        source_root = package / "source_release"
        source_root.mkdir()
        shutil.copy2(source, source_root / source.name)
        shutil.copy2(source_sidecar, source_root / source_sidecar.name)
        identity = {
            "schema_version": "1.0",
            "commit": commit,
            "branch": _git("branch", "--show-current"),
            "source_release": source.name,
            "source_release_sha256": expected_source,
            "docx_modified": False,
            "parent_results_recalculated": False,
        }
        (package / "PACKAGE_IDENTITY.json").write_text(
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files = sorted(
            path for path in package.rglob("*") if path.is_file()
        )
        checksum_lines = [
            f"{_sha256(path)}  {path.relative_to(package).as_posix()}"
            for path in files
        ]
        (package / "SHA256SUMS").write_text(
            "\n".join(checksum_lines) + "\n",
            encoding="ascii",
        )
        with zipfile.ZipFile(
            archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as output:
            for path in sorted(
                item for item in package.rglob("*") if item.is_file()
            ):
                info = zipfile.ZipInfo(
                    path.relative_to(package.parent).as_posix(),
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.external_attr = 0o100644 << 16
                output.writestr(
                    info,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        with tempfile.TemporaryDirectory() as extracted:
            with zipfile.ZipFile(archive) as packaged:
                packaged.extractall(extracted)
            checked = _verify_checksums(
                Path(extracted) / package.name
            )
    archive_sha = _sha256(archive)
    archive.with_suffix(".zip.sha256").write_text(
        f"{archive_sha}  {archive.name}\n",
        encoding="ascii",
    )
    manifest = {
        "commit": commit,
        "archive": archive.name,
        "sha256": archive_sha,
        "files_verified": checked,
        "source_release_sha256": expected_source,
    }
    (OUTPUT / "final_practical_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
