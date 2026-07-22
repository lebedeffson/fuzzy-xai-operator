from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

from .common import ARTIFACT_ROOT, ROOT, sha256_file, write_json


SOURCE_PATHS = (
    "framework/fuzzyxai/fuzzyxai/audit_h10",
    "baselines/h10",
    "experiments/h10",
    "tests/h10",
    "config/h10_v19_exploratory.yaml",
    "config/h10_v19_protocol.yaml",
    "data_seed/v19_identity_anchors.json",
    ".github/workflows/h10-audit.yml",
    "requirements.lock",
    "README.md",
    "Makefile",
    ".gitignore",
)


def _tracked_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files", *SOURCE_PATHS], cwd=ROOT, text=True)
    return [line for line in output.splitlines() if line]


def _write_zip(path: Path, files: list[Path], prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(files):
            relative = source.relative_to(ROOT)
            if ".h10_private" in str(relative) or "label_vault" in str(relative) or source.suffix == ".key":
                raise RuntimeError(f"forbidden H10 private file in package: {relative}")
            info = zipfile.ZipInfo(f"{prefix}/{relative.as_posix()}", date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())


def build() -> None:
    closure = ARTIFACT_ROOT / "closure"
    source_zip = closure / "fuzzyxai-h10-v19-source.zip"
    artifact_zip = closure / "fuzzyxai-h10-v19-artifacts.zip"
    _write_zip(source_zip, [ROOT / path for path in _tracked_files()], "fuzzyxai-h10-v19-source")
    artifact_files = [
        path
        for path in ARTIFACT_ROOT.rglob("*")
        if path.is_file() and path not in {source_zip, artifact_zip} and "label_vault" not in path.name and path.suffix != ".key"
    ]
    _write_zip(artifact_zip, artifact_files, "fuzzyxai-h10-v19-artifacts")
    checksums = {
        source_zip.name: sha256_file(source_zip),
        artifact_zip.name: sha256_file(artifact_zip),
    }
    write_json(closure / "h10_v19_SHA256.json", checksums)
    (closure / "h10_v19_SHA256.txt").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(checksums.items())), encoding="ascii")
    for zip_path in (source_zip, artifact_zip):
        with zipfile.ZipFile(zip_path) as archive:
            forbidden = [name for name in archive.namelist() if "label_vault" in name or name.endswith(".key") or ".h10_private" in name]
            if forbidden:
                raise RuntimeError(f"private H10 payload in {zip_path}: {forbidden}")


if __name__ == "__main__":
    build()
