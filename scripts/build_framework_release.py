"""Build and verify a clean source archive from the committed Git tree."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "release_artifacts"
FORBIDDEN_PARTS = {
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "site/dubnaxai",
}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
REQUIRED_PATHS = {
    "fuzzy-xai-operator/PROJECT_MEMORY.md",
    "fuzzy-xai-operator/RELEASE_STATUS.md",
    "fuzzy-xai-operator/TEST_REPORT.txt",
    "fuzzy-xai-operator/framework/fuzzyxai/operators_manifest.yaml",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/evidence/contracts.py",
    "fuzzy-xai-operator/framework/fuzzyxai/fuzzyxai/visualization/explanation_dashboard.py",
    "fuzzy-xai-operator/framework/fuzzyxai/matlab/+fuzzyxai/dashboard.m",
    "fuzzy-xai-operator/examples/object_85_training_trace.py",
}


def run_git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_archive(path: Path) -> tuple[int, list[str]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        name_set = set(names)
        missing = sorted(REQUIRED_PATHS - name_set)
        if missing:
            raise RuntimeError(f"release archive is missing required files: {missing}")
        forbidden = []
        for name in names:
            normalized = name.rstrip("/")
            if any(part in normalized for part in FORBIDDEN_PARTS):
                forbidden.append(name)
            if Path(normalized).suffix in FORBIDDEN_SUFFIXES:
                forbidden.append(name)
        if forbidden:
            raise RuntimeError(f"release archive contains generated or quarantined files: {sorted(set(forbidden))}")
        with tempfile.TemporaryDirectory() as temp_dir:
            archive.extractall(temp_dir)
            extracted_root = Path(temp_dir) / "fuzzy-xai-operator"
            if not (extracted_root / "pyproject.toml").is_file():
                raise RuntimeError("archive does not extract to one installable root")
    return len(names), missing


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    commit = run_git("rev-parse", "HEAD")
    short_commit = commit[:12]
    archive_path = OUTPUT_DIR / f"fuzzy-xai-operator-framework-{short_commit}.zip"
    subprocess.run(
        [
            "git",
            "archive",
            "--format=zip",
            "--prefix=fuzzy-xai-operator/",
            f"--output={archive_path}",
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
    )
    file_count, _ = validate_archive(archive_path)
    archive_hash = sha256(archive_path)
    checksum_path = archive_path.with_suffix(".zip.sha256")
    checksum_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="ascii")
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "branch": run_git("branch", "--show-current"),
        "archive": archive_path.name,
        "sha256": archive_hash,
        "file_count": file_count,
        "source": "git archive HEAD",
        "quarantined_site_included": False,
    }
    manifest_path = OUTPUT_DIR / "framework_release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: release_archive {archive_path}")
    print(f"PASS: release_sha256 {archive_hash}")
    print(f"PASS: release_cleanliness files={file_count}")


if __name__ == "__main__":
    main()
