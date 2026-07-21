"""Build a separate full doctoral archive from the committed Git index."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_artifacts"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    commit = git("rev-parse", "HEAD")
    if git("write-tree") != git("rev-parse", "HEAD^{tree}"):
        raise RuntimeError("doctoral archive index differs from HEAD; commit staged changes first")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"fuzzyxai-doctoral-research-archive-{commit[:12]}.zip"
    with tempfile.TemporaryDirectory() as temp_dir:
        export_root = Path(temp_dir) / "fuzzy-xai-operator"
        export_root.mkdir()
        subprocess.run(
            ["git", "checkout-index", "--all", f"--prefix={export_root}{os.sep}"],
            cwd=ROOT,
            check=True,
        )
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for source in sorted(path for path in export_root.rglob("*") if path.is_file()):
                archive.write(source, source.relative_to(export_root.parent))
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        if not names or any(not name.startswith("fuzzy-xai-operator/") for name in names):
            raise RuntimeError("doctoral archive must extract to one project root")
        required = {
            "fuzzy-xai-operator/PROJECT_MEMORY.md",
            "fuzzy-xai-operator/release_evidence/chapter4_empirical_validation/manifest_sha256.json",
        }
        missing = required - set(names)
        if missing:
            raise RuntimeError(f"doctoral archive missing required files: {sorted(missing)}")
    digest = sha256(target)
    target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "branch": git("branch", "--show-current"),
        "archive": target.name,
        "sha256": digest,
        "file_count": len(names),
        "source": "git checkout-index at HEAD",
        "archive_scope": "full doctoral research history; not the clean framework source release",
    }
    (OUTPUT / "doctoral_research_archive_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PASS: doctoral_archive {target}")
    print(f"PASS: doctoral_archive_sha256 {digest}")
    print(f"PASS: doctoral_archive_files {len(names)}")


if __name__ == "__main__":
    main()
