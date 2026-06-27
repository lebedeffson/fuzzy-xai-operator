from __future__ import annotations

import zipfile
import json
from pathlib import Path

from .common import ROOT, release_metadata


PACKAGE = ROOT / "fuzzyxai_final_audit_package.zip"
RUNTIME_PACKAGE = ROOT / "fuzzyxai_doctoral_runtime_release.zip"
VISUAL = ROOT / "visual_artifacts_latest.zip"

INCLUDE = [
    "fuzzyxai/audit",
    "tests/audit",
    "reports/audit",
    "reports/studio",
    "reports/studio_batch",
    "reports/chapter5/studio_tables",
    "configs/studio_scenarios",
    "docs/chapters",
    "figures",
    "visual_artifacts_latest.zip",
]

RUNTIME_INCLUDE = [
    "fuzzyxai",
    "apps",
    "configs/studio_scenarios",
    "patches",
    "reports/audit",
    "reports/studio",
    "reports/studio_batch",
    "reports/chapter5/studio_tables",
    "docs/chapters",
    "docs/DATA_AND_SCENARIO_POLICY.md",
    "tests/audit",
    "tests/test_studio_operator_engine.py",
    "tests/test_fuzzyxai_studio_demo_readiness.py",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    "README_DOCTORAL_RELEASE.md",
    "RELEASE_NOTES.md",
]

EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", "venv", "node_modules"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDE_NAMES = {".DS_Store"}


def _allowed(path: Path) -> bool:
    parts = set(path.parts)
    if parts & EXCLUDE_PARTS:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    if path.name in EXCLUDE_NAMES:
        return False
    return True


def _add_path(zf: zipfile.ZipFile, path: Path) -> None:
    if not path.exists():
        return
    if path.is_file():
        if _allowed(path):
            zf.write(path, path.relative_to(ROOT).as_posix())
        return
    for item in sorted(path.rglob("*")):
        if item.is_file() and _allowed(item):
            zf.write(item, item.relative_to(ROOT).as_posix())


def build_visual_zip() -> Path:
    if VISUAL.exists():
        VISUAL.unlink()
    with zipfile.ZipFile(VISUAL, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in ["figures", "reports/studio/fuzzyxai_studio_smoke.png", "reports/studio/hybrid_xiris_case_001.json", "reports/audit"]:
            _add_path(zf, ROOT / rel)
    return VISUAL


def build_audit_package() -> Path:
    build_visual_zip()
    if PACKAGE.exists():
        PACKAGE.unlink()
    with zipfile.ZipFile(PACKAGE, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in INCLUDE:
            _add_path(zf, ROOT / rel)
    return PACKAGE


def build_runtime_release() -> Path:
    if RUNTIME_PACKAGE.exists():
        RUNTIME_PACKAGE.unlink()
    with zipfile.ZipFile(RUNTIME_PACKAGE, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in RUNTIME_INCLUDE:
            _add_path(zf, ROOT / rel)
        _add_path(zf, PACKAGE)
        _add_path(zf, VISUAL)
        zf.writestr("RELEASE_METADATA.json", json.dumps(release_metadata(), ensure_ascii=False, indent=2))
    return RUNTIME_PACKAGE


def main() -> None:
    print(build_audit_package())
    print(build_runtime_release())


if __name__ == "__main__":
    main()
