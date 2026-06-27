from __future__ import annotations

import zipfile
from pathlib import Path

from fuzzyxai.audit.common import ROOT


OUT = ROOT / "FuzzyXAI_full_delivery_package.zip"


def ok(path: Path) -> bool:
    bad_parts = {"__pycache__", ".pytest_cache", ".venv", "node_modules"}
    if set(path.parts) & bad_parts:
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return path.name != ".DS_Store"


def main() -> None:
    if OUT.exists():
        OUT.unlink()
    include_files = [
        ROOT / "FuzzyXAI_FINAL_DELIVERY_REPORT.md",
        ROOT / "fuzzyxai_final_audit_package.zip",
        ROOT / "fuzzyxai_doctoral_runtime_release.zip",
        ROOT / "visual_artifacts_latest.zip",
        ROOT / "reports/practice_demo/FuzzyXAI_practice_demo_package.zip",
        ROOT / "reports/practice_demo/FuzzyXAI_practice_screenshots.zip",
    ]
    include_dirs = [
        ROOT / "reports/practice_demo",
        ROOT / "reports/dataset_audit",
        ROOT / "reports/training",
        ROOT / "reports/evaluation",
        ROOT / "reports/audit",
        ROOT / "reports/real_validation",
        ROOT / "reports/studio_batch",
        ROOT / "reports/chapter5/studio_tables",
        ROOT / "data/real_public",
    ]
    fixed = (2024, 1, 1, 0, 0, 0)
    seen: set[str] = set()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in include_files:
            if file.exists() and ok(file):
                arc = file.relative_to(ROOT).as_posix()
                zi = zipfile.ZipInfo(arc, fixed)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, file.read_bytes())
                seen.add(arc)
        for folder in include_dirs:
            if not folder.exists():
                continue
            for file in sorted(folder.rglob("*")):
                if file.is_file() and ok(file):
                    arc = file.relative_to(ROOT).as_posix()
                    if arc in seen:
                        continue
                    zi = zipfile.ZipInfo(arc, fixed)
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(zi, file.read_bytes())
                    seen.add(arc)
    print(OUT)


if __name__ == "__main__":
    main()
