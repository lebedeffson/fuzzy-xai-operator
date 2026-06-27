from __future__ import annotations

import zipfile
from pathlib import Path

from fuzzyxai.audit.common import ROOT


OUT = ROOT / "FuzzyXAI_full_delivery_package.zip"
PRACTICE_ROOT = ROOT / "reports/practice_demo"


def ok(path: Path) -> bool:
    bad_parts = {"__pycache__", ".pytest_cache", ".venv", "node_modules"}
    if set(path.parts) & bad_parts:
        return False
    if path.suffix in {".pyc", ".pyo"}:
        return False
    return path.name != ".DS_Store"


def archive_name(file: Path) -> str:
    if file.is_relative_to(PRACTICE_ROOT):
        return ("practice_demo" / file.relative_to(PRACTICE_ROOT)).as_posix()
    return file.relative_to(ROOT).as_posix()


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
                arc = archive_name(file)
                zi = zipfile.ZipInfo(arc, fixed)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, file.read_bytes())
                seen.add(arc)
        for folder in include_dirs:
            if not folder.exists():
                continue
            for file in sorted(folder.rglob("*")):
                if file.is_file() and ok(file):
                    arc = archive_name(file)
                    if arc in seen:
                        continue
                    zi = zipfile.ZipInfo(arc, fixed)
                    zi.compress_type = zipfile.ZIP_DEFLATED
                    zf.writestr(zi, file.read_bytes())
                    seen.add(arc)
    required = [
        "FuzzyXAI_FINAL_DELIVERY_REPORT.md",
        "practice_demo/README_PRACTICE_DEMO.md",
        "practice_demo/practice_manifest.json",
        "practice_demo/inputs/input_hybrid_xiris_eye_sample.json",
        "practice_demo/inputs/input_ecg_sample_signal.csv",
        "practice_demo/inputs/gis_layer_sample.json",
        "practice_demo/qa/QA_SCREENSHOT_REPORT.md",
        "practice_demo/qa/QA_PROOF_PACKAGE_SCHEMA.md",
        "practice_demo/qa/QA_DATASET_REQUIRED_FILES.md",
        "reports/real_validation/real_validation_report.md",
        "reports/real_validation/real_artifacts_manifest.json",
        "data/real_public/iris/close_up_human_iris.jpg",
        "data/real_public/ecg/mitdb_100_first_10s.csv",
    ]
    with zipfile.ZipFile(OUT) as zf:
        names = set(zf.namelist())
        missing = [name for name in required if name not in names]
        screenshot_count = sum(
            1 for name in names if name.startswith("practice_demo/screenshots/") and name.endswith(".png")
        )
    if missing or screenshot_count != 18:
        raise SystemExit(f"full delivery package invalid: missing={missing}, screenshots={screenshot_count}")
    print(OUT)


if __name__ == "__main__":
    main()
