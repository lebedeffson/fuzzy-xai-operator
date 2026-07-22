from __future__ import annotations

import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from .common import ARTIFACT_ROOT, ROOT, sha256_file


DELIVERABLE_ROOT = ROOT.parent / "fuzzy-xai-operator-deliverables"
FORBIDDEN_NAMES = {
    "reviewer_1.csv",
    "reviewer_2.csv",
    "opening_record.json",
    "sealed_test_truth.jsonl",
}


def _zip_files(output: Path, files: list[tuple[Path, str]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in sorted(files, key=lambda item: item[1]):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def package() -> Path:
    methodology = json.loads((ARTIFACT_ROOT / "closure" / "h10_final_gold_methodology_audit.json").read_text())
    if methodology["sealed_test_opened"]:
        raise RuntimeError("preconfirmatory package cannot include an opened sealed cycle")
    tracked_private = subprocess.check_output(
        ("git", "ls-files", ".h10_final_gold_private", "artifacts/h10_final_gold/adjudication/reviewer_1.csv", "artifacts/h10_final_gold/adjudication/reviewer_2.csv"),
        cwd=ROOT,
        text=True,
    ).strip()
    if tracked_private:
        raise RuntimeError(f"private Gold payload is tracked: {tracked_private}")
    subprocess.run((sys.executable, "scripts/build_framework_release.py"), cwd=ROOT, check=True)
    commit = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip()
    output_dir = DELIVERABLE_ROOT / f"h10-final-gold-{commit[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_source = ROOT / "release_artifacts" / f"fuzzyxai-source-release-{commit[:12]}.zip"
    source_zip = output_dir / "h10_final_gold_source.zip"
    shutil.copy2(generated_source, source_zip)
    artifact_files = []
    for path in ARTIFACT_ROOT.rglob("*"):
        if not path.is_file() or path.name in FORBIDDEN_NAMES:
            continue
        if "truth" in path.name.lower() or path.suffix == ".key":
            raise RuntimeError(f"private truth-like file reached public artifacts: {path}")
        artifact_files.append((path, path.relative_to(ARTIFACT_ROOT).as_posix()))
    artifacts_zip = output_dir / "h10_final_gold_artifacts.zip"
    _zip_files(artifacts_zip, artifact_files)
    handoff_files = [
        (source_zip, source_zip.name),
        (artifacts_zip, artifacts_zip.name),
        (ROOT / "config" / "h10_final_gold_protocol.yaml", "h10_final_gold_protocol.yaml"),
        (ARTIFACT_ROOT / "h10_final_gold_manifest.json", "h10_final_gold_manifest.json"),
        (ARTIFACT_ROOT / "closure" / "h10_final_gold_claim_registry.json", "h10_final_gold_claim_registry.json"),
        (ARTIFACT_ROOT / "closure" / "h10_final_gold_evidence_map.json", "h10_final_gold_evidence_map.json"),
        (ARTIFACT_ROOT / "closure" / "h10_final_gold_validation_report.md", "h10_final_gold_validation_report.md"),
        (ARTIFACT_ROOT / "closure" / "h10_final_gold_methodology_audit.json", "h10_final_gold_methodology_audit.json"),
        (ARTIFACT_ROOT / "closure" / "h10_final_gold_leakage_audit.json", "h10_final_gold_leakage_audit.json"),
    ]
    handoff_zip = output_dir / "h10_final_gold_preconfirmatory_handoff.zip"
    _zip_files(handoff_zip, handoff_files)
    checksums = []
    for path in (source_zip, artifacts_zip, handoff_zip):
        checksums.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "h10_final_gold_SHA256.txt").write_text("\n".join(checksums) + "\n", encoding="ascii")
    return output_dir


def main() -> None:
    print(package())


if __name__ == "__main__":
    main()
