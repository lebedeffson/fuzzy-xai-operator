from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from .common import ARTIFACTS, ROOT, read_jsonl, sha256_file, write_jsonl


FINAL = ROOT / "dissertation_artifacts" / "chapter4_v13" / "final"
OUTPUT_NAMES = (
    "Глава_4_FuzzyXAI_эмпирическая_редакция_v13.docx",
    "Глава_4_FuzzyXAI_эмпирическая_редакция_v13.pdf",
    "Глава_4_FuzzyXAI_v13_changelog.md",
    "Глава_4_FuzzyXAI_v13_validation_report.md",
    "Глава_4_FuzzyXAI_v13_evidence_map.json",
    "Глава_4_FuzzyXAI_v13_leakage_audit.json",
)


def _write_deterministic_zip(path: Path, files: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, name in sorted(files, key=lambda item: item[1]):
            info = zipfile.ZipInfo(name, date_time=(2026, 7, 22, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())


def build() -> dict[str, object]:
    validation = json.loads((ARTIFACTS / "manifests" / "validation.json").read_text(encoding="utf-8"))
    if not validation.get("passed"):
        raise RuntimeError("release requires a passing evidence validation")
    missing = [name for name in OUTPUT_NAMES if not (FINAL / name).exists()]
    if missing:
        raise RuntimeError(f"missing final files: {missing}")

    with tempfile.TemporaryDirectory(prefix="fuzzyxai-v13-release-") as temporary:
        temp = Path(temporary)
        sanitized = temp / "sanitized_explanation_metrics.jsonl"
        rows = []
        for row in read_jsonl(ARTIFACTS / "explanations" / "sealed_test.jsonl"):
            rows.append({key: value for key, value in row.items() if key != "canonical_payload"})
        write_jsonl(sanitized, rows)

        files: list[tuple[Path, str]] = []
        for name in OUTPUT_NAMES:
            files.append((FINAL / name, f"deliverables/{name}"))
        for relative in (
            "config/chapter4_v13_protocol.yaml",
            "config/chapter4_v13_protocol.sha256",
            "config/chapter4_v13_runtime.yaml",
            "config/chapter4_v13_case.yaml",
            "config/chapter4_v13_requirements.txt",
            "LICENSE",
            "README.md",
        ):
            files.append((ROOT / relative, f"source/{relative}"))
        for path in sorted((ROOT / "experiments" / "chapter4_v13").glob("*.py")):
            files.append((path, f"source/experiments/chapter4_v13/{path.name}"))
        for path in sorted((ROOT / "protocols" / "user_study").iterdir()):
            if path.is_file():
                files.append((path, f"source/protocols/user_study/{path.name}"))
        for relative in (
            "leakage_audit.json",
            "evidence_map.json",
            "validation_report.md",
            "policies/policy_results.csv",
            "policies/statistical_tests.json",
            "policies/summary.json",
            "route_faults/raw_results.jsonl",
            "route_faults/summary.csv",
            "route_faults/manifest.json",
            "runtime/raw_results.csv",
            "runtime/summary.csv",
            "runtime/manifest.json",
            "explanations/sealed_test_summary.json",
            "manifests/dataset_manifest.json",
            "manifests/model_manifest.json",
            "manifests/tables_manifest.json",
            "manifests/figures_manifest.json",
            "manifests/validation.json",
            "end_to_end_case/input_reference.json",
            "end_to_end_case/prediction.json",
            "end_to_end_case/diagnostic_state.json",
            "end_to_end_case/action.json",
            "end_to_end_case/stage_timings.json",
            "end_to_end_case/audit.md",
            "end_to_end_case/SHA256SUMS",
        ):
            source = ARTIFACTS / relative
            files.append((source, f"evidence/{relative}"))
        files.append((sanitized, "evidence/explanations/sanitized_explanation_metrics.jsonl"))
        for directory in ("tables", "figures"):
            for path in sorted((ARTIFACTS / directory).iterdir()):
                if path.is_file():
                    files.append((path, f"evidence/{directory}/{path.name}"))

        archive = FINAL / "fuzzyxai-chapter4-v13-artifacts.zip"
        _write_deterministic_zip(archive, files)
    sha_path = FINAL / "Глава_4_FuzzyXAI_v13_SHA256.txt"
    checks = [*OUTPUT_NAMES, "fuzzyxai-chapter4-v13-artifacts.zip"]
    sha_path.write_text("".join(f"{sha256_file(FINAL / name)}  {name}\n" for name in checks), encoding="utf-8")
    return {"archive": str(archive), "archive_sha256": sha256_file(archive), "files": len(files)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    result = build()
    print(f"PASS: release archive files={result['files']} sha256={result['archive_sha256']}")


if __name__ == "__main__":
    main()
