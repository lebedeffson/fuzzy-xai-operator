from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from merge_model_validation_reports import OPTIONAL_LIBRARIES, validate_runtime_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = ROOT / "release_evidence/model_universality"
REQUIRED_RUNTIME_LIBRARIES = {"sklearn", *OPTIONAL_LIBRARIES}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_checksums(directory: Path) -> None:
    for line in (directory / "checksums.sha256").read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"checksum mismatch: {path}")


def verify(evidence: Path, *, require_all_pass: bool = False) -> dict[str, Any]:
    _verify_checksums(evidence)
    matrix = _load(evidence / "model_support_matrix.json")
    rows = matrix.get("configurations", [])
    summary = matrix.get("summary", {})
    if len(rows) != summary.get("configuration_count"):
        raise ValueError("matrix row count does not match summary")
    with (evidence / "model_support_matrix.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != len(rows):
        raise ValueError("JSON and CSV model matrices have different row counts")
    serialized = json.dumps(matrix, ensure_ascii=False)
    if "not_installed_not_verified" in serialized or "installed_not_run_by_core_benchmark" in serialized:
        raise ValueError("legacy untyped adapter status remains in final matrix")

    reports = []
    for path in sorted((evidence / "adapter_reports").glob("adapter_report_*.json")):
        payload = _load(path)
        errors = validate_runtime_report(payload, source=path)
        if errors:
            raise ValueError("\n".join(errors))
        reports.append(payload)
    measured_libraries = {str(report["library"]) for report in reports if report["status"] == "pass"}
    python_versions = {
        str(report["library"]): {
            tuple(str(item["python_version"]).split(".")[:2])
            for item in reports
            if item["library"] == report["library"]
        }
        for report in reports
    }
    normalized_python_versions = {
        library: {".".join(version) for version in versions}
        for library, versions in python_versions.items()
    }
    if require_all_pass:
        missing = sorted(REQUIRED_RUNTIME_LIBRARIES - measured_libraries)
        if missing:
            raise ValueError(f"required runtime reports are missing or failed: {missing}")
        incomplete_versions = {
            library: sorted({"3.11", "3.12"} - normalized_python_versions.get(library, set()))
            for library in sorted(REQUIRED_RUNTIME_LIBRARIES)
            if normalized_python_versions.get(library, set()) != {"3.11", "3.12"}
        }
        if incomplete_versions:
            raise ValueError(f"runtime Python-version coverage is incomplete: {incomplete_versions}")
        failed_rows = [row["configuration_id"] for row in rows if row.get("status") != "pass"]
        if failed_rows:
            raise ValueError(f"matrix contains non-pass rows: {failed_rows}")
        if summary.get("version_conflicts"):
            raise ValueError(f"cross-version conflicts detected: {summary['version_conflicts']}")
    return {
        "configuration_count": len(rows),
        "runtime_report_count": len(reports),
        "measured_libraries": sorted(measured_libraries),
        "python_versions": normalized_python_versions,
        "status_counts": summary.get("status_counts", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify merged FuzzyXAI model-universality evidence.")
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--require-all-pass", action="store_true")
    args = parser.parse_args()
    result = verify(args.evidence, require_all_pass=args.require_all_pass)
    print(f"PASS: model_universality {result['configuration_count']} configurations")
    print(f"PASS: runtime_reports {result['runtime_report_count']}")
    print(f"PASS: runtime_libraries {','.join(result['measured_libraries'])}")


if __name__ == "__main__":
    main()
