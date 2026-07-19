from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "release_evidence/model_universality"
DEFAULT_OUTPUT = ROOT / "release_evidence/model_universality_unified"
OPTIONAL_LIBRARIES = ("xgboost", "lightgbm", "catboost", "torch", "tensorflow", "onnx")
REQUIRED_REPORT_FIELDS = {
    "schema_version",
    "report_id",
    "library",
    "library_version",
    "python_version",
    "operating_system",
    "environment",
    "model_family",
    "model_class",
    "task_type",
    "adapter",
    "adapter_id",
    "sample_size",
    "prediction_parity",
    "conformance",
    "graph_validation",
    "quality_gate",
    "api_checks",
    "human_explanation_checks",
    "native_channels",
    "surrogate_channels",
    "missing_channels",
    "duration_seconds",
    "warnings",
    "status",
    "artifact_sha256",
}
VALID_STATUSES = {
    "pass",
    "implemented_not_executed",
    "dependency_unavailable",
    "unsupported",
    "failed",
    "failed_version_inconsistency",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _report_checksum(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def validate_runtime_report(payload: dict[str, Any], *, source: Path | None = None) -> tuple[str, ...]:
    errors: list[str] = []
    missing = sorted(REQUIRED_REPORT_FIELDS - payload.keys())
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if payload.get("status") not in {"pass", "failed"}:
        errors.append(f"invalid runtime status: {payload.get('status')}")
    if payload.get("status") == "pass":
        for field in ("prediction_parity", "conformance", "graph_validation"):
            if float(payload.get(field, 0.0)) != 1.0:
                errors.append(f"pass report requires {field}=1.0")
        failed_apis = sorted(name for name, passed in payload.get("api_checks", {}).items() if not passed)
        if failed_apis:
            errors.append(f"pass report contains failed API checks: {', '.join(failed_apis)}")
        failed_human = sorted(
            name for name, passed in payload.get("human_explanation_checks", {}).items() if not passed
        )
        if failed_human:
            errors.append(f"pass report contains failed human-explanation checks: {', '.join(failed_human)}")
    expected = _report_checksum(payload)
    if payload.get("artifact_sha256") != expected:
        errors.append("artifact_sha256 does not match canonical report payload")
    if not isinstance(payload.get("api_checks"), dict) or not payload.get("api_checks"):
        errors.append("api_checks must be a non-empty object")
    prefix = f"{source}: " if source else ""
    return tuple(prefix + error for error in errors)


def _runtime_reports(paths: Iterable[Path]) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    seen_report_ids: dict[str, str] = {}
    for path in sorted(paths):
        payload = _load(path)
        report_errors = validate_runtime_report(payload, source=path)
        if report_errors:
            errors.extend(report_errors)
            continue
        report_id = str(payload["report_id"])
        checksum = str(payload["artifact_sha256"])
        if report_id in seen_report_ids:
            if seen_report_ids[report_id] != checksum:
                errors.append(f"{path}: conflicting duplicate report_id {report_id}")
            continue
        seen_report_ids[report_id] = checksum
        reports.append((path, payload))
    return reports, errors


def _core_rows(base: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary = _load(base / "summary.json")
    for path in sorted((base / "conformance_reports").glob("*.json")):
        payload = _load(path)
        config_id = str(payload["config_id"])
        if config_id == "callable_black_box":
            library, model_family = "python", "callable_black_box"
        elif config_id == "generic_predict_proba":
            library, model_family = "python", "predict_proba_compatible"
        elif config_id == "native_rule_model":
            library, model_family = "fuzzyxai-native", "rule_based"
        else:
            library, model_family = "sklearn", str(payload["model_family"])
        quality = dict(payload.get("quality", {}))
        human = dict(payload.get("human_explanation", {}))
        capability_values = payload.get("capabilities", {}).get("capabilities", {})
        descriptors = capability_values.get("channels", []) if isinstance(capability_values, dict) else []
        native_channels = [item["name"] for item in descriptors if item.get("available") and item.get("origin") == "native"]
        derived_channels = [
            item["name"]
            for item in descriptors
            if item.get("available") and item.get("origin") in {"derived", "derived_from_native"}
        ]
        surrogate_channels = [
            item["name"] for item in descriptors if item.get("available") and item.get("origin") == "surrogate"
        ]
        rows.append(
            {
                "configuration_id": config_id,
                "library": library,
                "library_version": summary.get("sklearn_version") if library == "sklearn" else None,
                "model_family": model_family,
                "model_class": payload["model_class"],
                "task_type": payload["task"],
                "adapter": payload["adapter_id"],
                "environment": "core-model-contracts",
                "python_versions": ["3.11", "3.12"],
                "status": "pass" if payload.get("status") == "pass" else "failed",
                "prediction_parity": 1.0 if payload.get("prediction_parity") else 0.0,
                "conformance": 1.0 if payload.get("conformance", {}).get("status") == "pass" else 0.0,
                "graph_validation": 1.0 if not payload.get("graph_errors") else 0.0,
                "quality_gate": quality.get("status", "partial"),
                "native_channels": native_channels,
                "derived_channels": derived_channels,
                "surrogate_channels": surrogate_channels,
                "missing_channels": list(payload.get("missing_channels", [])),
                "api_checks": {},
                "user_explanation_complete": all(human.get(key) for key in ("decision", "reliability", "recommended_action")),
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "report_path": f"core_reports/{path.name}",
            }
        )
    return rows


def _optional_rows(reports: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    by_library: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for _, payload in reports:
        if payload["library"] in OPTIONAL_LIBRARIES:
            by_library[str(payload["library"])].append(payload)
    rows: list[dict[str, Any]] = []
    for library in OPTIONAL_LIBRARIES:
        versions = by_library.get(library, [])
        if not versions:
            rows.append(
                {
                    "configuration_id": f"{library}_runtime",
                    "library": library,
                    "library_version": None,
                    "model_family": library,
                    "model_class": None,
                    "task_type": "binary_classification",
                    "adapter": None,
                    "environment": f"optional-runtime-{library}",
                    "python_versions": [],
                    "status": "implemented_not_executed",
                    "prediction_parity": None,
                    "conformance": None,
                    "graph_validation": None,
                    "quality_gate": "not_measured",
                    "native_channels": [],
                    "derived_channels": [],
                    "surrogate_channels": [],
                    "missing_channels": [],
                    "api_checks": {},
                    "user_explanation_complete": None,
                    "artifact_sha256": None,
                    "report_path": None,
                }
            )
            continue
        signatures = {
            (
                item["status"],
                float(item["prediction_parity"]),
                float(item["conformance"]),
                float(item["graph_validation"]),
                item["quality_gate"],
            )
            for item in versions
        }
        inconsistent = len(signatures) > 1
        representative = sorted(versions, key=lambda item: item["python_version"])[-1]
        status = "failed_version_inconsistency" if inconsistent else str(representative["status"])
        rows.append(
            {
                "configuration_id": f"{library}_runtime",
                "library": library,
                "library_version": representative["library_version"],
                "model_family": representative["model_family"],
                "model_class": representative["model_class"],
                "task_type": representative["task_type"],
                "adapter": representative["adapter"],
                "environment": representative["environment"],
                "python_versions": sorted({str(item["python_version"]) for item in versions}),
                "status": status,
                "prediction_parity": representative["prediction_parity"],
                "conformance": representative["conformance"],
                "graph_validation": representative["graph_validation"],
                "quality_gate": representative["quality_gate"],
                "native_channels": representative["native_channels"],
                "derived_channels": representative.get("derived_channels", []),
                "surrogate_channels": representative["surrogate_channels"],
                "missing_channels": representative["missing_channels"],
                "api_checks": representative["api_checks"],
                "user_explanation_complete": all(representative["human_explanation_checks"].values()),
                "artifact_sha256": representative["artifact_sha256"],
                "report_path": f"adapter_reports/adapter_report_{representative['report_id']}.json",
            }
        )
    return rows


def _version_conflicts(reports: list[tuple[Path, dict[str, Any]]]) -> set[str]:
    by_library: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    for _, payload in reports:
        by_library[str(payload["library"])].add(
            (
                payload["status"],
                float(payload["prediction_parity"]),
                float(payload["conformance"]),
                float(payload["graph_validation"]),
                payload["quality_gate"],
                tuple(sorted((str(name), bool(value)) for name, value in payload["api_checks"].items())),
            )
        )
    return {library for library, signatures in by_library.items() if len(signatures) > 1}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_matrix_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def _write_chapter4_family_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["library"]), str(row["model_family"]), str(row["task_type"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (library, family, task), items in sorted(grouped.items()):
        statuses = {str(item["status"]) for item in items}
        status = "pass" if statuses == {"pass"} else ", ".join(sorted(statuses))
        native = sorted({channel for item in items for channel in item.get("native_channels", [])})
        derived = sorted({channel for item in items for channel in item.get("derived_channels", [])})
        missing = sorted({channel for item in items for channel in item.get("missing_channels", [])})
        summary_rows.append(
            {
                "library": library,
                "model_family": family,
                "task_type": task,
                "configuration_count": len(items),
                "native_evidence": "; ".join(native) or "prediction",
                "additional_evidence": "; ".join(derived) or "not measured",
                "limitations": "; ".join(missing) or "reported in adapter evidence",
                "status": status,
            }
        )
    fields = list(summary_rows[0]) if summary_rows else []
    with (output / "chapter4_model_family_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# Model-family evidence summary for Chapter 4",
        "",
        "| Library | Model family | Task | Configurations | Native evidence | Additional evidence | Limitations | Status |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['library']} | {row['model_family']} | {row['task_type']} | {row['configuration_count']} | "
            f"{row['native_evidence']} | {row['additional_evidence']} | {row['limitations']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "Only rows marked `pass` are runtime-verified. Missing evidence channels remain explicit limitations.",
            "",
        ]
    )
    (output / "chapter4_model_family_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _api_payload(reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    rows = [
        {
            "report_id": payload["report_id"],
            "library": payload["library"],
            "python_version": payload["python_version"],
            "status": payload["status"],
            "checks": payload["api_checks"],
        }
        for _, payload in reports
    ]
    return {
        "schema_version": "1.0",
        "report_count": len(rows),
        "all_checks_pass": bool(rows) and all(item["status"] == "pass" and all(item["checks"].values()) for item in rows),
        "reports": rows,
        "claim_scope": "API checks establish execution and result-contract integrity, not domain validity.",
    }


def _quality_payload(rows: list[dict[str, Any]], reports: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    optional_quality = {str(payload["library"]): payload.get("quality", {}) for _, payload in reports}
    families = []
    for row in rows:
        quality = optional_quality.get(str(row["library"]), {})
        families.append(
            {
                "configuration_id": row["configuration_id"],
                "model_family": row["model_family"],
                "status": row["status"],
                "prediction_parity": row["prediction_parity"],
                "conformance": row["conformance"],
                "graph_validation": row["graph_validation"],
                "quality_gate": row["quality_gate"],
                "user_explanation_complete": row["user_explanation_complete"],
                "faithfulness": quality.get("faithfulness"),
                "fidelity": quality.get("fidelity"),
                "stability": quality.get("measured_top_reason_stability"),
                "completeness": quality.get("completeness"),
                "sparsity": quality.get("sparsity"),
                "provenance_complete": bool(row.get("artifact_sha256") and row.get("report_path")),
                "limitations": quality.get("limitations", []),
            }
        )
    return {
        "schema_version": "1.0",
        "family_count": len(families),
        "pass_count": sum(item["status"] == "pass" for item in families),
        "stability_threshold": 0.67,
        "families": families,
        "claim_scope": "Measured fields are reported only where the adapter runtime exposed the required evidence channel.",
    }


def _quality_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# FuzzyXAI explanation quality report",
        "",
        f"Verified configurations: {payload['pass_count']} / {payload['family_count']}.",
        "",
        "| Configuration | Family | Status | Parity | Conformance | Graph | Quality | Stability |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in payload["families"]:
        stability = "N/A" if row["stability"] is None else f"{row['stability']:.3f}"
        lines.append(
            f"| {row['configuration_id']} | {row['model_family']} | {row['status']} | "
            f"{row['prediction_parity'] if row['prediction_parity'] is not None else 'N/A'} | "
            f"{row['conformance'] if row['conformance'] is not None else 'N/A'} | "
            f"{row['graph_validation'] if row['graph_validation'] is not None else 'N/A'} | {row['quality_gate']} | {stability} |"
        )
    lines.extend(["", "Missing metrics are kept as N/A and are not imputed.", ""])
    return "\n".join(lines)


def _write_checksums(directory: Path) -> None:
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "checksums.sha256")
    (directory / "checksums.sha256").write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(directory).as_posix()}\n" for path in files),
        encoding="ascii",
    )


def merge_reports(base: Path, report_paths: Iterable[Path], output: Path, quality_output: Path) -> dict[str, Any]:
    reports, errors = _runtime_reports(report_paths)
    if errors:
        raise ValueError("\n".join(errors))
    conflicts = _version_conflicts(reports)
    rows = _core_rows(base) + _optional_rows(reports)
    if "sklearn" in conflicts:
        rows = [
            {**row, "status": "failed_version_inconsistency"} if row["library"] == "sklearn" else row
            for row in rows
        ]
    for row in rows:
        if row["status"] not in VALID_STATUSES:
            raise ValueError(f"invalid matrix status: {row['status']}")
    if output.exists():
        shutil.rmtree(output)
    if quality_output.exists():
        shutil.rmtree(quality_output)
    output.mkdir(parents=True, exist_ok=True)
    core_output = output / "core_reports"
    core_output.mkdir(parents=True, exist_ok=True)
    for source in sorted((base / "conformance_reports").glob("*.json")):
        shutil.copy2(source, core_output / source.name)
    report_output = output / "adapter_reports"
    report_output.mkdir(parents=True, exist_ok=True)
    for source, payload in reports:
        _write_json(report_output / f"adapter_report_{payload['report_id']}.json", payload)
    statuses = Counter(str(row["status"]) for row in rows)
    summary = {
        "schema_version": "1.0",
        "configuration_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "verified_optional_libraries": sorted(row["library"] for row in rows if row["library"] in OPTIONAL_LIBRARIES and row["status"] == "pass"),
        "missing_optional_reports": sorted(
            row["library"]
            for row in rows
            if row["library"] in OPTIONAL_LIBRARIES and row["status"] == "implemented_not_executed"
        ),
        "version_conflicts": sorted(conflicts),
        "release_claim": "Only rows with status=pass are runtime-verified; missing reports never become pass.",
    }
    _write_json(output / "model_support_matrix.json", {"summary": summary, "configurations": rows})
    _write_matrix_csv(output / "model_support_matrix.csv", rows)
    _write_chapter4_family_summary(output, rows)
    _write_json(output / "public_api_verification.json", _api_payload(reports))
    _write_json(output / "summary.json", summary)
    quality = _quality_payload(rows, reports)
    _write_json(quality_output / "explanation_quality_report.json", quality)
    quality_output.mkdir(parents=True, exist_ok=True)
    (quality_output / "explanation_quality_report.md").write_text(_quality_markdown(quality), encoding="utf-8")
    _write_checksums(quality_output)
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in {"manifest.json", "checksums.sha256"})
    manifest_files = [
        {"path": path.relative_to(output).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        for path in files
    ]
    _write_json(output / "manifest.json", {"schema_version": "1.0", "summary": summary, "files": manifest_files})
    manifest_files.append(
        {
            "path": "manifest.json",
            "sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
            "bytes": (output / "manifest.json").stat().st_size,
        }
    )
    (output / "checksums.sha256").write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in manifest_files), encoding="ascii")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independently measured FuzzyXAI model runtime reports.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quality-output", type=Path, default=ROOT / "release_evidence/explanation_quality")
    args = parser.parse_args()
    paths = args.reports_dir.rglob("adapter_report_*.json")
    summary = merge_reports(args.base, paths, args.output, args.quality_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
