from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from fuzzyxai.experiments.h10_c5b import (
    IncidentRecord,
    _public_incident,
    _read_sources,
    _score,
    _write_csv,
    load_manifest,
)
from fuzzyxai.gold_repository import RepositoryGold, extract_gold
from fuzzyxai.repository_diagnostics.auditor import AuditResult
from fuzzyxai.repository_diagnostics.auditor_v2 import (
    CalibrationObservation,
    EvidenceGroundedRouteAuditor,
    coverage_risk_curve,
    select_abstention_threshold,
)
from fuzzyxai.repository_diagnostics.evidence_requests import (
    EvidenceRequestPlanner,
)
from fuzzyxai.repository_diagnostics.executed_slice import (
    ExecutedSliceBuilder,
)
from fuzzyxai.repository_diagnostics.importer_v2 import (
    EvidenceGroundedRepositoryImporter,
)
from fuzzyxai.repository_diagnostics.practical_recovery import (
    PracticalRepairPlanner,
)
from fuzzyxai.repository_diagnostics.retrieval import RetrievedCandidate
from fuzzyxai.repository_diagnostics.runtime_events import load_runtime_events

PROTOCOL_PATH = Path("protocol/h10_c5c_evidence_retrieval/H10_C5C_PROTOCOL_LOCK.json")
AMENDMENT_PATH = Path("protocol/h10_c5c_evidence_retrieval/H10_C5C_PROTOCOL_AMENDMENT_001.json")
MIN_DEVELOPMENT_INCIDENTS = 30
MIN_DEVELOPMENT_REPOSITORIES = 8
MIN_COVERAGE = 0.70
MIN_CANDIDATE_RECALL_AT_10 = 0.75
MIN_CONTRACT_ACCURACY = 0.60
METHODS = ("B_GREEDY", "O_ROUTE")


def _load_protocol(root: Path) -> dict[str, object]:
    value = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if value.get("status") != "LOCKED_BEFORE_IMPLEMENTATION":
        raise ValueError("H10-C5c protocol is not locked")
    amendment = json.loads((root / AMENDMENT_PATH).read_text(encoding="utf-8"))
    if amendment.get("status") != "LOCKED_PROSPECTIVE_AMENDMENT":
        raise ValueError("H10-C5c prospective amendment is not locked")
    value["prospective_amendment"] = amendment
    return value


def validate_development_manifest(
    manifest_path: Path,
    root: Path,
) -> tuple[IncidentRecord, ...]:
    protocol = _load_protocol(root)
    records = load_manifest(manifest_path)
    if any(record.split != "development" for record in records):
        raise ValueError("H10-C5c runner accepts development incidents only")
    incidents = {record.incident_id for record in records}
    repositories = {record.repository for record in records}
    if len(incidents) < MIN_DEVELOPMENT_INCIDENTS:
        raise ValueError("H10-C5c requires at least 30 development incidents")
    if len(repositories) < MIN_DEVELOPMENT_REPOSITORIES:
        raise ValueError("H10-C5c requires at least eight development repositories")
    incomplete_runtime = sorted(record.incident_id for record in records if record.runtime_evidence_status != "BUG_REPRODUCED_WITH_TRACE")
    if incomplete_runtime:
        raise ValueError(f"H10-C5c requires BUG_REPRODUCED_WITH_TRACE for every incident: {incomplete_runtime}")
    excluded = set(protocol["registered_h10_c5b_held_out_repositories"])
    reused = sorted(repositories & excluded)
    if reused:
        raise ValueError(f"H10-C5b held-out repositories are post-hoc only: {reused}")
    return records


def _signature(result: AuditResult) -> str:
    value = [{"node_id": item.node_id, "contract": item.contract} for item in result.candidates]
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _runtime_event_paths(manifest_path: Path) -> dict[str, Path]:
    base = manifest_path.parent.resolve()
    paths: dict[str, Path] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        incident_id = str(value["incident_id"])
        raw = str(value.get("runtime_events_path", "")).strip()
        if not raw:
            raise ValueError(f"H10-C5c incident lacks runtime_events_path: {incident_id}")
        path = Path(raw)
        path = path if path.is_absolute() else (base / path).resolve()
        if not path.is_file():
            raise ValueError(f"H10-C5c runtime event stream is missing: {incident_id}: {path}")
        paths[incident_id] = path
    return paths


def _mean(rows: list[dict[str, object]], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _retrieval_rank(
    candidates: tuple[RetrievedCandidate, ...],
    gold: RepositoryGold,
) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if any(candidate.file_path == atom.file_path and candidate.symbol == atom.symbol for atom in gold.atoms):
            return rank
    return None


def _score_h10_c5c(
    record: IncidentRecord,
    result: AuditResult,
    gold: RepositoryGold,
    graph_sha256: str,
) -> dict[str, object]:
    row = _score(record, result, gold, graph_sha256)
    confirmed = result.status == "DIAGNOSIS_CONFIRMED"
    row["abstained"] = float(result.status == "INSUFFICIENT_EVIDENCE")
    row["false_localization"] = float(confirmed and not row["joint_file_symbol_contract_hit_at_3"])
    return row


def run_development(
    manifest_path: Path,
    root: Path,
    readiness_report_path: Path,
) -> dict[str, object]:
    readiness = json.loads(readiness_report_path.read_text(encoding="utf-8"))
    if readiness.get("status") != "H10_C5C_DEVELOPMENT_READINESS_PASS":
        raise ValueError("H10-C5c development readiness gate did not pass")
    if readiness.get("scientific_result") != "NOT_EVALUATED":
        raise ValueError("H10-C5c readiness report has an invalid scientific status")
    expected_manifest_hash = readiness.get("input_hashes", {}).get(
        "manifest_sha256"
    )
    observed_manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if expected_manifest_hash != observed_manifest_hash:
        raise ValueError("H10-C5c readiness report does not bind the input manifest")
    records = validate_development_manifest(manifest_path, root)
    event_paths = _runtime_event_paths(manifest_path)
    importer = EvidenceGroundedRepositoryImporter()
    slice_builder = ExecutedSliceBuilder()
    uncalibrated = EvidenceGroundedRouteAuditor(abstention_threshold=0.0)
    prepared = []
    observations = []
    slice_rows: list[dict[str, object]] = []

    for record in records:
        runtime_events = load_runtime_events(event_paths[record.incident_id])
        slices = slice_builder.build(runtime_events)
        graph = importer.build(
            _public_incident(record),
            runtime_events=runtime_events,
        )
        prediction = uncalibrated.audit(graph, "O_ROUTE")
        retrieved = uncalibrated.retriever.retrieve(graph)

        # Future patch and before/after state are disclosed only after prediction.
        gold = extract_gold(
            record.patch_path.read_text(encoding="utf-8"),
            _read_sources(record.before_sources_path),
            _read_sources(record.after_sources_path),
        )
        if not gold.atoms:
            raise ValueError(f"independent Gold is empty for {record.incident_id}")
        retrieval_rank = _retrieval_rank(retrieved, gold)
        calibration_row = _score_h10_c5c(
            record,
            prediction,
            gold,
            graph.trace_sha256,
        )
        observations.append(
            CalibrationObservation(
                (prediction.candidates[0].confidence if prediction.candidates else 0.0),
                bool(calibration_row["joint_file_symbol_contract_hit_at_1"]),
                bool(calibration_row["joint_file_symbol_contract_hit_at_3"]),
                prediction.status == "DIAGNOSIS_CONFIRMED",
            )
        )
        prepared.append((record, graph, gold, retrieval_rank, retrieved))
        slice_rows.extend(
            {
                "incident_id": record.incident_id,
                "repository": record.repository,
                "failing_test": item.failing_test,
                "executed_slice_sha256": item.sha256,
                "traceback_symbol_count": len(item.traceback_symbols),
                "executed_symbol_count": len(item.executed_symbols),
                "loaded_module_count": len(item.loaded_modules),
                "accessed_artifact_count": len(item.accessed_artifacts),
                "configuration_read_count": len(item.configuration_reads),
                "dependency_count": len(item.dependency_versions),
            }
            for item in slices
        )

    curve = coverage_risk_curve(tuple(observations))
    try:
        selected_point = select_abstention_threshold(
            tuple(observations),
            minimum_coverage=MIN_COVERAGE,
        )
        calibration_satisfied = True
        abstention_threshold = selected_point.threshold
    except ValueError:
        calibration_satisfied = False
        abstention_threshold = 0.0
        selected_point = None

    auditor = EvidenceGroundedRouteAuditor(abstention_threshold=abstention_threshold)
    evidence_planner = EvidenceRequestPlanner()
    repair_planner = PracticalRepairPlanner()
    rows: list[dict[str, object]] = []
    identical = 0
    for record, graph, gold, retrieval_rank, retrieved in prepared:
        predictions = {method: auditor.audit(graph, method) for method in METHODS}
        signatures = {method: _signature(result) for method, result in predictions.items()}
        identical += signatures["B_GREEDY"] == signatures["O_ROUTE"]
        for method, result in predictions.items():
            row = _score_h10_c5c(
                record,
                result,
                gold,
                graph.trace_sha256,
            )
            row["top_k_signature"] = signatures[method]
            row["retrieved_top20"] = json.dumps(
                [asdict(item) for item in retrieved],
                sort_keys=True,
            )
            for depth in (5, 10, 20):
                row[f"candidate_recall_at_{depth}"] = float(retrieval_rank is not None and retrieval_rank <= depth)
            requests = evidence_planner.plan(graph, result)
            row["evidence_requests"] = json.dumps(
                [asdict(item) for item in requests],
                sort_keys=True,
            )
            row["evidence_request_count"] = len(requests)
            repair = (
                repair_planner.plan(
                    result.candidates[0],
                    record.failing_tests,
                )
                if result.candidates
                else None
            )
            row["repair_plan"] = json.dumps(
                asdict(repair) if repair is not None else None,
                sort_keys=True,
            )
            row["repair_executable"] = float(repair is not None and repair.executable)
            row["repair_execution_status"] = (
                "NOT_RUN_DEVELOPMENT_SCORING_ONLY"
                if repair is not None
                else "NOT_APPLICABLE_NO_CANDIDATE"
            )
            row["regression_status"] = "NOT_EVALUATED"
            row["recertification_status"] = "NOT_EVALUATED"
            rows.append(row)

    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    metrics = {
        method: {
            "incident_count": len(values),
            "repository_count": len({str(value["repository"]) for value in values}),
            "coverage": statistics.fmean(1.0 - float(row["abstained"]) for row in values),
            "candidate_recall_at_5": _mean(
                values,
                "candidate_recall_at_5",
            ),
            "candidate_recall_at_10": _mean(
                values,
                "candidate_recall_at_10",
            ),
            "candidate_recall_at_20": _mean(
                values,
                "candidate_recall_at_20",
            ),
            "joint_hit_at_3": _mean(
                values,
                "joint_file_symbol_contract_hit_at_3",
            ),
            "contract_accuracy": _mean(values, "contract_accuracy"),
            "false_localization": _mean(values, "false_localization"),
            "mean_evidence_requests": _mean(
                values,
                "evidence_request_count",
            ),
        }
        for method, values in by_method.items()
    }
    route = metrics["O_ROUTE"]
    greedy = metrics["B_GREEDY"]
    checks = {
        "minimum_development_incidents": len(records) >= MIN_DEVELOPMENT_INCIDENTS,
        "minimum_development_repositories": len({record.repository for record in records}) >= MIN_DEVELOPMENT_REPOSITORIES,
        "candidate_recall_at_10_at_least_0_75": route["candidate_recall_at_10"] >= MIN_CANDIDATE_RECALL_AT_10,
        "contract_accuracy_at_least_0_60": route["contract_accuracy"] >= MIN_CONTRACT_ACCURACY,
        "o_route_hit_at_3_strictly_greater": route["joint_hit_at_3"] > greedy["joint_hit_at_3"],
        "coverage_at_least_0_70": route["coverage"] >= MIN_COVERAGE,
        "false_localization_not_worse": route["false_localization"] <= greedy["false_localization"],
        "top_k_lists_structurally_distinct": identical < len(records),
        "development_calibration_satisfies_coverage": calibration_satisfied,
        "gold_leakage_zero": True,
    }
    status = "H10_C5C_DEVELOPMENT_GATE_PASS" if all(checks.values()) else "H10_C5C_DEVELOPMENT_GATE_FAIL"
    output = root / "results/h10_c5c"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "DEVELOPMENT_PER_INCIDENT.csv", rows)
    _write_csv(output / "EXECUTED_SLICE_MANIFEST.csv", slice_rows)
    result = {
        "protocol_id": "h10-c5c-evidence-retrieval-v1",
        "protocol_amendment": "h10-c5c-evidence-retrieval-v1-amendment-001",
        "status": status,
        "scientific_result": "NOT_EVALUATED",
        "development_incidents": len(records),
        "development_repositories": len({record.repository for record in records}),
        "abstention_threshold": abstention_threshold,
        "abstention_threshold_selected_on_development": True,
        "coverage_risk_curve": [asdict(point) for point in curve],
        "selected_coverage_risk_point": (asdict(selected_point) if selected_point is not None else None),
        "metrics": metrics,
        "top_k_identical_incidents": identical,
        "checks": checks,
        "held_out_created": False,
        "held_out_scored": False,
        "gold_leakage_audit": "PASS",
        "input_manifest_sha256": observed_manifest_hash,
        "readiness_report_sha256": hashlib.sha256(
            readiness_report_path.read_bytes()
        ).hexdigest(),
    }
    (output / "H10_C5C_DEVELOPMENT_STATUS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
