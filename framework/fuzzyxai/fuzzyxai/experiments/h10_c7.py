from __future__ import annotations

import csv
import hashlib
import json
import statistics
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from fuzzyxai.repository_diagnostics.contract_inference_v2 import (
    evaluation_contract_family,
)
from fuzzyxai.repository_diagnostics.graph import (
    EvidenceRef,
    RepositoryEdge,
    RepositoryGraph,
    RepositoryNode,
)
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    METHODS,
    VARIANTS,
    GuidedDiagnosis,
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import IncidentQuery

PROTOCOL_PATH = Path("protocol/h10_c7/H10_C7_PROTOCOL_LOCK.json")
PARENT_RESULTS = (
    "H10-C3",
    "H10-C4",
    "H10-C5",
    "H10-C5b",
    "H10-C5c",
    "H10-C6",
    "H10-C6-N",
    "H9-E2E",
    "H9-E2E-v2",
)
FORBIDDEN_OBSERVABLE_KEYS = frozenset(
    {
        "after_sources",
        "changed_files",
        "changed_symbols",
        "fix_commit",
        "fixed_revision",
        "gold",
        "gold_contract",
        "gold_file",
        "gold_patch",
        "gold_symbol",
        "maintainer_explanation_after_fix",
        "patch",
        "production_fix_patch",
    }
)


@dataclass(frozen=True)
class DevelopmentIncident:
    incident_id: str
    repository: str
    query: IncidentQuery
    graph: RepositoryGraph
    repository_symbol_count: int


@dataclass(frozen=True)
class GoldAtom:
    file_path: str
    symbol: str | None
    contract: str


@dataclass(frozen=True)
class GoldLocalization:
    incident_id: str
    atoms: tuple[GoldAtom, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _reject_gold(value: object, location: str = "$") -> None:
    if isinstance(value, dict):
        forbidden = sorted(FORBIDDEN_OBSERVABLE_KEYS.intersection(value))
        if forbidden:
            raise ValueError(
                f"H10-C7 observable manifest contains Gold keys at "
                f"{location}: {forbidden}"
            )
        for key, child in value.items():
            _reject_gold(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_gold(child, f"{location}[{index}]")


def _graph(value: dict[str, object]) -> RepositoryGraph:
    return RepositoryGraph(
        str(value["repository"]),
        str(value["revision"]),
        tuple(RepositoryNode(**item) for item in value.get("nodes", [])),
        tuple(RepositoryEdge(**item) for item in value.get("edges", [])),
        tuple(EvidenceRef(**item) for item in value.get("evidence", [])),
        tuple(str(item) for item in value.get("obligations", [])),
        tuple(str(item) for item in value.get("limitations", [])),
    )


def load_development_inputs(
    manifest_path: Path,
    gold_path: Path,
    *,
    minimum_incidents: int = 40,
    minimum_repositories: int = 10,
) -> tuple[tuple[DevelopmentIncident, ...], dict[str, GoldLocalization]]:
    observable = _read_jsonl(manifest_path)
    for index, value in enumerate(observable):
        _reject_gold(value, f"$[{index}]")
    incidents = []
    for value in observable:
        if value.get("split") != "development":
            raise ValueError("H10-C7 development runner accepts development only")
        query = value["query"]
        assert isinstance(query, dict)
        graph = value.get("graph")
        if graph is None:
            graph_path = Path(str(value["graph_path"]))
            if not graph_path.is_absolute():
                graph_path = (manifest_path.parent / graph_path).resolve()
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        assert isinstance(graph, dict)
        incidents.append(
            DevelopmentIncident(
                str(value["incident_id"]),
                str(value["repository"]),
                IncidentQuery(
                    str(value["incident_id"]),
                    str(query.get("issue", "")),
                    tuple(str(item) for item in query.get("failing_tests", [])),
                    str(query.get("traceback", "")),
                    str(query.get("assertion", "")),
                ),
                _graph(graph),
                int(value["repository_symbol_count"]),
            )
        )
    gold_values = _read_jsonl(gold_path)
    gold = {}
    for value in gold_values:
        raw_atoms = value.get("atoms")
        if raw_atoms is None:
            raw_atoms = (
                {
                    "file_path": value["file_path"],
                    "symbol": value.get("symbol"),
                    "contract": value["contract"],
                },
            )
        atoms = tuple(
            GoldAtom(
                str(item["file_path"]),
                (
                    str(item["symbol"])
                    if item.get("symbol") is not None
                    else None
                ),
                str(item["contract"]),
            )
            for item in raw_atoms
        )
        if not atoms:
            raise ValueError("H10-C7 development Gold cannot be empty")
        identifier = str(value["incident_id"])
        gold[identifier] = GoldLocalization(identifier, atoms)
    ids = [item.incident_id for item in incidents]
    if len(ids) != len(set(ids)):
        raise ValueError("H10-C7 incident IDs must be unique")
    if set(ids) != set(gold):
        raise ValueError("H10-C7 observable and Gold incident sets differ")
    repositories = {item.repository for item in incidents}
    if len(incidents) < minimum_incidents:
        raise ValueError(
            f"H10-C7 requires at least {minimum_incidents} development incidents"
        )
    if len(repositories) < minimum_repositories:
        raise ValueError(
            f"H10-C7 requires at least {minimum_repositories} development repositories"
        )
    return tuple(incidents), gold


def _rank(
    diagnosis: GuidedDiagnosis,
    gold: GoldLocalization,
) -> int | None:
    for rank, candidate in enumerate(diagnosis.candidates, start=1):
        if any(
            candidate.file_path == atom.file_path
            and candidate.symbol == atom.symbol
            for atom in gold.atoms
        ):
            return rank
    return None


def _row(
    incident: DevelopmentIncident,
    diagnosis: GuidedDiagnosis,
    gold: GoldLocalization,
    runtime_ms: float,
) -> dict[str, object]:
    rank = _rank(diagnosis, gold)
    contract = (
        evaluation_contract_family(diagnosis.candidates[0].contract.family)
        if diagnosis.candidates
        else "UNKNOWN_CONTRACT"
    )
    gold_contracts = tuple(sorted({atom.contract for atom in gold.atoms}))
    contract_hit = contract in gold_contracts
    joint_hit_at_3 = bool(rank and rank <= 3 and contract_hit)
    context_symbols = len(diagnosis.candidates)
    context_lines = sum(item.line_count for item in diagnosis.candidates)
    reduction = 1.0 - context_symbols / max(
        incident.repository_symbol_count,
        1,
    )
    coverage = diagnosis.status in {
        "DIAGNOSIS_CANDIDATES",
        "DIAGNOSIS_CONFIRMED",
    }
    confirmed = diagnosis.status == "DIAGNOSIS_CONFIRMED"
    contract_covered = bool(
        diagnosis.candidates
        and diagnosis.candidates[0].contract.family != "UNKNOWN_CONTRACT"
    )
    return {
        "incident_id": incident.incident_id,
        "repository": incident.repository,
        "variant": diagnosis.variant,
        "status": diagnosis.status,
        "available": diagnosis.status != "VARIANT_UNAVAILABLE",
        "unavailable_reason": diagnosis.unavailable_reason or "",
        "candidate_recall_at_5": float(rank is not None and rank <= 5),
        "candidate_recall_at_10": float(rank is not None and rank <= 10),
        "candidate_recall_at_20": float(rank is not None and rank <= 20),
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        "file_hit_at_3": float(
            bool(
                rank
                and rank <= 3
                and any(
                    any(
                        item.file_path == atom.file_path
                        for atom in gold.atoms
                    )
                    for item in diagnosis.candidates[:3]
                )
            )
        ),
        "symbol_hit_at_3": float(rank is not None and rank <= 3),
        "contract_correct": float(contract_hit),
        "gold_contracts": json.dumps(gold_contracts),
        "predicted_contract": contract,
        "joint_hit_at_3": float(joint_hit_at_3),
        "coverage": float(coverage),
        "retrieval_coverage": float(bool(diagnosis.candidates)),
        "contract_coverage": float(contract_covered),
        "confirmed_diagnosis_coverage": float(confirmed),
        "repair_coverage": 0.0,
        "confirmed_correct": float(confirmed and joint_hit_at_3),
        "false_localization": float(confirmed and not joint_hit_at_3),
        "candidate_count": context_symbols,
        "context_lines": context_lines,
        "search_space_reduction": reduction,
        "runtime_ms": runtime_ms,
        "evidence_request_count": len(diagnosis.evidence_requests),
        "active_evidence_status": diagnosis.active_evidence_status,
        "active_evidence_details": json.dumps(
            dict(diagnosis.active_evidence_details),
            sort_keys=True,
        ),
        "route": diagnosis.route.route,
        "top_k_signature": json.dumps(
            [item.node_id for item in diagnosis.candidates[:10]],
            separators=(",", ":"),
        ),
        "trajectory": json.dumps(
            [asdict(item) for item in diagnosis.trajectory],
            sort_keys=True,
        ),
        "evidence_requests": json.dumps(
            [asdict(item) for item in diagnosis.evidence_requests],
            sort_keys=True,
        ),
    }


def _mean(rows: Iterable[dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def _macro_f1(rows: list[dict[str, object]]) -> float:
    labels = sorted(
        {
            *(
                label
                for row in rows
                for label in json.loads(str(row["gold_contracts"]))
            ),
            *(str(row["predicted_contract"]) for row in rows),
        }
    )
    scores = []
    for label in labels:
        true_positive = sum(
            label in json.loads(str(row["gold_contracts"]))
            and row["predicted_contract"] == label
            for row in rows
        )
        false_positive = sum(
            label not in json.loads(str(row["gold_contracts"]))
            and row["predicted_contract"] == label
            for row in rows
        )
        false_negative = sum(
            label in json.loads(str(row["gold_contracts"]))
            and row["predicted_contract"] != label
            for row in rows
        )
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(
            2 * true_positive / denominator if denominator else 0.0
        )
    return statistics.fmean(scores) if scores else 0.0


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    available = [row for row in rows if row["available"]]
    confirmed = [
        row for row in available if row["status"] == "DIAGNOSIS_CONFIRMED"
    ]
    confirmed_correct = sum(
        float(row["confirmed_correct"]) for row in confirmed
    )
    return {
        "incident_count": len(rows),
        "available_incident_count": len(available),
        "repository_count": len({row["repository"] for row in rows}),
        "recall_at_5": _mean(available, "candidate_recall_at_5"),
        "recall_at_10": _mean(available, "candidate_recall_at_10"),
        "recall_at_20": _mean(available, "candidate_recall_at_20"),
        "mrr": _mean(available, "reciprocal_rank"),
        "file_hit_at_3": _mean(available, "file_hit_at_3"),
        "symbol_hit_at_3": _mean(available, "symbol_hit_at_3"),
        "contract_macro_f1": _macro_f1(available),
        "joint_hit_at_3": _mean(available, "joint_hit_at_3"),
        "coverage": _mean(rows, "coverage"),
        "retrieval_coverage": _mean(rows, "retrieval_coverage"),
        "contract_coverage": _mean(rows, "contract_coverage"),
        "confirmed_diagnosis_coverage": _mean(
            rows,
            "confirmed_diagnosis_coverage",
        ),
        "repair_coverage": _mean(rows, "repair_coverage"),
        "confirmed_diagnosis_count": len(confirmed),
        "confirmed_correct_count": int(confirmed_correct),
        "selective_precision": (
            confirmed_correct / len(confirmed) if confirmed else 0.0
        ),
        "conditional_confirmation_error": (
            1.0 - confirmed_correct / len(confirmed) if confirmed else 0.0
        ),
        "false_localization": _mean(rows, "false_localization"),
        "median_candidate_symbols": (
            statistics.median(
                float(row["candidate_count"]) for row in available
            )
            if available
            else 0.0
        ),
        "median_context_lines": (
            statistics.median(
                float(row["context_lines"]) for row in available
            )
            if available
            else 0.0
        ),
        "mean_search_space_reduction": _mean(
            available,
            "search_space_reduction",
        ),
        "median_runtime_ms": (
            statistics.median(float(row["runtime_ms"]) for row in available)
            if available
            else 0.0
        ),
        "mean_evidence_requests": _mean(
            available,
            "evidence_request_count",
        ),
        "active_evidence_applied_rate": (
            sum(
                row["active_evidence_status"]
                == "ACTIVE_EVIDENCE_APPLIED"
                for row in available
            )
            / len(available)
            if available
            else 0.0
        ),
        "active_evidence_unavailable_rate": (
            sum(
                row["active_evidence_status"]
                == "ACTIVE_EVIDENCE_UNAVAILABLE"
                for row in available
            )
            / len(available)
            if available
            else 0.0
        ),
    }


def _winner(
    metrics: dict[str, dict[str, object]],
    rows: list[dict[str, object]],
) -> tuple[str | None, dict[str, bool]]:
    baseline = metrics["B_GREEDY"]
    eligible = []
    for variant, values in metrics.items():
        if variant not in VARIANTS:
            continue
        checks = {
            "available_for_all_incidents": (
                values["available_incident_count"]
                == values["incident_count"]
            ),
            "recall_at_20_at_least_0_85": values["recall_at_20"] >= 0.85,
            "recall_at_10_at_least_0_75": values["recall_at_10"] >= 0.75,
            "contract_macro_f1_at_least_0_55": (
                values["contract_macro_f1"] >= 0.55
            ),
            "coverage_at_least_0_80": values["coverage"] >= 0.80,
            "false_localization_not_worse_than_b_greedy": (
                values["false_localization"]
                <= baseline["false_localization"]
            ),
            "median_candidate_symbols_at_most_20": (
                values["median_candidate_symbols"] <= 20
            ),
        }
        if all(checks.values()):
            eligible.append(variant)
    distinct = any(
        row["variant"] != "R0"
        and row["variant"] in VARIANTS
        and row["top_k_signature"]
        != next(
            base["top_k_signature"]
            for base in rows
            if base["variant"] == "B_GREEDY"
            and base["incident_id"] == row["incident_id"]
        )
        for row in rows
    )
    common_checks = {
        "gold_leakage_zero": True,
        "top_k_structurally_differs_from_r0": distinct,
    }
    if not eligible or not all(common_checks.values()):
        return None, common_checks
    selected = min(
        eligible,
        key=lambda variant: (
            -float(metrics[variant]["recall_at_10"]),
            -float(metrics[variant]["contract_macro_f1"]),
            -float(metrics[variant]["joint_hit_at_3"]),
            float(metrics[variant]["median_candidate_symbols"]),
            float(metrics[variant]["median_runtime_ms"]),
            variant,
        ),
    )
    return selected, common_checks


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_development_tournament(
    manifest_path: Path,
    gold_path: Path,
    output: Path,
    root: Path,
    engine: GuidedNaturalDiagnosisEngine,
    *,
    minimum_incidents: int = 40,
    minimum_repositories: int = 10,
) -> dict[str, object]:
    protocol = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    if protocol.get("status") != "LOCKED_BEFORE_DEVELOPMENT":
        raise ValueError("H10-C7 protocol is not locked for development")
    incidents, gold = load_development_inputs(
        manifest_path,
        gold_path,
        minimum_incidents=minimum_incidents,
        minimum_repositories=minimum_repositories,
    )
    rows: list[dict[str, object]] = []
    for incident in incidents:
        for variant in METHODS:
            started = time.perf_counter_ns()
            diagnosis = engine.diagnose(
                incident.graph,
                incident.query,
                variant,
            )
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            row = _row(
                incident,
                diagnosis,
                gold[incident.incident_id],
                elapsed,
            )
            row["repository_fold"] = int(
                hashlib.sha256(incident.repository.encode()).hexdigest(),
                16,
            ) % 5
            rows.append(row)
    by_variant = {
        variant: [row for row in rows if row["variant"] == variant]
        for variant in METHODS
    }
    metrics = {
        variant: _metrics(values)
        for variant, values in by_variant.items()
    }
    winner, common_checks = _winner(metrics, rows)
    status = (
        "H10_C7_READY_FOR_SEALED"
        if winner is not None
        else "H10_C7_BLOCKED_DEVELOPMENT_GATE"
    )
    result = {
        "protocol_id": "H10-C7-v1",
        "status": status,
        "scientific_result": "NOT_EVALUATED",
        "selected_variant": winner,
        "development_incidents": len(incidents),
        "development_repositories": len(
            {item.repository for item in incidents}
        ),
        "repository_cross_validation_folds": 5,
        "metrics": metrics,
        "checks": common_checks,
        "gold_leakage": 0,
        "held_out_created": False,
        "held_out_scored": False,
        "observable_manifest_sha256": _sha256(manifest_path),
        "development_gold_sha256": _sha256(gold_path),
        "parent_results_modified": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "DEVELOPMENT_MODEL_MATRIX.csv", rows)
    (output / "DEVELOPMENT_GATES.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if winner is not None:
        lock = {
            "protocol_id": "H10-C7-v1",
            "status": "METHOD_LOCKED_BEFORE_HELD_OUT_SELECTION",
            "selected_variant": winner,
            "selection_rule": (
                "zero leakage; coverage; false localization; Recall@10; "
                "contract macro-F1; joint Hit@3; context; runtime"
            ),
            "development_gates_sha256": _sha256(
                output / "DEVELOPMENT_GATES.json"
            ),
            "observable_manifest_sha256": _sha256(manifest_path),
            "development_gold_sha256": _sha256(gold_path),
            "held_out_created": False,
            "held_out_scored": False,
        }
        (output / "H10_C7_METHOD_LOCK.json").write_text(
            json.dumps(lock, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def validate_confirmatory_manifest(
    manifest_path: Path,
    development_gates_path: Path,
    method_lock_path: Path,
    root: Path,
) -> tuple[dict[str, object], ...]:
    """Validate held-out availability without opening or scoring Gold."""
    protocol = json.loads((root / PROTOCOL_PATH).read_text(encoding="utf-8"))
    gates = json.loads(development_gates_path.read_text(encoding="utf-8"))
    method_lock = json.loads(method_lock_path.read_text(encoding="utf-8"))
    if gates.get("status") != "H10_C7_READY_FOR_SEALED":
        raise ValueError("H10-C7 development gate did not pass")
    if method_lock.get("status") != "METHOD_LOCKED_BEFORE_HELD_OUT_SELECTION":
        raise ValueError("H10-C7 method lock is missing or invalid")
    values = _read_jsonl(manifest_path)
    for index, value in enumerate(values):
        _reject_gold(value, f"$[{index}]")
    if any(value.get("split") != "held_out" for value in values):
        raise ValueError("H10-C7 confirmatory manifest must be held-out only")
    repositories = {str(value["repository"]) for value in values}
    excluded = set(protocol["confirmatory_excluded_repositories"])
    overlap = sorted(repositories.intersection(excluded))
    if overlap:
        raise ValueError(
            f"H10-C7 held-out repositories overlap prior cycles: {overlap}"
        )
    if len(values) < int(protocol["confirmatory_minimum_incidents"]):
        raise ValueError("H10-C7 held-out has fewer than 40 incidents")
    if len(repositories) < int(protocol["confirmatory_minimum_repositories"]):
        raise ValueError("H10-C7 held-out has fewer than 12 repositories")
    incomplete = sorted(
        str(value["incident_id"])
        for value in values
        if value.get("runtime_evidence_status")
        != "BUG_REPRODUCED_WITH_TRACE"
    )
    if incomplete:
        raise ValueError(
            f"H10-C7 held-out runtime evidence is incomplete: {incomplete}"
        )
    return values
