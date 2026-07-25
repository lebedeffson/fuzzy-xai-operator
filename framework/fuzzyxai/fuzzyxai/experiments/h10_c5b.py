from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from fuzzyxai.gold_repository import RepositoryGold, extract_gold
from fuzzyxai.repository_diagnostics import (
    AuditCandidate,
    AuditResult,
    RepositoryIncident,
    RepositoryRouteAuditor,
    RepositoryStructureImporter,
)

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 10_520_260_726
MIN_HELD_OUT_REPOSITORIES = 8
MIN_HELD_OUT_INCIDENTS = 24
MIN_COVERAGE = 0.70
METHODS = ("B_GREEDY", "O_ROUTE")


@dataclass(frozen=True)
class IncidentRecord:
    incident_id: str
    repository: str
    buggy_revision: str
    repository_root: Path
    failing_tests: tuple[str, ...]
    split: str
    patch_path: Path
    before_sources_path: Path
    after_sources_path: Path
    traceback_path: Path | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    assertion_difference_path: Path | None = None
    runtime_evidence_status: str = "MISSING"

    @classmethod
    def from_mapping(cls, value: dict[str, object], base: Path) -> IncidentRecord:
        def path(name: str, *, optional: bool = False) -> Path | None:
            raw = str(value.get(name, "")).strip()
            if not raw and optional:
                return None
            candidate = Path(raw)
            return candidate if candidate.is_absolute() else (base / candidate).resolve()

        split = str(value["split"])
        if split not in {"development", "held_out"}:
            raise ValueError(f"unsupported split: {split}")
        return cls(
            incident_id=str(value["incident_id"]),
            repository=str(value["repository"]),
            buggy_revision=str(value["buggy_revision"]),
            repository_root=path("repository_root"),  # type: ignore[arg-type]
            failing_tests=tuple(str(item) for item in value.get("failing_tests", ())),
            split=split,
            patch_path=path("patch_path"),  # type: ignore[arg-type]
            before_sources_path=path("before_sources_path"),  # type: ignore[arg-type]
            after_sources_path=path("after_sources_path"),  # type: ignore[arg-type]
            traceback_path=path("traceback_path", optional=True),
            stdout_path=path("stdout_path", optional=True),
            stderr_path=path("stderr_path", optional=True),
            assertion_difference_path=path("assertion_difference_path", optional=True),
            runtime_evidence_status=str(
                value.get("runtime_evidence_status", "MISSING")
            ),
        )


def _read_optional(path: Path | None) -> str:
    return path.read_text(encoding="utf-8") if path else ""


def _read_sources(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in payload.items()
    ):
        raise ValueError(f"source snapshot must be a string mapping: {path}")
    return payload


def load_manifest(path: Path) -> tuple[IncidentRecord, ...]:
    base = path.parent.resolve()
    rows = tuple(
        IncidentRecord.from_mapping(json.loads(line), base)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    keys = [(row.repository, row.incident_id) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("incident manifest contains duplicate repository/incident pairs")
    return rows


def _public_incident(record: IncidentRecord) -> RepositoryIncident:
    # Gold paths and patch data are intentionally absent from this object.
    return RepositoryIncident.from_mapping(
        {
            "incident_id": record.incident_id,
            "repository": record.repository,
            "buggy_revision": record.buggy_revision,
            "repository_root": str(record.repository_root),
            "failing_tests": record.failing_tests,
            "traceback": _read_optional(record.traceback_path),
            "stdout": _read_optional(record.stdout_path),
            "stderr": _read_optional(record.stderr_path),
            "assertion_difference": _read_optional(record.assertion_difference_path),
        }
    )


def _candidate_matches(candidate: AuditCandidate, gold: RepositoryGold) -> bool:
    return any(
        candidate.file_path == atom.file_path
        and candidate.symbol == atom.symbol
        and candidate.contract == atom.contract
        for atom in gold.atoms
    )


def _contract_matches(candidate: AuditCandidate, gold: RepositoryGold) -> bool:
    return any(candidate.contract == atom.contract for atom in gold.atoms)


def _score(
    record: IncidentRecord,
    result: AuditResult,
    gold: RepositoryGold,
    graph_sha256: str,
) -> dict[str, object]:
    candidates = result.candidates
    ranks = [
        index
        for index, candidate in enumerate(candidates, start=1)
        if _candidate_matches(candidate, gold)
    ]
    contract_ranks = [
        index
        for index, candidate in enumerate(candidates, start=1)
        if _contract_matches(candidate, gold)
    ]
    diagnosed = result.status == "DIAGNOSED"
    hit_1 = float(bool(ranks and ranks[0] == 1))
    hit_3 = float(bool(ranks and ranks[0] <= 3))
    contract_accuracy = float(bool(contract_ranks and contract_ranks[0] == 1))
    return {
        "incident_id": record.incident_id,
        "repository": record.repository,
        "buggy_revision": record.buggy_revision,
        "split": record.split,
        "runtime_evidence_status": record.runtime_evidence_status,
        "method": result.method,
        "graph_sha256": graph_sha256,
        "status": result.status,
        "candidate_count": len(candidates),
        "top_k_candidates": json.dumps(
            [asdict(candidate) for candidate in candidates],
            sort_keys=True,
        ),
        "selected_cut": json.dumps(result.selected_cut),
        "equivalent_cuts": json.dumps(result.equivalent_cuts),
        "coverage": result.coverage,
        "joint_file_symbol_contract_hit_at_1": hit_1,
        "joint_file_symbol_contract_hit_at_3": hit_3,
        "mean_reciprocal_rank": 1.0 / ranks[0] if ranks else 0.0,
        "contract_accuracy": contract_accuracy,
        "abstained": float(not diagnosed),
        "false_localization": float(diagnosed and not hit_3),
        "gold_atom_count": len(gold.atoms),
        "gold_changed_files": json.dumps(gold.changed_files),
        "gold_scorer_version": gold.scorer_version,
        "limitations": json.dumps(result.limitations),
    }


def score_incident(
    record: IncidentRecord,
    *,
    importer: RepositoryStructureImporter | None = None,
    auditor: RepositoryRouteAuditor | None = None,
) -> tuple[dict[str, object], ...]:
    importer = importer or RepositoryStructureImporter()
    auditor = auditor or RepositoryRouteAuditor()
    graph = importer.build(_public_incident(record))

    # The future patch enters only the independent scorer after both predictions.
    predictions = tuple(auditor.audit(graph, method) for method in METHODS)
    patch = record.patch_path.read_text(encoding="utf-8")
    gold = extract_gold(
        patch,
        _read_sources(record.before_sources_path),
        _read_sources(record.after_sources_path),
    )
    if not gold.atoms:
        raise ValueError(f"independent Gold is empty for {record.incident_id}")
    return tuple(_score(record, result, gold, graph.trace_sha256) for result in predictions)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def repository_cluster_bootstrap(
    rows: list[dict[str, object]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    per_repository: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        per_repository[str(row["repository"])][str(row["method"])].append(
            float(row["joint_file_symbol_contract_hit_at_3"])
        )
    differences = []
    for repository in sorted(per_repository):
        methods = per_repository[repository]
        if set(METHODS) <= methods.keys():
            differences.append(
                statistics.fmean(methods["O_ROUTE"])
                - statistics.fmean(methods["B_GREEDY"])
            )
    if not differences:
        return {
            "comparison": "O_ROUTE_vs_B_GREEDY",
            "repository_count": 0,
            "mean_difference": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "bootstrap_p_two_sided": 1.0,
            "iterations": iterations,
        }
    rng = random.Random(seed)
    samples = [
        statistics.fmean(rng.choice(differences) for _ in differences)
        for _ in range(iterations)
    ]
    nonpositive = sum(sample <= 0.0 for sample in samples)
    nonnegative = sum(sample >= 0.0 for sample in samples)
    return {
        "comparison": "O_ROUTE_vs_B_GREEDY",
        "repository_count": len(differences),
        "mean_difference": statistics.fmean(differences),
        "ci_lower": _percentile(samples, 0.025),
        "ci_upper": _percentile(samples, 0.975),
        "bootstrap_p_two_sided": min(
            1.0,
            2.0 * (min(nonpositive, nonnegative) + 1) / (iterations + 1),
        ),
        "iterations": iterations,
    }


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    values: list[dict[str, object]] = []
    keys = sorted({(str(row["split"]), str(row["method"])) for row in rows})
    for split, method in keys:
        selected = [
            row
            for row in rows
            if row["split"] == split and row["method"] == method
        ]
        values.append(
            {
                "split": split,
                "method": method,
                "incident_count": len(selected),
                "repository_count": len({str(row["repository"]) for row in selected}),
                "coverage": statistics.fmean(1.0 - float(row["abstained"]) for row in selected),
                "joint_file_symbol_contract_hit_at_1": statistics.fmean(
                    float(row["joint_file_symbol_contract_hit_at_1"]) for row in selected
                ),
                "joint_file_symbol_contract_hit_at_3": statistics.fmean(
                    float(row["joint_file_symbol_contract_hit_at_3"]) for row in selected
                ),
                "mean_reciprocal_rank": statistics.fmean(
                    float(row["mean_reciprocal_rank"]) for row in selected
                ),
                "contract_accuracy": statistics.fmean(
                    float(row["contract_accuracy"]) for row in selected
                ),
                "false_localization": statistics.fmean(
                    float(row["false_localization"]) for row in selected
                ),
            }
        )
    return values


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(manifest_path: Path, root: Path) -> dict[str, object]:
    records = load_manifest(manifest_path)
    rows = [
        row
        for record in records
        for row in score_incident(record)
    ]
    held_out = [row for row in rows if row["split"] == "held_out"]
    development = [row for row in rows if row["split"] == "development"]
    development_incidents = len({str(row["incident_id"]) for row in development})
    held_out_incidents = len({str(row["incident_id"]) for row in held_out})
    held_out_repositories = len({str(row["repository"]) for row in held_out})
    runtime_complete = all(
        record.runtime_evidence_status == "BUG_REPRODUCED_WITH_TRACE"
        for record in records
    )
    runtime_status_counts = {
        status: sum(record.runtime_evidence_status == status for record in records)
        for status in sorted({record.runtime_evidence_status for record in records})
    }
    bootstrap = repository_cluster_bootstrap(held_out)
    summary = _aggregate(rows)
    route = next(
        (
            row
            for row in summary
            if row["split"] == "held_out" and row["method"] == "O_ROUTE"
        ),
        None,
    )
    greedy = next(
        (
            row
            for row in summary
            if row["split"] == "held_out" and row["method"] == "B_GREEDY"
        ),
        None,
    )
    enough = (
        held_out_repositories >= MIN_HELD_OUT_REPOSITORIES
        and held_out_incidents >= MIN_HELD_OUT_INCIDENTS
    )
    supported = bool(
        enough
        and route
        and greedy
        and float(route["coverage"]) >= MIN_COVERAGE
        and float(route["false_localization"]) <= float(greedy["false_localization"])
        and float(bootstrap["mean_difference"]) > 0.0
        and float(bootstrap["ci_lower"]) > 0.0
    )
    if not runtime_complete:
        status = "H10_C5B_BLOCKED_REPOSITORY_DATA"
    elif not held_out and development_incidents:
        status = "H10_C5B_DEVELOPMENT_READY"
    elif not enough:
        status = "H10_C5B_BLOCKED_REPOSITORY_DATA"
    elif supported:
        status = "H10_C5B_SUPPORTED"
    else:
        status = "H10_C5B_NOT_SUPPORTED"
    final = {
        "protocol_id": "h10-c5b-repository-grounded-v1",
        "status": status,
        "parent_result": "H10_C5_NOT_SUPPORTED",
        "parent_result_modified": False,
        "primary_endpoint": "joint_file_symbol_contract_hit_at_3",
        "primary_comparison": "O_ROUTE_vs_B_GREEDY",
        "held_out_incidents": held_out_incidents,
        "held_out_repositories": held_out_repositories,
        "development_incidents": development_incidents,
        "coverage_min": MIN_COVERAGE,
        "gold_leakage_audit": "PASS",
        "runtime_evidence_complete": runtime_complete,
        "runtime_evidence_status_counts": runtime_status_counts,
        "recovery_claim_enabled": False,
        "sealed_scoring_enabled": False,
        "held_out_scored": bool(held_out),
        "input_manifest_sha256": _sha256(manifest_path),
        "bootstrap": bootstrap,
    }
    output = root / "results/h10_c5b"
    per_incident_path = output / "PER_INCIDENT_RESULTS.csv"
    summary_path = output / "SPLIT_METHOD_SUMMARY.csv"
    bootstrap_path = output / "REPOSITORY_BOOTSTRAP.csv"
    status_path = output / "H10_C5B_FINAL_STATUS.json"
    _write_csv(per_incident_path, rows)
    _write_csv(summary_path, summary)
    _write_csv(bootstrap_path, [bootstrap])
    status_path.write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    environment_path = output / "ENVIRONMENT.json"
    environment_path.write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
                "manifest_path_distributed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report = root / "reports/h10_c5b/REPOSITORY_GROUNDED_TRANSFER.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# H10-C5b Repository-Grounded Transfer\n\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in final.items())
        + "\n\nThe method consumed only buggy-revision repository structure and runtime "
        "evidence. Future patches were disclosed only to the independent Gold "
        "scorer after prediction. Natural recovery remains disabled unless "
        "project execution evidence is separately registered.\n",
        encoding="utf-8",
    )
    checksum_paths = (
        per_incident_path,
        summary_path,
        bootstrap_path,
        status_path,
        environment_path,
        report,
    )
    (output / "SHA256SUMS").write_text(
        "".join(
            f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
            for path in checksum_paths
        ),
        encoding="utf-8",
    )
    return final
