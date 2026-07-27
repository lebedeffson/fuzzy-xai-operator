from __future__ import annotations

import csv
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path

from fuzzyxai.experiments.h10_c5b import (
    IncidentRecord,
    _public_incident,
    _read_sources,
    _score,
)
from fuzzyxai.experiments.h10_c5c import validate_development_manifest
from fuzzyxai.gold_repository import GoldRepairAtom, RepositoryGold, extract_gold
from fuzzyxai.repository_diagnostics.auditor import AuditResult
from fuzzyxai.repository_diagnostics.auditor_v2 import (
    EvidenceGroundedRouteAuditor,
)
from fuzzyxai.repository_diagnostics.contract_inference import (
    SUPPORTED_CONTRACTS,
    ContractInference,
    EvidenceGroundedContractInferer,
)
from fuzzyxai.repository_diagnostics.graph import RepositoryGraph, RepositoryNode
from fuzzyxai.repository_diagnostics.importer_v2 import (
    EvidenceGroundedRepositoryImporter,
)
from fuzzyxai.repository_diagnostics.retrieval import (
    CandidateFeatures,
    EvidenceGroundedCandidateRetriever,
    RetrievedCandidate,
)
from fuzzyxai.repository_diagnostics.runtime_events import (
    load_runtime_events,
)

LOCK_PATH = Path(
    "protocol/h10_c5c_evidence_retrieval/"
    "H10_C5C_POSTHOC_ORACLE_DECOMPOSITION_LOCK.json"
)
VARIANTS = (
    "BASELINE",
    "ORACLE_CANDIDATE",
    "ORACLE_CONTRACT",
    "ORACLE_BOTH",
)


class _FixedRetriever:
    def __init__(self, candidates: tuple[RetrievedCandidate, ...]) -> None:
        self.candidates = candidates

    def retrieve(
        self,
        graph: RepositoryGraph,
    ) -> tuple[RetrievedCandidate, ...]:
        del graph
        return self.candidates


class _OracleContractInferer:
    def __init__(self, gold: RepositoryGold) -> None:
        self.gold = gold

    def infer(
        self,
        graph: RepositoryGraph,
        candidate: RetrievedCandidate,
    ) -> ContractInference:
        del graph
        contract = _gold_contract_for_candidate(candidate, self.gold)
        supported = contract in SUPPORTED_CONTRACTS
        return ContractInference(
            contract,
            0.0,
            1.0 if supported else 0.0,
            ("posthoc_oracle_contract",),
            supported,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_lock(root: Path) -> dict[str, object]:
    value = json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))
    if value.get("status") != "LOCKED_BEFORE_POSTHOC_ORACLE_EXECUTION":
        raise ValueError("H10-C5c post-hoc oracle lock is invalid")
    if value.get("oracle_results_may_support_scientific_claim") is not False:
        raise ValueError("oracle analysis must remain non-confirmatory")
    if value.get("method_or_gate_changed") is not False:
        raise ValueError("post-hoc oracle analysis cannot change the method")
    if (
        value.get("held_out_created") is not False
        or value.get("held_out_scored") is not False
    ):
        raise ValueError("post-hoc oracle analysis cannot use held-out data")
    return value


def _runtime_event_paths(manifest_path: Path) -> dict[str, Path]:
    base = manifest_path.parent.resolve()
    paths = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        raw = Path(str(value["runtime_events_path"]))
        paths[str(value["incident_id"])] = (
            raw if raw.is_absolute() else (base / raw).resolve()
        )
    return paths


def _gold_contract_for_candidate(
    candidate: RetrievedCandidate,
    gold: RepositoryGold,
) -> str:
    exact = next(
        (
            atom.contract
            for atom in gold.atoms
            if atom.file_path == candidate.file_path
            and atom.symbol == candidate.symbol
        ),
        None,
    )
    if exact is not None:
        return exact
    return min(atom.contract for atom in gold.atoms)


def _gold_node(
    graph: RepositoryGraph,
    atom: GoldRepairAtom,
) -> RepositoryNode | None:
    exact = next(
        (
            node
            for node in graph.nodes
            if node.file_path == atom.file_path
            and node.symbol == atom.symbol
        ),
        None,
    )
    if exact is not None:
        return exact
    if atom.symbol is None:
        return next(
            (
                node
                for node in graph.nodes
                if node.kind == "file"
                and node.file_path == atom.file_path
            ),
            None,
        )
    return None


def _with_oracle_nodes(
    graph: RepositoryGraph,
    gold: RepositoryGold,
) -> tuple[RepositoryGraph, dict[tuple[str, str | None], RepositoryNode]]:
    nodes = list(graph.nodes)
    by_source = {}
    for atom in gold.atoms:
        key = (atom.file_path, atom.symbol)
        if key in by_source:
            continue
        node = _gold_node(graph, atom)
        if node is None:
            file_node = next(
                (
                    candidate
                    for candidate in graph.nodes
                    if candidate.kind == "file"
                    and candidate.file_path == atom.file_path
                ),
                None,
            )
            digest = hashlib.sha256(
                f"{atom.file_path}\0{atom.symbol}".encode()
            ).hexdigest()[:16]
            node = RepositoryNode(
                f"posthoc_oracle:{digest}",
                "posthoc_oracle_symbol",
                graph.repository,
                atom.file_path,
                atom.symbol,
                {
                    "posthoc_oracle_only": True,
                    "semantic_tokens": tuple(
                        sorted(
                            token.lower()
                            for token in (atom.symbol or "").replace(".", "_").split("_")
                            if token
                        )
                    ),
                },
                file_node.evidence_refs if file_node is not None else (),
            )
            nodes.append(node)
        by_source[key] = node
    return (
        replace(
            graph,
            nodes=tuple(sorted(nodes, key=lambda item: item.node_id)),
        ),
        by_source,
    )


def _oracle_retrieved_pool(
    graph: RepositoryGraph,
    observed: tuple[RetrievedCandidate, ...],
    gold: RepositoryGold,
) -> tuple[RepositoryGraph, tuple[RetrievedCandidate, ...]]:
    augmented, nodes = _with_oracle_nodes(graph, gold)
    score = max(
        (candidate.retrieval_score for candidate in observed),
        default=1.0,
    )
    confidence = max(
        (candidate.confidence for candidate in observed),
        default=1.0,
    )
    oracle_sources = set(nodes)
    values = [
        candidate
        for candidate in observed
        if (candidate.file_path, candidate.symbol) not in oracle_sources
    ]
    for source, node in sorted(
        nodes.items(),
        key=lambda item: (item[0][0], item[0][1] or ""),
    ):
        values.append(
            RetrievedCandidate(
                node.node_id,
                node.repository,
                node.file_path,
                node.symbol,
                score,
                confidence,
                graph.obligations,
                node.evidence_refs,
                (),
                CandidateFeatures(),
            )
        )
    return augmented, tuple(
        sorted(
            values,
            key=lambda item: (
                -item.retrieval_score,
                -len(item.covered_obligations),
                item.file_path or "",
                item.symbol or "",
                item.node_id,
            ),
        )
    )


def _auditor(
    threshold: float,
    retrieved: tuple[RetrievedCandidate, ...],
    contract_inferer: EvidenceGroundedContractInferer
    | _OracleContractInferer,
) -> EvidenceGroundedRouteAuditor:
    return EvidenceGroundedRouteAuditor(
        abstention_threshold=threshold,
        retriever=_FixedRetriever(retrieved),
        contract_inferer=contract_inferer,
    )


def _signature(result: AuditResult) -> str:
    value = [
        {
            "node_id": candidate.node_id,
            "contract": candidate.contract,
        }
        for candidate in result.candidates
    ]
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def _gold_cut_hit(
    result: AuditResult,
    gold: RepositoryGold,
) -> bool:
    by_id = {candidate.node_id: candidate for candidate in result.candidates}
    return any(
        candidate.file_path == atom.file_path
        and candidate.symbol == atom.symbol
        and candidate.contract == atom.contract
        for node_id in result.selected_cut
        if (candidate := by_id.get(node_id)) is not None
        for atom in gold.atoms
    )


def _variant_row(
    record: IncidentRecord,
    variant: str,
    result: AuditResult,
    gold: RepositoryGold,
    *,
    oracle_candidate_injected: bool,
) -> dict[str, object]:
    scored = _score(
        record,
        result,
        gold,
        "POSTHOC_ORACLE_GRAPH_NOT_A_METHOD_RESULT",
    )
    return {
        "incident_id": record.incident_id,
        "repository": record.repository,
        "variant": variant,
        "status": result.status,
        "joint_hit_at_3": scored[
            "joint_file_symbol_contract_hit_at_3"
        ],
        "joint_hit_at_1": scored[
            "joint_file_symbol_contract_hit_at_1"
        ],
        "selected_cut_contains_gold": float(
            _gold_cut_hit(result, gold)
        ),
        "candidate_count": len(result.candidates),
        "oracle_candidate_injected": oracle_candidate_injected,
        "gold_contract_registered": all(
            atom.contract in SUPPORTED_CONTRACTS
            for atom in gold.atoms
        ),
        "gold_sources": json.dumps(
            sorted(
                {
                    (atom.file_path, atom.symbol)
                    for atom in gold.atoms
                },
                key=lambda item: (item[0], item[1] or ""),
            )
        ),
        "gold_contracts": json.dumps(
            sorted({atom.contract for atom in gold.atoms})
        ),
        "top_k_signature": _signature(result),
        "selected_cut": json.dumps(result.selected_cut),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _interpret(metrics: dict[str, dict[str, float]]) -> str:
    baseline = metrics["BASELINE"]["joint_hit_at_3"]
    candidate = metrics["ORACLE_CANDIDATE"]["joint_hit_at_3"]
    contract = metrics["ORACLE_CONTRACT"]["joint_hit_at_3"]
    both = metrics["ORACLE_BOTH"]["joint_hit_at_3"]
    candidate_gain = candidate > baseline
    contract_gain = contract > baseline
    both_gain = both > max(candidate, contract)
    if candidate_gain and not contract_gain:
        return "RETRIEVAL_DOMINANT"
    if contract_gain and not candidate_gain:
        return "CONTRACT_INFERENCE_DOMINANT"
    if both_gain:
        return "RETRIEVAL_CONTRACT_INTERACTION"
    if both <= baseline:
        return "DOWNSTREAM_GRAPH_OR_OPTIMIZATION_LIMIT"
    return "MIXED_RETRIEVAL_AND_CONTRACT_LIMIT"


def run_posthoc_oracle_decomposition(
    manifest_path: Path,
    development_status_path: Path,
    baseline_results_path: Path,
    root: Path,
    output: Path,
) -> dict[str, object]:
    lock = _load_lock(root)
    development_status = json.loads(
        development_status_path.read_text(encoding="utf-8")
    )
    if (
        development_status.get("status")
        != "H10_C5C_DEVELOPMENT_GATE_FAIL"
        or development_status.get("scientific_result") != "NOT_EVALUATED"
        or development_status.get("held_out_created") is not False
        or development_status.get("held_out_scored") is not False
    ):
        raise ValueError("post-hoc analysis requires the closed development gate")
    if _sha256(development_status_path) != lock["development_status_sha256"]:
        raise ValueError("development status does not match oracle lock")
    if _sha256(manifest_path) != lock["input_manifest_sha256"]:
        raise ValueError("development manifest does not match oracle lock")
    if _sha256(baseline_results_path) != lock["baseline_results_sha256"]:
        raise ValueError("development baseline does not match oracle lock")

    records = validate_development_manifest(manifest_path, root)
    event_paths = _runtime_event_paths(manifest_path)
    with baseline_results_path.open(
        encoding="utf-8",
        newline="",
    ) as stream:
        committed_rows = {
            row["incident_id"]: row
            for row in csv.DictReader(stream)
            if row["method"] == "O_ROUTE"
        }
    if set(committed_rows) != {
        record.incident_id for record in records
    }:
        raise ValueError("committed baseline rows do not match manifest")

    threshold = float(development_status["abstention_threshold"])
    importer = EvidenceGroundedRepositoryImporter()
    retriever = EvidenceGroundedCandidateRetriever()
    rows = []
    for record in records:
        events = load_runtime_events(event_paths[record.incident_id])
        graph = importer.build(
            _public_incident(record),
            runtime_events=events,
        )
        observed = retriever.retrieve(graph)

        # Gold is opened only for this explicitly post-hoc development analysis.
        gold = extract_gold(
            record.patch_path.read_text(encoding="utf-8"),
            _read_sources(record.before_sources_path),
            _read_sources(record.after_sources_path),
        )
        oracle_graph, oracle_pool = _oracle_retrieved_pool(
            graph,
            observed,
            gold,
        )
        variants = {
            "BASELINE": _auditor(
                threshold,
                observed,
                EvidenceGroundedContractInferer(),
            ).audit(graph, "O_ROUTE"),
            "ORACLE_CANDIDATE": _auditor(
                threshold,
                oracle_pool,
                EvidenceGroundedContractInferer(),
            ).audit(oracle_graph, "O_ROUTE"),
            "ORACLE_CONTRACT": _auditor(
                threshold,
                observed,
                _OracleContractInferer(gold),
            ).audit(graph, "O_ROUTE"),
            "ORACLE_BOTH": _auditor(
                threshold,
                oracle_pool,
                _OracleContractInferer(gold),
            ).audit(oracle_graph, "O_ROUTE"),
        }
        committed_signature = committed_rows[
            record.incident_id
        ]["top_k_signature"]
        if _signature(variants["BASELINE"]) != committed_signature:
            raise ValueError(
                "baseline reproduction mismatch: "
                f"{record.incident_id}"
            )
        for variant in VARIANTS:
            rows.append(
                _variant_row(
                    record,
                    variant,
                    variants[variant],
                    gold,
                    oracle_candidate_injected=(
                        variant
                        in {"ORACLE_CANDIDATE", "ORACLE_BOTH"}
                    ),
                )
            )

    metrics = {
        variant: {
            "incident_count": len(
                [
                    row
                    for row in rows
                    if row["variant"] == variant
                ]
            ),
            "joint_hit_at_3": statistics.fmean(
                float(row["joint_hit_at_3"])
                for row in rows
                if row["variant"] == variant
            ),
            "joint_hit_at_1": statistics.fmean(
                float(row["joint_hit_at_1"])
                for row in rows
                if row["variant"] == variant
            ),
            "selected_cut_contains_gold": statistics.fmean(
                float(row["selected_cut_contains_gold"])
                for row in rows
                if row["variant"] == variant
            ),
            "diagnosis_coverage": statistics.fmean(
                float(row["status"] == "DIAGNOSIS_CONFIRMED")
                for row in rows
                if row["variant"] == variant
            ),
        }
        for variant in VARIANTS
    }
    summary = {
        "analysis_id": lock["analysis_id"],
        "analysis_type": lock["analysis_type"],
        "status": "POSTHOC_DECOMPOSITION_COMPLETE",
        "scientific_result": "NOT_EVALUATED",
        "scientific_claim_permitted": False,
        "official_development_status_modified": False,
        "held_out_created": False,
        "held_out_scored": False,
        "incident_count": len(records),
        "repository_count": len(
            {record.repository for record in records}
        ),
        "metrics": metrics,
        "interpretation": _interpret(metrics),
        "input_hashes": {
            "manifest_sha256": _sha256(manifest_path),
            "development_status_sha256": _sha256(
                development_status_path
            ),
            "baseline_results_sha256": _sha256(
                baseline_results_path
            ),
            "oracle_lock_sha256": _sha256(root / LOCK_PATH),
        },
        "limitations": [
            "Oracle interventions use disclosed development Gold.",
            "Oracle values are diagnostic upper bounds, not method performance.",
            "No held-out data were created or scored.",
            "No statistical hypothesis test is performed.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "ORACLE_DECOMPOSITION_PER_INCIDENT.csv",
        rows,
    )
    (output / "ORACLE_DECOMPOSITION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary
