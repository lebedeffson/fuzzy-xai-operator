from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from fuzzyxai.experiments.h10_c7 import GoldLocalization
from fuzzyxai.experiments.h10_c7r import HeldOutInputs
from fuzzyxai.repository_diagnostics.graph import RepositoryGraph
from fuzzyxai.repository_diagnostics.guided_diagnosis import (
    GuidedNaturalDiagnosisEngine,
)
from fuzzyxai.repository_diagnostics.guided_retrieval import (
    R9_SOURCE_KINDS,
    RankedFile,
    RankedSymbol,
    documents_from_graph,
)

CAUSAL_EVENT_KINDS = frozenset(
    {
        "argument_value",
        "assertion_operand",
        "exception",
        "last_writer",
        "return_value",
        "value_flow",
    }
)
REQUIRED_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "sequence_id",
        "timestamp_ns",
        "thread_id",
        "call_depth",
        "test_id",
        "kind",
        "source_file",
        "occurrence_count",
    }
)
REQUIRED_CORE_KINDS = frozenset({"coverage", "call", "traceback_frame"})
REQUIRED_CAUSAL_OBSERVATIONS = frozenset(
    {"assertion_operand", "exception", "argument_value", "return_value"}
)


@dataclass(frozen=True)
class RuntimeReadiness:
    status: str
    event_count: int
    test_count: int
    max_sequence_id: int
    event_kinds: tuple[str, ...]
    causal_event_kinds: tuple[str, ...]
    missing_fields: tuple[str, ...]
    chronology_errors: tuple[str, ...]
    occurrence_count_total: int
    aggregated_event_count: int
    full_tail_end_preserved: bool
    has_core_runtime: bool
    has_causal_observation: bool
    has_value_provenance: bool

    @property
    def ready(self) -> bool:
        return self.status == "R10_RUNTIME_READY"

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


def read_raw_runtime_rows(path: Path) -> tuple[dict[str, object], ...]:
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"runtime rows must be JSON objects: {path}")
    return rows


def audit_runtime_rows(
    rows: Sequence[Mapping[str, object]],
) -> RuntimeReadiness:
    """Fail closed when chronology was reconstructed rather than collected."""
    missing = sorted(
        {
            field
            for row in rows
            for field in REQUIRED_EVENT_FIELDS
            if field not in row
        }
    )
    chronology_errors: list[str] = []
    previous: dict[str, int] = defaultdict(lambda: -1)
    event_ids: set[str] = set()
    kinds: Counter[str] = Counter()
    occurrence_total = 0
    aggregated_events = 0
    max_sequence = -1
    tests: set[str] = set()

    for index, row in enumerate(rows):
        event_id = str(row.get("event_id", ""))
        test_id = str(row.get("test_id", ""))
        sequence = _integer(row.get("sequence_id"), fallback=-1)
        first = _integer(row.get("first_sequence_id"), fallback=sequence)
        last = _integer(row.get("last_sequence_id"), fallback=sequence)
        occurrence = _integer(row.get("occurrence_count"), fallback=0)
        timestamp = _integer(row.get("timestamp_ns"), fallback=0)
        kind = str(row.get("kind", ""))

        if not event_id or event_id in event_ids:
            chronology_errors.append(f"row[{index}]:invalid_or_duplicate_event_id")
        event_ids.add(event_id)
        if not test_id:
            chronology_errors.append(f"row[{index}]:missing_test_id")
        else:
            tests.add(test_id)
        if sequence < 0 or sequence <= previous[test_id]:
            chronology_errors.append(f"row[{index}]:non_monotonic_sequence")
        if first < 0 or last < first or sequence < first or sequence > last:
            chronology_errors.append(f"row[{index}]:invalid_sequence_interval")
        if timestamp <= 0:
            chronology_errors.append(f"row[{index}]:missing_monotonic_timestamp")
        if occurrence <= 0:
            chronology_errors.append(f"row[{index}]:invalid_occurrence_count")
        previous[test_id] = max(previous[test_id], last)
        occurrence_total += max(occurrence, 0)
        aggregated_events += int(occurrence > 1 or last > first)
        max_sequence = max(max_sequence, last)
        kinds[kind] += 1

    kind_set = set(kinds)
    causal = kind_set & CAUSAL_EVENT_KINDS
    core = REQUIRED_CORE_KINDS.issubset(kind_set)
    causal_observation = bool(causal & REQUIRED_CAUSAL_OBSERVATIONS)
    provenance = bool(causal & {"last_writer", "value_flow"})
    tail_end_preserved = bool(rows) and not any(
        error.endswith(
            (
                "non_monotonic_sequence",
                "invalid_sequence_interval",
            )
        )
        for error in chronology_errors
    )
    ready = (
        bool(rows)
        and not missing
        and not chronology_errors
        and core
        and causal_observation
    )
    return RuntimeReadiness(
        status=(
            "R10_RUNTIME_READY"
            if ready
            else "R10_RUNTIME_RECOLLECTION_REQUIRED"
        ),
        event_count=len(rows),
        test_count=len(tests),
        max_sequence_id=max_sequence,
        event_kinds=tuple(sorted(kind_set)),
        causal_event_kinds=tuple(sorted(causal)),
        missing_fields=tuple(missing),
        chronology_errors=tuple(chronology_errors[:100]),
        occurrence_count_total=occurrence_total,
        aggregated_event_count=aggregated_events,
        full_tail_end_preserved=tail_end_preserved,
        has_core_runtime=core,
        has_causal_observation=causal_observation,
        has_value_provenance=provenance,
    )


def audit_runtime_file(path: Path) -> RuntimeReadiness:
    return audit_runtime_rows(read_raw_runtime_rows(path))


def enrich_graph_with_source_excerpts(
    graph: RepositoryGraph,
    repository_root: Path,
) -> RepositoryGraph:
    """Add observable source spans in the R10 collector, outside frozen importers."""
    source_cache: dict[str, tuple[str, ...]] = {}
    nodes = []
    for node in graph.nodes:
        if not node.file_path:
            nodes.append(node)
            continue
        path = repository_root / node.file_path
        if path.suffix != ".py" or not path.is_file():
            nodes.append(node)
            continue
        lines = source_cache.get(node.file_path)
        if lines is None:
            lines = tuple(
                path.read_text(encoding="utf-8", errors="replace").splitlines()
            )
            source_cache[node.file_path] = lines
        start = max(1, _integer(node.attributes.get("lineno"), fallback=1))
        end = max(
            start,
            _integer(node.attributes.get("end_lineno"), fallback=start + 40),
        )
        excerpt = "\n".join(lines[start - 1 : min(end, start + 160)])
        attributes = {
            **node.attributes,
            "source_excerpt": excerpt[:12000],
        }
        nodes.append(replace(node, attributes=attributes))
    return replace(graph, nodes=tuple(nodes))


def summarize_runtime_readiness(
    incidents: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    statuses = [str(item["status"]) for item in incidents]
    return {
        "protocol_id": "H10-C7R-R10-development-v1",
        "incident_count": len(incidents),
        "ready_incidents": statuses.count("R10_RUNTIME_READY"),
        "recollection_required_incidents": statuses.count(
            "R10_RUNTIME_RECOLLECTION_REQUIRED"
        ),
        "all_incidents_ready": bool(incidents)
        and all(status == "R10_RUNTIME_READY" for status in statuses),
        "scientific_result": "NOT_EVALUATED",
        "development_scored": False,
        "new_held_out_created": False,
        "new_held_out_scored": False,
    }


def score_r10_variants(
    inputs: HeldOutInputs,
    *,
    engine: GuidedNaturalDiagnosisEngine,
    variants: Sequence[str],
) -> list[dict[str, object]]:
    """Score frozen R10 stages without using Gold in any feature channel."""
    rows: list[dict[str, object]] = []
    for case in inputs.cases:
        documents = documents_from_graph(
            case.graph,
            case.runtime_events,
            source_kinds=R9_SOURCE_KINDS,
        )
        files = engine.r10_file_retriever.rank(
            case.query,
            documents,
            case.runtime_events,
            limit=engine.r10_config.file_limit,
        )
        pool = engine.r10_pool_builder.build(
            case.query,
            files,
            documents,
            symbols_per_file=engine.r10_config.symbols_per_file,
            pool_limit=engine.r10_config.pool_limit,
        )
        gold = inputs.gold[case.incident_id]
        for variant in variants:
            ranking, unavailable = engine._r10_ranking(
                variant,
                case.query,
                documents,
                case.runtime_events,
            )
            rows.append(
                {
                    "incident_id": case.incident_id,
                    "repository": case.repository,
                    "variant": variant,
                    "available": not bool(unavailable),
                    "unavailable_reason": unavailable or "",
                    "file_rank": _gold_file_rank(files, gold) or 0,
                    "file_hit_at_10": float(
                        bool((_gold_file_rank(files, gold) or 0) <= 10)
                        and bool(_gold_file_rank(files, gold))
                    ),
                    "file_hit_at_20": float(bool(_gold_file_rank(files, gold))),
                    "pool_rank": _gold_symbol_rank(pool, gold) or 0,
                    "pool_hit_at_200": float(bool(_gold_symbol_rank(pool, gold))),
                    "symbol_rank": _gold_symbol_rank(ranking, gold) or 0,
                    "symbol_hit_at_20": float(
                        bool(_gold_symbol_rank(ranking, gold))
                    ),
                    "file_candidate_count": len(files),
                    "pool_candidate_count": len(pool),
                    "final_candidate_count": len(ranking),
                    "top_20": [item.node_id for item in ranking],
                    "contract_reordered_localization": False,
                }
            )
    return rows


def select_loro_variant(
    rows: Sequence[Mapping[str, object]],
    *,
    variants: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Select a variant on other repositories, then score the held repository."""
    repositories = sorted({str(row["repository"]) for row in rows})
    selected: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for held_repository in repositories:
        training = [
            row for row in rows if row["repository"] != held_repository
        ]
        scores = {
            variant: statistics.fmean(
                float(row["symbol_hit_at_20"])
                for row in training
                if row["variant"] == variant and bool(row["available"])
            )
            if any(
                row["variant"] == variant and bool(row["available"])
                for row in training
            )
            else -1.0
            for variant in variants
        }
        selected_variant = max(
            variants,
            key=lambda variant: (scores[variant], variant),
        )
        held = [
            dict(row)
            for row in rows
            if row["repository"] == held_repository
            and row["variant"] == selected_variant
        ]
        selected.extend(held)
        folds.append(
            {
                "held_repository": held_repository,
                "selected_variant": selected_variant,
                "training_repositories": [
                    repository
                    for repository in repositories
                    if repository != held_repository
                ],
                "training_symbol_recall_at_20": scores[selected_variant],
                "test_incidents": len(held),
                "test_symbol_recall_at_20": statistics.fmean(
                    float(row["symbol_hit_at_20"]) for row in held
                ),
            }
        )
    return selected, folds


def summarize_r10(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    repositories = sorted({str(row["repository"]) for row in rows})
    per_repository = [
        statistics.fmean(
            float(row["symbol_hit_at_20"])
            for row in rows
            if row["repository"] == repository
        )
        for repository in repositories
    ]
    return {
        "incident_count": len(rows),
        "repository_count": len(repositories),
        "file_recall_at_10": _mean(rows, "file_hit_at_10"),
        "file_recall_at_20": _mean(rows, "file_hit_at_20"),
        "symbol_pool_recall_at_200": _mean(rows, "pool_hit_at_200"),
        "symbol_recall_at_20": _mean(rows, "symbol_hit_at_20"),
        "repository_recall_lower_quartile": _lower_quartile(per_repository),
        "coverage": statistics.fmean(bool(row["available"]) for row in rows),
        "contract_reordering_count": sum(
            bool(row["contract_reordered_localization"]) for row in rows
        ),
        "selected_variants": dict(
            sorted(Counter(str(row["variant"]) for row in rows).items())
        ),
    }


def development_gates(summary: Mapping[str, object]) -> dict[str, bool]:
    return {
        "file_recall_at_10_at_least_0_90": (
            float(summary["file_recall_at_10"]) >= 0.90
        ),
        "file_recall_at_20_at_least_0_97": (
            float(summary["file_recall_at_20"]) >= 0.97
        ),
        "symbol_pool_recall_at_200_at_least_0_95": (
            float(summary["symbol_pool_recall_at_200"]) >= 0.95
        ),
        "symbol_recall_at_20_at_least_0_85": (
            float(summary["symbol_recall_at_20"]) >= 0.85
        ),
        "repository_recall_lower_quartile_at_least_0_75": (
            float(summary["repository_recall_lower_quartile"]) >= 0.75
        ),
        "coverage_at_least_0_90": float(summary["coverage"]) >= 0.90,
        "contract_does_not_reorder_localization": (
            int(summary["contract_reordering_count"]) == 0
        ),
    }


def _gold_file_rank(
    files: Sequence[RankedFile],
    gold: GoldLocalization,
) -> int | None:
    for rank, candidate in enumerate(files, start=1):
        if any(candidate.file_path == atom.file_path for atom in gold.atoms):
            return rank
    return None


def _gold_symbol_rank(
    candidates: Sequence[RankedSymbol],
    gold: GoldLocalization,
) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if any(
            candidate.file_path == atom.file_path
            and candidate.symbol == atom.symbol
            for atom in gold.atoms
        ):
            return rank
    return None


def _mean(rows: Sequence[Mapping[str, object]], name: str) -> float:
    return statistics.fmean(float(row[name]) for row in rows)


def _lower_quartile(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, (len(ordered) - 1) // 4)]


def _integer(value: object, *, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
