from __future__ import annotations

"""Independent adjudication oracle for H10 v19.

This module deliberately has no imports from ``fuzzyxai.audit_h10`` and does
not reuse the evaluated classifier, localizer, diagnostic-cut solver, repair
planner, or taxonomy tables. Truth is derived from an immutable mutation log
and an independently implemented exhaustive hitting-set oracle.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable


@dataclass(frozen=True)
class MutationSpec:
    parent: str
    leaf: str
    fields_by_severity: dict[str, tuple[str, ...]]
    source_nodes: tuple[str, ...]


CATALOG: tuple[MutationSpec, ...] = (
    MutationSpec("artifact_integrity", "hash_corruption", {"subtle": ("artifact_sha256",), "moderate": ("artifact_sha256",), "severe": ("artifact_sha256",)}, ("canonical_artifact",)),
    MutationSpec("artifact_integrity", "missing_artifact", {"subtle": ("artifact_uri",), "moderate": ("artifact_uri",), "severe": ("artifact_uri",)}, ("artifact_store",)),
    MutationSpec("artifact_integrity", "version_mismatch", {"subtle": ("model_version",), "moderate": ("model_version", "artifact_model_id"), "severe": ("model_version", "artifact_model_id", "model_id")}, ("model_registry",)),
    MutationSpec("semantic_compatibility", "model_explainer_mismatch", {"subtle": ("explainer_model_family",), "moderate": ("model_family", "explainer_model_family"), "severe": ("model_family", "explainer_model_family", "explainer_version")}, ("compatibility_registry",)),
    MutationSpec("semantic_compatibility", "preprocessing_mismatch", {"subtle": ("preprocessing_signature",), "moderate": ("preprocessing_signature", "dependency_digest"), "severe": ("preprocessing_signature", "dependency_digest", "canonical_source_id")}, ("preprocessor_registry",)),
    MutationSpec("semantic_compatibility", "dictionary_mismatch", {"subtle": ("dictionary_version",), "moderate": ("dictionary_version", "canonical_source_id"), "severe": ("dictionary_version", "canonical_source_id", "dependency_digest")}, ("dictionary_registry",)),
    MutationSpec("reference_context", "wrong_reference_population", {"subtle": ("reference_population",), "moderate": ("reference_population", "deployment_context"), "severe": ("reference_population", "deployment_context", "calibration_version")}, ("reference_registry",)),
    MutationSpec("reference_context", "stale_calibration", {"subtle": ("calibration_age_days",), "moderate": ("calibration_version", "calibration_age_days"), "severe": ("calibration_version", "calibration_age_days", "reference_population")}, ("calibration_registry",)),
    MutationSpec("reference_context", "deployment_context_mismatch", {"subtle": ("deployment_context",), "moderate": ("deployment_context", "reference_population"), "severe": ("deployment_context", "reference_population", "calibration_version")}, ("deployment_registry",)),
    MutationSpec("provenance", "missing_source", {"subtle": ("source_uri",), "moderate": ("source_uri", "dependency_digest"), "severe": ("source_uri", "dependency_digest", "canonical_source_id")}, ("provenance_registry",)),
    MutationSpec("provenance", "broken_dependency", {"subtle": ("dependency_digest",), "moderate": ("dependency_digest", "canonical_source_id"), "severe": ("dependency_digest", "canonical_source_id", "artifact_sha256")}, ("dependency_registry",)),
    MutationSpec("provenance", "cross_model_artifact_mix", {"subtle": ("artifact_model_id",), "moderate": ("artifact_model_id", "model_id"), "severe": ("artifact_model_id", "model_id", "model_version")}, ("artifact_binding_registry",)),
    MutationSpec("reduction", "excessive_information_loss", {"subtle": ("reduction_loss",), "moderate": ("reduction_loss", "projection_type"), "severe": ("reduction_loss", "projection_type", "canonical_source_id")}, ("reduction_registry",)),
    MutationSpec("reduction", "unsupported_projection", {"subtle": ("projection_type",), "moderate": ("projection_type", "canonical_source_id"), "severe": ("projection_type", "canonical_source_id", "reduction_loss")}, ("reduction_registry",)),
    MutationSpec("reduction", "lost_canonical_link", {"subtle": ("canonical_source_id",), "moderate": ("canonical_source_id", "artifact_uri"), "severe": ("canonical_source_id", "artifact_uri", "artifact_sha256")}, ("canonical_registry",)),
)

SPEC_BY_LEAF = {item.leaf: item for item in CATALOG}


@dataclass(frozen=True)
class MutationOperation:
    leaf: str
    parent: str
    severity: str
    modified_fields: tuple[str, ...]
    source_nodes: tuple[str, ...]


@dataclass(frozen=True)
class OracleTruth:
    case_id: str
    route_status: str
    parent_families: tuple[str, ...]
    leaf_types: tuple[str, ...]
    source_nodes: tuple[str, ...]
    optimal_cuts: tuple[tuple[str, ...], ...]
    optimal_cost: float
    repair_sets: tuple[tuple[str, ...], ...]
    unknown: bool
    insufficient_evidence: bool
    severity: str
    composite: bool
    mutation_log: tuple[MutationOperation, ...]

    @property
    def diagnostic_cut(self) -> tuple[str, ...]:
        return self.optimal_cuts[0] if self.optimal_cuts else ()

    @property
    def repair_set(self) -> tuple[str, ...]:
        return self.repair_sets[0] if self.repair_sets else ()


def independent_optimal_cuts(
    invalid_paths: Iterable[Iterable[str]], costs: dict[str, float]
) -> tuple[tuple[tuple[str, ...], ...], float]:
    """Enumerate all minimum-cost hitting sets without H10 solver code."""
    paths = tuple(frozenset(path) for path in invalid_paths if tuple(path))
    if not paths:
        return ((),), 0.0
    nodes = tuple(sorted(frozenset().union(*paths)))
    best_cost = float("inf")
    best: list[tuple[str, ...]] = []
    for size in range(1, len(nodes) + 1):
        found_at_size = False
        for combo in combinations(nodes, size):
            chosen = set(combo)
            if not all(chosen & path for path in paths):
                continue
            cost = float(sum(costs.get(node, 1.0) for node in combo))
            if cost < best_cost - 1e-12:
                best_cost = cost
                best = [tuple(sorted(combo))]
                found_at_size = True
            elif abs(cost - best_cost) <= 1e-12:
                best.append(tuple(sorted(combo)))
                found_at_size = True
        # A larger set can be cheaper when costs differ, so do not stop solely by size.
        if found_at_size and all(costs.get(node, 1.0) >= 0.0 for node in nodes):
            lower_bound = sum(sorted(costs.get(node, 1.0) for node in nodes)[: size + 1]) if size < len(nodes) else float("inf")
            if lower_bound > best_cost:
                break
    if not best:
        raise ValueError("independent oracle found no hitting set")
    return tuple(sorted(set(best))), best_cost


def build_truth(
    *,
    case_id: str,
    operations: tuple[MutationOperation, ...],
    dependency_paths: tuple[tuple[str, ...], ...],
    repair_costs: dict[str, float],
    unknown: bool,
    insufficient: bool,
) -> OracleTruth:
    modified = {field for operation in operations for field in operation.modified_fields}
    invalid_paths = tuple(frozenset(node for node in path if node in modified) for path in dependency_paths)
    invalid_paths = tuple(path for path in invalid_paths if path)
    if modified and not invalid_paths:
        invalid_paths = tuple(frozenset((field,)) for field in sorted(modified))
    cuts, cost = independent_optimal_cuts(invalid_paths, repair_costs)
    parents = tuple(dict.fromkeys(operation.parent for operation in operations))
    leaves = tuple(operation.leaf for operation in operations)
    sources = tuple(dict.fromkeys(node for operation in operations for node in operation.source_nodes))
    severity_order = {"subtle": 1, "moderate": 2, "severe": 3}
    severity = max((operation.severity for operation in operations), key=severity_order.get, default="none")
    return OracleTruth(
        case_id=case_id,
        route_status="insufficient_evidence" if insufficient else "invalid",
        parent_families=parents,
        leaf_types=leaves,
        source_nodes=sources,
        optimal_cuts=cuts,
        optimal_cost=cost,
        repair_sets=(sources,),
        unknown=unknown,
        insufficient_evidence=insufficient,
        severity=severity,
        composite=len(operations) > 1,
        mutation_log=operations,
    )


def to_dict(truth: OracleTruth) -> dict[str, Any]:
    return {
        "case_id": truth.case_id,
        "route_status": truth.route_status,
        "parent_families": list(truth.parent_families),
        "leaf_types": list(truth.leaf_types),
        "source_nodes": list(truth.source_nodes),
        "optimal_cuts": [list(item) for item in truth.optimal_cuts],
        "optimal_cost": truth.optimal_cost,
        "repair_sets": [list(item) for item in truth.repair_sets],
        "unknown": truth.unknown,
        "insufficient_evidence": truth.insufficient_evidence,
        "severity": truth.severity,
        "composite": truth.composite,
        "mutation_log": [
            {
                "leaf": item.leaf,
                "parent": item.parent,
                "severity": item.severity,
                "modified_fields": list(item.modified_fields),
                "source_nodes": list(item.source_nodes),
            }
            for item in truth.mutation_log
        ],
    }


def from_dict(payload: dict[str, Any]) -> OracleTruth:
    return OracleTruth(
        case_id=payload["case_id"],
        route_status=payload["route_status"],
        parent_families=tuple(payload["parent_families"]),
        leaf_types=tuple(payload["leaf_types"]),
        source_nodes=tuple(payload["source_nodes"]),
        optimal_cuts=tuple(tuple(item) for item in payload["optimal_cuts"]),
        optimal_cost=float(payload["optimal_cost"]),
        repair_sets=tuple(tuple(item) for item in payload["repair_sets"]),
        unknown=bool(payload["unknown"]),
        insufficient_evidence=bool(payload["insufficient_evidence"]),
        severity=payload["severity"],
        composite=bool(payload["composite"]),
        mutation_log=tuple(
            MutationOperation(
                leaf=item["leaf"],
                parent=item["parent"],
                severity=item["severity"],
                modified_fields=tuple(item["modified_fields"]),
                source_nodes=tuple(item["source_nodes"]),
            )
            for item in payload["mutation_log"]
        ),
    )
