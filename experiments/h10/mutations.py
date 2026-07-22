from __future__ import annotations

import random
from dataclasses import replace
from hashlib import sha256
from typing import Any

from fuzzyxai.audit_h10.models import RouteObservation

from .oracle_v19 import MutationOperation, OracleTruth, SPEC_BY_LEAF, build_truth

MutationTruth = OracleTruth


def _mutated_string(original: str, leaf: str, severity: str, field: str) -> str:
    token = sha256(f"{leaf}:{severity}:{field}:{original}".encode()).hexdigest()[:12]
    if severity == "subtle":
        # Preserve most of the lexical prefix to make distance-based diagnosis nontrivial.
        prefix = original[: max(1, int(len(original) * 0.85))]
        return f"{prefix}:{token[:4]}"
    if severity == "moderate":
        return f"alt:{token}:{original[-8:]}"
    return f"foreign:{leaf}:{token}"


def _value(field: str, original: Any, leaf: str, severity: str) -> Any:
    if field == "calibration_age_days":
        return {"subtle": 31.0, "moderate": 120.0, "severe": 720.0}[severity]
    if field == "reduction_loss":
        return {"subtle": 0.101, "moderate": 0.40, "severe": 0.99}[severity]
    if field == "artifact_sha256":
        text = str(original)
        keep = {"subtle": 60, "moderate": 32, "severe": 0}[severity]
        return text[:keep] + sha256(f"{leaf}:{severity}:{text}".encode()).hexdigest()[keep:]
    if field == "artifact_uri" and leaf == "missing_artifact" and severity == "severe":
        return None
    if field == "source_uri" and leaf == "missing_source" and severity == "severe":
        return None
    if field == "canonical_source_id" and severity == "severe" and leaf in {"lost_canonical_link", "missing_source"}:
        return None
    if isinstance(original, str):
        return _mutated_string(original, leaf, severity, field)
    return None if severity == "severe" else f"mutated:{leaf}:{severity}"


def mutate_route(
    route: RouteObservation,
    leaves: tuple[str, ...],
    severity: str,
    *,
    unknown: bool = False,
    insufficient: bool = False,
) -> tuple[RouteObservation, MutationTruth]:
    observed = dict(route.expected)
    operations: list[MutationOperation] = []
    for leaf in leaves:
        spec = SPEC_BY_LEAF[leaf]
        fields = spec.fields_by_severity[severity]
        for field in fields:
            observed[field] = _value(field, route.expected.get(field), leaf, severity)
        operations.append(MutationOperation(leaf, spec.parent, severity, fields, spec.source_nodes))
    if insufficient:
        candidate_fields = [field for operation in operations for field in operation.modified_fields if field in route.mandatory_fields]
        field = candidate_fields[0] if candidate_fields else "source_uri"
        observed[field] = None
    mutated = replace(route, observed=observed)
    case_id = f"{route.route_id}:{'+'.join(leaves)}:{severity}:{int(unknown)}:{int(insufficient)}"
    truth = build_truth(
        case_id=case_id,
        operations=tuple(operations),
        dependency_paths=route.dependency_paths,
        repair_costs=route.repair_costs,
        unknown=unknown,
        insufficient=insufficient,
    )
    return mutated, truth


def valid_truth(route: RouteObservation) -> MutationTruth:
    return MutationTruth(
        case_id=f"{route.route_id}:valid",
        route_status="valid",
        parent_families=(),
        leaf_types=(),
        source_nodes=(),
        optimal_cuts=((),),
        optimal_cost=0.0,
        repair_sets=((),),
        unknown=False,
        insufficient_evidence=False,
        severity="none",
        composite=False,
        mutation_log=(),
    )


def make_cases(
    routes: list[RouteObservation],
    *,
    seed: int,
    known_leaves: tuple[str, ...],
    held_out_leaves: tuple[str, ...],
    include_valid: bool = True,
) -> list[tuple[RouteObservation, MutationTruth]]:
    rng = random.Random(seed)
    cases: list[tuple[RouteObservation, MutationTruth]] = []
    severities = ("subtle", "moderate", "severe")
    for index, route in enumerate(routes):
        selector = int(sha256(f"{seed}:{route.object_id}".encode()).hexdigest()[:8], 16) % 12
        if include_valid and selector in {0, 1}:
            cases.append((replace(route, route_id=f"{route.route_id}:valid"), valid_truth(route)))
            continue
        unknown = selector in {8, 9} and bool(held_out_leaves)
        pool = held_out_leaves if unknown else known_leaves
        leaf = pool[rng.randrange(len(pool))]
        leaves = (leaf,)
        if selector in {5, 9, 10}:
            second_pool = known_leaves if unknown else pool
            second = second_pool[rng.randrange(len(second_pool))]
            if second != leaf:
                leaves += (second,)
        insufficient = selector == 11
        severity = severities[(index + selector) % len(severities)]
        mutated, truth = mutate_route(route, leaves, severity, unknown=unknown, insufficient=insufficient)
        cases.append((replace(mutated, route_id=truth.case_id), truth))
    return cases
