"""Evidence-first dissertation conformance gate for FuzzyXAI chapters 1-4.

This module CERTIFIES THE COMPUTATIONAL EXPLANATION ROUTE, not the correctness
of its prediction, the causal truth of posthoc attributions, or medical safety.
It consumes an existing FuzzyXAI ExplanationViewModel, without altering model
inference, explanation, diagnostics or the established risk observer.

All checks are deterministic, fail closed and preserve missing/not-applicable
semantics. No missing measurement is replaced by a fabricated zero.
"""
from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .doctoral_catalog import catalog_lookup

SCHEMA = "FuzzyXAI-DoctoralCertification/1.0"
RISK_COMPONENTS = ("rho_p", "u_M", "one_minus_I_pre", "Delta", "chi_R")
ACTIONS = ("accept", "lower_confidence", "request_more_data", "defer_to_human", "block")
SOURCE_TYPES = frozenset({"intrinsic", "builtin", "posthoc"})
OBJECT_TYPES = frozenset({"causal", "data", "representation", "logic", "features"})
AUDIENCES = frozenset({"end_user", "regulator", "researcher", "developer", "auditor"})
# u_fou is uncertainty in a *type-2 membership grade*. It is NOT u_int,
# which denotes an interval-valued type-1 representation.
_TYPE1_PROFILE_FLAGS = frozenset({"u_num", "u_ling", "u_int", "u_exp", "u_conf", "u_trace", "u_shift", "u_cf", "u_multi", "u_time", "u_user", "u_src"})
PROFILE_FLAGS = _TYPE1_PROFILE_FLAGS | {"u_fou"}
CLASS_COVERAGE: dict[str, frozenset[str]] = {
    "F0": frozenset({"u_num", "u_ling"}),
    "F_int": frozenset({"u_num", "u_ling", "u_int"}),
    "F_IT2": frozenset({"u_num", "u_ling", "u_int", "u_fou"}),
    "F_H": frozenset({"u_num", "u_ling", "u_exp"}),
    "F_N_src": frozenset({"u_num", "u_ling", "u_conf", "u_src"}),
    "FML_user": frozenset({"u_num", "u_ling", "u_int", "u_conf", "u_src", "u_user"}),
    # No type-1 class can certify a type-2 footprint merely by renaming it.
    "FML_audit": _TYPE1_PROFILE_FLAGS,
}
# Complexity is a declared reference ordering, NOT a measured runtime claim.
CLASS_COSTS = {"F0": (1, 1), "F_int": (2, 2), "F_IT2": (4, 4), "F_H": (3, 3), "F_N_src": (3, 3), "FML_user": (5, 5), "FML_audit": (8, 8)}
_ACTION_RANK = {"accept": 0, "lower_confidence": 1, "request_more_data": 2,
                "defer_to_human": 3, "review": 3, "audit": 3, "audit_report": 3, "block": 4,
                "insufficient_evidence": 3}
_SHA256 = __import__("re").compile(r"^[0-9a-fA-F]{64}$")


def canonical_json(value: object) -> bytes:
    """Deterministic, strict JSON; reject NaN/Inf and ambiguous other objects."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def certifier_code_sha256() -> str:
    """Hash exact installed certifier bytes, not source-model correctness."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def finite_number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric, not {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def unit(value: object, context: str) -> float:
    result = finite_number(value, context)
    if not 0 <= result <= 1:
        raise ValueError(f"{context} must be in [0,1]")
    return result


def positive_cost(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a numeric cost")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{context} must be finite and positive")
    return result


def fuzzy_and(values: Sequence[float], operator: str = "minimum") -> float:
    """Three declared T-norms; absence of conditions cannot imply truth."""
    if not values:
        raise ValueError("at least one measured antecedent degree is required")
    parsed = [unit(v, "antecedent degree") for v in values]
    if operator == "minimum":
        return min(parsed)
    if operator == "product":
        return math.prod(parsed)
    if operator == "lukasiewicz":
        return max(0.0, sum(parsed) - (len(parsed) - 1))
    raise ValueError("unsupported T-norm")


@dataclass(frozen=True)
class Violation:
    code: str
    path: str
    detail: str
    kind: str  # incomplete | invalid
    causes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Taxonomy:
    method_id: str
    source: str
    object: str
    audience: str
    canonical_configuration: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Taxonomy:
        result = cls(method_id=str(value.get("method_id", "")).strip(),
                     source=str(value.get("source", "")),
                     object=str(value.get("object", "")),
                     audience=str(value.get("audience", "")),
                     canonical_configuration=str(value.get("canonical_configuration", "")).strip())
        if not result.method_id or not result.canonical_configuration:
            raise ValueError("method_id and canonical_configuration must be registered")
        if result.source not in SOURCE_TYPES or result.object not in OBJECT_TYPES or result.audience not in AUDIENCES:
            raise ValueError("invalid taxonomy axis: source/object/audience")
        return result


@dataclass(frozen=True)
class GatePolicy:
    require_dataset_version: bool = True
    require_membership_policy: bool = True
    require_graph: bool = True
    require_claim_refs: bool = True
    require_rule_interface: bool = True
    require_tnorm_evidence: bool = True
    require_membership_replay: bool = True
    require_representation: bool = True
    require_representation_execution: bool = True
    require_risk: bool = True
    require_model_fingerprint: bool = True
    require_source_hashes: bool = True
    allow_controlled_fixture: bool = False
    alignment_tolerance: float = 1e-6
    risk_tolerance: float = 1e-6
    reduction_epsilon: float = 0.15
    t_norm: str = "minimum"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> GatePolicy:
        opts = dict(value or {})
        unknown = set(opts).difference(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown policy keys: {sorted(unknown)}")
        result = cls(**opts)
        for key in ("alignment_tolerance", "risk_tolerance", "reduction_epsilon"):
            unit(getattr(result, key), key)
        fuzzy_and([0.5, 0.6], result.t_norm)
        for key in ("require_dataset_version", "require_membership_policy", "require_graph", "require_claim_refs",
                    "require_rule_interface", "require_tnorm_evidence", "require_membership_replay", "require_representation", "require_representation_execution", "require_risk", "require_model_fingerprint", "require_source_hashes",
                    "allow_controlled_fixture"):
            if type(getattr(result, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _try_unit(value: Any, path: str, defects: list[Violation], *, cause: str | None = None) -> float | None:
    if value is None:
        defects.append(Violation("D_missing", path, "required measurement was not provided", "incomplete", (cause or path,)))
        return None
    try:
        return unit(value, path)
    except ValueError as exc:
        defects.append(Violation("D_invalid_numeric", path, str(exc), "invalid", (cause or path,)))
        return None


def _add(defects: list[Violation], code: str, path: str, detail: str, *, incomplete: bool = False, causes: tuple[str, ...] | None = None) -> None:
    defects.append(Violation(code, path, detail, "incomplete" if incomplete else "invalid", causes or (path,)))


def _validate_taxonomy(taxonomy: Mapping[str, Any], defects: list[Violation]) -> dict[str, str] | None:
    try:
        coord = Taxonomy.from_mapping(taxonomy)
        record = catalog_lookup(coord.method_id)
        if record is not None and any(getattr(coord, key) != record[key] for key in ("source", "object", "audience")):
            raise ValueError(f"method {coord.method_id} disagrees with the canonical configuration in Appendix A.5")
        if record is not None and coord.canonical_configuration != record["label"]:
            raise ValueError(f"method {coord.method_id} has a noncanonical title/configuration")
        return {**asdict(coord), "catalog_status": "registered" if record is not None else "external_configuration_self_declared"}
    except (TypeError, ValueError) as exc:
        _add(defects, "D_taxonomy", "taxonomy", str(exc), incomplete=True)
        return None


def _validate_trace(trace: Mapping[str, Any], policy: GatePolicy, defects: list[Violation]) -> dict[str, Any]:
    required = ("input_sha256", "explain_plan_sha256")
    if policy.require_model_fingerprint:
        required += ("model_fingerprint",)
    for key in required if policy.require_source_hashes else ():
        value = trace.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            _add(defects, "D_trace_hash", f"trace.{key}", "missing or malformed sha256", incomplete=True)
    if policy.require_dataset_version and not _clean(trace.get("dataset_version")):
        _add(defects, "D_dataset_version", "trace.dataset_version", "dataset version must be recorded", incomplete=True)
    if not _clean(trace.get("adapter_id")):
        _add(defects, "D_adapter", "trace.adapter_id", "model adapter identity is missing", incomplete=True)
    object_ids = _sequence(trace.get("object_ids"))
    if not object_ids or any(not _clean(v) for v in object_ids) or len(set(str(v) for v in object_ids)) != len(object_ids):
        _add(defects, "D_object_id", "trace.object_ids", "unique non-empty object ids required", incomplete=True)
    stamp = _clean(trace.get("generated_at"))
    if not stamp:
        _add(defects, "D_time", "trace.generated_at", "generation timestamp is missing", incomplete=True)
    else:
        try:
            parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("timezone required")
        except ValueError:
            _add(defects, "D_time", "trace.generated_at", "timestamp must be an ISO8601 value with timezone")
    if trace.get("result_origin") == "controlled_fixture" and not policy.allow_controlled_fixture:
        _add(defects, "D_fixture_scope", "trace.result_origin", "fixture cannot satisfy measured-evidence certification", incomplete=True)
    return {key: trace.get(key) for key in (*required, "dataset_version", "adapter_id", "object_ids", "generated_at", "result_origin")}


def _validate_graph_and_claims(payload: Mapping[str, Any], policy: GatePolicy, defects: list[Violation]) -> dict[str, Any]:
    graph = _mapping(payload.get("explanation_graph"))
    nodes = _sequence(graph.get("nodes"))
    edges = _sequence(graph.get("edges"))
    claims = _sequence(payload.get("claims"))
    if policy.require_graph and not nodes:
        _add(defects, "D_graph_missing", "explanation_graph.nodes", "graph evidence missing", incomplete=True)
    node_map: dict[str, Mapping[str, Any]] = {}
    for i, node in enumerate(nodes):
        node = _mapping(node)
        node_id = _clean(node.get("node_id"))
        if not node_id or node_id in node_map:
            _add(defects, "D_node_identity", f"explanation_graph.nodes[{i}]", "duplicate or empty node identifier")
        else:
            node_map[node_id] = node
    adjacency: dict[str, set[str]] = {key: set() for key in node_map}
    for i, edge in enumerate(edges):
        edge = _mapping(edge)
        start, end = _clean(edge.get("source")), _clean(edge.get("target"))
        if start not in node_map or end not in node_map:
            _add(defects, "D_dangling_edge", f"explanation_graph.edges[{i}]", "edge endpoint missing", causes=(start or end or "graph",))
        else:
            adjacency[start].add(end)
    tr = _mapping(payload.get("trace"))
    model_node = _mapping(node_map.get("model_artifact:root"))
    model_digest = _mapping(model_node.get("payload")).get("model_fingerprint")
    if model_digest and tr.get("model_fingerprint") and model_digest != tr["model_fingerprint"]:
        _add(defects, "D_model_lineage", "explanation_graph.model_artifact:root", "model fingerprint differs between trace and graph")
    dataset_node = _mapping(node_map.get("dataset:root"))
    dataset_version = _mapping(dataset_node.get("payload")).get("dataset_version")
    if dataset_version and tr.get("dataset_version") and dataset_version != tr["dataset_version"]:
        _add(defects, "D_dataset_lineage", "explanation_graph.dataset:root", "dataset version differs between trace and graph")
    # Iterative DAG check: deep evidence graphs must not hit recursion limits.
    indegree: dict[str, int] = {key: 0 for key in adjacency}
    for successors in adjacency.values():
        for target in successors:
            indegree[target] += 1
    frontier = [key for key, count in indegree.items() if count == 0]
    visited_count = 0
    while frontier:
        node = frontier.pop()
        visited_count += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                frontier.append(target)
    if visited_count != len(adjacency):
        _add(defects, "D_graph_cycle", "explanation_graph.edges", "directed provenance cycle detected")
    # External evidence identifiers are not themselves proof nodes.  They
    # must be witnessed by a materialized graph node linked to the claim.
    ref_witnesses: dict[str, set[str]] = {name: {name} for name in node_map}
    for node_id, node in node_map.items():
        for external_ref in _sequence(node.get("evidence_refs")):
            ref_witnesses.setdefault(str(external_ref), set()).add(node_id)
    status_counts: dict[str, int] = {}
    claim_ids: set[str] = set()
    for i, claim in enumerate(claims):
        claim = _mapping(claim)
        claim_id = _clean(claim.get("claim_id"))
        if not claim_id or claim_id in claim_ids:
            _add(defects, "D_claim_identity", f"claims[{i}].claim_id", "missing or duplicate claim identifier")
        claim_ids.add(claim_id)
        if policy.require_graph and f"claim:{claim_id}" not in node_map:
            _add(defects, "D_claim_node_missing", f"claims[{i}]", "materialized claim node absent from provenance graph", incomplete=True)
        status = _clean(claim.get("evidence_status", claim.get("status")))
        if status not in {"supported", "not_applicable", "contested", "insufficient_evidence"}:
            _add(defects, "D_claim_status", f"claims[{i}].evidence_status", "unsupported claim evidence status")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "contested":
            _add(defects, "D_contested_claim", f"claims[{i}]", "contested claim needs expert/audit resolution", incomplete=True)
        elif status == "insufficient_evidence":
            _add(defects, "D_insufficient_claim", f"claims[{i}]", "an explicitly ungrounded claim cannot be certified", incomplete=True)
        identifiers = _sequence(claim.get("evidence_refs"))
        if status == "supported" and not identifiers:
            _add(defects, "D_claim_ungrounded", f"claims[{i}]", "supported claim lacks evidence", causes=(f"claim:{i}",))
        if status == "contested" and not (identifiers or _sequence(claim.get("counter_evidence_refs"))):
            _add(defects, "D_counter_evidence", f"claims[{i}]", "contested claim lacks supporting/counter evidence")
        if policy.require_claim_refs:
            for ref in identifiers:
                if _clean(ref) not in ref_witnesses:
                    _add(defects, "D_unknown_evidence", f"claims[{i}].evidence_refs", "reference not contained in graph: " + str(ref), incomplete=True, causes=(str(ref),))
    def reachable(start: str, target: str) -> bool:
        if start == target:
            return True
        frontier = [start]
        visited: set[str] = set()
        while frontier:
            node = frontier.pop()
            if node == target:
                return True
            if node not in visited:
                visited.add(node)
                frontier.extend(adjacency.get(node, ()))
        return False
    if policy.require_claim_refs:
        for i, claim in enumerate(claims):
            claim = _mapping(claim)
            claim_id = _clean(claim.get("claim_id"))
            claim_node = f"claim:{claim_id}"
            if claim_node in node_map:
                for ref in _sequence(claim.get("evidence_refs")):
                    witnesses = ref_witnesses.get(_clean(ref), set())
                    if witnesses and not any(reachable(source, claim_node) for source in witnesses):
                        _add(defects, "D_unlinked_claim", f"claims[{i}]", "referenced evidence cannot reach claim node", causes=(str(ref),))
    if policy.require_graph:
        if "action" not in node_map or not reachable("prediction", "action"):
            _add(defects, "D_action_lineage", "explanation_graph", "no directed prediction-to-action path", incomplete=True)
        if "prediction" not in node_map or not any(
            name.startswith("data:") and reachable(name, "prediction") for name in node_map
        ):
            _add(defects, "D_input_prediction_lineage", "explanation_graph", "no data-to-prediction evidence path", incomplete=True)
    if not claims:
        _add(defects, "D_claims_missing", "claims", "no declarative claims have been materialized", incomplete=True)
    return {"n_nodes": len(nodes), "n_edges": len(edges), "n_claims": len(claims), "claim_status_counts": status_counts,
            "missing_evidence": list(_sequence(graph.get("missing_evidence")))}


def _validate_fuzzy(fuzzy: Mapping[str, Any], policy: GatePolicy, defects: list[Violation]) -> dict[str, Any]:
    memberships = _mapping(fuzzy.get("memberships"))
    policies = _mapping(fuzzy.get("membership_policies"))
    def visit(node: Any, prefix: str) -> None:
        if isinstance(node, Mapping):
            for key, v in node.items():
                visit(v, f"{prefix}.{key}")
        else:
            _try_unit(node, prefix, defects)
    visit(memberships, "fuzzy.memberships")
    if policy.require_membership_policy and memberships and not policies:
        _add(defects, "D_membership_origin", "fuzzy.membership_policies", "membership functions were not disclosed", incomplete=True)
    for name, item in policies.items():
        spec = _mapping(item)
        if not _clean(spec.get("origin")) or not _clean(spec.get("version")):
            _add(defects, "D_membership_origin", f"fuzzy.membership_policies.{name}", "origin and version required", incomplete=True)
        if spec.get("origin") not in {"expert", "calibrated", "preset", "imported"}:
            _add(defects, "D_membership_origin", f"fuzzy.membership_policies.{name}.origin", "unknown membership origin")
        for i, term in enumerate(_sequence(spec.get("terms"))):
            t = _mapping(term)
            params = _sequence(t.get("parameters"))
            if t.get("function") != "triangular" or len(params) != 3:
                _add(defects, "D_membership_shape", f"fuzzy.membership_policies.{name}.terms[{i}]", "only declared triangular terms checked")
                continue
            try:
                a, b, c = (finite_number(v, "membership parameter") for v in params)
                if a > b or b > c:
                    raise ValueError("parameters not ordered")
            except ValueError as exc:
                _add(defects, "D_membership_shape", f"fuzzy.membership_policies.{name}.terms[{i}]", str(exc))
    return {"membership_groups": len(memberships), "registered_policies": len(policies)}


def triangular_membership(x: float, params: Sequence[float]) -> float:
    """Exact triangular (including both shoulder) membership contract."""
    if len(params) != 3:
        raise ValueError("triangular membership requires three ordered parameters")
    a, b, c = [finite_number(v, "membership parameter") for v in params]
    if not a <= b <= c:
        raise ValueError("triangular membership parameters are not ordered")
    x = finite_number(x, "membership feature value")
    if x < a or x > c:
        return 0.0
    if x == b or (a == b and x == a) or (b == c and x == c):
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


def _validate_rule_interface(payload: Mapping[str, Any], policy: GatePolicy,
                             defects: list[Violation]) -> dict[str, int]:
    """Check the chapter-2 rule/projection contract without fabricating rules."""
    layers = _mapping(payload.get("layers"))
    fuzzy = _mapping(payload.get("fuzzy"))
    memberships = _mapping(fuzzy.get("memberships"))
    membership_policies = _mapping(fuzzy.get("membership_policies"))
    rules = _sequence(layers.get("rules"))
    activations = _sequence(layers.get("fuzzy_rule_activations"))
    rule_ids = [_clean(_mapping(item).get("rule_id")) for item in rules]
    if not all(rule_ids) or len(set(rule_ids)) != len(rule_ids):
        _add(defects, "D_rule_identity", "layers.rules", "rule identifiers must be non-empty and unique")
    graph_nodes = {_clean(_mapping(item).get("node_id")) for item in
                   _sequence(_mapping(payload.get("explanation_graph")).get("nodes"))}
    known_objects = {str(item) for item in _sequence(_mapping(payload.get("trace")).get("object_ids"))}
    for i, rule_id in enumerate(rule_ids):
        if policy.require_graph and rule_id and f"rule:{rule_id}" not in graph_nodes:
            _add(defects, "D_rule_node_missing", f"layers.rules[{i}]", "rule has no provenance graph node", incomplete=True)
    if policy.require_rule_interface:
        if not memberships:
            _add(defects, "D_term_projection", "fuzzy.memberships", "linguistic degree projection absent", incomplete=True)
        if not rules:
            _add(defects, "D_rule_projection", "layers.rules", "no native or certified surrogate rule evidence", incomplete=True)
        if not activations:
            _add(defects, "D_rule_activation", "layers.fuzzy_rule_activations", "rule activation measurement absent", incomplete=True)
    for i, item in enumerate(activations):
        row = _mapping(item)
        rid = _clean(row.get("rule_id"))
        oid = _clean(row.get("object_id"))
        if not rid or rid not in rule_ids:
            _add(defects, "D_activation_unregistered_rule", f"layers.fuzzy_rule_activations[{i}].rule_id", "activation does not refer to a measured rule")
        if not oid or oid not in known_objects:
            _add(defects, "D_activation_unknown_object", f"layers.fuzzy_rule_activations[{i}].object_id", "activation object not in the same explanation trace")
        if policy.require_graph and rid and oid and f"fuzzy_rule:{oid}:{rid}" not in graph_nodes and f"rule:{rid}" not in graph_nodes:
            _add(defects, "D_activation_node_missing", f"layers.fuzzy_rule_activations[{i}]", "activation has no materialized graph witness", incomplete=True)
        activation = _try_unit(row.get("activation_strength"), f"layers.fuzzy_rule_activations[{i}].activation_strength", defects)
        terms = _sequence(row.get("terms"))
        if policy.require_tnorm_evidence and not terms:
            _add(defects, "D_activation_terms_missing", f"layers.fuzzy_rule_activations[{i}].terms",
                 "per-term measured membership degrees needed for T-norm verification", incomplete=True)
        degree_values: list[float] = []
        for j, term in enumerate(terms):
            term = _mapping(term)
            val = _try_unit(term.get("membership_degree"),
                            f"layers.fuzzy_rule_activations[{i}].terms[{j}].membership_degree", defects)
            if val is not None:
                degree_values.append(val)
            if policy.require_membership_replay:
                feature, term_label = _clean(term.get("feature")), _clean(term.get("term"))
                spec = _mapping(membership_policies.get(feature))
                declared = next((_mapping(x) for x in _sequence(spec.get("terms")) if _mapping(x).get("label") == term_label), {})
                position = f"layers.fuzzy_rule_activations[{i}].terms[{j}]"
                if not declared:
                    _add(defects, "D_term_policy_missing", position, "membership function for this antecedent was not registered", incomplete=True)
                elif term.get("feature_value") is None:
                    _add(defects, "D_term_input_missing", position, "feature value needed to replay the declared membership function", incomplete=True)
                elif declared.get("function") == "triangular":
                    try:
                        raw = finite_number(term["feature_value"], "feature value")
                        observed = triangular_membership(raw, _sequence(declared.get("parameters")))
                        if val is not None and abs(observed - val) > policy.alignment_tolerance:
                            _add(defects, "D_membership_replay", position, "measured degree differs from declared membership function")
                    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
                        _add(defects, "D_membership_input", position, str(exc))
                else:
                    _add(defects, "D_membership_unsupported", position, "unsupported function for replay", incomplete=True)
        # Cross-check a feature's observed membership against the projection
        # only when the linguistic label identifies ONE antecedent in this
        # record (otherwise the unqualified label is ambiguous).
        projected = _mapping(memberships.get(oid))
        term_labels = [_clean(_mapping(t).get("term")) for t in terms]
        for t in terms:
            one = _mapping(t)
            label = _clean(one.get("term"))
            named_key = label if label in projected else f"{_clean(one.get('feature'))}:{label}"
            if label and term_labels.count(label) == 1 and named_key in projected:
                try:
                    lhs = unit(projected[named_key], f"fuzzy.memberships.{oid}.{named_key}")
                    rhs = unit(one.get("membership_degree"), "rule antecedent membership")
                    if abs(lhs - rhs) > policy.alignment_tolerance:
                        _add(defects, "D_membership_projection_mismatch", f"fuzzy.memberships.{oid}.{named_key}",
                             "linguistic projection disagrees with this object's measured rule antecedent")
                except ValueError as exc:
                    _add(defects, "D_membership_projection_invalid", f"fuzzy.memberships.{oid}.{named_key}", str(exc))
        if activation is not None and terms and len(degree_values) == len(terms):
            t_value = fuzzy_and(degree_values, policy.t_norm)
            if abs(activation - t_value) > policy.alignment_tolerance:
                _add(defects, "D_tnorm_inconsistent", f"layers.fuzzy_rule_activations[{i}].activation_strength",
                     "observed activation disagrees with preregistered T-norm")
    return {"rules": len(rules), "activations": len(activations)}


def select_representation(profile: Sequence[str], *, audience: str, measured_delta: float | None = None,
                          reduction_epsilon: float = 0.15) -> dict[str, Any]:
    """Minimal sufficient class for the declared, FINITE chapter-3 profile.

    Complexity costs are explicit policy ordering; epsilon checks measured
    reduction loss when a reduction is declared, never a guessed delta.
    """
    flags = set(str(flag) for flag in profile)
    if not flags or flags - PROFILE_FLAGS:
        raise ValueError(f"unknown or empty uncertainty profile: {sorted(flags - PROFILE_FLAGS)}")
    if audience not in AUDIENCES:
        raise ValueError("unknown audience")
    eps = unit(reduction_epsilon, "reduction_epsilon")
    delta = None if measured_delta is None else unit(measured_delta, "measured_delta")
    if delta is not None and delta > eps:
        return {"selected_class": None, "status": "D_reduction", "profile": sorted(flags),
                "delta": delta, "epsilon": eps, "alternatives": []}
    options = [name for name, coverage in CLASS_COVERAGE.items() if flags <= coverage]
    if audience == "end_user":
        options.sort(key=lambda item: (CLASS_COSTS[item][0], CLASS_COSTS[item][1], item))
    else:
        options.sort(key=lambda item: (CLASS_COSTS[item][1], CLASS_COSTS[item][0], item))
    return {"selected_class": options[0] if options else None,
            "status": "selected" if options else "D_choice", "profile": sorted(flags),
            "delta": delta, "epsilon": eps, "alternatives": options,
            "cost_basis": "declared_complexity_order_not_measured_runtime"}



def _validate_interval_type2_representation(
    representation: Mapping[str, Any],
    profile: Sequence[str],
    payload: Mapping[str, Any],
    policy: GatePolicy,
    defects: list[Violation],
) -> dict[str, Any]:
    """Require independently replayed type-2 FOU evidence for the F_IT2 class.

    A claimed label, already computed action, or in-process self-SHA is NOT a
    source of Type-2 certification. An auditor explicitly supplies per-object
    numeric receipts and a separately registered model SHA256; the receipts
    are retained in the audit certificate for deterministic verification.
    This is computational replay, not cryptographic author authentication.
    """
    guard = _mapping(_mapping(payload.get("risk")).get("it2_guard"))
    if "u_fou" not in profile:
        if guard:
            _add(defects, "D_it2_profile_omits_fou", "representation.profile",
                 "the native IT2 FOU was measured but excluded from the declared uncertainty profile")
        return {"status": "not_applicable"}
    trace = _mapping(payload.get("trace"))
    if trace.get("adapter_id") != "interval_type2_v2":
        _add(defects, "D_it2_wrong_adapter", "trace.adapter_id",
             "F_IT2 requires real native interval-type-2 inference")
    graph = _mapping(payload.get("explanation_graph"))
    nodes = {_clean(_mapping(node).get("node_id")) for node in _sequence(graph.get("nodes"))}
    edges = [(str(_mapping(e).get("source", "")), str(_mapping(e).get("target", "")))
             for e in _sequence(graph.get("edges"))]
    if "it2:guard" not in nodes or ("it2:guard", "action") not in edges:
        _add(defects, "D_it2_graph_witness", "explanation_graph",
             "the actual type-2 guard must be materialized and connected to the final action", incomplete=True)
    if guard.get("status") != "verified":
        _add(defects, "D_it2_not_verified", "risk.it2_guard",
             "IT2 guard must explicitly verify native numeric inference and source identity", incomplete=True)
    cases = _sequence(guard.get("cases"))
    if not cases:
        _add(defects, "D_it2_cases_missing", "risk.it2_guard.cases",
             "no per-object type-2 inference was observed", incomplete=True)
    anchor = representation.get("it2_trusted_model_sha256")
    if not isinstance(anchor, str) or not _SHA256.fullmatch(anchor):
        _add(defects, "D_it2_anchor_missing", "representation.it2_trusted_model_sha256",
             "separately registered SHA256 model fingerprint required for independent replay",
             incomplete=True)
        return {"status": "insufficient_evidence", "replayed_cases": 0}
    declared_fingerprint = _clean(trace.get("model_fingerprint"))
    if anchor.lower() != declared_fingerprint.lower():
        _add(defects, "D_it2_anchor_mismatch", "trace.model_fingerprint",
             "native runtime model fingerprint differs from the independently declared anchor")
    receipts = _sequence(representation.get("it2_receipts"))
    if not receipts:
        _add(defects, "D_it2_receipts_missing", "representation.it2_receipts",
             "raw numerical IT2 receipts needed for independent FOU recomputation", incomplete=True)
        return {"status": "insufficient_evidence", "replayed_cases": 0,
                "anchor_status": "declared_but_not_replayed"}
    if len(receipts) != len(cases):
        _add(defects, "D_it2_case_count", "representation.it2_receipts",
             "number of receipts differs from the actual runtime cases")
    from .interval_type2 import verify_it2_receipt
    model = _mapping(payload.get("model"))
    metadata = _mapping(model.get("metadata"))
    published_inferences = _sequence(metadata.get("it2_inference"))
    trace_ids = _sequence(trace.get("object_ids"))
    if len(cases) != len(trace_ids):
        _add(defects, "D_it2_trace_count", "trace.object_ids",
             "runtime Type-2 cases are not one-to-one with original trace identities")
    if len(published_inferences) != len(cases):
        _add(defects, "D_it2_prediction_count", "model.metadata.it2_inference",
             "number of native model outputs differs from the verified interval cases")
    seen_ids: set[str] = set()
    accepted = 0
    for index, supplied in enumerate(receipts):
        item = _mapping(supplied)
        if not item:
            _add(defects, "D_it2_receipt_invalid", f"representation.it2_receipts[{index}]",
                 "receipt is not a mapping")
            continue
        oid = _clean(item.get("source_id"))
        if not oid or oid in seen_ids:
            _add(defects, "D_it2_duplicate_identity", f"representation.it2_receipts[{index}]",
                 "duplicate or absent source identifier")
            continue
        seen_ids.add(oid)
        check = verify_it2_receipt(item, trusted_model_sha256=anchor.lower())
        if not check.get("valid"):
            _add(defects, "D_it2_replay_failed", f"representation.it2_receipts[{index}]",
                 "the independent rule/FOU replay failed: " + str(check.get("errors")))
            continue
        if index >= len(cases):
            _add(defects, "D_it2_unmatched_receipt", f"representation.it2_receipts[{index}]",
                 "an extra receipt is not present in runtime evidence")
            continue
        actual = _mapping(cases[index])
        actual_inference = _mapping(published_inferences[index]) if index < len(published_inferences) else {}
        if (
            item.get("source_id") != actual.get("object_id")
            or (index < len(trace_ids) and oid != str(trace_ids[index]))
            or item.get("dataset_version") != trace.get("dataset_version")
            or item.get("dataset_version") != actual.get("dataset_version")
            or item.get("model_sha256") != actual.get("model_sha256")
            or item.get("receipt_sha256") != actual.get("receipt_sha256")
            or item.get("input_sha256") != actual.get("input_sha256")
            or not actual.get("replay_verified")
        ):
            _add(defects, "D_it2_identity_mismatch", f"representation.it2_receipts[{index}]",
                 "independently replayed receipt differs from object, model, data or provenance identity")
            continue
        evidence_inference = _mapping(item.get("inference"))
        if (
            canonical_json(evidence_inference) != canonical_json(actual_inference)
            or canonical_json(_mapping(item.get("decision"))) != canonical_json(_mapping(actual.get("decision")))
        ):
            _add(defects, "D_it2_value_mismatch", f"representation.it2_receipts[{index}]",
                 "runtime interval inference or its action differs from independently replayed receipt")
            continue
        accepted += 1
    if len(seen_ids) != len(cases):
        _add(defects, "D_it2_incomplete_batch", "representation.it2_receipts",
             "not every observed Type-2 case has a unique replayed receipt", incomplete=True)
    return {
        "status": "replayed" if accepted == len(cases) and len(receipts) == len(cases) else "incomplete",
        "replayed_cases": accepted,
        "total_cases": len(cases),
        "anchor_status": "matched_declared_model_sha256" if anchor.lower() == declared_fingerprint.lower() else "mismatch",
        "scope": "type2_parameter_FOU_replay_not_calibration_or_authentication",
    }


def _validate_representation(value: Mapping[str, Any], taxonomy: Mapping[str, str] | None,
                             policy: GatePolicy, diagnostics: Sequence[Any], defects: list[Violation],
                             payload: Mapping[str, Any]) -> dict[str, Any]:
    profile = _sequence(value.get("profile"))
    if not profile:
        if policy.require_representation:
            _add(defects, "D_profile_missing", "representation.profile", "explicit uncertainty profile missing", incomplete=True)
        return {"status": "not_provided"}
    audience = taxonomy["audience"] if taxonomy is not None else "auditor"
    measured_delta = value.get("measured_delta")
    try:
        selected = select_representation(profile, audience=audience, measured_delta=measured_delta,
                                         reduction_epsilon=policy.reduction_epsilon)
    except (ValueError, TypeError) as exc:
        _add(defects, "D_choice", "representation.profile", str(exc))
        return {"status": "invalid"}
    if selected["status"] != "selected":
        _add(defects, "D_choice", "representation", "no admissible reduction/representation", incomplete=False)
    requested = value.get("selected_class")
    if requested is not None and requested != selected["selected_class"]:
        _add(defects, "D_choice", "representation.selected_class", f"declared class {requested!r} differs from minimal admissible {selected['selected_class']!r}")
    disagreement = _mapping(payload.get("disagreement"))
    risk = _mapping(payload.get("risk"))
    structural_conflict = _mapping(risk.get("components")).get("chi_R") == 1
    explicit_conflict = any("conflict" in str(_mapping(item).get("code", _mapping(item).get("diagnostic_id", ""))).lower() for item in diagnostics)
    if (explicit_conflict or structural_conflict) and "u_conf" not in profile:
        _add(defects, "D_profile_omits_conflict", "representation.profile", "measured structural conflict was omitted from profile")
    source_conflict = any("source_conflict" in str(_mapping(item).get("code", _mapping(item).get("diagnostic_id", ""))).lower() for item in diagnostics)
    if source_conflict and "u_src" not in profile:
        _add(defects, "D_profile_omits_source", "representation.profile", "measured source conflict was omitted from profile")
    if _sequence(_mapping(payload.get("explanation_graph")).get("missing_evidence")) and "u_trace" not in profile:
        _add(defects, "D_profile_omits_trace", "representation.profile", "unresolved evidence gap omitted from profile")
    # The prescribed representation must actually appear as a measured/executed
    # route state; policy selection alone does not prove it was instantiated.
    if policy.require_representation_execution and selected.get("selected_class") is not None:
        fuzzy = _mapping(payload.get("fuzzy"))
        emitted = _clean(fuzzy.get("selected_representation"))
        witnessed: list[str] = []
        for state in _sequence(payload.get("route")):
            state = _mapping(state)
            output = _mapping(state.get("output_values"))
            if "selected_class" not in output:
                continue
            provenance = _clean(state.get("value_source"))
            if provenance in {"computed", "measured"}:
                witnessed.append(_clean(output["selected_class"]))
            else:
                _add(defects, "D_representation_untrusted_source", "route",
                     "route class was not produced by a computed/measured operator", incomplete=True)
        if not emitted or not witnessed:
            _add(defects, "D_representation_not_executed", "fuzzy.selected_representation",
                 "class must be present in both the executed route and its explanation projection", incomplete=True)
        elif emitted != selected["selected_class"] or any(item != emitted for item in witnessed):
            _add(defects, "D_representation_mismatch", "fuzzy.selected_representation",
                 f"executed class/projection disagrees with the selected {selected['selected_class']!r}")
    selected["it2_verification"] = _validate_interval_type2_representation(
        value, profile, payload, policy, defects
    )
    return selected


def _validate_alignment(disagreement: Mapping[str, Any], policy: GatePolicy, defects: list[Violation]) -> dict[str, Any]:
    ap = _mapping(disagreement.get("alignment_policy"))
    rp = _mapping(disagreement.get("reduction_policy"))
    gamma_status = disagreement.get("alignment_status")
    reduction_status = disagreement.get("reduction_status")
    output: dict[str, Any] = {"alignment_status": gamma_status, "reduction_status": reduction_status}
    if ap.get("applicable") is True:
        gamma = _try_unit(disagreement.get("gamma"), "disagreement.gamma", defects)
        gamma_max = _try_unit(disagreement.get("gamma_max"), "disagreement.gamma_max", defects)
        if gamma_status != "measured":
            _add(defects, "D_alignment_not_measured", "disagreement.alignment_status", "alignment declared applicable but not measured", incomplete=True)
        if gamma is not None and gamma_max is not None and gamma > gamma_max + policy.alignment_tolerance:
            _add(defects, "D_alignment", "disagreement.gamma", "semantic divergence exceeds declared bound")
        weights = _mapping(disagreement.get("beta"))
        parts = _mapping(disagreement.get("components"))
        if parts and weights and set(parts) <= set(weights):
            try:
                denom = sum(unit(weights[key], "beta weight") for key in parts)
                if denom <= 0:
                    raise ValueError("no active positive alignment weights")
                recon = sum(unit(parts[key], "alignment component") * unit(weights[key], "beta weight") for key in parts) / denom
                if gamma is not None and abs(round(recon, 6) - gamma) > policy.alignment_tolerance:
                    _add(defects, "D_alignment_reconstruction", "disagreement.gamma", "stored gamma inconsistent with its own components")
            except ValueError as exc:
                _add(defects, "D_alignment_components", "disagreement.components", str(exc))
        output.update({"gamma": gamma, "gamma_max": gamma_max})
    elif ap.get("applicable") is False:
        if gamma_status == "measured":
            # Can have extra evidence; never call it an error when not required.
            output["additional_alignment_evidence"] = True
    else:
        _add(defects, "D_alignment_contract", "disagreement.alignment_policy", "applicability not declared", incomplete=True)
    if rp.get("applicable") is True:
        delta = _try_unit(disagreement.get("delta"), "disagreement.delta", defects)
        if reduction_status != "measured":
            _add(defects, "D_reduction_not_measured", "disagreement.reduction_status", "reduction required but not measured", incomplete=True)
        if delta is not None and delta > policy.reduction_epsilon:
            _add(defects, "D_reduction", "disagreement.delta", "measured loss exceeds registered reduction epsilon")
        output["delta"] = delta
    elif rp.get("applicable") is not False:
        _add(defects, "D_reduction_contract", "disagreement.reduction_policy", "reduction applicability not declared", incomplete=True)
    return output


def _validate_risk(risk: Mapping[str, Any], policy: GatePolicy, defects: list[Violation]) -> dict[str, Any]:
    status = risk.get("status")
    action = _clean(risk.get("action"))
    output: dict[str, Any] = {"status": status, "action": action}
    if not policy.require_risk:
        return output
    if status != "complete":
        _add(defects, "D_risk_incomplete", "risk.status", "canonical risk was not computed in full", incomplete=True)
        if risk.get("rho") is not None:
            _add(defects, "D_false_complete_risk", "risk.rho", "a partial score is labeled canonical rho")
    components = _mapping(risk.get("components"))
    weights = _mapping(risk.get("risk_weights"))
    thresholds = _mapping(risk.get("risk_thresholds"))
    if set(components) != set(RISK_COMPONENTS) or set(weights) != set(RISK_COMPONENTS):
        _add(defects, "D_risk_components", "risk.components", "exactly five P19 components and weights required", incomplete=True)
        return output
    parsed: dict[str, float] = {}
    parsed_weights: dict[str, float] = {}
    for name in RISK_COMPONENTS:
        value = _try_unit(components.get(name), f"risk.components.{name}", defects)
        weight = _try_unit(weights.get(name), f"risk.risk_weights.{name}", defects)
        if value is not None:
            parsed[name] = value
        if weight is not None:
            parsed_weights[name] = weight
    if len(parsed_weights) == 5 and abs(sum(parsed_weights.values()) - 1) > 1e-9:
        _add(defects, "D_risk_weight_sum", "risk.risk_weights", "weights do not sum to one")
    critical = risk.get("chi_r_crit")
    if isinstance(critical, bool) or critical not in (0, 1, 0.0, 1.0):
        _add(defects, "D_critical_flag", "risk.chi_r_crit", "critical flag must be precisely 0 or 1")
        critical = None
    if "chi_R" in parsed and parsed["chi_R"] not in (0.0, 1.0):
        _add(defects, "D_rupture_indicator", "risk.components.chi_R", "structural rupture indicator must be 0 or 1")
    if critical == 1 and "chi_R" in parsed and parsed["chi_R"] != 1.0:
        _add(defects, "D_critical_mismatch", "risk.chi_r_crit", "critical override requires a structural rupture")
    rho = _try_unit(risk.get("rho"), "risk.rho", defects) if status == "complete" else None
    if len(parsed) == 5 and len(parsed_weights) == 5 and rho is not None:
        recomputed = sum(parsed[name] * parsed_weights[name] for name in RISK_COMPONENTS)
        output.update({"rho_reported": rho, "rho_recomputed": recomputed})
        if abs(recomputed - rho) > policy.risk_tolerance:
            _add(defects, "D_risk_arithmetic", "risk.rho", "stored rho does not match canonical weighted sum")
        try:
            t = [unit(thresholds.get(f"theta_{i}"), f"theta_{i}") for i in range(1, 5)]
            if not t[0] < t[1] < t[2] < t[3]:
                raise ValueError("thresholds must be strictly increasing")
            candidate = ACTIONS[next((i for i,v in enumerate(t) if rho < v), 4)]
            output["candidate_action"] = candidate
            if action not in _ACTION_RANK:
                _add(defects, "D_action_unknown", "risk.action", "unknown action code")
            elif critical == 1 and action != "block":
                _add(defects, "D_critical_not_blocked", "risk.action", "critical rupture overrides all thresholds")
            elif critical == 0 and _ACTION_RANK[action] < _ACTION_RANK[candidate]:
                _add(defects, "D_action_too_permissive", "risk.action", "action is more permissive than declared risk thresholds")
        except (ValueError, TypeError) as exc:
            _add(defects, "D_threshold_contract", "risk.risk_thresholds", str(exc), incomplete=True)
    if critical == 1 and action != "block":
        _add(defects, "D_critical_not_blocked", "risk.action", "critical override was bypassed")
    return output


def _validate_native_fuzzy_model(payload: Mapping[str, Any], policy: GatePolicy,
                                 defects: list[Violation]) -> dict[str, Any]:
    """Independent replay of a native fuzzy-logistic predictor from disclosed rules.

    A stored Gamma=0 is not enough; we reconstruct the actual prediction from
    fitted model coefficients, per-object antecedent memberships and intercept.
    This optional contract only applies to the explicitly versioned native
    predictor. No inferred causal or clinical validity follows from replay.
    """
    model = _mapping(payload.get("model"))
    if model.get("model_type") != "NativeFuzzyLogistic-v2":
        return {"status": "not_applicable"}
    manifest = _mapping(model.get("parameter_manifest"))
    if not manifest or manifest.get("model_type") != "NativeFuzzyLogistic-v2":
        _add(defects, "D_native_manifest", "model.parameter_manifest", "native model parameters missing", incomplete=True)
        return {"status": "missing"}
    fingerprint = _mapping(payload.get("trace")).get("model_fingerprint")
    if sha256(manifest) != fingerprint:
        _add(defects, "D_native_fingerprint", "trace.model_fingerprint", "parameter manifest hash differs from model fingerprint")
    layers = _mapping(payload.get("layers"))
    declared_rules = {_clean(_mapping(r).get("rule_id")): _mapping(r) for r in _sequence(layers.get("rules"))}
    activations = {_clean(_mapping(r).get("rule_id")): _mapping(r) for r in _sequence(layers.get("fuzzy_rule_activations"))}
    try:
        logit = finite_number(manifest.get("intercept_log_odds"), "fuzzy intercept")
        for feat in _sequence(manifest.get("rules")):
            feat = _mapping(feat)
            index = int(feat["index"])
            for term in ("low", "high"):
                rid = f"f{index:02d}:{term}"
                activation, rule = activations.get(rid), declared_rules.get(rid)
                if not activation or not rule:
                    raise ValueError(f"native rule {rid} absent from measured evidence")
                coefficient = finite_number(feat[f"{term}_consequent"], f"fitted coefficient:{rid}")
                if abs(finite_number(rule.get("consequent"), f"disclosed coefficient:{rid}") - coefficient) > policy.risk_tolerance:
                    _add(defects, "D_native_rule_parameter", f"layers.rules.{rid}", "disclosed coefficient differs from fitted model")
                if rule.get("consequent_semantics") != "log_odds":
                    _add(defects, "D_native_rule_semantics", f"layers.rules.{rid}", "coefficient presented with wrong mathematical meaning")
                activation_value = unit(activation.get("activation_strength"), f"activation:{rid}")
                logit += coefficient * activation_value
        # sigmoid with overflow protection (logits may be large).
        reconstructed = 1.0 / (1.0 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1.0 + math.exp(logit))
        reported = unit(model.get("predicted_malignancy_probability"), "model predicted probability")
        graph_nodes = _sequence(_mapping(payload.get("explanation_graph")).get("nodes"))
        graph_value = next((_mapping(_mapping(node).get("payload")).get("probability_malignant")
                            for node in graph_nodes if _mapping(node).get("node_id") == "prediction"), None)
        if abs(unit(graph_value, "graph prediction") - reported) > policy.risk_tolerance:
            _add(defects, "D_native_graph_prediction", "explanation_graph.prediction", "graph prediction differs from model output")
        risk_value = _mapping(_mapping(payload.get("risk")).get("components")).get("rho_p")
        if abs(unit(risk_value, "risk rho_p") - reported) > policy.risk_tolerance:
            _add(defects, "D_native_risk_prediction", "risk.components.rho_p", "risk uses a different model probability")
        if abs(reconstructed - reported) > policy.risk_tolerance:
            _add(defects, "D_native_rule_reconstruction", "model.predicted_malignancy_probability",
                 "native rule contributions and fitted intercept do not reconstruct actual prediction")
        gamma = _mapping(payload.get("disagreement")).get("gamma")
        if abs(unit(gamma, "alignment gamma") - abs(reconstructed - reported)) > policy.alignment_tolerance:
            _add(defects, "D_native_gamma_replay", "disagreement.gamma", "reported alignment loss differs from independent reconstruction")
        return {"status": "replayed", "reconstructed_probability": reconstructed,
                "reported_probability": reported, "probability_gap": abs(reconstructed-reported)}
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        _add(defects, "D_native_replay_incomplete", "model.parameter_manifest", str(exc), incomplete=True)
        return {"status": "incomplete"}


def _validate_cross_operator_consistency(payload: Mapping[str, Any], policy: GatePolicy,
                                          defects: list[Violation]) -> dict[str, Any]:
    """Bind chapter 2 graph and chapter 3 risk to the SAME measured route."""
    risk = _mapping(payload.get("risk"))
    components = _mapping(risk.get("components"))
    disagreement = _mapping(payload.get("disagreement"))
    output: dict[str, Any] = {}
    if not policy.require_risk:
        return output
    expected_i = disagreement.get("pre_interpretability")
    if expected_i is None:
        _add(defects, "D_i_pre_missing", "disagreement.pre_interpretability",
             "pre-risk interpretability must have explicit provenance", incomplete=True)
    else:
        model_i = _try_unit(expected_i, "disagreement.pre_interpretability", defects)
        component = _try_unit(components.get("one_minus_I_pre"), "risk.components.one_minus_I_pre", defects)
        if model_i is not None and component is not None:
            output["i_pre_reported"] = model_i
            if abs((1.0 - model_i) - component) > policy.risk_tolerance:
                _add(defects, "D_i_pre_risk_mismatch", "risk.components.one_minus_I_pre",
                     "risk interpretability term disagrees with pre-risk explanation state")
    if _mapping(disagreement.get("reduction_policy")).get("applicable") is True:
        delta = disagreement.get("delta")
        risk_delta = components.get("Delta")
        if delta is None or risk_delta is None:
            _add(defects, "D_risk_reduction_missing", "risk.components.Delta",
                 "measured reduction and risk reduction component must both exist", incomplete=True)
        else:
            left = _try_unit(delta, "disagreement.delta", defects)
            right = _try_unit(risk_delta, "risk.components.Delta", defects)
            if left is not None and right is not None and abs(left - right) > policy.risk_tolerance:
                _add(defects, "D_risk_reduction_mismatch", "risk.components.Delta",
                     "risk reduction term disagrees with executed reduction operator")
    graph_nodes = _sequence(_mapping(payload.get("explanation_graph")).get("nodes"))
    graph_action = next((_mapping(_mapping(node).get("payload")).get("action") for node in graph_nodes
                         if _mapping(node).get("node_id") == "action"), None)
    if graph_action is None:
        _add(defects, "D_graph_action_missing", "explanation_graph.action",
             "action node must preserve the actual observer action", incomplete=True)
    elif graph_action != risk.get("action"):
        _add(defects, "D_graph_action_mismatch", "explanation_graph.action",
             "action graph and risk observer disagree")
    output["graph_action"] = graph_action
    return output


def _diagnostic_cut(issues: Sequence[Violation], costs: Mapping[str, float] | None = None) -> dict[str, Any]:
    if not issues:
        return {"atoms": [], "cost": 0.0, "optimality_proven": True, "scope": "certificate_obligations"}
    rows = [frozenset(x.causes) for x in issues]
    atoms = sorted(set().union(*rows))
    pricing = {key: positive_cost((costs or {}).get(key, 1), key) for key in atoms}
    if len(atoms) <= 18:
        best: tuple[float, tuple[str, ...]] | None = None
        for n in range(1, len(atoms)+1):
            for combo in itertools.combinations(atoms, n):
                if not all(set(combo) & row for row in rows):
                    continue
                weight = sum(pricing[key] for key in combo)
                candidate = (weight, combo)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        return {"atoms": list(best[1]), "cost": best[0], "optimality_proven": True,
                "scope": "certificate_obligations", "solver": "exact_enumeration"}
    uncovered = set(range(len(rows)))
    selected: list[str] = []
    while uncovered:
        atom = max((key for key in atoms if key not in selected),
                   key=lambda key: (sum(key in rows[i] for i in uncovered) / pricing[key], -pricing[key], key))
        selected.append(atom)
        uncovered = {i for i in uncovered if atom not in rows[i]}
    return {"atoms": sorted(selected), "cost": sum(pricing[k] for k in selected),
            "optimality_proven": False, "scope": "certificate_obligations", "solver": "greedy_upper_bound"}


# These are *content* fields, unlike integrity hashes, offsets or measured
# scalar values. Redaction happens before the receipt is persisted, regardless
# of whether the caller supplied a ModelExplanationResult or a raw mapping.
_RAW_FIELDS = frozenset({
    "raw_text", "raw_excerpt", "highlighted_html", "raw_values", "raw_signal",
    "raw_objects", "raw_rows", "reference_data", "image_png_base64",
    "attribution_png_base64", "attribution_overlay_png_base64", "attribution_array",
})


def _strip_raw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): ("[raw payload omitted]" if str(key) in _RAW_FIELDS else _strip_raw(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_raw(item) for item in value]
    return value


def _extract_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "view_model") and hasattr(result.view_model, "to_dict"):
        return _strip_raw(result.view_model.to_dict(include_raw=False))
    if isinstance(result, Mapping):
        return _strip_raw(result)
    raise TypeError("expected ModelExplanationResult or serialized ExplanationViewModel mapping")


def certify_payload(payload: Mapping[str, Any], *, taxonomy: Mapping[str, Any],
                    representation: Mapping[str, Any], policy: Mapping[str, Any] | None = None,
                    repair_costs: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Issue a structural certificate. Status valid != correct prediction.

    Receipts contain only the already-redacted, client-provided view model.
    Verification is replay-based; a digest does not authenticate an author.
    """
    plan = GatePolicy.from_mapping(policy)
    sanitized = _extract_payload(payload)
    # Reject non-JSON-safe values rather than making a misleading digest.
    canonical_json(sanitized)
    issues: list[Violation] = []
    # The consumer can run diagnostic subchecks, but must never receive a
    # FULL chapter-1-to-4 certificate after disabling an essential gate.
    relaxed = sorted(name for name in GatePolicy.__dataclass_fields__
                     if name.startswith("require_") and not getattr(plan, name))
    if relaxed:
        _add(issues, "D_policy_relaxed", "policy", "full doctoral certification requires all gates: " + ", ".join(relaxed), incomplete=True)
    coord = _validate_taxonomy(taxonomy, issues)
    trace = _validate_trace(_mapping(sanitized.get("trace")), plan, issues)
    graph = _validate_graph_and_claims(sanitized, plan, issues)
    fuzzy = _validate_fuzzy(_mapping(sanitized.get("fuzzy")), plan, issues)
    rules = _validate_rule_interface(sanitized, plan, issues)
    alignment = _validate_alignment(_mapping(sanitized.get("disagreement")), plan, issues)
    risk = _validate_risk(_mapping(sanitized.get("risk")), plan, issues)
    native_model_replay = _validate_native_fuzzy_model(sanitized, plan, issues)
    cross_operator = _validate_cross_operator_consistency(sanitized, plan, issues)
    selected = _validate_representation(representation, coord, plan, _sequence(sanitized.get("diagnostics")), issues, sanitized)
    delta_from_provenance = _mapping(sanitized.get("disagreement")).get("delta")
    if representation.get("measured_delta") is not None and delta_from_provenance is not None:
        try:
            declared_delta = unit(representation["measured_delta"], "representation.measured_delta")
            executed_delta = unit(delta_from_provenance, "disagreement.delta")
            if abs(declared_delta - executed_delta) > plan.alignment_tolerance:
                _add(issues, "D_reduction_provenance", "representation.measured_delta", "declared reduction loss differs from executed operator")
        except ValueError as exc:
            _add(issues, "D_reduction_provenance_invalid", "representation.measured_delta", str(exc))
    if graph["missing_evidence"]:
        _add(issues, "D_missing_evidence_channels", "explanation_graph.missing_evidence",
             "graph declares missing evidence channels", incomplete=True)
    if coord is not None and coord["object"] == "causal":
        channels = _sequence(_mapping(sanitized.get("explanation_level")).get("available_channels"))
        if not any("causal" in str(channel).lower() or "intervention" in str(channel).lower() for channel in channels):
            _add(issues, "D_causal_claim", "taxonomy.object", "causal object is not backed by an intervention channel", incomplete=True)
    status = "invalid" if any(item.kind == "invalid" for item in issues) else "insufficient_evidence" if issues else "valid"
    action = _clean(_mapping(sanitized.get("risk")).get("action"))
    decision = {
        "schema_version": SCHEMA,
        "scope": "computational_route_compatibility_not_prediction_accuracy_or_causality",
        "status": status,
        "route_certified": status == "valid",
        "auto_accept_permitted": status == "valid" and action == "accept",
        "action": action or "insufficient_evidence",
        "taxonomy": coord,
        "trace": trace,
        "graph": graph,
        "fuzzy": fuzzy,
        "rule_interface": rules,
        "alignment": alignment,
        "risk": risk,
        "native_model_replay": native_model_replay,
        "cross_operator": cross_operator,
        "representation": selected,
        "issues": [item.to_dict() for item in issues],
        "diagnostic_cut": _diagnostic_cut(issues, repair_costs),
        "not_validated": ["source model correctness", "out-of-sample accuracy", "clinical safety", "causal identifiability"],
    }
    bindings = {"payload": sanitized, "taxonomy": dict(taxonomy), "representation": dict(representation),
                "policy": asdict(plan), "repair_costs": dict(repair_costs or {})}
    core = {"bindings": bindings, "decision": decision,
            "certifier_code_sha256": certifier_code_sha256()}
    return {**core, "sha256": sha256(core)}


def certify_result(result: Any, *, taxonomy: Mapping[str, Any], representation: Mapping[str, Any],
                   policy: Mapping[str, Any] | None = None, repair_costs: Mapping[str, float] | None = None) -> dict[str, Any]:
    return certify_payload(_extract_payload(result), taxonomy=taxonomy, representation=representation,
                           policy=policy, repair_costs=repair_costs)


def verify_certificate(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Recalculate ALL checks; reject tampering with any input or decision.

    SHA256 proves content equality to the received receipt only, not authorship.
    """
    try:
        bindings = _mapping(receipt["bindings"])
        expected = certify_payload(_mapping(bindings["payload"]), taxonomy=_mapping(bindings["taxonomy"]),
                                   representation=_mapping(bindings["representation"]),
                                   policy=_mapping(bindings["policy"]),
                                   repair_costs=_mapping(bindings.get("repair_costs")))
        digest_matches = hmac.compare_digest(str(receipt.get("sha256", "")), expected["sha256"])
        decision_matches = canonical_json(receipt.get("decision")) == canonical_json(expected["decision"])
        code_matches = hmac.compare_digest(str(receipt.get("certifier_code_sha256", "")), certifier_code_sha256())
        return {"valid": digest_matches and decision_matches and code_matches,
                "recomputed_status": expected["decision"]["status"],
                "digest_matches": digest_matches, "decision_matches": decision_matches,
                "certifier_code_matches": code_matches,
                "scope": "replay_and_integrity_not_authentication"}
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return {"valid": False, "error": str(exc), "scope": "replay_and_integrity_not_authentication"}


def project_for_audience(receipt: Mapping[str, Any], audience: str = "auditor") -> dict[str, Any]:
    """Different projections of ONE immutable result; never fresh inference."""
    if audience not in {"end_user", "developer", "auditor"}:
        raise ValueError("audience must be end_user, developer, auditor")
    checked = verify_certificate(receipt)
    if not checked["valid"]:
        raise ValueError("cannot present a tampered or unverifiable certificate")
    core = _mapping(receipt["decision"])
    base = {"status": core["status"], "action": core["action"],
            "auto_accept_permitted": core["auto_accept_permitted"],
            "scope": core["scope"], "issues": core["issues"]}
    if audience == "end_user":
        return {"audience": audience, **base, "limitation": "Техническая согласованность не подтверждает правильность прогноза"}
    if audience == "developer":
        return {"audience": audience, **base, "representation": core["representation"],
                "alignment": core["alignment"], "risk": core["risk"], "diagnostic_cut": core["diagnostic_cut"]}
    return {"audience": audience, **base, "certificate_sha256": receipt["sha256"],
            "trace": core["trace"], "taxonomy": core["taxonomy"], "graph": core["graph"],
            "representation": core["representation"], "alignment": core["alignment"],
            "risk": core["risk"], "diagnostic_cut": core["diagnostic_cut"],
            "not_validated": core["not_validated"]}


def compare_recertifications(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Record independent re-verification; does not execute repairs."""
    if not verify_certificate(before)["valid"] or not verify_certificate(after)["valid"]:
        raise ValueError("one or both certificates fail replay verification")
    a, b = _mapping(before["decision"]), _mapping(after["decision"])
    old = {(v["code"], v["path"]) for v in a["issues"]}
    new = {(v["code"], v["path"]) for v in b["issues"]}
    return {"before_sha256": before["sha256"], "after_sha256": after["sha256"],
            "changed_input": before["bindings"]["payload"] != after["bindings"]["payload"],
            "before_status": a["status"], "after_status": b["status"],
            "resolved": sorted([list(v) for v in old - new]),
            "remaining": sorted([list(v) for v in new]),
            "new_issues": sorted([list(v) for v in new - old]),
            "recertified": b["route_certified"],
            "scope": "independent_reverification_no_automatic_repair"}


__all__ = ["Taxonomy", "GatePolicy", "Violation", "fuzzy_and", "select_representation",
           "certify_payload", "certify_result", "verify_certificate", "project_for_audience",
           "compare_recertifications", "canonical_json", "sha256"]
