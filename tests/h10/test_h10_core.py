from __future__ import annotations

from experiments.h10.mutations import mutate_route
from experiments.h10.routes import build_route
from fuzzyxai.audit_h10 import H10Auditor
from fuzzyxai.audit_h10.diagnostic_cut import DiagnosticCutSolver


def _auditor() -> H10Auditor:
    base = build_route("development", "tabular", "object-1")
    samples = []
    for leaf in ("hash_corruption", "preprocessing_mismatch", "stale_calibration", "broken_dependency"):
        for severity in ("subtle", "moderate", "severe"):
            route, _ = mutate_route(base, (leaf,), severity)
            samples.append((route, leaf))
    return H10Auditor.create(threshold_known=0.4, threshold_anomaly=1.0).fit(samples)


def test_exact_weighted_hitting_set_prefers_lower_total_cost() -> None:
    paths = (frozenset(("a", "b")), frozenset(("b", "c")))
    result = DiagnosticCutSolver().solve(paths, {"a": 1.0, "b": 3.0, "c": 1.0})
    assert result.optimal
    assert result.cut_nodes == ("a", "c")
    assert result.total_cost == 2.0


def test_large_graph_uses_approximate_solver_and_covers_every_path() -> None:
    paths = (frozenset(("a", "b")), frozenset(("b", "c")))
    result = DiagnosticCutSolver(exact_node_limit=1).solve(paths, {"a": 1.0, "b": 3.0, "c": 1.0})
    assert result.optimal is False
    assert result.solver == "greedy_weighted_hitting_set"
    assert result.covered_invalid_paths == len(paths)


def test_repair_is_recertified_and_trace_is_deterministic() -> None:
    base = build_route("development", "tabular", "object-2")
    route, _ = mutate_route(base, ("preprocessing_mismatch",), "moderate")
    auditor = _auditor()
    first = auditor.diagnose(route)
    second = auditor.diagnose(route)
    assert first.recertified is True
    assert first.trace == second.trace
    assert first.diagnostic_cut.cut_nodes == ("preprocessing_signature",)


def test_insufficient_evidence_abstains() -> None:
    base = build_route("development", "text", "object-3")
    route, _ = mutate_route(base, ("missing_source",), "severe", insufficient=True)
    result = _auditor().diagnose(route)
    assert result.route_status == "insufficient_evidence"
    assert result.fault.abstained_at_leaf
