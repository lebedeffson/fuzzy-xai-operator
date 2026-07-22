from __future__ import annotations

from baselines.h10_gold import IndependentIfElse, TypedRouteWithoutReasoning
from fuzzyxai.audit_h10.gold_benchmark import FullH10GoldAuditor

from experiments.h10_gold.pipelines import pipeline_graphs


def _case(observed: dict) -> dict:
    clean = pipeline_graphs()["tabular_tree_a"]
    costs = {f"node:{node['id']}": node["repair_cost"] for node in clean["nodes"]}
    return {
        "case_id": "case-1",
        "pipeline_id": "tabular_tree_a",
        "modality": "tabular",
        "split": "development",
        "registered_graph": clean,
        "observed_graph": observed,
        "repair_costs": costs,
    }


def test_all_methods_accept_clean_route() -> None:
    clean = pipeline_graphs()["tabular_tree_a"]
    for method in (IndependentIfElse(), TypedRouteWithoutReasoning(), FullH10GoldAuditor()):
        assert method.diagnose(_case(clean))["route_status"] == "valid"


def test_full_h10_trace_is_deterministic() -> None:
    clean = pipeline_graphs()["tabular_tree_a"]
    observed = {**clean, "nodes": [dict(node) for node in clean["nodes"]]}
    observed["nodes"][1]["version"] = "stale-v2"
    method = FullH10GoldAuditor()
    first = method.diagnose(_case(observed))
    second = method.diagnose(_case(observed))
    assert first["trace"] == second["trace"]
    assert first["recertified"] is True
